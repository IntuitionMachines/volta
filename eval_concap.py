# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import json
import time
import datetime
import random
import logging
import argparse
import pickle
import gc
from io import open

import numpy as np

from tqdm import tqdm
import torch
import torch.distributed as dist

from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

from volta.config import BertConfig
from volta.datasets import ConceptCapLoaderTrain, ConceptCapLoaderVal
from volta.train_utils import freeze_layers, tbLogger, summary_parameters, save, resume


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--annotations_path", default="datasets/conceptual_caption/annotations", type=str,
                        help="The corpus annotations directory.")
    parser.add_argument("--features_path", default="datasets/conceptual_caption/imgfeats", type=str,
                        help="The corpus image features directory.")
    # Model
    parser.add_argument("--from_pretrained", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, roberta-base, ...")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased, ...")
    parser.add_argument("--config_file", type=str, default="config/vilbert_base.json",
                        help="The config file which specified the model details.")
    parser.add_argument("--resume_file", default="", type=str,
                        help="Resume from checkpoint")
    parser.add_argument("--resume_from_next_epoch", action="store_true")

    # Output
    parser.add_argument("--output_dir", default="checkpoints", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--logdir", default="logs", type=str,
                        help="The logging directory where the training logs will be written.")
    # Text
    parser.add_argument("--max_seq_length", default=36, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case", action='store_true', default=True,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    # Training
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", dest="grad_acc_steps", type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    # Scheduler
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_steps", default=None, type=float,
                        help="Number of training steps to perform linear learning rate warmup for. "
                             "It overwrites --warmup_proportion.")
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization")
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus.")
    parser.add_argument("--num_workers", type=int, default=25,
                        help="Number of workers in the dataloader.")
    parser.add_argument("--distributed", action="store_true",
                        help="whether use chunck for parallel training.")
    # Objective
    parser.add_argument("--objective", default=0, type=int,
                        help="Which objective to use \n"
                             "0: with ITM loss, \n"
                             "1: with ITM loss; for the not aligned pair, no masking objective, \n"
                             "2: without ITM loss, do not sample negative pair, \n"
                             "3: without ITM loss, but still sample both positive and negative pairs, domain confusion \n"
                             "4: without ITM loss, ALWAYS sample negative pairs, domain confusion")
    parser.add_argument("--add_multi_label_loss_t", action="store_true")
    parser.add_argument("--add_multi_label_loss_v", action="store_true")
    parser.add_argument("--add_kl_entropy_reg", action="store_true")
    parser.add_argument("--add_kl_dist_matching", action="store_true")
    parser.add_argument("--add_gradient_penalty", action="store_true")
    parser.add_argument("--aligned_pairs_proportion", default=1.0, type=float,
                        help="how much proportion of aligned data will be used in training")

    # UDA Training Specifics
    parser.add_argument("--freeze_bert", action="store_true",
                        help="whether freeze bert when training generator")
    parser.add_argument("--caption_availability", default=1., type=float,
                        help="to use how many portion of image-text pairs")

    # Optimizer
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_betas", default=(0.9, 0.98), nargs="+", type=float,
                        help="Betas for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay for Adam optimizer.")
    parser.add_argument("--clip_grad_norm", default=0.0, type=float,
                        help="Clip gradients within the specified range.")

    return parser.parse_args()

def count_objects(obj_labels):
    obj_labels = obj_labels.tolist()
    obj_labels = [item for sublist in obj_labels for item in sublist]   # flattened list
    return {item: obj_labels.count(item) for item in set(obj_labels)}

def main():
    args = parse_args()
    # Devices
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")  # Init distributed backend for sychronizing nodes/GPUs
        args.local_rank = local_rank

    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True
    logger.info(f"device: {device} n_gpu: {n_gpu}, distributed training: {bool(args.local_rank != -1)}, "
                f"local_rank: {args.local_rank}")

    # Load config
    config = BertConfig.from_json_file(args.config_file)
    config.objective = args.objective

    cache = 5000
    args.train_batch_size = args.train_batch_size // args.grad_acc_steps
    if dist.is_available() and args.local_rank != -1:
        num_replicas = dist.get_world_size()
        args.train_batch_size = args.train_batch_size // num_replicas
        args.num_workers = args.num_workers // num_replicas
        cache = cache // num_replicas
        args.num_train_epochs = args.num_train_epochs // num_replicas
        args.seed = args.seed + args.local_rank

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Datasets
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    valid_dataset = ConceptCapLoaderVal(args.annotations_path, args.features_path, tokenizer, args.bert_model,
                                        seq_len=args.max_seq_length, batch_size=args.train_batch_size, num_workers=8,
                                        objective=args.objective, add_global_imgfeat=config.add_global_imgfeat,
                                        num_locs=config.num_locs,
                                        visual_target_categories_file=config.visual_target_categories_file,
                                        caption_availability=1.)

    # Model
    if config.image_embeddings == 'mix_uniter':
        from volta.encoders_mm import BertForVLPreTraining
    else:
        from volta.encoders import BertForVLPreTraining

    if args.from_pretrained:
        type_vocab_size = config.type_vocab_size
        config.type_vocab_size = 2
        model = BertForVLPreTraining.from_pretrained(args.from_pretrained, config=config,
                                                     default_gpu=default_gpu, from_hf=True)
        # Resize type embeddings
        model.bert.embeddings.token_type_embeddings = \
            model._get_resized_embeddings(model.bert.embeddings.token_type_embeddings, type_vocab_size)
        config.type_vocab_size = type_vocab_size
    else:
        model = BertForVLPreTraining(config)

    # Optimization details
    freeze_layers(model)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    bert_weight_name = json.load(open("config/" + args.from_pretrained + "_weight_name.json", "r"))
    if not args.from_pretrained:
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
    else:
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if key[12:] in bert_weight_name:
                    lr = args.learning_rate * 0.1
                else:
                    lr = args.learning_rate

                if any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": 0.0}]

                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": args.weight_decay}]
        if default_gpu:
            print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=args.adam_betas)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=1, t_total=1)

    # Resume training
    start_iter_id, global_step, start_epoch, tb_logger, _ = \
        resume(args.resume_file, model, optimizer, scheduler, None,
               resume_from_next_epoch=args.resume_from_next_epoch,
               loading_only_for_eval=True)

    # Move to GPU(s)
    #model.cuda()
    device = torch.cuda.current_device()
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                #state[k] = v.cuda()
                state[k] = v.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Save starting model
    #save(save_path, logger, -1, model, optimizer, scheduler, global_step, tb_logger, default_gpu)

    # Print summary
    if default_gpu:
        summary_parameters(model, logger)

    add_domain_confusion_loss = False
    if args.objective == 3 or args.objective == 4:
        add_domain_confusion_loss = True

    # Do the evaluation
    torch.set_grad_enabled(False)
    numBatches = len(valid_dataset)
    model.eval()

    obj_counts = {}
    out_dir = os.path.basename(os.path.dirname(args.resume_file))
    out_dir = os.path.join(os.getcwd(), 'logs/representations/{}'.format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for step, batch in tqdm(enumerate(valid_dataset)):
        if step > 5:
            break
        image_ids = batch[-1]
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch[:-1])
        if len(batch) == 16:
            input_ids, input_mask, segment_ids, lm_label_ids, is_match, \
            image_feat, image_loc, image_cls, obj_labels, obj_confs, \
            attr_labels, attr_confs, image_attrs, image_label, image_mask, caption_avail = batch
        elif len(batch) == 18:
            input_ids, input_mask, segment_ids, lm_label_ids, is_match, \
            image_feat, image_loc, image_cls, obj_labels, obj_confs, \
            attr_labels, attr_confs, image_attrs, image_label, image_mask, \
            obj_tokens, attr_tokens, caption_avail = batch
        else:
            raise ValueError

        # counting how many objects in each VG class
        for k, v in count_objects(obj_labels).items():
            if k in obj_counts:
                obj_counts[k] += v
            else:
                obj_counts[k] = v

        sequence_output_t, \
        sequence_output_v, \
        pooled_output_t, \
        pooled_output_v, \
        all_attention_mask, \
        discriminator_score_t, \
        discriminator_score_v = model.module.get_representations(input_ids,
                                                                 image_feat,
                                                                 image_loc,
                                                                 segment_ids,
                                                                 input_mask,
                                                                 image_mask,
                                                                 lm_label_ids,
                                                                 image_label,
                                                                 image_cls,
                                                                 obj_labels,
                                                                 obj_confs,
                                                                 attr_labels,
                                                                 attr_confs,
                                                                 image_attrs,
                                                                 is_match,
                                                                 caption_avail,
                                                                 output_all_attention_masks=True,
                                                                 output_all_encoded_layers=True,
                                                                 add_multi_label_loss_t=args.add_multi_label_loss_t,
                                                                 add_multi_label_loss_v=args.add_multi_label_loss_v,
                                                                 add_domain_confusion_loss=add_domain_confusion_loss,
                                                                 add_kl_entropy_reg=args.add_kl_entropy_reg,
                                                                 add_kl_dist_matching=args.add_kl_dist_matching)

        att_masks_t, att_masks_v = all_attention_mask
        layers_of_interest = [0, 4, 8, 11]

        '''att_masks_tx = [mask['intra_attn'].cpu().numpy() for i, mask in enumerate(att_masks_t)
                        if i in layers_of_interest]     # 12 * (batch, num_TE, n_tokens, n_tokens)
        att_masks_vx = [mask['intra_attn'].cpu().numpy() for i, mask in enumerate(att_masks_v)
                        if i in layers_of_interest]     # 12 * (batch, num_TE, n_tokens, n_tokens)'''

        # store masked object representations
        #lm_label_ids[:, 0] = 1  # keep [CLS]
        lm_keep = lm_label_ids != -1
        sequence_output_tx = [seq.cpu().numpy() for i, seq in enumerate(sequence_output_t)
                              if i in layers_of_interest]
        #ones = torch.ones([image_label.shape[0], 1], device=image_label.device, dtype=image_label.dtype)
        #image_label = torch.cat([ones, image_label], dim=1)
        im_keep = image_label == 1
        sequence_output_vx = [seq.cpu().numpy() for i, seq in enumerate(sequence_output_v)
                              if i in layers_of_interest]  # 12 * (batch, n_regions, 768)

        '''sequence_output_tx = [seq.cpu().numpy() for i, seq in enumerate(sequence_output_t)
                              if i in layers_of_interest]   # 12 * (batch, n_tokens, 768)
        sequence_output_vx = [seq.cpu().numpy() for i, seq in enumerate(sequence_output_v)
                              if i in layers_of_interest]   # 12 * (batch, n_regions, 768)'''
        output_dict = {'sequence_output_tx': sequence_output_tx,    # (batch, n_tokens, dim)
                       'sequence_output_vx': sequence_output_vx,    # (batch, n_tokens, dim)
                       'obj_labels': obj_labels.cpu().numpy(),
                       'image_label': image_label.cpu().numpy(),
                       'lm_label_ids': lm_label_ids.cpu().numpy(),
                       'input_ids': input_ids.cpu().numpy(),
                       'is_match': is_match.cpu().numpy(),
                       'discriminator_score_t': discriminator_score_t[:, 0].cpu().numpy(),
                       'discriminator_score_v': discriminator_score_v[:, 0].cpu().numpy(),
                       'image_ids': image_ids}

        out_name = os.path.join(out_dir, '%05d.pkl' % step)
        with open(out_name, 'wb') as f:
            pickle.dump(output_dict, f, pickle.HIGHEST_PROTOCOL)
        '''
        store sequence_output_t, sequence_output_v, obj_labels, image_feat, all_attention_mask?
        '''

        '''batch_size = input_ids.size(0)
        masked_loss_t, masked_loss_v, pair_match_loss, discriminator_loss_t, discriminator_loss_v, \
        multilabel_loss_t, multilabel_loss_v, kl_entropy_t, kl_entropy_v, knn_kldiv = \
            model(input_ids, image_feat, image_loc, segment_ids, input_mask, image_mask, lm_label_ids, image_label,
                  image_cls, obj_labels, obj_confs, attr_labels, attr_confs, image_attrs, is_match, caption_avail,
                  add_multi_label_loss_t=args.add_multi_label_loss_t,
                  add_multi_label_loss_v=args.add_multi_label_loss_v,
                  add_domain_confusion_loss=add_domain_confusion_loss,
                  add_kl_entropy_reg=args.add_kl_entropy_reg,
                  add_kl_dist_matching=args.add_kl_dist_matching)

        loss = config.masked_loss_t_weight * masked_loss_t + \
               config.masked_loss_v_weight * masked_loss_v
        if args.objective == 3:
            loss += (pair_match_loss + config.adversarial_weight * (discriminator_loss_t + discriminator_loss_v))
        elif args.objective == 4:
            pair_match_loss = pair_match_loss.detach()
            loss += config.adversarial_weight * (discriminator_loss_t + discriminator_loss_v)
        else:
            loss += pair_match_loss

        if args.add_multi_label_loss_t:
            loss += multilabel_loss_t

        if args.add_multi_label_loss_v:
            loss += multilabel_loss_v

        if args.add_kl_entropy_reg:
            loss += config.kl_entropy_weight * (kl_entropy_t + kl_entropy_v)

        if args.add_kl_dist_matching:
            loss += config.knn_kl_weight * knn_kldiv

        if n_gpu > 1:
            loss = loss.mean()
            masked_loss_t = masked_loss_t.mean()
            masked_loss_v = masked_loss_v.mean()
            pair_match_loss = pair_match_loss.mean()
            if args.objective == 3 or args.objective == 4:
                discriminator_loss_t = discriminator_loss_t.mean()
                discriminator_loss_v = discriminator_loss_v.mean()
            if args.add_multi_label_loss_t:
                multilabel_loss_t = multilabel_loss_t.mean()
            if args.add_multi_label_loss_v:
                multilabel_loss_v = multilabel_loss_v.mean()
            if args.add_kl_entropy_reg:
                kl_entropy_t = kl_entropy_t.mean()
                kl_entropy_v = kl_entropy_v.mean()
            if args.add_kl_dist_matching:
                knn_kldiv = knn_kldiv.mean()'''

    sorted_obj_counts = {k: v for k, v in sorted(obj_counts.items(), key=lambda item: -item[1])}    # obj_id: [0, 1599]
    print(sorted_obj_counts)

    out_name = os.path.join(out_dir, 'sorted_obj_counts.pkl')
    with open(out_name, 'wb') as f:
        pickle.dump(sorted_obj_counts, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
