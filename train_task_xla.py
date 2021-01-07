# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import json
import time
import datetime
import yaml
import random
import logging
import argparse
from io import open
from tqdm import tqdm
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.distributed as dist
# from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

from pytorch_transformers.optimization import AdamW, WarmupConstantSchedule, WarmupLinearSchedule

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
except ImportError:
    print('torch_xla package not found.')

from volta.config import BertConfig
from volta.optimization import RAdam
from volta.encoders import BertForVLTasks
from volta.train_utils_xla import freeze_layers, tbLogger, summary_parameters, save, resume
from volta.task_utils_xla import LoadDataset, LoadLoss, ForwardModelsTrain, ForwardModelsVal


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--from_pretrained", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--config_file", default="config/vilbert_base.json", type=str,
                        help="The config file which specified the model details.")
    parser.add_argument("--resume_file", default="", type=str,
                        help="Resume from checkpoint")
    # Output
    parser.add_argument("--output_dir", default="save", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--logdir", default="logs", type=str,
                        help="The logging directory where the training logs will be written.")
    parser.add_argument("--logfreq", default=50, type=int,
                        help="log onto tensorboard every n steps.")
    parser.add_argument("--save_name", default="", type=str,
                        help="save name for training.")
    # Task
    parser.add_argument("--tasks_config_file", default="config_tasks/vilbert_trainval_tasks.yml", type=str,
                        help="The config file which specified the tasks details.")
    parser.add_argument("--task", default="", type=str,
                        help="training task number")
    # Text
    parser.add_argument("--do_lower_case", action='store_true', default=True,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    # Training
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", dest="grad_acc_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--drop_last", action="store_true",
                        help="whether to drop last incomplete batch")
    # Scheduler
    parser.add_argument("--lr_scheduler", default="warmup_linear", type=str,
                        help="whether use learning rate scheduler.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_steps", default=None, type=float,
                        help="Number of training steps to perform linear learning rate warmup for. "
                             "It overwrites --warmup_proportion.")
    # Seed
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for initialization")
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers in the dataloader.")
    parser.add_argument("--in_memory", action="store_true",
                        help="whether use chunck for parallel training.")
    # Optimization
    parser.add_argument("--optim", default="AdamW", type=str,
                        help="what to use for the optimization.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_betas", default=(0.9, 0.999), nargs="+", type=float,
                        help="Betas for Adam optimizer.")
    parser.add_argument("--adam_correct_bias", default=False, action='store_true',
                        help="Correct bias for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay for Adam optimizer.")
    parser.add_argument("--clip_grad_norm", default=0.0, type=float,
                        help="Clip gradients within the specified range.")

    # Distributed (Torch_XLA)
    parser.add_argument("--tpu_ip_address", default=None, type=str,
                        help="tpu ip address")
    parser.add_argument("--tpu_port", default="8470", type=str,
                        help="port")
    parser.add_argument("--nprocs", type=int, default=8,
                        help="number of tpu cores (assuming one process one core")

    return parser.parse_args()


def train(device_id, args):
    # XLA Devices
    device = xm.xla_device()
    local_rank = xm.get_ordinal()
    n_tpu = xm.xrt_world_size()
    dist_is_available = True if n_tpu > 1 else False
    default_tpu = local_rank == 0
    logger.info(f"device: {device} n_tpu: {n_tpu}, distributed training: {dist_is_available}, "
                f"default_tpu: {default_tpu}, device_id: {device_id}")

    # Load config
    config = BertConfig.from_json_file(args.config_file)

    # Load task config
    with open(args.tasks_config_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))
    task_id = args.task.strip()
    task = "TASK" + task_id
    task_name = task_cfg[task]["name"]
    base_lr = task_cfg[task]["lr"]
    if task_cfg[task].get("fusion_method", None):
        # VL-BERT pooling for VQA
        config.fusion_method = task_cfg[task]["fusion_method"]

    # Output dirs
    if args.save_name:
        prefix = "-" + args.save_name
    else:
        prefix = ""

    now = datetime.datetime.fromtimestamp(time.time()).strftime('-%Y-%m%d-%H-%M')
    timestamp = (task_name + "_" + args.config_file.split("/")[1].split(".")[0] + prefix + now)
    save_path = os.path.join(args.output_dir, timestamp)
    if default_tpu:
        logger.info('tb log path: {}'.format(save_path))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save all the hidden parameters.
        with open(os.path.join(save_path, "command.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)

    # Seed
    random.seed(args.seed + device_id)
    np.random.seed(args.seed + device_id)
    torch.manual_seed(args.seed + device_id)

    # Dataset
    batch_size, task2num_iters, dset_train, dset_val, dl_train, dl_val, \
    sampler_train = LoadDataset(args, config, task_cfg, args.task, local_rank, world_size=n_tpu)

    # Logging
    logdir = os.path.join(args.logdir, timestamp)
    tb_logger = tbLogger(logdir, save_path, [task_name], [task], task2num_iters, args.grad_acc_steps,
                         save_logger=default_tpu)

    if not os.path.exists(args.output_dir) and default_tpu:
        os.makedirs(args.output_dir)

    # Model
    if "roberta" in args.bert_model:
        config.model = "roberta"
    model = BertForVLTasks.from_pretrained(args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task])
    # Move to TPU(s)
    model = model.to(device)

    if task_cfg[task].get("embed_clf", None):
        logger.info('Initializing classifier weight for %s from pretrained word embeddings...' % task)
        answers_word_embed = []
        for k, v in model.state_dict().items():
            if 'bert.embeddings.word_embeddings.weight' in k:
                word_embeddings = v.detach().clone()
                break
        for answer, label in sorted(dset_train.ans2label.items()):
            a_tokens = dset_train._tokenizer.tokenize(answer)
            a_ids = dset_train._tokenizer.convert_tokens_to_ids(a_tokens)
            if len(a_ids):
                a_word_embed = (torch.stack([word_embeddings[a_id] for a_id in a_ids], dim=0)).mean(dim=0)
            else:
                a_tokens = dset_train._tokenizer.tokenize("<unk>")
                a_id = dset_train._tokenizer.convert_tokens_to_ids(a_tokens)[0]
                a_word_embed = word_embeddings[a_id]
            answers_word_embed.append(a_word_embed)
        answers_word_embed_tensor = torch.stack(answers_word_embed, dim=0)
        for name, module in model.named_modules():
            if name.endswith('clfs_dict.%s.logit_fc.3' % task):
                module.weight.data = answers_word_embed_tensor.to(device=module.weight.data.device)

    # Optimization details
    freeze_layers(model)

    criterion = LoadLoss(task_cfg, args.task)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if "vil_" in key:
                lr = 1e-4
            else:
                lr = base_lr
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": 0.0}]
            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": args.weight_decay}]
    #if default_tpu:
    #    print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))
    if args.optim == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=base_lr,
                          eps=args.adam_epsilon,
                          betas=args.adam_betas,
                          correct_bias=args.adam_correct_bias)
    elif args.optim == "RAdam":
        optimizer = RAdam(optimizer_grouped_parameters, lr=base_lr)
    num_train_optim_steps = (task2num_iters[task] * args.num_train_epochs // args.grad_acc_steps)
    warmup_steps = args.warmup_steps or args.warmup_proportion * num_train_optim_steps
    if args.lr_scheduler == "warmup_linear":
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optim_steps)
    else:
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmup_steps)

    # Resume training
    start_iter_id, global_step, start_epoch, tb_logger, max_score = \
        resume(args.resume_file, model, optimizer, scheduler, tb_logger)

    # Pre-evaluate
    #evaluate(config, dl_val, task_cfg, device, task, model, criterion, start_epoch, default_tpu, tb_logger)

    # Save starting model
    save(save_path, logger, -1, model, optimizer, scheduler, global_step, tb_logger, default_tpu)

    # Print summary
    if default_tpu:
        summary_parameters(model, logger)

    print("***** rank {}: Running training *****".format(local_rank))
    print("  Num Iters: ", task2num_iters[task], flush=True)
    print("  Batch size: ", batch_size, flush=True)
    print("  Num steps: %d" % num_train_optim_steps, flush=True)

    # Train
    for epoch_id in tqdm(range(start_epoch, args.num_train_epochs), desc="Epoch"):
        model.train()
        sampler_train.set_epoch(epoch_id)
        #pl_loader = pl.ParallelLoader(dl_train, [device]).per_device_loader(device)
        for step, batch in tqdm(enumerate(dl_train)):
        #for step, batch in tqdm(enumerate(pl_loader)):
            iter_id = start_iter_id + step + (epoch_id * len(dl_train))
            #iter_id = start_iter_id + step + (epoch_id * len(pl_loader))

            loss, score = ForwardModelsTrain(config, task_cfg, device, task, batch, model, criterion)
            if args.grad_acc_steps > 1:
                loss = loss / args.grad_acc_steps
            loss.backward()

            if (step + 1) % args.grad_acc_steps == 0:
                # Clip gradient
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                xm.optimizer_step(optimizer, barrier=True)

                if global_step < warmup_steps or args.lr_scheduler == "warmup_linear":
                    scheduler.step()

                model.zero_grad()
                global_step += 1

                #if default_tpu and iter_id % args.logfreq == 0:
                if iter_id % args.logfreq == 0:
                    xm.add_step_closure(tb_logger.step_train, args=(epoch_id, iter_id, loss, score,
                                                                    optimizer.param_groups[0]["lr"], task, "train"))

                #tb_logger.step_train(epoch_id, iter_id, loss, score, optimizer.param_groups[0]["lr"], task, "train")

            #if (step % (20 * args.grad_acc_steps) == 0) and step != 0 and default_tpu:
            #    tb_logger.showLossTrain()

            # Decide whether to evaluate task
            if iter_id != 0 and iter_id % task2num_iters[task] == 0:
                score = evaluate(config, dl_val, task_cfg, device, task, model, criterion, epoch_id, default_tpu, tb_logger)
                if score > max_score:
                    max_score = score
                    save(save_path, logger, epoch_id, model, optimizer, scheduler,
                         global_step, tb_logger, default_tpu, max_score)

        save(save_path, logger, epoch_id, model, optimizer, scheduler, global_step, tb_logger, default_tpu, max_score)

    if default_tpu:
        tb_logger.txt_close()


def node_run(device_id, args):
    train(device_id, args)
    xm.rendezvous('finishing up')

def evaluate(config, dataloader_val, task_cfg, device, task_id, model, criterion, epoch_id, default_tpu, tb_logger):
    model.eval()
    pl_loader = pl.ParallelLoader(dataloader_val, [device]).per_device_loader(device)
    for i, batch in tqdm(enumerate(pl_loader)):
        loss, score, batch_size = ForwardModelsVal(config, task_cfg, device, task_id, batch, model, criterion)
        xm.mark_step()
        if tb_logger:
            tb_logger.step_val(epoch_id, loss, score, task_id, batch_size, "val")

    score = -1
    if tb_logger:
        score = tb_logger.showLossVal(task_id)

    model.train()
    return score


if __name__ == "__main__":
    args = parse_args()
    assert args.tpu_ip_address is not None
    assert args.tpu_port is not None
    print('Using TPU {}, port: {}'.format(args.tpu_ip_address, args.tpu_port))
    print('Spawning {} process(es) (1 process 1 tpu core)'.format(args.nprocs))
    os.environ["XRT_TPU_CONFIG"] = f"tpu_worker;0;{args.tpu_ip_address}:{args.tpu_port}"
    os.environ['XLA_USE_BF16'] = "0"
    torch.set_default_tensor_type('torch.FloatTensor')
    xmp.spawn(node_run, args=(args,), nprocs=args.nprocs, start_method='fork')
