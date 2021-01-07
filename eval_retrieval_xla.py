# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import yaml
import random
import logging
import argparse
from io import open
from tqdm import tqdm
from easydict import EasyDict as edict
import gc
import pickle

import numpy as np

import torch

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
except ImportError:
    print('torch_xla package not found.')

from volta.config import BertConfig
from volta.encoders import BertForVLTasks, BertForVLPreTraining
from volta.task_utils_xla import LoadDatasetEval

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

mp_outformat = 'results_{}.pkl'

def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--from_pretrained", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--config_file", default="config/bert_config.json", type=str,
                        help="The config file which specified the model details.")
    # Output
    parser.add_argument("--output_dir", default="results", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--save_name", default="", type=str,
                        help="save name for training.")
    # Task
    parser.add_argument("--tasks_config_file", default="config_tasks/vilbert_trainval_tasks.yml", type=str,
                        help="The config file which specified the tasks details.")
    parser.add_argument("--task", default="", type=str,
                        help="training task number")
    # Text
    parser.add_argument("--do_lower_case", default=True, type=bool,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    # Evaluation
    parser.add_argument("--split", default="", type=str,
                        help="which split to use.")
    parser.add_argument("--zero_shot", action="store_true",
                        help="Zero-shot evaluation.")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="batch size.")
    parser.add_argument("--drop_last", action="store_true",
                        help="whether to drop last incomplete batch")
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers in the dataloader.")
    parser.add_argument("--in_memory", default=False, type=bool,
                        help="whether use chunck for parallel training.")
    parser.add_argument("--use_chunk", default=0, type=float,
                        help="whether use chunck for parallel training.")

    # Distributed (Torch_XLA)
    parser.add_argument("--tpu_ip_address", default=None, type=str,
                        help="tpu ip address")
    parser.add_argument("--tpu_port", default="8470", type=str,
                        help="port")
    parser.add_argument("--nprocs", type=int, default=8,
                        help="number of tpu cores (assuming one process one core")

    return parser.parse_args()


def forward(args, batch, model, task, device):
    with torch.no_grad():
        batch = tuple(t.to(device) for t in batch)
        #batch, features, spatials, image_mask = batch
        #question, input_mask, segment_ids, target, caption_idx, image_idx = batch
        features, spatials, image_mask, question, input_mask, segment_ids = batch

        features = features.squeeze(0)
        spatials = spatials.squeeze(0)
        image_mask = image_mask.squeeze(0)
        question = question.repeat(features.shape[0], 1)
        segment_ids = segment_ids.repeat(features.shape[0], 1)
        input_mask = input_mask.repeat(features.shape[0], 1)

        if args.zero_shot:
            _, _, vil_logit, _, _ = model(question, features, spatials, segment_ids, input_mask, image_mask)
            return torch.softmax(vil_logit, dim=1)[:, 0].view(-1)
        else:
            vil_logit, _, _, _ = model(question, features, spatials, task, segment_ids, input_mask, image_mask)
            return vil_logit.view(-1)

def collect_results(score_matrix, target_matrix, num_images, num_captions):
    rank_matrix = np.ones(num_captions) * num_images
    results = []
    for caption_idx in range(num_captions):
        argsorted_score = np.argsort(-score_matrix[caption_idx])
        rank = np.where(
            (
                    argsorted_score == np.where(target_matrix[caption_idx] == 1)[0][0]
            )
            == 1
        )[0][0]
        rank_matrix[caption_idx] = rank

        results.append(argsorted_score.tolist()[:20])

    # Report final metrics
    r1 = 100.0 * np.sum(rank_matrix < 1) / len(rank_matrix)
    r5 = 100.0 * np.sum(rank_matrix < 5) / len(rank_matrix)
    r10 = 100.0 * np.sum(rank_matrix < 10) / len(rank_matrix)

    medr = np.floor(np.median(rank_matrix) + 1)
    meanr = np.mean(rank_matrix) + 1

    print("************************************************", flush=True)
    print("****************Image Retrieval*****************", flush=True)
    print("************************************************", flush=True)
    print(
        "Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
        % (r1, r5, r10, medr, meanr)
    )
    print("************************************************", flush=True)

    # Text Retrieval
    rank_matrix = np.zeros(num_images)
    for image_idx in range(num_images):
        ranks = []
        tgt_captions = np.where(target_matrix[:, image_idx] == 1)[0]
        sorted_scores = np.argsort(-score_matrix[:, image_idx])
        for tgt_caption in tgt_captions:
            ranks.append(np.where((sorted_scores == tgt_caption) == 1)[0][0])
        rank_matrix[image_idx] = min(ranks)

    r1 = 100.0 * np.sum(rank_matrix < 1) / len(rank_matrix)
    r5 = 100.0 * np.sum(rank_matrix < 5) / len(rank_matrix)
    r10 = 100.0 * np.sum(rank_matrix < 10) / len(rank_matrix)

    medr = np.floor(np.median(rank_matrix) + 1)
    meanr = np.mean(rank_matrix) + 1

    print("************************************************", flush=True)
    print("****************Text Retrieval******************", flush=True)
    print("************************************************", flush=True)
    print(
        "Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
        % (r1, r5, r10, medr, meanr)
    )
    print("************************************************", flush=True)

def collect_results_on_device(score_matrix, target_matrix, num_images, num_captions):
    rank_matrix = torch.ones(num_captions) * num_images
    for caption_idx in tqdm(range(num_captions)):
        argsorted_score = torch.argsort(-score_matrix[caption_idx])
        # only one target == 1 in target_matrix
        cond_vector = (argsorted_score == torch.where(target_matrix[caption_idx] == 1)[0][0])
        # xm.master_print("argsorted_score {}".format(argsorted_score))
        rank = torch.where(cond_vector == 1)[0][0]
        rank_matrix[caption_idx] = rank
    # ********************************
    # Report final metrics
    r1 = 100.0 * torch.sum(rank_matrix < 1) / len(rank_matrix)
    r5 = 100.0 * torch.sum(rank_matrix < 5) / len(rank_matrix)
    r10 = 100.0 * torch.sum(rank_matrix < 10) / len(rank_matrix)

    medr = torch.floor(torch.median(rank_matrix) + 1)
    meanr = torch.mean(rank_matrix) + 1

    print("************************************************", flush=True)
    print("****************Image Retrieval*****************", flush=True)
    print("************************************************", flush=True)
    print(
        "Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
        % (r1, r5, r10, medr, meanr)
    )
    print("************************************************", flush=True)

def run_eval(device_id, args):
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

    # Output dirs
    if "/" in args.from_pretrained:
        timeStamp = args.from_pretrained.split("/")[1]
    else:
        timeStamp = args.from_pretrained
    savePath = os.path.join(args.output_dir, timeStamp)
    if default_tpu and not os.path.exists(savePath):
        os.makedirs(savePath)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset
    batch_size, task2num_iters, dset_val, dl_val = LoadDatasetEval(args, config, task_cfg, args.task,
                                                                   local_rank, world_size=n_tpu)

    # Model
    if args.zero_shot:
        config.visual_target_weights = [0, 0, 0, 0, 0, 0, 0]
        model = BertForVLPreTraining.from_pretrained(args.from_pretrained, config=config)
    else:
        model = BertForVLTasks.from_pretrained(args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task])

    model = xmp.MpModelWrapper(model)
    # Move to TPU(s)
    model = model.to(device)
    # Evaluate
    model.eval()

    num_captions = len(dset_val._caption_entries)
    num_images = len(dset_val._image_entries)

    score_matrix = np.zeros((num_captions, num_images))
    target_matrix = np.zeros((num_captions, num_images)).astype(np.bool)
    caption_visited = np.zeros((num_captions, num_images))

    # Print summary
    # Get parallel loader
    loader = dl_val
    #if n_tpu > 1:
    #    loader = pl.ParallelLoader(loader, [device]).per_device_loader(device)

    for i, batch in tqdm(enumerate(loader)):
        features, spatials, image_mask, question, input_mask, segment_ids, target, caption_idx, image_idx = batch
        target = target[0]
        caption_idx = caption_idx[0]
        image_idx = image_idx[0]

        half_num_images = int(num_images / 2)
        vil_logit = forward(args, tuple(batch[0:6]), model, task, device)

        xm.mark_step()
        score_matrix[caption_idx, image_idx * half_num_images:(image_idx + 1) * half_num_images] = vil_logit.cpu().numpy()
        target_matrix[caption_idx, image_idx * half_num_images:(image_idx + 1) * half_num_images] = (target == 1).cpu().numpy()
        caption_visited[caption_idx, image_idx * half_num_images: (image_idx + 1) * half_num_images] += 1

    # Collect stats from TPUs
    outname = mp_outformat.format(local_rank)
    outpath = os.path.join(args.output_dir, outname)
    outdict = {'score_matrix': score_matrix, 'target_matrix': target_matrix, 'caption_visited': caption_visited,
               'num_images': num_images, 'num_captions': num_captions}
    print('Dumping {}'.format(outpath))
    with open(outpath, 'wb') as fp:
        pickle.dump(outdict, fp)

    '''if device_id == 0:
            if args.split:
                json_path = os.path.join(savePath, args.split)
            else:
                json_path = os.path.join(savePath, task_cfg[task_id]["val_split"])
            json.dump(results, open(json_path + "_result.json", "w"))
            json.dump(others, open(json_path + "_others.json", "w"))'''

    return outpath

def node_run(device_id, args):
    run_eval(device_id, args)

def collect_results_from_mp(args):
    for i in range(args.nprocs):
        filepath = os.path.join(args.output_dir, mp_outformat.format(i))
        with open(filepath, 'rb') as fp:
            result_dict = pickle.load(fp)

        if i == 0:
            score_matrix = result_dict['score_matrix']
            target_matrix = result_dict['target_matrix']
            caption_visited = result_dict['caption_visited']
            num_images = result_dict['num_images']
            num_captions = result_dict['num_captions']
        else:
            score_matrix += result_dict['score_matrix']
            caption_visited += result_dict['caption_visited']
            target_matrix |= result_dict['target_matrix']

    caption_visited = np.maximum(caption_visited, 1)
    score_matrix /= caption_visited
    collect_results(score_matrix, target_matrix, num_images, num_captions)


if __name__ == "__main__":
    args = parse_args()
    assert args.tpu_ip_address is not None
    assert args.tpu_port is not None
    print('Using TPU {}, port: {}'.format(args.tpu_ip_address, args.tpu_port))
    print('Spawning {} process(es) (1 process 1 tpu core)'.format(args.nprocs))
    os.environ["XRT_TPU_CONFIG"] = f"tpu_worker;0;{args.tpu_ip_address}:{args.tpu_port}"
    os.environ['XLA_USE_BF16'] = "0"
    xmp.spawn(node_run, args=(args,), nprocs=args.nprocs)

    collect_results_from_mp(args)

