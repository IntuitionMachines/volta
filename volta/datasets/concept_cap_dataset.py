# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import random
import logging

import numpy as np
import tensorpack.dataflow as td

import torch
import torch.distributed as dist
from datasets import load_dataset

import msgpack_numpy
msgpack_numpy.patch()

MAX_MSGPACK_LEN = 1000000000

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    gt_boxes_area = (
            (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).reshape(1, K)

    anchors_area = (
            (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    ).reshape(N, 1)

    boxes = np.repeat(anchors.reshape(N, 1, 4), K, axis=1)
    query_boxes = np.repeat(gt_boxes.reshape(1, K, 4), N, axis=0)

    iw = (
            np.minimum(boxes[:, :, 2], query_boxes[:, :, 2])
            - np.maximum(boxes[:, :, 0], query_boxes[:, :, 0])
            + 1
    )
    iw[iw < 0] = 0

    ih = (
            np.minimum(boxes[:, :, 3], query_boxes[:, :, 3])
            - np.maximum(boxes[:, :, 1], query_boxes[:, :, 1])
            + 1
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(
        self,
        image_feat=None,
        image_cls=None,
        obj_labels=None,
        obj_confs=None,
        attr_labels=None,
        attr_confs=None,
        image_attrs=None,
        caption=None,
        is_next=None,
        lm_labels=None,
        image_loc=None,
        num_boxes=None,
        overlaps=None,
        obj_tokens=None,
        attr_tokens=None,
        caption_avail_label=None
    ):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.image_feat = image_feat
        self.caption = caption
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model
        self.image_loc = image_loc
        self.image_cls = image_cls
        self.obj_labels = obj_labels    # (label, conf)
        self.obj_confs = obj_confs
        self.attr_labels = attr_labels  # (label, conf)
        self.attr_confs = attr_confs
        self.image_attrs = image_attrs
        self.num_boxes = num_boxes
        self.overlaps = overlaps
        self.obj_tokens = obj_tokens
        self.attr_tokens = attr_tokens
        self.caption_avail_label = caption_avail_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids=None,
        input_mask=None,
        segment_ids=None,
        is_next=None,
        lm_label_ids=None,
        image_feat=None,
        image_cls=None,
        obj_labels=None,
        obj_confs=None,
        attr_labels=None,
        attr_confs=None,
        image_attrs=None,
        image_loc=None,
        image_label=None,
        image_mask=None,
        obj_tokens=None,
        attr_tokens=None,
        masked_label=None,
        caption_avail_label=None
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids
        self.image_feat = image_feat
        self.image_loc = image_loc
        self.image_label = image_label
        self.image_cls = image_cls
        self.obj_labels = obj_labels
        self.obj_confs = obj_confs
        self.attr_labels = attr_labels
        self.attr_confs = attr_confs
        self.image_attrs = image_attrs
        self.image_mask = image_mask
        self.obj_tokens = obj_tokens
        self.attr_tokens = attr_tokens
        self.masked_label = masked_label
        self.caption_avail_label = caption_avail_label


class ConceptCapLoaderTrain(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the 
            GPU, which is faster).
    """

    def __init__(
        self,
        annotations_path,
        features_path,
        tokenizer,
        bert_model,
        seq_len,
        batch_size=512,
        num_workers=25,
        cache=10000,
        local_rank=-1,
        objective=0,
        num_locs=5,
        add_global_imgfeat=None,
        visual_target_categories_file=None,
        caption_availability=1.,
        remove_CLS_token=False,
        remove_SEP_token=False,
    ):
        if dist.is_available() and local_rank != -1:
            rank = dist.get_rank()
            #lmdb_file = os.path.join(features_path, "training_feat_part_" + str(rank) + ".lmdb")
            lmdb_file = os.path.join(features_path, "training_feat_all.lmdb")
        else:
            lmdb_file = os.path.join(features_path, "training_feat_all.lmdb")

            print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)
        ds = td.LocallyShuffleData(ds, cache)
        caption_path = os.path.join(annotations_path, "caption_train.json")

        self.caption_availability_dict = self._label_caption_availability(caption_path, caption_availability)
        preprocess_function = BertPreprocessBatch(
            caption_path,
            tokenizer,
            bert_model,
            seq_len,
            36,
            self.num_dataset,
            objective=objective,
            num_locs=num_locs,
            visual_target_categories_file=visual_target_categories_file,
            caption_availability=self.caption_availability_dict,
            ext_corpus=[],
            remove_CLS_token=remove_CLS_token,
            remove_SEP_token=remove_SEP_token
        )

        ds = td.PrefetchData(ds, 10000, 1)
        ds = td.MapData(ds, preprocess_function)

        ds = td.PrefetchDataZMQ(ds, num_workers)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.add_global_imgfeat = add_global_imgfeat
        self.num_locs = num_locs
        self.tokenize_visual_categories = preprocess_function.tokenize_visual_categories
        self.preprocess_function = preprocess_function

    def __iter__(self):
        for batch in self.ds.get_data():
            # obj_labels \in [0, 1599], "background" class not included
            if self.tokenize_visual_categories:
                input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, \
                    image_cls, obj_labels, obj_confs, attr_labels, attr_confs, image_attrs, image_label, \
                image_mask, masked_label, obj_tokens, attr_tokens, caption_avail, image_id = batch
            else:
                input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, \
                image_cls, obj_labels, obj_confs, attr_labels, attr_confs, image_attrs, image_label, \
                image_mask, masked_label, caption_avail, image_id = batch

            batch_size = input_ids.shape[0]

            if self.add_global_imgfeat == "first":
                sum_count = np.sum(masked_label == 0, axis=1, keepdims=True)
                sum_count[sum_count == 0] = 1
                g_image_feat = np.sum(image_feat, axis=1) / sum_count
                image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), image_feat], axis=1)
                image_feat = np.array(image_feat, dtype=np.float32)

                g_loc = [0, 0, 1, 1] + [1]*(self.num_locs - 4)
                g_image_loc = np.repeat(np.array([g_loc], dtype=np.float32), batch_size, axis=0)
                image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=1), image_loc], axis=1)

                image_loc = np.array(image_loc, dtype=np.float32)
                g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
                image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            elif self.add_global_imgfeat == "last":
                sum_count = np.sum(masked_label == 0, axis=1, keepdims=True)
                sum_count[sum_count == 0] = 1
                g_image_feat = np.sum(image_feat, axis=1) / sum_count
                image_feat = np.concatenate([image_feat, np.expand_dims(g_image_feat, axis=1)], axis=1)
                image_feat = np.array(image_feat, dtype=np.float32)

                g_loc = [0, 0, 1, 1] + [1]*(self.num_locs - 4)
                g_image_loc = np.repeat(np.array([g_loc], dtype=np.float32), batch_size, axis=0)
                image_loc = np.concatenate([image_loc, np.expand_dims(g_image_loc, axis=1)], axis=1)

                image_loc = np.array(image_loc, dtype=np.float32)
                g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
                image_mask = np.concatenate([image_mask, g_image_mask], axis=1)

            if self.tokenize_visual_categories:
                batch = (
                    input_ids,
                    input_mask,
                    segment_ids,
                    lm_label_ids,
                    is_next,
                    image_feat,
                    image_loc,
                    image_cls,
                    obj_labels,
                    obj_confs,
                    attr_labels,
                    attr_confs,
                    image_attrs,
                    image_label,
                    image_mask,
                    obj_tokens,
                    attr_tokens,
                    caption_avail,
                    image_id
                )
            else:
                batch = (
                    input_ids,
                    input_mask,
                    segment_ids,
                    lm_label_ids,
                    is_next,
                    image_feat,
                    image_loc,
                    image_cls,
                    obj_labels,
                    obj_confs,
                    attr_labels,
                    attr_confs,
                    image_attrs,
                    image_label,
                    image_mask,
                    caption_avail,
                    image_id
                )

            yield batch
            #yield [torch.tensor(data) for data in batch] + [image_id]

    def __len__(self):
        return self.ds.size()

    def _label_caption_availability(self, caption_path, caption_availability):
        image_ids = list(json.load(open(caption_path, "r")).keys())
        num_captions = len(image_ids)
        availability = (np.random.rand(num_captions) < caption_availability).astype(np.int)
        return {id: avail for id, avail in zip(image_ids, availability)}


class ConceptCapLoaderVal(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the 
            GPU, which is faster).
    """

    def __init__(
            self,
            annotations_path,
            features_path,
            tokenizer,
            bert_model,
            seq_len,
            batch_size=512,
            num_workers=25,
            cache=5000,
            objective=0,
            num_locs=5,
            add_global_imgfeat=True,
            visualization=False,
            visual_target_categories_file=None,
            caption_availability=1.,
            remove_CLS_token=False,
            remove_SEP_token=False,
    ):
        lmdb_file = os.path.join(features_path, "validation_feat_all.lmdb")
        caption_path = os.path.join(annotations_path, "caption_valid.json")
        print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)
        self.caption_availability_dict = self._label_caption_availability(caption_path, caption_availability)
        preprocess_function = BertPreprocessBatch(
            caption_path,
            tokenizer,
            bert_model,
            seq_len,
            36,
            self.num_dataset,
            visualization=visualization,
            objective=objective,
            num_locs=num_locs,
            visual_target_categories_file=visual_target_categories_file,
            caption_availability=self.caption_availability_dict,
            remove_CLS_token=remove_CLS_token,
            remove_SEP_token=remove_SEP_token
        )

        ds = td.MapData(ds, preprocess_function)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.add_global_imgfeat = add_global_imgfeat
        self.num_locs = num_locs
        self.tokenize_visual_categories = preprocess_function.tokenize_visual_categories

    def __iter__(self):
        for batch in self.ds.get_data():
            if self.tokenize_visual_categories:
                input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, \
                    image_cls, obj_labels, obj_confs, attr_labels, attr_confs, image_attrs, image_label, \
                image_mask, masked_label, obj_tokens, attr_tokens, caption_avail, image_id = batch
            else:
                input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, \
                image_cls, obj_labels, obj_confs, attr_labels, attr_confs, image_attrs, image_label, \
                image_mask, masked_label, caption_avail, image_id = batch

            batch_size = input_ids.shape[0]

            if self.add_global_imgfeat:
                sum_count = np.sum(masked_label == 0, axis=1, keepdims=True)
                sum_count[sum_count == 0] = 1
                g_image_feat = np.sum(image_feat, axis=1) / sum_count
                image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), image_feat], axis=1)
                image_feat = np.array(image_feat, dtype=np.float32)

                g_loc = [0, 0, 1, 1] + [1]*(self.num_locs - 4)
                g_image_loc = np.repeat(np.array([g_loc], dtype=np.float32), batch_size, axis=0)
                image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=1), image_loc], axis=1)

                image_loc = np.array(image_loc, dtype=np.float32)
                g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
                image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            if self.tokenize_visual_categories:
                batch = (
                    input_ids,
                    input_mask,
                    segment_ids,
                    lm_label_ids,
                    is_next,
                    image_feat,
                    image_loc,
                    image_cls,
                    obj_labels,
                    obj_confs,
                    attr_labels,
                    attr_confs,
                    image_attrs,
                    image_label,
                    image_mask,
                    obj_tokens,
                    attr_tokens,
                    caption_avail
                )
            else:
                batch = (
                    input_ids,
                    input_mask,
                    segment_ids,
                    lm_label_ids,
                    is_next,
                    image_feat,
                    image_loc,
                    image_cls,
                    obj_labels,
                    obj_confs,
                    attr_labels,
                    attr_confs,
                    image_attrs,
                    image_label,
                    image_mask,
                    caption_avail
                )
            yield tuple([torch.tensor(data) for data in batch] + [image_id])

    def __len__(self):
        return self.ds.size()

    def _label_caption_availability(self, caption_path, caption_availability):
        image_ids = list(json.load(open(caption_path, "r")).keys())
        num_captions = len(image_ids)
        availability = (np.random.rand(num_captions) < caption_availability).astype(np.int)
        return {id: avail for id, avail in zip(image_ids, availability)}


class BertPreprocessBatch(object):
    def __init__(
            self,
            caption_path,
            tokenizer,
            bert_model,
            seq_len,
            region_len,
            data_size,
            split="Train",
            visualization=False,
            objective=0,
            num_locs=5,
            visual_target_categories_file=None,
            caption_availability={},
            ext_corpus=[],
            remove_CLS_token=False,
            remove_SEP_token=False,
    ):

        self.split = split
        self.seq_len = seq_len
        self.region_len = region_len
        self.tokenizer = tokenizer
        self.num_caps = data_size
        # todo: add captions from other corpus here
        self.captions = list(json.load(open(caption_path, "r")).values())
        self._add_captions(ext_corpus)

        self.visualization = visualization
        self.objective = objective
        self.bert_model = bert_model
        self.num_locs = num_locs
        self.vis_categories = None
        self.vis_att_categories = None

        self.get_visual_categories(visual_target_categories_file)
        self.tokenize_visual_categories = True if (self.vis_categories is not None and
                                                   self.vis_att_categories is not None) else False
        self.caption_availability = caption_availability
        self.remove_CLS_token = remove_CLS_token
        self.remove_SEP_token = remove_SEP_token

    def _img_token_to_name(self, obj_labels, attr_labels):
        obj_tokens = [self.vis_category_to_tokenIds[i] for i in obj_labels]
        attr_tokens = [self.vis_att_category_to_tokenIds[i] for i in attr_labels]

        return obj_tokens, attr_tokens

    def _add_captions(self, datasets, num=2500000):
        for ds in datasets:
            logger.info('sampling {} sentences from {}'.format(num, ds))
            self.captions += random.choices(load_dataset(ds)['train']['text'], k=num)
            logger.info('sampled {} sentences from {}'.format(num, ds))

        # update num of captions
        self.num_caps = len(self.captions)
        logger.info('total number of captions becomes {}'.format(self.num_caps))

    def __call__(self, data):
        image_feature_wp, image_cls_wp, obj_labels, obj_confs, attr_labels, attr_confs, attr_scores, \
            image_location_wp, num_boxes, image_h, image_w, image_id, caption = data

        caption_avail_label = self.caption_availability[image_id]

        image_feature = np.zeros((self.region_len, 2048), dtype=np.float32)
        image_cls = np.zeros((self.region_len, 1601), dtype=np.float32)
        image_attrs = np.zeros((self.region_len, 401), dtype=np.float32)
        image_location = np.zeros((self.region_len, self.num_locs), dtype=np.float32)

        # calculate the IOU here.
        overlaps = iou(image_location_wp, image_location_wp)

        num_boxes = int(num_boxes)
        image_feature[:num_boxes] = image_feature_wp
        image_cls[:num_boxes] = image_cls_wp
        image_attrs[:num_boxes] = attr_scores
        image_location[:num_boxes, :4] = image_location_wp
        obj_labels = obj_labels[:num_boxes]
        obj_confs = obj_confs[:num_boxes]
        attr_labels = attr_labels[:num_boxes]
        attr_confs = attr_confs[:num_boxes]

        if self.tokenize_visual_categories:
            obj_tokens, attr_tokens = self._img_token_to_name(obj_labels, attr_labels)

        if self.num_locs == 5:
            image_location[:, 4] = (
                (image_location[:, 3] - image_location[:, 1])
                * (image_location[:, 2] - image_location[:, 0])
                / (float(image_w) * float(image_h))
            )

        # Normalize the box locations (to 0 ~ 1)
        image_location[:, 0] = image_location[:, 0] / float(image_w)
        image_location[:, 1] = image_location[:, 1] / float(image_h)
        image_location[:, 2] = image_location[:, 2] / float(image_w)
        image_location[:, 3] = image_location[:, 3] / float(image_h)

        caption, label = self.random_cap(caption, caption_avail_label)
        tokens_caption = self.tokenizer.encode(caption)

        example = dict(image_feat=image_feature,
                       image_cls=image_cls,
                       obj_labels=obj_labels,
                       obj_confs=obj_confs,
                       attr_labels=attr_labels,
                       attr_confs=attr_confs,
                       image_attrs=image_attrs,
                       caption=tokens_caption,
                       is_next=label,
                       image_loc=image_location,
                       num_boxes=num_boxes,
                       overlaps=overlaps,
                       caption_avail_label=caption_avail_label)

        if self.tokenize_visual_categories:
            example.update(dict(obj_tokens=obj_tokens, attr_tokens=attr_tokens))

        cur_example = InputExample(**example)
        # transform sample to features
        cur_features = self.convert_example_to_features(cur_example, self.seq_len, self.tokenizer, self.region_len)

        if self.tokenize_visual_categories:
            cur_tensors = (
                cur_features.input_ids,
                cur_features.input_mask,
                cur_features.segment_ids,
                cur_features.lm_label_ids,
                cur_features.is_next,
                cur_features.image_feat,
                cur_features.image_loc,
                cur_features.image_cls,
                cur_features.obj_labels,
                cur_features.obj_confs,
                cur_features.attr_labels,
                cur_features.attr_confs,
                cur_features.image_attrs,
                cur_features.image_label,
                cur_features.image_mask,
                cur_features.masked_label,
                cur_features.obj_tokens,
                cur_features.attr_tokens,
                cur_features.caption_avail_label,
                image_id,
            )
        else:
            cur_tensors = (
                cur_features.input_ids,
                cur_features.input_mask,
                cur_features.segment_ids,
                cur_features.lm_label_ids,
                cur_features.is_next,
                cur_features.image_feat,
                cur_features.image_loc,
                cur_features.image_cls,
                cur_features.obj_labels,
                cur_features.obj_confs,
                cur_features.attr_labels,
                cur_features.attr_confs,
                cur_features.image_attrs,
                cur_features.image_label,
                cur_features.image_mask,
                cur_features.masked_label,
                cur_features.caption_avail_label,
                image_id,
            )

        return cur_tensors

    def get_visual_categories(self, path, max_length=5):
        import json
        self.vis_categories, self.vis_att_categories = None, None
        if path is not None and os.path.exists(path):
            with open(path, 'rb') as rp:
                categories = json.load(rp)
            self.vis_categories = [item['name'] for item in categories['categories']]
            self.vis_att_categories = [item['name'] for item in categories['attCategories']]
            logger.info('found {} / {} obj / att categories in {}'.format(len(self.vis_categories),
                                                                          len(self.vis_att_categories),
                                                                          path))
            '''
                e.g. category 0 is 'yolk' (whose id is 0)
                self.vis_category_to_tokenIds[0] converts 0 to a sequence of language tokens [10930, 13687, 0, 0, 0], 
                where 0 are [PAD] tokens. Number of zeros is according to `max_length`.
            '''
            self.vis_category_to_tokenIds = []
            self.vis_att_category_to_tokenIds = []
            for item in categories['categories']:
                tokens = self.tokenizer.encode(item['name'])
                out_tokens = [0] * max_length
                out_tokens[0:len(tokens)] = tokens
                self.vis_category_to_tokenIds.append(out_tokens)
            self.vis_category_to_tokenIds = np.array(self.vis_category_to_tokenIds)
            for item in categories['attCategories']:
                tokens = self.tokenizer.encode(item['name'])
                out_tokens = [0] * max_length
                out_tokens[0:len(tokens)] = tokens
                self.vis_att_category_to_tokenIds.append(out_tokens)
            self.vis_att_category_to_tokenIds = np.array(self.vis_att_category_to_tokenIds)

    def random_cap(self, caption, caption_avail_label=None):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.

        (updates): now aligned pairs are labeled as 1 and 0 for not aligned pairs
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """

        if self.visualization:
            return caption, 1

        if (self.objective == 0 or self.objective == 1 or self.objective == 3) and random.random() > 0.5:
            caption = self.get_random_caption()
            label = 0
        elif self.objective == 4 or self.objective == 5:
            # objective 4, always sample negative pairs
            caption = self.get_random_caption()
            label = 0
        else:
            # objective 2, do not sample negative pairs
            label = 1

        return caption, label

    def get_random_caption(self):
        """
        Get random caption from another document for nextSentence task.
        :return: str, content of one line
        """
        # add the hard negative mining objective here.
        rand_doc_idx = random.randint(0, self.num_caps - 1)
        caption = self.captions[rand_doc_idx]
        return caption

    def convert_example_to_features(self, example, max_seq_length, tokenizer, max_region_length):
        """
        """
        image_feat = example.image_feat
        tokens = example.caption
        image_loc = example.image_loc
        image_cls = example.image_cls
        num_boxes = int(example.num_boxes)
        overlaps = example.overlaps

        if not self.remove_CLS_token or not self.remove_SEP_token:
            self._truncate_seq_pair(tokens, max_seq_length - 2)
        else:
            self._truncate_seq_pair(tokens, max_seq_length)

        tokens, tokens_label = self.random_word(tokens, tokenizer)
        image_feat, image_loc, image_label, masked_label = self.random_region(
            image_feat, image_loc, num_boxes, overlaps
        )

        if not self.remove_CLS_token or not self.remove_SEP_token:
            # concatenate lm labels and account for CLS and SEP: [CLS] tokens [SEP]
            lm_label_ids = [-1] + tokens_label + [-1]
            tokens = tokenizer.add_special_tokens_single_sentence(tokens)

        segment_ids = [0] * len(tokens)
        input_ids = tokens

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)
        image_mask = [1] * num_boxes
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)
            image_label.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        while len(example.obj_tokens) < max_region_length:
            num_tokens = len(example.obj_tokens[0])
            example.obj_tokens.append([0] * num_tokens)

        while len(example.attr_tokens) < max_region_length:
            num_tokens = len(example.attr_tokens[0])
            example.attr_tokens.append([0] * num_tokens)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length
        assert len(image_mask) == max_region_length
        assert len(image_label) == max_region_length

        inputs = dict(input_ids=np.array(input_ids),
                      input_mask=np.array(input_mask),
                      segment_ids=np.array(segment_ids),
                      lm_label_ids=np.array(lm_label_ids),
                      is_next=np.array(example.is_next),
                      image_feat=image_feat,
                      image_cls=image_cls,
                      obj_labels=example.obj_labels,
                      obj_confs=example.obj_confs,
                      attr_labels=example.attr_labels,
                      attr_confs=example.attr_confs,
                      image_attrs=example.image_attrs,
                      image_loc=image_loc,
                      image_label=np.array(image_label),
                      image_mask=np.array(image_mask),
                      masked_label=masked_label,
                      caption_avail_label=example.caption_avail_label)

        if self.tokenize_visual_categories:
            assert len(example.obj_tokens) == max_region_length
            assert len(example.attr_tokens) == max_region_length
            inputs.update(dict(obj_tokens=np.array(example.obj_tokens), attr_tokens=np.array(example.attr_tokens)))

        features = InputFeatures(**inputs)
        return features

    def _truncate_seq_pair(self, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break
            tokens_b.pop()

    def random_word(self, tokens, tokenizer):
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability

            if prob < 0.15 and (not self.visualization):
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = np.random.randint(len(tokenizer))

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return tokens, output_label

    def random_region(self, image_feat, image_loc, num_boxes, overlaps):
        """
        """
        output_label = []
        masked_label = np.zeros((image_feat.shape[0]))

        for i in range(num_boxes):
            prob = random.random()
            # mask token with 15% probability

            if prob < 0.15 and not self.visualization:
                prob /= 0.15

                # 90% randomly change token to mask token
                if prob < 0.9:
                    image_feat[i] = 0
                # mask the overlap regions into zeros
                masked_label = np.logical_or(masked_label, overlaps[i] > 0.4)

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return image_feat, image_loc, output_label, masked_label
