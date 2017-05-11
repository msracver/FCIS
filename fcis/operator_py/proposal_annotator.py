# --------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Haozhi Qi, Guodong Zhang
# --------------------------------------------------------

import cv2
import mxnet as mx
import numpy as np

from bbox.bbox_transform import bbox_overlaps, bbox_transform, remove_repetition
from bbox.bbox_regression import expand_bbox_regression_targets
from mask.mask_transform import intersect_box_mask
import cPickle


class ProposalAnnotatorOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, mask_size, binary_thresh, batch_images, batch_rois, cfg, fg_fraction):
        super(ProposalAnnotatorOperator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._fg_fraction = fg_fraction
        self._mask_size = mask_size
        self._binary_thresh = binary_thresh
        self._cfg = cfg

    def forward(self, is_train, req, in_data, out_data, aux):
        all_rois = in_data[0].asnumpy()
        gt_boxes = in_data[1].asnumpy()
        gt_masks = in_data[2].asnumpy()

        # when applying OHEM, batch rois is set to -1
        if self._batch_rois == -1:
            rois_per_image = all_rois.shape[0] + gt_boxes.shape[0]
            fg_rois_per_image = rois_per_image
        else:
            assert self._batch_rois % self._batch_images == 0
            rois_per_image = self._batch_rois / self._batch_images
            fg_rois_per_image = np.round(self._fg_fraction * rois_per_image).astype(int)

        # include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))
        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'

        rois, labels, bbox_targets, bbox_weights, mask_reg_targets = \
            self.sample_rois(all_rois, fg_rois_per_image, rois_per_image, self._num_classes,
                             self._cfg, gt_boxes=gt_boxes, gt_masks=gt_masks)

        for ind, val in enumerate([rois, labels, bbox_targets, bbox_weights, mask_reg_targets]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)

    def sample_rois(self, rois, fg_rois_per_image, rois_per_image, num_classes, cfg,
                    labels=None, overlaps=None, bbox_targets=None, gt_boxes=None, gt_masks=None):
        if labels is None:
            overlaps = bbox_overlaps(rois[:, 1:].astype(np.float),
                                     gt_boxes[:, :4].astype(np.float))
            gt_assignment = overlaps.argmax(axis=1)
            overlaps = overlaps.max(axis=1)
            labels = gt_boxes[gt_assignment, 4]

        # foreground RoI with FG_THRESH overlap
        fg_indexes = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        if cfg.TRAIN.IGNORE_GAP:
            keep_inds = remove_repetition(rois[fg_indexes, 1:])
            fg_indexes = fg_indexes[keep_inds]

        # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
        fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
        # Sample foreground regions without replacement
        if len(fg_indexes) > fg_rois_per_this_image:
            fg_indexes = np.random.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_indexes = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        if cfg.TRAIN.IGNORE_GAP:
            keep_inds = remove_repetition(rois[bg_indexes, 1:])
            bg_indexes = bg_indexes[keep_inds]

        # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
        # Sample foreground regions without replacement
        if len(bg_indexes) > bg_rois_per_this_image:
            bg_indexes = np.random.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

        # indexes selected
        keep_indexes = np.append(fg_indexes, bg_indexes)

        # pad more to ensure a fixed minibatch size
        while keep_indexes.shape[0] < rois_per_image:
            gap = np.minimum(len(rois), rois_per_image - keep_indexes.shape[0])
            if cfg.TRAIN.GAP_SELECT_FROM_ALL:
                gap_indexes = np.random.choice(range(len(rois)), size=gap, replace=False)
            else:
                bg_full_indexes = list(set(range(len(rois))) - set(fg_indexes))
                gap_indexes = np.random.choice(bg_full_indexes, size=gap, replace=False)
            keep_indexes = np.append(keep_indexes, gap_indexes)

        # select labels
        labels = labels[keep_indexes]
        # set labels of bg_rois to be 0
        labels[fg_rois_per_this_image:] = 0
        rois = rois[keep_indexes]

        # load or compute bbox target
        if bbox_targets is not None:
            bbox_target_data = bbox_targets[keep_indexes, :]
        else:
            targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :4])
            if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
                targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
                           / np.array(cfg.TRAIN.BBOX_STDS))
            bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

        bbox_targets, bbox_weights = \
            expand_bbox_regression_targets(bbox_target_data, num_classes, cfg)

        if cfg.TRAIN.IGNORE_GAP:
            valid_rois_per_this_image = fg_rois_per_this_image+bg_rois_per_this_image
            labels[valid_rois_per_this_image:] = -1
            bbox_weights[valid_rois_per_this_image:] = 0

        # masks
        # debug_gt_image_buffer = cv2.imread('debug_im_buffer.jpg')
        mask_reg_targets = -np.ones((len(keep_indexes), 1, self._mask_size, self._mask_size))
        for idx, obj in enumerate(fg_indexes):
            gt_roi = np.round(gt_boxes[gt_assignment[obj], :-1]).astype(int)
            ex_roi = np.round(rois[idx, 1:]).astype(int)
            gt_mask = gt_masks[gt_assignment[obj]]
            mask_reg_target = intersect_box_mask(ex_roi, gt_roi, gt_mask)
            mask_reg_target = cv2.resize(mask_reg_target.astype(np.float), (self._mask_size, self._mask_size))
            mask_reg_target = mask_reg_target >= self._binary_thresh
            mask_reg_targets[idx, ...] = mask_reg_target

        return rois, labels, bbox_targets, bbox_weights, mask_reg_targets


@mx.operator.register('proposal_annotator')
class ProposalAnnotatorProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, mask_size, binary_thresh, batch_images, batch_rois, cfg, fg_fraction='0.25'):
        super(ProposalAnnotatorProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._fg_fraction = float(fg_fraction)
        self._mask_size = int(mask_size)
        self._binary_thresh = float(binary_thresh)
        self._cfg = cPickle.loads(cfg)

    def list_arguments(self):
        return ['rois', 'gt_boxes', 'gt_masks']

    def list_outputs(self):
        return ['rois_output', 'label', 'bbox_target', 'bbox_weight', 'mask_reg_targets']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]
        gt_masks_shape = in_shape[2]

        rois = rpn_rois_shape[0] + gt_boxes_shape[0] if self._batch_rois == -1 else self._batch_rois

        output_rois_shape = (rois, 5)
        label_shape = (rois,)
        bbox_target_shape = (rois, self._num_classes * 4)
        bbox_weight_shape = (rois, self._num_classes * 4)
        mask_reg_targets_shape = (rois, 1, self._mask_size, self._mask_size)

        return [rpn_rois_shape, gt_boxes_shape, gt_masks_shape], \
               [output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape, mask_reg_targets_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalAnnotatorOperator(self._num_classes, self._mask_size, self._binary_thresh,
                                         self._batch_images, self._batch_rois, self._cfg, self._fg_fraction)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
