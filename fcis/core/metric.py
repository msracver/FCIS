# --------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Haozhi Qi, Guodong Zhang
# --------------------------------------------------------

import mxnet as mx
import numpy as np


def get_rpn_names():
    pred = ['rpn_cls_prob', 'rpn_bbox_loss']
    label = ['rpn_label', 'rpn_bbox_target', 'rpn_bbox_weight']
    return pred, label


def get_rcnn_names(cfg):
    pred = ['rcnn_cls_prob', 'rcnn_bbox_loss', 'fcis_mask_loss', 'fcis_mask_label']
    label = ['rcnn_label', 'rcnn_bbox_target', 'rcnn_bbox_weight']
    if cfg.TRAIN.END2END:
        pred.append('rcnn_label')
        rpn_pred, rpn_label = get_rpn_names()
        pred = rpn_pred + pred
        label = rpn_label
    return pred, label


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetric, self).__init__('RPNLogLoss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss')].asnumpy()
        label = labels[self.label.index('rpn_label')].asnumpy()
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


class FCISAccMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(FCISAccMetric, self).__init__('FCISAcc')
        self.e2e = cfg.TRAIN.END2END
        self.pred, self.label = get_rcnn_names(cfg)
        self.cfg = cfg

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class FCISAccFGMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(FCISAccFGMetric, self).__init__('FCISAccFG')
        self.e2e = cfg.TRAIN.END2END
        self.pred, self.label = get_rcnn_names(cfg)
        self.cfg = cfg

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label > 0)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class FCISLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(FCISLogLossMetric, self).__init__('FCISLogLoss')
        self.e2e = cfg.TRAIN.END2END
        self.pred, self.label = get_rcnn_names(cfg)
        self.cfg = cfg

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class FCISL1LossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(FCISL1LossMetric, self).__init__('FCISL1Loss')
        self.e2e = cfg.TRAIN.END2END
        self.pred, self.label = get_rcnn_names(cfg)
        self.cfg = cfg

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        if self.e2e:
            label = preds[self.pred.index('rcnn_label')].asnumpy()
        else:
            label = labels[self.label.index('rcnn_label')].asnumpy()

        # calculate num_inst
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


class FCISMaskLossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(FCISMaskLossMetric, self).__init__('FCISMaskLoss')
        self.e2e = cfg.TRAIN.END2END
        self.pred, self.label = get_rcnn_names(cfg)
        self.cfg = cfg

    def update(self, labels, preds):
        mask_loss = preds[self.pred.index('fcis_mask_loss')]
        if self.e2e:
            label = preds[self.pred.index('fcis_mask_label')]
        else:
            raise NotImplementedError
        mask_size = mask_loss.shape[2]
        label = label.asnumpy().astype('int32').reshape((-1))
        mask_loss = mx.nd.transpose(mask_loss.reshape((mask_loss.shape[0], mask_loss.shape[1], mask_size * mask_size)), axes=(0, 2, 1))
        mask_loss = mask_loss.reshape((label.shape[0], 2))
        mask_loss = mask_loss.asnumpy()
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = mask_loss[keep_inds, label]
        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)

        self.sum_metric += cls_loss
        self.num_inst += len(keep_inds)
