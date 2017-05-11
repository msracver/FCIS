# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Guodong Zhang
# --------------------------------------------------------

"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np
import cPickle


class BoxAnnotatorOHEMOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, num_reg_classes, roi_per_img, cfg):
        super(BoxAnnotatorOHEMOperator, self).__init__()
        self._num_classes = num_classes
        self._num_reg_classes = num_reg_classes
        self._roi_per_img = roi_per_img
        self._cfg = cfg

    def forward(self, is_train, req, in_data, out_data, aux):

        cls_score    = in_data[0]
        seg_pred     = in_data[1]
        bbox_pred    = in_data[2]
        labels       = in_data[3].asnumpy()
        mask_targets = in_data[4].asnumpy()
        bbox_targets = in_data[5]
        bbox_weights = in_data[6]

        per_roi_loss_cls = mx.nd.SoftmaxActivation(cls_score) + 1e-14
        per_roi_loss_cls = per_roi_loss_cls.asnumpy()
        per_roi_loss_cls = per_roi_loss_cls[np.arange(per_roi_loss_cls.shape[0], dtype='int'), labels.astype('int')]
        per_roi_loss_cls = -1 * np.log(per_roi_loss_cls)
        per_roi_loss_cls = np.reshape(per_roi_loss_cls, newshape=(-1,))
        remove_inds = np.where(labels == -1)
        per_roi_loss_cls[remove_inds] = 0

        SoftmaxOutput = mx.nd.SoftmaxActivation(seg_pred, mode='channel') + 1e-14
        shape = SoftmaxOutput.shape
        label = mask_targets.astype('int').reshape((-1))
        SoftmaxOutput = mx.nd.transpose(SoftmaxOutput.reshape((shape[0], shape[1], shape[2] * shape[3])),axes=(0, 2, 1))
        SoftmaxOutput = SoftmaxOutput.reshape((label.shape[0], 2)).asnumpy() + 1e-14
        keep_inds = np.where(label != -1)[0]
        per_roi_loss_seg = np.zeros((label.shape[0]))
        per_roi_loss_seg[keep_inds] = -1 * np.log(SoftmaxOutput[keep_inds, label[keep_inds]])
        per_roi_loss_seg = np.average(per_roi_loss_seg.reshape((shape[0],-1)), axis=1)

        per_roi_loss_bbox = bbox_weights * mx.nd.smooth_l1((bbox_pred - bbox_targets), scalar=1.0)
        per_roi_loss_bbox = mx.nd.sum(per_roi_loss_bbox, axis=1).asnumpy()

        top_k_per_roi_loss = np.argsort(self._cfg.TRAIN.LOSS_WEIGHT[0] * per_roi_loss_cls +
                                        self._cfg.TRAIN.LOSS_WEIGHT[1] * per_roi_loss_seg +
                                        self._cfg.TRAIN.LOSS_WEIGHT[2] * per_roi_loss_bbox)
        labels_ohem = labels
        labels_ohem[top_k_per_roi_loss[::-1][self._roi_per_img:]] = -1
        mask_targets_ohem = mask_targets
        mask_targets_ohem[top_k_per_roi_loss[::-1][self._roi_per_img:],:,:,:] = -1
        bbox_weights_ohem = bbox_weights.asnumpy()
        bbox_weights_ohem[top_k_per_roi_loss[::-1][self._roi_per_img:]] = 0

        labels_ohem = mx.nd.array(labels_ohem)
        mask_targets_ohem = mx.nd.array(mask_targets_ohem)
        bbox_weights_ohem = mx.nd.array(bbox_weights_ohem)

        for ind, val in enumerate([labels_ohem, mask_targets_ohem, bbox_weights_ohem]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register('BoxAnnotatorOHEM')
class BoxAnnotatorOHEMProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, num_reg_classes, roi_per_img, cfg):
        super(BoxAnnotatorOHEMProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._num_reg_classes = int(num_reg_classes)
        self._roi_per_img = int(roi_per_img)
        self._cfg = cPickle.loads(cfg)

    def list_arguments(self):
        return ['cls_score', 'seg_pred', 'bbox_pred', 'labels', 'mask_targets', 'bbox_targets', 'bbox_weights']

    def list_outputs(self):
        return ['labels_ohem', 'mask_targets_ohem', 'bbox_weights_ohem']

    def infer_shape(self, in_shape):
        labels_shape = in_shape[3]
        mask_targets_shape = in_shape[4]
        bbox_weights_shape = in_shape[6]

        return in_shape, \
               [labels_shape, mask_targets_shape, bbox_weights_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return BoxAnnotatorOHEMOperator(self._num_classes, self._num_reg_classes, self._roi_per_img,
                                        self._cfg)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
