# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Guodong Zhang
# --------------------------------------------------------

"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np
from distutils.util import strtobool
from bbox.bbox_transform import bbox_pred, clip_boxes


class BoxParserOperator(mx.operator.CustomOp):
    def __init__(self, b_clip_boxes, bbox_class_agnostic, bbox_means, bbox_stds):
        super(BoxParserOperator, self).__init__()
        self._b_clip_boxes = b_clip_boxes
        self._bbox_class_agnostic = bbox_class_agnostic
        self._bbox_means = np.fromstring(bbox_means[1:-1], dtype=float, sep=',')
        self._bbox_stds = np.fromstring(bbox_stds[1:-1], dtype=float, sep=',')

    def forward(self, is_train, req, in_data, out_data, aux):

        bottom_rois     = in_data[0].asnumpy()
        bbox_delta      = in_data[1].asnumpy()
        cls_prob        = in_data[2].asnumpy()
        im_info         = in_data[3].asnumpy()

        num_rois = bottom_rois.shape[0]
        # 1. judge if bbox class-agnostic
        # 2. if not, calculate bbox_class_idx
        if self._bbox_class_agnostic:
            bbox_class_idx = np.ones((num_rois))  # (num_rois, 1) zeros
        else:
            bbox_class_idx = np.argmax(cls_prob[:,1:], axis=1) + 1
        bbox_class_idx = bbox_class_idx[:, np.newaxis] * 4
        bbox_class_idx = np.hstack((bbox_class_idx,bbox_class_idx+1,bbox_class_idx+2,bbox_class_idx+3))

        # 3. get bbox_pred given bbox_class_idx
        rows = np.arange(num_rois, dtype=np.intp)
        bbox_delta = bbox_delta[rows[:,np.newaxis], bbox_class_idx.astype(np.intp)]

        # 4. calculate bbox_delta by bbox_pred[i] * std[i] + mean[i]
        means = np.array(self._bbox_means)
        stds = np.array(self._bbox_stds)
        vx = bbox_delta[:, 0] * stds[0] + means[0]
        vy = bbox_delta[:, 1] * stds[1] + means[1]
        vw = bbox_delta[:, 2] * stds[2] + means[2]
        vh = bbox_delta[:, 3] * stds[3] + means[3]
        bbox_delta = np.hstack((vx[:, np.newaxis], vy[:, np.newaxis], vw[:, np.newaxis], vh[:, np.newaxis]))

        # 6. calculate top_rois by bbox_pred
        proposal = bbox_pred(bottom_rois[:, 1:], bbox_delta)

        # 7. clip boxes
        if self._b_clip_boxes:
            proposal = clip_boxes(proposal, im_info[0, :2])

        output = bottom_rois
        output[:,1:] = proposal

        for ind, val in enumerate([output]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register('BoxParser')
class BoxParserProp(mx.operator.CustomOpProp):
    def __init__(self, b_clip_boxes, bbox_class_agnostic, bbox_means='(0,0,0,0)', bbox_stds='(0.1,0.1,0.2,0.2)'):
        super(BoxParserProp, self).__init__(need_top_grad=False)
        self._b_clip_boxes = strtobool(b_clip_boxes)
        self._bbox_class_agnostic = strtobool(bbox_class_agnostic)
        self._bbox_means = bbox_means
        self._bbox_stds = bbox_stds

    def list_arguments(self):
        return ['bottom_rois', 'bbox_delta', 'cls_prob', 'im_info']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        output_shape = in_shape[0]

        return in_shape, [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return BoxParserOperator(self._b_clip_boxes, self._bbox_class_agnostic,
                                 self._bbox_means, self._bbox_stds)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
