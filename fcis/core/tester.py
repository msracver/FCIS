# --------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Haozhi Qi, Haochen Zhang, Guodong Zhang, Yi Li
# --------------------------------------------------------

import cPickle
import os
import time
import mxnet as mx
import numpy as np

from module import MutableModule
from utils import image
from bbox.bbox_transform import bbox_pred, clip_boxes, filter_boxes
from nms.nms import py_nms_wrapper
from utils.PrefetchingIter import PrefetchingIter
from mask.mask_transform import gpu_mask_voting, cpu_mask_voting


class Predictor(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context, max_data_shapes=max_data_shapes)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def predict(self, data_batch):
        self._mod.forward(data_batch)
        return [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]


def im_detect(predictor, data_batch, data_names, scales, cfg):
    output_all = predictor.predict(data_batch)
    data_dict_all = [dict(zip(data_names, data_batch.data[i])) for i in xrange(len(data_batch.data))]
    scores_all = []
    pred_boxes_all = []
    pred_masks_all = []
    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        if cfg.TEST.HAS_RPN:
            rois = output['rois_output'].asnumpy()[:, 1:]
        else:
            raise NotImplementedError
        # save output
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        pred_masks = output['seg_pred_output'].asnumpy()

        if cfg.TEST.ITER == 2 and cfg.TEST.MIN_DROP_SIZE > 0:
            keep_inds = filter_boxes(rois, cfg.TEST.MIN_DROP_SIZE)
            rois = rois[keep_inds, :]
            scores = scores[keep_inds, :]
            pred_masks = pred_masks[keep_inds, ...]

        # we used scaled image & roi to train, so it is necessary to transform them back
        pred_boxes = rois / scale

        scores_all.append(scores)
        pred_boxes_all.append(pred_boxes)
        pred_masks_all.append(pred_masks)

    return scores_all, pred_boxes_all, pred_masks_all, data_dict_all


def pred_eval(predictor, test_data, imdb, cfg, vis=False, thresh=1e-3, logger=None, ignore_cache=False):

    det_file = os.path.join(imdb.result_path, imdb.name + '_detections.pkl')
    seg_file = os.path.join(imdb.result_path, imdb.name + '_masks.pkl')

    if os.path.exists(det_file) and os.path.exists(seg_file) and not ignore_cache:
        with open(det_file, 'rb') as f:
            all_boxes = cPickle.load(f)
        with open(seg_file, 'rb') as f:
            all_masks = cPickle.load(f)
    else:
        assert vis or not test_data.shuffle
        data_names = [k[0] for k in test_data.provide_data[0]]

        if not isinstance(test_data, PrefetchingIter):
            test_data = PrefetchingIter(test_data)

        # function pointers
        nms = py_nms_wrapper(cfg.TEST.NMS)
        mask_voting = gpu_mask_voting if cfg.TEST.USE_GPU_MASK_MERGE else cpu_mask_voting

        max_per_image = 100 if cfg.TEST.USE_MASK_MERGE else -1
        num_images = imdb.num_images
        all_boxes = [[[] for _ in xrange(num_images)]
                     for _ in xrange(imdb.num_classes)]
        all_masks = [[[] for _ in xrange(num_images)]
                     for _ in xrange(imdb.num_classes)]

        idx = 0
        t = time.time()
        for data_batch in test_data:
            t1 = time.time() - t
            t = time.time()

            scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
            scores_all, boxes_all, masks_all, data_dict_all = im_detect(predictor, data_batch, data_names, scales, cfg)
            im_shapes = [data_batch.data[i][0].shape[2:4] for i in xrange(len(data_batch.data))]

            t2 = time.time() - t
            t = time.time()

            # post processing
            for delta, (scores, boxes, masks, data_dict) in enumerate(zip(scores_all, boxes_all, masks_all, data_dict_all)):

                if not cfg.TEST.USE_MASK_MERGE:
                    for j in range(1, imdb.num_classes):
                        indexes = np.where(scores[:, j] > thresh)[0]
                        cls_scores = scores[indexes, j, np.newaxis]
                        cls_masks = masks[indexes, 1, :, :]
                        try:
                            if cfg.CLASS_AGNOSTIC:
                                cls_boxes = boxes[indexes, :]
                            else:
                                raise Exception()
                        except:
                            cls_boxes = boxes[indexes, j * 4:(j + 1) * 4]

                        cls_dets = np.hstack((cls_boxes, cls_scores))
                        keep = nms(cls_dets)
                        all_boxes[j][idx + delta] = cls_dets[keep, :]
                        all_masks[j][idx + delta] = cls_masks[keep, :]
                else:
                    masks = masks[:, 1:, :, :]
                    im_height = np.round(im_shapes[delta][0] / scales[delta]).astype('int')
                    im_width = np.round(im_shapes[delta][1] / scales[delta]).astype('int')
                    boxes = clip_boxes(boxes, (im_height, im_width))
                    result_mask, result_box = mask_voting(masks, boxes, scores, imdb.num_classes,
                                                          max_per_image, im_width, im_height,
                                                          cfg.TEST.NMS, cfg.TEST.MASK_MERGE_THRESH,
                                                          cfg.BINARY_THRESH)
                    for j in xrange(1, imdb.num_classes):
                        all_boxes[j][idx+delta] = result_box[j]
                        all_masks[j][idx+delta] = result_mask[j][:,0,:,:]

                if vis:
                    boxes_this_image = [[]] + [all_boxes[j][idx + delta] for j in range(1, imdb.num_classes)]
                    masks_this_image = [[]] + [all_masks[j][idx + delta] for j in range(1, imdb.num_classes)]
                    vis_all_mask(data_dict['data'].asnumpy(), boxes_this_image, masks_this_image, imdb.classes, scales[delta], cfg)

            idx += test_data.batch_size
            t3 = time.time() - t
            t = time.time()

            print 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, imdb.num_images, t1, t2, t3)
            if logger:
                logger.info('testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, imdb.num_images, t1, t2, t3))
            
        with open(det_file, 'wb') as f:
            cPickle.dump(all_boxes, f, protocol=cPickle.HIGHEST_PROTOCOL)
        with open(seg_file, 'wb') as f:
            cPickle.dump(all_masks, f, protocol=cPickle.HIGHEST_PROTOCOL)

    info_str = imdb.evaluate_sds(all_boxes, all_masks)
    if logger:
        logger.info('evaluate detections: \n{}'.format(info_str))

        
def vis_all_mask(im_array, detections, masks, class_names, scale, cfg):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import matplotlib.pyplot as plt
    import random
    import cv2
    import os
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    plt.cla()
    plt.axis('off')
    plt.imshow(im)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        dets = detections[j]
        msks = masks[j]
        for det, msk in zip(dets, msks):
            if det[-1] < 0.7:
                continue
            color = (random.random(), random.random(), random.random())  # generate a random color
            bbox = det[:4] * scale
            cod = np.zeros(4).astype(int)
            cod[0] = int(bbox[0])
            cod[1] = int(bbox[1])
            cod[2] = int(bbox[2])
            cod[3] = int(bbox[3])
            if im[cod[1]:cod[3], cod[0]:cod[2], 0].size > 0:
                msk = cv2.resize(msk, im[cod[1]:cod[3], cod[0]:cod[2], 0].T.shape)
                bimsk = msk > cfg.BINARY_THRESH
                bimsk = bimsk.astype(int)
                bimsk = np.repeat(bimsk[:, :, np.newaxis], 3, axis=2)
                mskd = im[cod[1]:cod[3], cod[0]:cod[2], :] * bimsk
                clmsk = np.ones(bimsk.shape) * bimsk
                clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
                clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
                clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
                im[cod[1]:cod[3], cod[0]:cod[2], :] = im[cod[1]:cod[3], cod[0]:cod[2], :] + 0.8 * clmsk - 0.8 * mskd
            score = det[-1]
            plt.gca().text((bbox[2]+bbox[0])/2, bbox[1],
                           '{:s} {:.3f}'.format(name, score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.imshow(im)
    plt.show()
    
