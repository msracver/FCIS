# --------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Haozhi Qi
# --------------------------------------------------------

import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "gpu_mv.hpp":
    void _mv(np.float32_t* all_boxes, np.float32_t* all_masks, np.int32_t all_boxes_num, np.int32_t* candidate_inds, np.int32_t* candidate_start, np.float32_t* candidate_weights, np.int32_t candidate_num, np.float32_t binary_thresh, np.int32_t image_height, np.int32_t image_width, np.int32_t box_dim, np.int32_t mask_size, np.int32_t result_num, np.float32_t* result_mask, np.int32_t* result_box, np.int32_t device_id);

# boxes: n * 4
# masks: n * 1 * 21 * 21
# scores: n * 21
def mv(np.ndarray[np.float32_t, ndim=2] all_boxes,
                np.ndarray[np.float32_t, ndim=4] all_masks,
                np.ndarray[np.int32_t, ndim=1] candidate_inds,
                np.ndarray[np.int32_t, ndim=1] candidate_start,
                np.ndarray[np.float32_t, ndim=1] candidate_weights,
                np.float32_t binary_thresh,
                np.int32_t image_height,
                np.int32_t image_width,
                np.int32_t device_id = 0):
    cdef int all_box_num = all_boxes.shape[0]
    cdef int boxes_dim = all_boxes.shape[1]
    cdef int mask_size = all_masks.shape[3]
    cdef int candidate_num = candidate_inds.shape[0]
    cdef int result_num = candidate_start.shape[0]
    cdef np.ndarray[np.float32_t, ndim=4] \
        result_mask = np.zeros((result_num, 1, all_masks.shape[2], all_masks.shape[3]), dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=2] \
        result_box = np.zeros((result_num, boxes_dim), dtype=np.int32)
    if all_boxes.shape[0] > 0:
        _mv(&all_boxes[0, 0], &all_masks[0, 0, 0, 0], all_box_num, &candidate_inds[0], &candidate_start[0], &candidate_weights[0], candidate_num, binary_thresh, image_height, image_width, boxes_dim, mask_size, candidate_start.shape[0], &result_mask[0,0,0,0], &result_box[0,0], device_id)
    return result_mask, result_box
