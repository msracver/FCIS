// ------------------------------------------------------------------
// Fully Convolutional Instance-aware Semantic Segmentation
// Copyright (c) 2017 Microsoft
// Licensed under The Apache-2.0 License [see LICENSE for details]
// Written by Haozhi Qi
// ------------------------------------------------------------------

#include "gpu_mv.hpp"
#include <iostream>

const int CAFFE_CUDA_NUM_THREADS = 512;

inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}


#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

__device__ float bilinear_interpolate(const float* bottom_data,
                                      const int input_height, const int input_width,
                                      float inverse_y, float inverse_x) {

  // deal with cases that inverse elements are out of feature map boundary
  if (inverse_y <= 0) inverse_y = 0;
  if (inverse_x <= 0) inverse_x = 0;
  
  int h_low = (int) inverse_y;
  int w_low = (int) inverse_x;
  int h_high;
  int w_high;

  // handle boundary case
  if (h_low >= input_height - 1) {
    h_high = h_low = input_height - 1;
    inverse_y = (float) h_low;
  } else {
    h_high = h_low + 1;
  }

  if (w_low >= input_width - 1) {
    w_high = w_low = input_width - 1;
    inverse_x = (float) w_low;
  } else {
    w_high = w_low + 1;
  }

  float lh = inverse_y - h_low;
  float lw = inverse_x - w_low;
  float hh = 1 - lh, hw = 1 - lw;
  // corner point of interpolation
  float v1 = bottom_data[h_low * input_width + w_low];
  float v2 = bottom_data[h_low * input_width + w_high];
  float v3 = bottom_data[h_high * input_width + w_low];
  float v4 = bottom_data[h_high * input_width + w_high];
  // weight for each corner
  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  // do bilinear interpolation
  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

__global__ void mask_render(const int nthreads, const float* input_box, const float* input_mask, const int box_dim, const int mask_size,
                 const int image_height, const int image_width, float* target_buffer) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // target buffer's size if (n * h * w)
    int w = index % image_width;
    int h = (index / image_width) % image_height;
    int n = index / image_width / image_height;
    // get the n-th boxes
    const float* offset_box = input_box + n * box_dim;
    const float* offset_mask = input_mask + n * mask_size * mask_size;
    const float box_x1 = offset_box[0];
    const float box_y1 = offset_box[1];
    const float box_x2 = offset_box[2];
    const float box_y2 = offset_box[3];
    // check whether pixel is out of box bound
    if (w < box_x1 || w > box_x2 || h < box_y1 || h > box_y2) {
      target_buffer[index] = 0.0;
      continue;
    }
    const float box_width = box_x2 - box_x1 + 1.0;
    const float box_height = box_y2 - box_y1 + 1.0;
    const float ratio_w = (float) mask_size / box_width;
    const float ratio_h = (float) mask_size / box_height;
    const float inverse_x = ((float) w - box_x1 + 0.5) * ratio_w - 0.5;
    const float inverse_y = ((float) h - box_y1 + 0.5) * ratio_h - 0.5;

    target_buffer[index] = bilinear_interpolate(offset_mask, mask_size, mask_size, inverse_y, inverse_x);
  }
}

__global__ void mask_aggregate(const int nthreads, const float* render_mask, float* aggregate_mask, const int* candidate_inds, const int* candidate_starts, const float* candidate_weights, const int image_height, const int image_width, const float binary_thresh) {
  // render_mask: num_boxes * image_height * image_width
  // aggregate_mask: output_num * image_height * image_width
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % image_width;
    int h = (index / image_width) % image_height;
    int n = index / image_width / image_height;
    // get candidate_inds, candidate_start
    int candidate_start = (n == 0) ? 0 : candidate_starts[n-1];
    int candidate_end = candidate_starts[n];
    // output value will be summation of (mask * mask_weight)
    float val = 0.0;
    for (int i = candidate_start; i < candidate_end; ++i) {
      int input_mask_ind = candidate_inds[i];
      int offset_render_mask = (input_mask_ind * image_height + h) * image_width + w;
      const float mask_val = render_mask[offset_render_mask] >= binary_thresh ? 1.f : 0.f;
      val += (mask_val * candidate_weights[i]); 
    }
    aggregate_mask[index] = val;
  }
}

__global__ void reduce_mask_col(const int nthreads, const float* masks, int image_height, int image_width, const float binary_thresh, bool* output_buffer) {
  // nthreads will be output_num * image_width
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % image_width;
    int n = index / image_width;
    output_buffer[index] = false;
    for (int i = 0; i < image_height; ++i) {
      if (masks[(n * image_height + i) * image_width + w] >= binary_thresh) {
        output_buffer[index] = true;
        break;
      }
    }
  }
}

__global__ void reduce_mask_row(const int nthreads, const float* masks, int image_height, int image_width, const float binary_thresh, bool* output_buffer) {
  // nthreads will be output_num * image_width
  CUDA_KERNEL_LOOP(index, nthreads) {
    int h = index % image_height;
    int n = index / image_height;
    output_buffer[index] = false;
    for (int i = 0; i < image_width; ++i) {
      if (masks[(n * image_height + h) * image_width + i] >= binary_thresh) {
        output_buffer[index] = true;
        break;
      }
    }
  }
}

__global__ void reduce_bounding_x(const int nthreads, const bool* reduced_col, int* output_buffer, const int image_width) {
  // nthreads will be output_num * 2
  CUDA_KERNEL_LOOP(index, nthreads) {
    int x = index % 2;
    int n = index / 2;
    output_buffer[index] = image_width / 2;
    if (x == 0) {
      for (int i = 0; i < image_width; ++i) {
        if (reduced_col[n * image_width + i]) {
          output_buffer[index] = i;
          break;
        }
      }   
    } else {
      for (int i = image_width - 1; i >= 0; --i) {
        if (reduced_col[n * image_width + i]) {
          output_buffer[index] = i;
          break;
        }
      }
    }
  }
}

__global__ void reduce_bounding_y(const int nthreads, const bool* reduced_row, int* output_buffer, const int image_height) {
  // nthreads will be output_num * 2
  CUDA_KERNEL_LOOP(index, nthreads) {
    int x = index % 2;
    int n = index / 2;
    output_buffer[index] = image_height / 2;
    if (x == 0) {
      for (int i = 0; i < image_height; ++i) {
        if (reduced_row[n * image_height + i]) {
          output_buffer[index] = i;
          break;
        }
      }   
    } else {
      for (int i = image_height - 1; i >= 0; --i) {
        if (reduced_row[n * image_height + i]) {
          output_buffer[index] = i;
          break;
        }
      }
    }
  }
}


__global__ void mask_resize(const int nthreads, const float* original_mask, const int* bounding_x, const int* bounding_y, float* resized_mask, const int mask_size, const int image_height, const int image_width) {
  // output size should be result_num * mask_size * mask_size
  // original_mask should be result_num * image_height * image_width
  // bounding_x should be result_num * 2
  // bounding_y should be result_num * 2
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % mask_size;
    int h = (index / mask_size) % mask_size;
    int n = index / mask_size / mask_size;
    int bbox_x1 = bounding_x[n * 2];
    int bbox_x2 = bounding_x[n * 2 + 1];
    int bbox_y1 = bounding_y[n * 2];
    int bbox_y2 = bounding_y[n * 2 + 1];
    float bbox_width = bbox_x2 - bbox_x1 + 1.0;
    float bbox_height = bbox_y2 - bbox_y1 + 1.0;
    float ratio_w = bbox_width / static_cast<float>(mask_size);
    float ratio_h = bbox_height / static_cast<float>(mask_size);
    float inverse_x = bbox_x1 + static_cast<float>(w + 0.5) * ratio_w - 0.5;
    float inverse_y = bbox_y1 + static_cast<float>(h + 0.5) * ratio_h - 0.5;
    const float* offset_mask = original_mask + n * image_height * image_width;
    resized_mask[index] = bilinear_interpolate(offset_mask, image_height, image_width, inverse_y, inverse_x);
  }
}

void _mv(const float* all_boxes, const float* all_masks, const int all_boxes_num, const int* candidate_inds, const int* candidate_start, const float* candidate_weights, const int candidate_num, const float binary_thresh, const int image_height, const int image_width, const int box_dim, const int mask_size, const int result_num, float* finalize_output_mask, int* finalize_output_box, const int device_id) {

  // allocate device memory
  float* dev_boxes = NULL;
  float* dev_masks = NULL;
  int* dev_candidate_inds = NULL;
  float* dev_candidate_weights = NULL;
  int* dev_candidate_start = NULL;
  
  CUDA_CHECK(cudaMalloc(&dev_boxes, all_boxes_num * box_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dev_boxes, all_boxes, all_boxes_num * box_dim * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&dev_masks, all_boxes_num * mask_size * mask_size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dev_masks, all_masks, all_boxes_num * mask_size * mask_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&dev_candidate_inds, candidate_num * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dev_candidate_inds, candidate_inds, candidate_num * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&dev_candidate_weights, candidate_num * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dev_candidate_weights, candidate_weights, candidate_num * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&dev_candidate_start, result_num * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dev_candidate_start, candidate_start, result_num * sizeof(int),
                        cudaMemcpyHostToDevice));

  // 1. Masks are of size mask_size x mask_size, to do aggregation
  //    first resize them to image scale (image_height x image_width)
  //    result n x image_height x image_width buffer
  const int render_mask_num = all_boxes_num * image_height * image_width;
  float* dev_render_mask = NULL;
  CUDA_CHECK(cudaMalloc(&dev_render_mask, render_mask_num * sizeof(float)));
  
  mask_render<<<CAFFE_GET_BLOCKS(render_mask_num), CAFFE_CUDA_NUM_THREADS>>> (render_mask_num, dev_boxes, dev_masks, box_dim, mask_size, image_height, image_width, dev_render_mask);
  CUDA_POST_KERNEL_CHECK;

  // 2. After we get above buffer, we need to merge certain masks
  //    to get new masks according to candidate_weights and candidate_inds
  //    new_mask = \sum (old_mask * old_mask_weight)
  const int output_mask_num = result_num * image_height * image_width;
  float* dev_output_mask = NULL;
  CUDA_CHECK(cudaMalloc(&dev_output_mask, output_mask_num * sizeof(float)));
  mask_aggregate<<<CAFFE_GET_BLOCKS(output_mask_num), CAFFE_CUDA_NUM_THREADS>>> (output_mask_num, dev_render_mask, dev_output_mask, dev_candidate_inds, dev_candidate_start, dev_candidate_weights, image_height, image_width, binary_thresh);

  CUDA_POST_KERNEL_CHECK;

  // 3. After we get new masks buffer (result_num * image_height * image_width)
  //    we then find the mask boundary, this is achieved by two reduction operation
  //    then the tight mask boundary can be obtained
  int reduced_col_num = result_num * image_width;
  bool* reduced_col_buffer = NULL;
  CUDA_CHECK(cudaMalloc(&reduced_col_buffer, reduced_col_num * sizeof(bool)));
  reduce_mask_col<<<CAFFE_GET_BLOCKS(reduced_col_num), CAFFE_CUDA_NUM_THREADS>>> (reduced_col_num, dev_output_mask, image_height, image_width, binary_thresh, reduced_col_buffer);
  
  int reduced_bound_x_num = result_num * 2;
  int* reduced_bound_x = NULL;
  CUDA_CHECK(cudaMalloc(&reduced_bound_x, reduced_bound_x_num * sizeof(int)));
  reduce_bounding_x<<<CAFFE_GET_BLOCKS(reduced_bound_x_num), CAFFE_CUDA_NUM_THREADS>>> (reduced_bound_x_num, reduced_col_buffer, reduced_bound_x, image_width);
  
  // find vertical boundary
  int reduced_row_num = result_num * image_height;
  bool* reduced_row_buffer = NULL;
  CUDA_CHECK(cudaMalloc(&reduced_row_buffer, reduced_row_num * sizeof(bool)));
  reduce_mask_row<<<CAFFE_GET_BLOCKS(reduced_row_num), CAFFE_CUDA_NUM_THREADS>>> (reduced_row_num, dev_output_mask, image_height, image_width, binary_thresh, reduced_row_buffer);
  
  int reduced_bound_y_num = result_num * 2;
  int* reduced_bound_y = NULL;
  CUDA_CHECK(cudaMalloc(&reduced_bound_y, reduced_bound_y_num * sizeof(int)));
  reduce_bounding_y<<<CAFFE_GET_BLOCKS(reduced_bound_y_num), CAFFE_CUDA_NUM_THREADS>>> (reduced_bound_y_num, reduced_row_buffer, reduced_bound_y, image_height);

  // 4. Once we get tight mask boundary, we could use it to resize masks back
  //    to mask_size x mask_size
  const int resized_mask_num = result_num * mask_size * mask_size;
  float* resized_mask = NULL;
  CUDA_CHECK(cudaMalloc(&resized_mask, resized_mask_num * sizeof(float)));
  mask_resize<<<CAFFE_GET_BLOCKS(resized_mask_num), CAFFE_CUDA_NUM_THREADS>>> (resized_mask_num, dev_output_mask, reduced_bound_x, reduced_bound_y, resized_mask, mask_size, image_height, image_width);

  // copy back boxes to cpu
  int* cpu_bound_x = (int*) malloc(reduced_bound_x_num * sizeof(int));
  cudaMemcpy(cpu_bound_x, reduced_bound_x, reduced_bound_x_num * sizeof(int), cudaMemcpyDeviceToHost);
  int* cpu_bound_y = (int*) malloc(reduced_bound_y_num * sizeof(int));
  cudaMemcpy(cpu_bound_y, reduced_bound_y, reduced_bound_y_num * sizeof(int), cudaMemcpyDeviceToHost);
  int cnt = 0;
  for (int i = 0; i < result_num; i ++) {
    finalize_output_box[i*4] = cpu_bound_x[cnt];
    finalize_output_box[i*4+1] = cpu_bound_y[cnt];
    finalize_output_box[i*4+2] = cpu_bound_x[cnt+1];
    finalize_output_box[i*4+3] = cpu_bound_y[cnt+1];
    cnt += 2;
  }
  // copy back masks to cpu
  CUDA_CHECK(cudaMemcpy(finalize_output_mask, resized_mask, resized_mask_num * sizeof(float), 
                        cudaMemcpyDeviceToHost));
  
  // free gpu memories     
  CUDA_CHECK(cudaFree(dev_boxes));
  CUDA_CHECK(cudaFree(dev_masks));
  CUDA_CHECK(cudaFree(dev_candidate_inds));
  CUDA_CHECK(cudaFree(dev_candidate_start));
  CUDA_CHECK(cudaFree(dev_candidate_weights));
  CUDA_CHECK(cudaFree(dev_render_mask));
  CUDA_CHECK(cudaFree(resized_mask));
  CUDA_CHECK(cudaFree(dev_output_mask));
  CUDA_CHECK(cudaFree(reduced_col_buffer));
  CUDA_CHECK(cudaFree(reduced_bound_x));
  CUDA_CHECK(cudaFree(reduced_row_buffer));
  CUDA_CHECK(cudaFree(reduced_bound_y));
}
