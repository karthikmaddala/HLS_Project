#ifndef LAYERS_H_
#define LAYERS_H_

#include "CNN.h"
#include <stddef.h>

void k2c_conv2d(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* kernel,
                const k2c_tensor* bias, const size_t * stride, const size_t * dilation, 
                float* output_array, float* input_array, float* kernel_array, float* bias_array);
void k2c_maxpool2d(k2c_tensor* output, const k2c_tensor* input, const size_t * pool_size,
                   const size_t * stride, float* output_array, float* input_array);
void k2c_dense(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* kernel,
               const k2c_tensor* bias, float * fwork, float* output_array, float* input_array, float* kernel_array, float* bias_array);
void k2c_flatten(k2c_tensor *output, const k2c_tensor* input, float* output_array, float* input_array);


#endif
