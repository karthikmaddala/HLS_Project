#ifndef MAXPOOL_H_
#define MAXPOOL_H_
#include <stddef.h>
void k2c_maxpool2d(float output_array[2704],
                size_t output_ndim,
                size_t output_numel,
                size_t output_shape[5],

                const float input_array[10816],
                const size_t input_ndim,
                const size_t input_numel,
                const size_t input_shape[5],

                const size_t pool_size[2],
                const size_t stride[2]);

#endif