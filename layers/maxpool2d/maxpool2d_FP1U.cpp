#include <stddef.h>
#include <float.h>
#include "maxpool.h"

void k2c_maxpool2d(float output_array[2704],
                size_t output_ndim,
                size_t output_numel,
                size_t output_shape[5],

                const float input_array[10816],
                const size_t input_ndim,
                const size_t input_numel,
                const size_t input_shape[5],

                const size_t pool_size[2],
                const size_t stride[2]) {

    const size_t channels = input_shape[2];

    // Flatten loops and unroll for maximum parallelism
    for (size_t i = 0; i < channels; ++i) {
        #pragma HLS loop_tripcount min=26 max=26
        for (size_t j = 0; j < output_shape[1]; ++j) {
            #pragma HLS loop_tripcount min=13 max=13
            for (size_t l = 0; l < output_shape[0]; ++l) {
                #pragma HLS loop_tripcount min=13 max=13

                float max_val = -FLT_MAX; // Initialize to the minimum possible float value
                size_t base_input_index = (l * stride[0] * input_shape[1] + j * stride[1]) * channels + i;
                size_t base_output_index = (l * output_shape[1] + j) * channels + i;

                #pragma HLS PIPELINE II=1
                for (size_t n = 0; n < pool_size[0]; ++n) {
#pragma HLS loop_tripcount min=2 max=2
                    #pragma HLS UNROLL
                    for (size_t p = 0; p < pool_size[1]; ++p) {
#pragma HLS loop_tripcount min=2 max=2
                        #pragma HLS UNROLL
                        size_t input_index = base_input_index + (n * input_shape[1] + p) * channels;
                        if (input_array[input_index] > max_val) {
                            max_val = input_array[input_index];
                        }
                    }
                }
                output_array[base_output_index] = max_val;
            }
        }
    }
}
