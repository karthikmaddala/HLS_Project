#include <stddef.h>
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

    // Partition the arrays to allow parallel access
    #pragma HLS ARRAY_PARTITION variable=output_array cyclic factor=2
    #pragma HLS ARRAY_PARTITION variable=input_array cyclic factor=2

    // Main loops
    for (size_t i = 0; i < channels; ++i) {
        #pragma HLS loop_tripcount min=26 max=26
        for (size_t j = 0, k = 0; j < output_shape[1] * channels; j += channels, k += channels * stride[1]) {
            #pragma HLS loop_tripcount min=13 max=13
            for (size_t l = 0, m = 0; l < output_numel; l += channels * output_shape[1], m += channels * input_shape[1] * stride[0]) {
                #pragma HLS loop_tripcount min=8 max=8
                #pragma HLS PIPELINE II=1
                float max_val = input_array[m + k + i];
                for (size_t n = 0; n < pool_size[1] * channels; n += channels) {
                    #pragma HLS loop_tripcount min=2 max=2
                    for (size_t p = 0; p < pool_size[0] * channels * input_shape[1]; p += channels * input_shape[1]) {
                        #pragma HLS loop_tripcount min=2 max=2
                        if (max_val < input_array[m + k + i + n + p]) {
                            max_val = input_array[m + k + i + n + p];
                        }
                    }
                }
                output_array[l + j + i] = max_val;
            }
        }
    }
}