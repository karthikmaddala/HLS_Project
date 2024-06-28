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

    // Define tile sizes
    const size_t TILE_SIZE_I = 4;
    const size_t TILE_SIZE_J = 4;
    const size_t TILE_SIZE_L = 4;

    // Main loops with tiling and loop interchange
    for (size_t tl = 0; tl < output_numel; tl += TILE_SIZE_L * channels * output_shape[1]) {
        #pragma HLS loop_tripcount min=2 max=2
        for (size_t tj = 0; tj < output_shape[1] * channels; tj += TILE_SIZE_J * channels) {
            #pragma HLS loop_tripcount min=4 max=4
            for (size_t ti = 0; ti < channels; ti += TILE_SIZE_I) {
                #pragma HLS loop_tripcount min=7 max=7

                for (size_t l = tl, m = tl / (channels * output_shape[1]) * stride[0] * channels * input_shape[1]; l < tl + TILE_SIZE_L * channels * output_shape[1] && l < output_numel; l += channels * output_shape[1], m += channels * input_shape[1] * stride[0]) {
                    #pragma HLS loop_tripcount min=2 max=2
                    for (size_t j = tj, k = tj / channels * stride[1] * channels; j < tj + TILE_SIZE_J * channels && j < output_shape[1] * channels; j += channels, k += channels * stride[1]) {
                        #pragma HLS loop_tripcount min=4 max=4
                        for (size_t i = ti; i < ti + TILE_SIZE_I && i < channels; ++i) {
                            #pragma HLS loop_tripcount min=4 max=4
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
        }
    }
}
