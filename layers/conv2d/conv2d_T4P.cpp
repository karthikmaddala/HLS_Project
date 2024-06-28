#include <stddef.h>
#include <math.h>
#include "conv2d.h"


static const size_t tile_size = 4;
/** output_array is the Pointer to array of tensor values flattened in row major order and so on for other variables */
/** output_ndim is the Rank of the tensor (number of dimensions). and so on for other variables */
/** output_numel Number of elements in the tensor and so on for other variables */
/** output_shape is a Array of size 5 where each element is the size of the tensor in each dimension and so on for other variables */

size_t min(size_t a, size_t b) {
    return (a <= b) ? a: b;
}

void k2c_conv2d(float output_array[10816],
                size_t output_ndim,
                size_t output_numel,
                size_t output_shape[5],

                const float input_array[784],
                const size_t input_ndim,
                const size_t input_numel,
                const size_t input_shape[5],
                
                const float kernel_array[144],
                const size_t kernel_ndim,
                const size_t kernel_numel,
                const size_t kernel_shape[5],


                const float bias_array[16],
                const size_t bias_ndim,
                const size_t bias_numel,
                const size_t bias_shape[5],

                const size_t stride[2], 
                const size_t dilation[2]) {

    // memset(output_array,0,output_numel*sizeof(output_array[0]));

   for (int foo = 0; foo < output_numel; foo++) {
#pragma HLS loop_tripcount min=10816 max=10816
#pragma HLS pipeline rewind
        output_array[foo] = 0;
    }

    const size_t out_rows = output_shape[0];
    const size_t out_cols = output_shape[1];
    const size_t out_channels = output_shape[2];
    const size_t in_channels = input_shape[2];
    const size_t kernel_rows = kernel_shape[0];
    const size_t kernel_cols = kernel_shape[1];


    // Main tiled loops
loop1:    for (size_t x0_tile = 0; x0_tile < out_rows; x0_tile += tile_size) {
#pragma HLS loop_tripcount min=26/tile_size max=26/tile_size
// #pragma HLS loop_flatten off
loop2:        for (size_t x1_tile = 0; x1_tile < out_cols; x1_tile += tile_size) {
#pragma HLS loop_tripcount min=26/tile_size max=26/tile_size
// #pragma HLS loop_flatten off
loop3:            for (size_t z0 = 0; z0 < kernel_rows; ++z0) {
#pragma HLS loop_tripcount min=3 max=3
// #pragma HLS loop_flatten off
loop4:                for (size_t z1 = 0; z1 < kernel_cols; ++z1) {
#pragma HLS loop_tripcount min=3 max=3
// #pragma HLS loop_flatten off
loop5:                    for (size_t q_tile = 0; q_tile < in_channels; q_tile += tile_size) {
#pragma HLS loop_tripcount min=16/tile_size max=16/tile_size
#pragma HLS pipeline
loop6:                        for (size_t k_tile = 0; k_tile < out_channels; k_tile += tile_size) {
#pragma HLS loop_tripcount min=1/tile_size max=1/tile_size

                            // Inner tiled loops

tiled_loop1:                            for (size_t x0 = x0_tile; x0 < min(x0_tile + tile_size, out_rows); ++x0) {
#pragma HLS loop_tripcount min=tile_size max=tile_size
tiled_loop2:                                for (size_t x1 = x1_tile; x1 < min(x1_tile + tile_size, out_cols); ++x1) {
#pragma HLS loop_tripcount min=tile_size max=tile_size
tiled_loop3:                                    for (size_t q = q_tile; q < min(q_tile + tile_size, in_channels); ++q) {
#pragma HLS loop_tripcount min=tile_size max=tile_size
tiled_loop4:                                        for (size_t k = k_tile; k < min(k_tile + tile_size, out_channels); ++k) {
#pragma HLS loop_tripcount min=tile_size max=tile_size
// #pragma HLS loop_flatten
                                            output_array[x0*(out_channels*out_cols) + x1*(out_channels) + k] +=
                                                kernel_array[z0*(kernel_shape[3]*kernel_shape[2]*kernel_shape[1]) + z1*(kernel_shape[3]*kernel_shape[2]) + q*(kernel_shape[3]) + k] *
                                                input_array[(x0*stride[0] + dilation[0]*z0)*(input_shape[2]*input_shape[1]) + (x1*stride[1] + dilation[1]*z1)*(input_shape[2]) + q];
                                        }
                                    }
                                }
                            }

                        }
                    }
                }
            }
        }
    }

    // k2c_bias_add(output_array, output_numel, bias_array, bias_numel);
loop_bias1:    for (size_t i=0; i<output_numel; i+=bias_numel) {
#pragma HLS loop_tripcount min=10816 max=10816
#pragma HLS pipeline rewind
loop_bias2:        for (size_t j=0; j<bias_numel; ++j) {
#pragma HLS loop_tripcount min=16 max=16
#pragma HLS unroll
            output_array[i+j] += bias_array[j];
        }
    }
}
