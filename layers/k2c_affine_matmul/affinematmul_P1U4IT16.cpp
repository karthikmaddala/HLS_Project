#include <stddef.h>
#include "k2c_affine_matmul.h"

#define TILE_SIZE 16

void k2c_affine_matmul(float C[64*128], const float A[68*256], const float B[256*128], const float d[128],
                       const size_t outrows, const size_t outcols, const size_t innerdim) {

    // Make sure the output is empty
    for (size_t foo = 0; foo < outrows * outcols; ++foo) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=4
#pragma HLS loop_tripcount min=8192 max=8192
        C[foo] = 0;
    }

    for (size_t jj = 0; jj < outcols; jj += TILE_SIZE) {
#pragma HLS loop_tripcount min=8 max=8
        for (size_t ii = 0; ii < outrows; ii += TILE_SIZE) {
#pragma HLS loop_tripcount min=4 max=4
            for (size_t kk = 0; kk < innerdim; kk += TILE_SIZE) {
#pragma HLS loop_tripcount min=16 max=16
                for (size_t j = jj; j < jj + TILE_SIZE && j < outcols; ++j) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=4
#pragma HLS loop_tripcount min=16 max=16
                    for (size_t i = ii; i < ii + TILE_SIZE && i < outrows; ++i) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=4
#pragma HLS loop_tripcount min=16 max=16
                        const size_t outrowidx = i * outcols;
                        const size_t inneridx = i * innerdim;
                        float sum = 0; // Temporary variable to hold the sum
                        for (size_t k = kk; k < kk + TILE_SIZE && k < innerdim; ++k) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=4
#pragma HLS loop_tripcount min=16 max=16
                            sum += A[inneridx + k] * B[k * outcols + j];
                        }
                        C[outrowidx + j] += sum;
                    }
                }
            }
        }
    }

    // Add bias term
    for (size_t i = 0; i < outrows; ++i) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=4
#pragma HLS loop_tripcount min=64 max=64
        for (size_t j = 0; j < outcols; ++j) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=4
#pragma HLS loop_tripcount min=128 max=128
            C[i * outcols + j] += d[j];
        }
    }
}
