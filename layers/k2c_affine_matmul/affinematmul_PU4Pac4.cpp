#include <stddef.h>
#include "k2c_affine_matmul.h"

void k2c_affine_matmul(float C[64*128], const float A[68*256], const float B[256*128], const float d[128],
                       const size_t outrows, const size_t outcols, const size_t innerdim) {

    // Partition the arrays for better parallelism
#pragma HLS ARRAY_PARTITION variable=A cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=B cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=C cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=d complete

    // Make sure the output is empty
    for (size_t foo = 0; foo < outrows * outcols; ++foo) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=4
#pragma HLS loop_tripcount min=8192 max=8192
        C[foo] = 0;
    }

    for (size_t i = 0; i < outrows; ++i) {
#pragma HLS loop_tripcount min=64 max=64
        const size_t outrowidx = i * outcols;
        const size_t inneridx = i * innerdim;
        for (size_t j = 0; j < outcols; ++j) {
#pragma HLS loop_tripcount min=128 max=128
            float sum = 0; // Temporary variable to hold the sum
            for (size_t k = 0; k < innerdim; ++k) {
#pragma HLS loop_tripcount min=256 max=256
#pragma HLS pipeline II=1
#pragma HLS unroll factor=4
                sum += A[inneridx + k] * B[k * outcols + j];
            }
            C[outrowidx + j] = sum + d[j];
        }
    }
}