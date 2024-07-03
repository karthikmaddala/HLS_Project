#include <stddef.h>
#include "matmul.h"

#define TILE_SIZE 4

void k2c_matmul(float C[64*128], const float A[68*256], const float B[256*128], const size_t outrows,
                const size_t outcols, const size_t innerdim) {

    // Initialize the output matrix C to zero
    for (size_t foo = 0; foo < outrows * outcols; ++foo) {
#pragma HLS loop_tripcount min=8192 max=8192
#pragma HLS pipeline
#pragma HLS unroll factor=4
        C[foo] = 0;
    }

    // Perform matrix multiplication with loop tiling
    for (size_t jj = 0; jj < outcols; jj += TILE_SIZE) {
#pragma HLS loop_tripcount min=8 max=8
        for (size_t ii = 0; ii < outrows; ii += TILE_SIZE) {
#pragma HLS loop_tripcount min=4 max=4
            for (size_t j = jj; j < jj + TILE_SIZE && j < outcols; ++j) {
#pragma HLS loop_tripcount min=16 max=16
#pragma HLS pipeline II=1
                for (size_t i = ii; i < ii + TILE_SIZE && i < outrows; ++i) {
#pragma HLS loop_tripcount min=16 max=16
#pragma HLS pipeline II=1
                    const size_t outrowidx = i * outcols;
                    const size_t inneridx = i * innerdim;

                    float sum = 0; // Temporary variable to accumulate the sum
                    for (size_t k = 0; k < innerdim; ++k) {
#pragma HLS loop_tripcount min=256 max=256
#pragma HLS unroll factor=4
                        sum += A[inneridx + k] * B[k * outcols + j];
                    }
                    C[outrowidx + j] = sum; // Write the accumulated sum to C
                }
            }
        }
    }
}
