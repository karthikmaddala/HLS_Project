#include <stddef.h>
#include "relu.h"

void k2c_relu_func(float x[10000], const size_t size) {
#pragma HLS array_partition variable=x cyclic factor=10

    for (size_t i=0; i < size; ++i) {
#pragma HLS loop_tripcount min=10000 max=10000
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
        if (x[i] <= 0.0f) {
            x[i] = 0.0f;
        }
    }
}
