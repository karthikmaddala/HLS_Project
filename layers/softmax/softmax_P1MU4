#include <stddef.h>
#include <math.h>
#include "softmax.h"

void k2c_softmax_func(float x[10000], const size_t size) {

    float xmax = x[0];
    float sum = 0;

    // Find xmax with pipelining and unrolling
    for (size_t i = 0; i < size; ++i) {
        #pragma HLS loop_tripcount min=10000 max=10000
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=4
        if (x[i] > xmax) {
            xmax = x[i];
        }
    }

    // Compute exponentials and partial sum with pipelining and unrolling
    for (size_t i = 0; i < size; ++i) {
        #pragma HLS loop_tripcount min=10000 max=10000
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=4
        x[i] = expf(x[i] - xmax);
        sum += x[i];
    }

    sum = 1.0f / sum;

    // Normalize with pipelining and unrolling
    for (size_t i = 0; i < size; ++i) {
        #pragma HLS loop_tripcount min=10000 max=10000
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=4
        x[i] = x[i] * sum;
    }
}
