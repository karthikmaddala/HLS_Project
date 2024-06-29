#include <stddef.h>
#include <math.h>
#include "softmax.h"

void k2c_softmax_func(float * x, const size_t size) {
    #pragma HLS INTERFACE s_axilite port=return
    #pragma HLS INTERFACE s_axilite port=size
    #pragma HLS INTERFACE m_axi depth=10000 port=x

    float xmax = x[0];
    float sum = 0;

    // Loop to find xmax with pipelining and unrolling
    for (size_t i = 0; i < size; ++i) {
        #pragma HLS loop_tripcount min=10000 max=10000
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=4
        if (x[i] > xmax) {
            xmax = x[i];
        }
    }

    // Loop to compute exponentials with pipelining and unrolling
    for (size_t i = 0; i < size; ++i) {
        #pragma HLS loop_tripcount min=10000 max=10000
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=4
        x[i] = expf(x[i] - xmax);
    }

    // Loop to sum exponentials with pipelining and unrolling
    for (size_t i = 0; i < size; ++i) {
        #pragma HLS loop_tripcount min=10000 max=10000
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=4
        sum += x[i];
    }

    sum = 1.0f / sum;

    // Loop to normalize with pipelining and unrolling
    for (size_t i = 0; i < size; ++i) {
        #pragma HLS loop_tripcount min=10000 max=10000
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=4
        x[i] = x[i] * sum;
    }
}
