#include <stddef.h>
#include <math.h>
#include "softmax.h"

void k2c_softmax_func(float x[10000], const size_t size) {

    float xmax = x[0];
    float sum = 0;

    // Find xmax and compute exponentials in one loop
    for (size_t i=0; i < size; ++i) {
        #pragma HLS loop_tripcount min=10000 max=10000
        #pragma HLS PIPELINE II=1
        if (x[i] > xmax) {
            xmax = x[i];
        }
    }

    // Compute exponentials and partial sum in one loop
    for (size_t i=0; i < size; ++i) {
        #pragma HLS loop_tripcount min=10000 max=10000
        #pragma HLS PIPELINE II=1
        x[i] = expf(x[i] - xmax);
        sum += x[i];
    }

    sum = 1.0f / sum;

    // Normalize in one loop
    for (size_t i=0; i < size; ++i) {
        #pragma HLS loop_tripcount min=10000 max=10000
        #pragma HLS PIPELINE II=1
        x[i] = x[i] * sum;
    }
}
