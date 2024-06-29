#include <stddef.h>
#include <math.h>
#include "softmax.h"

void k2c_softmax_func(float * x, const size_t size) {

    float xmax = x[0];
    float sum = 0;
    for (size_t i=0; i < size; ++i) {
        #pragma HLS loop_tripcount min=10000 max=10000
        if (x[i]>xmax) {
            xmax = x[i];
        }
    }

    for (size_t i=0; i < size; ++i) {
        #pragma HLS loop_tripcount min=10000 max=10000
        x[i] = expf(x[i]-xmax);
    }

    for (size_t i=0; i < size; ++i) {
        #pragma HLS loop_tripcount min=10000 max=10000
        sum += x[i];
    }

    sum = 1.0f/sum;
    for (size_t i=0; i < size; ++i) {
        #pragma HLS loop_tripcount min=10000 max=10000
        x[i] = x[i]*sum;
    }
}
