#include <stddef.h>
#include "relu.h"

void k2c_relu_func(float * x, const size_t size) {

    for (size_t i=0; i < size; ++i) {
#pragma HLS loop_tripcount min=10000 max=10000
        if (x[i] <= 0.0f) {
            x[i] = 0.0f;
        }
    }
}