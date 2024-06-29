#include <stddef.h>
#include <math.h>
#include <stddef.h>
#include "tanh.h"

void k2c_tanh_func(float x[10000], const size_t size) {
		#pragma HLS ARRAY_PARTITION variable=x  block factor=4 dim=1
		#pragma HLS PIPELINE II=1

    for (size_t i=0; i<size; ++i) {
		#pragma HLS UNROLL factor=4
        #pragma HLS loop_tripcount min=10000 max=10000
        x[i] = tanhf(x[i]);
    }
}

