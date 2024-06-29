#include <stddef.h>
#include <math.h>
#include <stddef.h>
#include "tanh.h"

void k2c_tanh_func(float x[10000], const size_t size) {
		
    for (size_t i=0; i<size; ++i) {
	    #pragma HLS PIPELINE II=1
		#pragma HLS UNROLL factor=4
        #pragma HLS loop_tripcount min=10000 max=10000
        x[i] = tanhf(x[i]);
    }
}

