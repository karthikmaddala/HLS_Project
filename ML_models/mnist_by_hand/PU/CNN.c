#include <math.h>
#include <string.h>
#include "CNN.h"
#include "layers.h"
#include "helpers.h"
#include "data.h"

k2c_tensor input_1_input;
float input_1_input_array[784];

k2c_tensor dense_output;
float dense_output_array[10];

void CNN(k2c_tensor_wa* input_1_input_wa, k2c_tensor_wa* dense_output_wa) {

    // #pragma HLS INTERFACE m_axi port = input_1_input_wa depth = 800
    // #pragma HLS INTERFACE m_axi port = dense_output_wa depth = 20

//#pragma HLS array_partition variable=conv2d_kernel_array complete
//#pragma HLS array_partition variable=dense_kernel_array cyclic factor=338

    for (int i = 0; i < 784; i++) {
#pragma HLS pipeline
#pragma HLS unroll factor=4
        input_1_input_array[i] = input_1_input_wa->array[i];
    }

    input_1_input.ndim = input_1_input_wa->ndim;
	input_1_input.numel = input_1_input_wa->numel;
	for (size_t unga = 0; unga < K2C_MAX_NDIM; ++unga) {
#pragma HLS unroll
		input_1_input.shape[unga] = input_1_input_wa->shape[unga];
	}

	dense_output.ndim = 1;
	dense_output.numel = 10;
	dense_output.shape[0] = 10;
	dense_output.shape[1] = 1;
	dense_output.shape[2] = 1;
	dense_output.shape[3] = 1;
	dense_output.shape[4] = 1;

    k2c_conv2d(&conv2d_output,&input_1_input,&conv2d_kernel,
               &conv2d_bias,conv2d_stride,conv2d_dilation, conv2d_output_array, input_1_input_array, conv2d_kernel_array, conv2d_bias_array);

    k2c_maxpool2d(&max_pooling2d_output,&conv2d_output,max_pooling2d_pool_size,
                  max_pooling2d_stride, max_pooling2d_output_array, conv2d_output_array);

    k2c_flatten(&flatten_output,&max_pooling2d_output, flatten_output_array, max_pooling2d_output_array);

    k2c_tensor dropout_output;
    dropout_output.ndim = flatten_output.ndim; // copy data into output struct
    dropout_output.numel = flatten_output.numel;
    // memcpy_hls(dropout_output.shape,flatten_output.shape,K2C_MAX_NDIM*sizeof(size_t));
    for (int k = 0; k < 5; k++) {
#pragma HLS unroll
        dropout_output.shape[k] = flatten_output.shape[k];
    }
    // dropout_output.array = &flatten_output_array[0]; // rename for clarity

    k2c_dense(&dense_output,&dropout_output,&dense_kernel,
              &dense_bias,dense_fwork, dense_output_array, flatten_output_array, dense_kernel_array, dense_bias_array);

    for (int j = 0; j < 10; j++) {
#pragma HLS unroll
        dense_output_wa->array[j] = dense_output_array[j];
    }

}

void CNN_initialize() {

}

void CNN_terminate() {

}
