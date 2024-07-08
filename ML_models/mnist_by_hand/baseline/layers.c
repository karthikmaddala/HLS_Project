#include "layers.h"
#include "helpers.h"
#include <stddef.h>

void k2c_conv2d(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* kernel,
                const k2c_tensor* bias, const size_t * stride, const size_t * dilation, 
                float* output_array, float* input_array, float* kernel_array, float* bias_array) {

    // memset(output_array,0,output->numel*sizeof(output_array[0]));
    for (int foo = 0; foo < output->numel; foo++) {
#pragma HLS loop_tripcount min=10816 max=10816
        output_array[foo] = 0;
    }

    const size_t out_rows = output->shape[0];
    const size_t out_cols = output->shape[1];
    const size_t out_channels = output->shape[2];
    const size_t in_channels = input->shape[2];

    for (size_t x0=0; x0 < out_rows; ++x0) {
#pragma HLS loop_tripcount min=26 max=26
        for (size_t x1=0; x1 < out_cols; ++x1) {
#pragma HLS loop_tripcount min=26 max=26
            for (size_t z0=0; z0 < kernel->shape[0]; ++z0) {
#pragma HLS loop_tripcount min=3 max=3
                for (size_t z1=0; z1 < kernel->shape[1]; ++z1) {
#pragma HLS loop_tripcount min=3 max=3
                    for (size_t q=0; q < in_channels; ++q) {
#pragma HLS loop_tripcount min=1 max=1
                        for (size_t k=0; k < out_channels; ++k) {
#pragma HLS loop_tripcount min=16 max=16
                            output_array[x0*(output->shape[2]*output->shape[1])
                                          + x1*(output->shape[2]) + k] +=
                                              kernel_array[z0*(kernel->shape[3]*kernel->shape[2]*kernel->shape[1])
                                                            + z1*(kernel->shape[3]*kernel->shape[2])
                                                            + q*(kernel->shape[3]) + k]*
                                              input_array[(x0*stride[0]
                                                            + dilation[0]*z0)*(input->shape[2]*input->shape[1])
                                                           + (x1*stride[1] + dilation[1]*z1)*(input->shape[2]) + q];
                        }
                    }
                }
            }
        }
    }

    k2c_bias_add(output,bias, output_array, bias_array);
    k2c_relu_func(output_array,output->numel);
}

void k2c_maxpool2d(k2c_tensor* output, const k2c_tensor* input, const size_t * pool_size,
                   const size_t * stride, float* output_array, float* input_array) {


    const size_t channels = input->shape[2];
    // i,j,l output indices
    /// i, k, m input indices
    for (size_t i=0; i< channels; ++i) {
#pragma HLS loop_tripcount min=16 max=16
        for (size_t j=0, k=0; j<output->shape[1]*channels;
                j+=channels, k+=channels*stride[1]) {
#pragma HLS loop_tripcount min=13 max=13
            for (size_t l=0, m=0; l<output->numel; l+=channels*output->shape[1],
                    m+=channels*input->shape[1]*stride[0]) {
#pragma HLS loop_tripcount min=13 max=13
                output_array[l+j+i] = input_array[m+k+i];
                for (size_t n=0; n<pool_size[1]*channels; n+=channels) {
#pragma HLS loop_tripcount min=2 max=2
                    for (size_t p=0; p<pool_size[0]*channels*input->shape[1];
                            p+=channels*input->shape[1]) {
#pragma HLS loop_tripcount min=2 max=2
                        if (output_array[l+j+i] < input_array[m+k+i+n+p]) {
                            output_array[l+j+i] = input_array[m+k+i+n+p];
                        }
                    }
                }
            }
        }
    }
}


void k2c_dense(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* kernel,
               const k2c_tensor* bias, float * fwork, float* output_array, float* input_array, float* kernel_array, float* bias_array) {

    if (input->ndim <=2) {
        size_t outrows;

        if (input->ndim>1) {
            outrows = input->shape[0];
        }
        else {
            outrows = 1;
        }
        const size_t outcols = kernel->shape[1];
        const size_t innerdim = kernel->shape[0];
        const size_t outsize = outrows*outcols;
//#pragma HLS dataflow
        k2c_affine_matmul(output_array,input_array,kernel_array,bias_array,
                          outrows,outcols,innerdim);
        k2c_softmax_func(output_array,outsize);
    }
    else {
        const size_t axesA[1] = {input->ndim-1};
        const size_t axesB[1] = {0};
        const size_t naxes = 1;
        const int normalize = 0;
//#pragma HLS dataflow
        k2c_dot(output, input, kernel, axesA, axesB, naxes, normalize, fwork, output_array, input_array, kernel_array);
        k2c_bias_add(output, bias, output_array, bias_array);
        k2c_softmax_func(output_array, output->numel);
    }
}

void k2c_flatten(k2c_tensor *output, const k2c_tensor* input, float* output_array, float* input_array) {

    for (size_t j = 0; j < input->numel; j++) {
//#pragma HLS pipeline
#pragma HLS loop_tripcount min=2704 max=2704
    	output_array[j] = input_array[j];
    }
    for (size_t i=0; i<input->ndim; ++i) {
#pragma HLS loop_tripcount min=5 max=5
        output->shape[i] = 1;
    }
    output->shape[0] = input->numel;
    output->numel = input->numel;
    output->ndim = 1;
}
