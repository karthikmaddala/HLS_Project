#include <stdio.h>
#include <stdlib.h>
#include "conv2d.h"

// Function to initialize data with random values
void initialize_data(float *data, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
}


void k2c_conv2d_test(float output_array[10],
                size_t output_ndim,
                size_t output_numel,
                size_t output_shape[5],

                const float input_array[784],
                const size_t input_ndim,
                const size_t input_numel,
                const size_t input_shape[5],
                
                const float kernel_array[144],
                const size_t kernel_ndim,
                const size_t kernel_numel,
                const size_t kernel_shape[5],


                const float bias_array[16],
                const size_t bias_ndim,
                const size_t bias_numel,
                const size_t bias_shape[5],

                const size_t stride[2], 
                const size_t dilation[2]) {

    // memset(output_array,0,output_numel*sizeof(output_array[0]));

    for (size_t foo = 0; foo < output_numel; ++foo) {
        output_array[foo] = 0;
    }

    const size_t out_rows = output_shape[0];
    const size_t out_cols = output_shape[1];
    const size_t out_channels = output_shape[2];
    const size_t in_channels = input_shape[2];

    for (size_t x0=0; x0 < out_rows; ++x0) {
        for (size_t x1=0; x1 < out_cols; ++x1) {
            for (size_t z0=0; z0 < kernel_shape[0]; ++z0) {
                for (size_t z1=0; z1 < kernel_shape[1]; ++z1) {
                    for (size_t q=0; q < in_channels; ++q) {
                        for (size_t k=0; k < out_channels; ++k) {
                            output_array[x0*(output_shape[2]*output_shape[1])
                                          + x1*(output_shape[2]) + k] +=
                                              kernel_array[z0*(kernel_shape[3]*kernel_shape[2]*kernel_shape[1])
                                                            + z1*(kernel_shape[3]*kernel_shape[2])
                                                            + q*(kernel_shape[3]) + k]*
                                              input_array[(x0*stride[0]
                                                            + dilation[0]*z0)*(input_shape[2]*input_shape[1])
                                                           + (x1*stride[1] + dilation[1]*z1)*(input_shape[2]) + q];
                        }
                    }
                }
            }
        }
    }
    // k2c_bias_add(output_array, output_numel, bias_array, bias_numel);
    for (size_t i=0; i<output_numel; i+=bias_numel) {
        for (size_t j=0; j<bias_numel; ++j) {
            output_array[i+j] += bias_array[j];
        }
    }
}

int main() {
    // Define sizes based on the given information
    size_t conv2d_output_ndim = 3;
    size_t conv2d_output_numel = 10816;
    size_t conv2d_output_shape[5] = {26, 26, 16, 1, 1};

    size_t conv2d_kernel_ndim = 4;
    size_t conv2d_kernel_numel = 144;
    size_t conv2d_kernel_shape[5] = {3, 3, 1, 16, 1};

    size_t test1_input_1_input_ndim = 3;
    size_t test1_input_1_input_numel = 784;
    size_t test1_input_1_input_shape[5] = {28, 28, 1, 1, 1};

    size_t bias_shape[5] = {16, 1, 1, 1, 1}; // Assuming bias shape based on output channels
    size_t bias_numel = 16; // Number of biases based on output channels
    size_t stride[2] = {1, 1}; // Assuming stride of 1
    size_t dilation[2] = {1, 1}; // Assuming no dilation

    // Declare arrays for inputs, kernel, output, and bias
    float input[784];
    float kernel[144];
    float output[10816];
    float bias[16];

    // Initialize input, kernel, and bias with random values
    initialize_data(input, test1_input_1_input_numel);
    initialize_data(kernel, conv2d_kernel_numel);
    initialize_data(bias, bias_numel);

    // Perform the convolution
    k2c_conv2d(output, conv2d_output_ndim, conv2d_output_numel, conv2d_output_shape,
               input, test1_input_1_input_ndim, test1_input_1_input_numel, test1_input_1_input_shape,
               kernel, conv2d_kernel_ndim, conv2d_kernel_numel, conv2d_kernel_shape,
               bias, 1, bias_numel, bias_shape,
               stride, dilation);

    float test_output[10816];

    k2c_conv2d_test(test_output, conv2d_output_ndim, conv2d_output_numel, conv2d_output_shape,
               input, test1_input_1_input_ndim, test1_input_1_input_numel, test1_input_1_input_shape,
               kernel, conv2d_kernel_ndim, conv2d_kernel_numel, conv2d_kernel_shape,
               bias, 1, bias_numel, bias_shape,
               stride, dilation);

    int mismatch = 0;

    // Print some of the output values for verification
    for (size_t i = 0; i < 10; i++) { // Print first 10 values for example
        if (output[i] != test_output[i]) {
            mismatch++;
        }
    }
    printf("Number of mismatches: %d\n", mismatch);

    return 0;
}
