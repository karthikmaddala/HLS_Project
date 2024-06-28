#include <stdio.h>
#include <stdlib.h>
#include "header/maxpool2d.h"

// Function to initialize data with random values
void initialize_data(float *data, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
}


void k2c_maxpool2d_test(float output_array[2704],
                size_t output_ndim,
                size_t output_numel,
                size_t output_shape[5],

                const float input_array[10816],
                const size_t input_ndim,
                const size_t input_numel,
                const size_t input_shape[5],

                const size_t pool_size[2],
                const size_t stride[2]) {


    const size_t channels = input_shape[2];
    // i,j,l output indices
    /// i, k, m input indices
    for (size_t i=0; i< channels; ++i) {
        for (size_t j=0, k=0; j<output_shape[1]*channels;
                j+=channels, k+=channels*stride[1]) {
            for (size_t l=0, m=0; l<output_numel; l+=channels*output_shape[1],
                    m+=channels*input_shape[1]*stride[0]) {
                output_array[l+j+i] = input_array[m+k+i];
                for (size_t n=0; n<pool_size[1]*channels; n+=channels) {
                    for (size_t p=0; p<pool_size[0]*channels*input_shape[1];
                            p+=channels*input_shape[1]) {
                        if (output_array[l+j+i] < input_array[m+k+i+n+p]) {
                            output_array[l+j+i] = input_array[m+k+i+n+p];
                        }
                    }
                }
            }
        }
    }
}

int main() {
    // Define sizes based on the given information
    size_t conv2d_output_ndim = 3;
    size_t conv2d_output_numel = 10816;
    size_t conv2d_output_shape[5] = {26, 26, 16, 1, 1};

    size_t max_pooling2d_stride[2] = {2,2};
    size_t max_pooling2d_pool_size[2] = {2,2};

    size_t max_pooling2d_output_ndim = 3;
    size_t max_pooling2d_output_numel = 2704;
    size_t max_pooling2d_output_shape[5] = {13,13,16, 1, 1};
    // k2c_tensor max_pooling2d_output = {3, 2704, {13,13,16, 1, 1}};


    float output[2704];
    float input[10816];

    initialize_data(input, conv2d_output_numel);

    k2c_maxpool2d(output, max_pooling2d_output_ndim, max_pooling2d_output_numel, max_pooling2d_output_shape,
    input, conv2d_output_ndim, conv2d_output_numel, conv2d_output_shape, max_pooling2d_pool_size, max_pooling2d_stride);

    float test_output[2704];

    k2c_maxpool2d(test_output, max_pooling2d_output_ndim, max_pooling2d_output_numel, max_pooling2d_output_shape,
    input, conv2d_output_ndim, conv2d_output_numel, conv2d_output_shape, max_pooling2d_pool_size, max_pooling2d_stride);
    int mismatch = 0;

    for (size_t i = 0; i < 2704; i++) {
        if (output[i] != test_output[i]) {
            mismatch++;
        }
    }
    printf("Number of mismatches: %d\n", mismatch);

    return 0;
}