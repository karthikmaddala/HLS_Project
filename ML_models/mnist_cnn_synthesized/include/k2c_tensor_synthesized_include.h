#ifndef K2C_TENSOR_SYNTHESIZED_INCLUDE_H_
#define K2C_TENSOR_SYNTHESIZED_INCLUDE_H_
#pragma once
#include <stdlib.h>
#define K2C_MAX_NDIM 5
#define MAX_ARR_LENGTH 21632


typedef struct k2c_tensor //tensor with array
{
    float array[MAX_ARR_LENGTH];

    /** Rank of the tensor (number of dimensions). */
    size_t ndim;

    /** Number of elements in the tensor. */
    size_t numel;

    /** Array, size of the tensor in each dimension. */
    size_t shape[K2C_MAX_NDIM];
}k2c_tensor;

k2c_tensor conv2d_output = {0};

k2c_tensor dense_kernel = {0};

k2c_tensor conv2d_1_kernel = {0};

//#undef MAX_ARR_LENGTH
//#define MAX_ARR_LENGTH 7744

k2c_tensor conv2d_1_output = {0};

k2c_tensor max_pooling2d_output = {0};

k2c_tensor max_pooling2d_1_output = {0};

k2c_tensor flatten_output = {0};

//#undef MAX_ARR_LENGTH
//#define MAX_ARR_LENGTH 784

k2c_tensor conv2d_kernel = {0};

k2c_tensor conv2d_bias = {0};

k2c_tensor conv2d_1_bias = {0};

k2c_tensor dense_bias = {0};

//test_bench cases

k2c_tensor test1_input_1_input = {0};
k2c_tensor keras_dense_test1 = {0};
k2c_tensor c_dense_test1 = {0};
k2c_tensor test2_input_1_input = {0};
k2c_tensor keras_dense_test2 = {0};
k2c_tensor c_dense_test2 = {0};
k2c_tensor test3_input_1_input = {0};
k2c_tensor keras_dense_test3 = {0};
k2c_tensor c_dense_test3 = {0};
k2c_tensor test4_input_1_input = {0};
k2c_tensor keras_dense_test4 = {0};
k2c_tensor c_dense_test4 = {0};
k2c_tensor test5_input_1_input = {0};
k2c_tensor keras_dense_test5 = {0};
k2c_tensor c_dense_test5 = {0};
k2c_tensor test6_input_1_input = {0};
k2c_tensor keras_dense_test6 = {0};
k2c_tensor c_dense_test6 = {0};
k2c_tensor test7_input_1_input = {0};
k2c_tensor keras_dense_test7 = {0};
k2c_tensor c_dense_test7 = {0};
k2c_tensor test8_input_1_input = {0};
k2c_tensor keras_dense_test8 = {0};
k2c_tensor c_dense_test8 = {0};
k2c_tensor test9_input_1_input = {0};
k2c_tensor keras_dense_test9 = {0};
k2c_tensor c_dense_test9 = {0};
k2c_tensor test10_input_1_input = {0};
k2c_tensor keras_dense_test10 = {0};
k2c_tensor c_dense_test10 = {0};

#endif
