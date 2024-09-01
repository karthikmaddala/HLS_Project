
#pragma once
#include <stdlib.h>


/**
 * Rank of largest keras2c tensors.
 * mostly used to ensure a standard size for the tensor.shape array.
 */
#define K2C_MAX_NDIM 5
#define MAX_ARRAY_SIZE 1690


/**
 * tensor type for keras2c.
 */
struct k2c_tensor
{
    /** Rank of the tensor (number of dimensions). */
    size_t ndim;

    /** Number of elements in the tensor. */
    size_t numel;

    /** Array, size of the tensor in each dimension. */
    size_t shape[K2C_MAX_NDIM];
};


typedef struct k2c_tensor_wa
{
    float array[MAX_ARRAY_SIZE];
    size_t ndim;
    size_t numel;
    size_t shape[K2C_MAX_NDIM];
}k2c_tensor_wa;

typedef struct k2c_tensor k2c_tensor;

void finalmusic1(k2c_tensor_wa* conv2d_4_input_input, k2c_tensor_wa* dense_3_output);
void finalmusic1_initialize(); 
void finalmusic1_terminate(); 
