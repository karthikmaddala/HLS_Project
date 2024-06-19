#pragma once
#include <stdlib.h>
#define K2C_MAX_NDIM 5
#define MAX_ARR_LENGTH 1000

/* tensor type for keras2c.*/
struct k2c_tensor
{
    /** Pointer to array of tensor values flattened in row major order. */
    float array[MAX_ARR_LENGTH];

    /** Rank of the tensor (number of dimensions). */
    size_t ndim;

    /** Number of elements in the tensor. */
    size_t numel;

    /** Array, size of the tensor in each dimension. */
    size_t shape[K2C_MAX_NDIM];
};
typedef struct k2c_tensor k2c_tensor;