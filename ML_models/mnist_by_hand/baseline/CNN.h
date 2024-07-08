#ifndef CNN_H_
#define CNN_H_
#include <stddef.h>
#define K2C_MAX_NDIM 5
typedef struct k2c_tensor
{
    /** Rank of the tensor (number of dimensions). */
    size_t ndim;

    /** Number of elements in the tensor. */
    size_t numel;

    /** Array, size of the tensor in each dimension. */
    size_t shape[K2C_MAX_NDIM];
}k2c_tensor;

typedef struct k2c_tensor_wa //tensor with array
{
    float array[784];

    /** Rank of the tensor (number of dimensions). */
    size_t ndim;

    /** Number of elements in the tensor. */
    size_t numel;

    /** Array, size of the tensor in each dimension. */
    size_t shape[K2C_MAX_NDIM];
}k2c_tensor_wa;

void CNN(k2c_tensor_wa* input_1_input, k2c_tensor_wa* dense_output);

void CNN_initialize();
void CNN_terminate();

#endif
