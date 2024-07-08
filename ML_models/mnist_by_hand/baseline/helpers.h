#ifndef HELPERS_H_
#define HELPERS_H_

#include <stddef.h>

#define tile_size 4

size_t min(size_t a, size_t b);
void k2c_idx2sub(const size_t idx, size_t * sub, const size_t * shape, const size_t ndim);
size_t k2c_sub2idx(const size_t * sub, const size_t * shape, const size_t ndim);
void k2c_relu_func(float * x, const size_t size);
void k2c_softmax_func(float * x, const size_t size);
void k2c_matmul(float * C, const float * A, const float * B, const size_t outrows, const size_t outcols, const size_t innerdim);
void k2c_affine_matmul(float * C, const float * A, const float * B, const float * d, const size_t outrows,const size_t outcols, const size_t innerdim);
void k2c_dot(k2c_tensor* C, const k2c_tensor* A, const k2c_tensor* B, const size_t * axesA, const size_t * axesB, const size_t naxes, const int normalize, float * fwork, float* C_array, float* A_array, float* B_array);
void k2c_bias_add(k2c_tensor* A, const k2c_tensor* b, float* A_array, float* b_array);

#endif
