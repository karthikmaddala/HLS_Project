#ifndef DOT_H_
#define DOT_H_
#include <stddef.h>
#define K2C_MAX_NDIM 5

size_t k2c_sub2idx(const size_t * sub, const size_t * shape, const size_t ndim);
void k2c_matmul(float * C, const float * A, const float * B, const size_t outrows,
                const size_t outcols, const size_t innerdim);
void k2c_idx2sub(const size_t idx, size_t * sub, const size_t * shape, const size_t ndim);

void k2c_dot(float* C_array,
            size_t C_ndim,
            size_t C_numel,
            size_t C_shape[5], 

            const float* A_array,
            const size_t A_ndim,
            const size_t A_numel,
            const size_t A_shape[5], 

            const float* B_array,
            const size_t B_ndim,
            const size_t B_numel,
            const size_t B_shape[5], 

            const size_t * axesA,
            const size_t * axesB, 
            const size_t naxes, 
            const int normalize, 
            float * fwork);

#endif