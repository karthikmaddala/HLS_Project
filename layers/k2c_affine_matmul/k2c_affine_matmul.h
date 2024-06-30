#ifndef MATMUL_H_
#define MATMUL_H_

#include <stddef.h>

void k2c_affine_matmul(float * C, const float * A, const float * B, const float * d,
                       const size_t outrows,const size_t outcols, const size_t innerdim);

#endif