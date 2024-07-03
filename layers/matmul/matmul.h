#ifndef MATMUL_H_
#define MATMUL_H_

void k2c_matmul(float C[64*128], const float A[68*256], const float B[256*128], const size_t outrows,
                const size_t outcols, const size_t innerdim);

#endif