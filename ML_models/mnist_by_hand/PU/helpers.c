#include "CNN.h"
#include "helpers.h"
#include <math.h>

size_t min(size_t a, size_t b) {
	return (a <= b) ? a : b;
}

void k2c_idx2sub(const size_t idx, size_t * sub, const size_t * shape, const size_t ndim) {

    size_t idx2 = idx;
    for (int i=ndim-1; i>=0; --i) {
#pragma HLS unroll factor=2
#pragma HLS loop_tripcount min=5 max=5
        sub[i] = idx2%shape[i];
        idx2 /= shape[i];
    }
}

size_t k2c_sub2idx(const size_t * sub, const size_t * shape, const size_t ndim) {

    size_t idx = 0;
    size_t temp = 0;
    for (size_t i=0; i<ndim; ++i) {
#pragma HLS loop_tripcount min=5 max=5
#pragma HLS unroll factor=2
        temp = sub[i];
        for (size_t j=ndim-1; j>i; --j) {
#pragma HLS loop_tripcount min=5 max=5
            temp *= shape[j];
        }
        idx += temp;
    }
    return idx;
}

void k2c_relu_func(float * x, const size_t size) {

    for (size_t i=0; i < size; ++i) {
//#pragma HLS pipeline
#pragma HLS loop_tripcount min=10816 max=10816
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
        if (x[i] <= 0.0f) {
            x[i] = 0.0f;
        }
    }
}

void k2c_softmax_func(float * x, const size_t size) {

    float xmax = x[0];
    float sum = 0;

    // Find xmax with pipelining and unrolling
    for (size_t i1 = 0; i1 < size; ++i1) {
        #pragma HLS loop_tripcount min=10 max=10
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=2
        if (x[i1] > xmax) {
            xmax = x[i1];
        }
    }

    // Compute exponentials and partial sum with pipelining and unrolling
    for (size_t i2 = 0; i2 < size; ++i2) {
        #pragma HLS loop_tripcount min=10 max=10
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=2
        x[i2] = expf(x[i2] - xmax);
        sum += x[i2];
    }

    sum = 1.0f / sum;

    // Normalize with pipelining and unrolling
    for (size_t i3 = 0; i3 < size; ++i3) {
        #pragma HLS loop_tripcount min=10 max=10
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=2
        x[i3] = x[i3] * sum;
    }
}

//void k2c_matmul(float *C, const float *A, const float *B, const size_t outrows, const size_t outcols, const size_t innerdim)
//{
//
//    // Initialize the output matrix
//    for (size_t a10 = 0; a10 < outrows * outcols; a10++)
//    {
//        C[a10] = 0;
//    }
//
//    // Perform matrix multiplication with tiling
//    for (size_t i = 0; i < outrows; i += tile_size)
//    {
//        for (size_t j = 0; j < outcols; j += tile_size)
//        {
//            for (size_t k = 0; k < innerdim; k += tile_size)
//            {
//                // For each tile
//                size_t i_end = min(i + tile_size, outrows);
//                size_t j_end = min(j + tile_size, outcols);
//                size_t k_end = min(k + tile_size, innerdim);
//#pragma HLS pipeline
//                for (size_t ii = i; ii < i_end; ++ii)
//                {
//                    size_t outrowidx = ii * outcols;
//                    size_t inneridx = ii * innerdim;
//
//                    for (size_t kk = k; kk < k_end; ++kk)
//                    {
//                        float a_val = A[inneridx + kk];
//
//                        for (size_t jj = j; jj < j_end; ++jj)
//                        {
//                            C[outrowidx + jj] += a_val * B[kk * outcols + jj];
//                        }
//                    }
//                }
//            }
//        }
//    }
//}

//void k2c_affine_matmul(float * C, const float * A, const float * B, const float * d,
//                       const size_t outrows,const size_t outcols, const size_t innerdim) {
//
//    // make sure output is empty
//    // memset(C, 0, outrows*outcols*sizeof(C[0]));
//    for (int foo = 0; foo < outrows*outcols; foo++) {
//        C[foo] = 0;
//    }
//    for (size_t ii = 0 ; ii < outrows; ii += tile_size) {
//		for (size_t jj = 0; jj < outcols; jj += tile_size) {
//			for (size_t kk = 0;  kk < innerdim; kk += tile_size) {
//#pragma HLS pipeline
//				for (size_t i = ii; i < min(ii + tile_size, outrows); ++i) {
//					const size_t outrowidx = i*outcols;
//					const size_t inneridx = i*innerdim;
//					for (size_t j = jj; j < min(jj + tile_size, outcols); ++j) {
//                        float sum = 0.0f;
//						for (size_t k = kk; k < min(kk + tile_size, innerdim); ++k) {
//							sum += A[inneridx+k] * B[k*outcols+j];
//						}
//						C[outrowidx+j] += sum;
//					}
//				}
//#pragma HLS pipeline
//                for (size_t ll = ii; ll < min(ii + tile_size, outrows); ++ll) {
//                    const size_t outrowidx_b = ll * outcols;
//                    for (size_t mm = jj; mm < min(jj + tile_size, outcols); ++mm) {
//                        C[outrowidx_b + mm] += d[mm];
//                    }
//                }
//			}
//		}
//	}
//}

void k2c_affine_matmul(float *C, const float *A, const float *B, const float *d, const size_t outrows, const size_t outcols, const size_t innerdim)
{

    // make sure output is empty
    // memset(C, 0, outrows*outcols*sizeof(C[0]));
    for (size_t p = 0; p < outrows * outcols; p++)
    {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=2
#pragma HLS loop_tripcount min=27040 max=27040
        C[p] = 0;
    }
    for (size_t i = 0; i < outrows; ++i)
    {
#pragma HLS loop_tripcount min=2704 max=2704
        const size_t outrowidx = i * outcols;
        const size_t inneridx = i * innerdim;
        for (size_t j = 0; j < outcols; ++j)
        {
#pragma HLS loop_tripcount min=10 max=10
        	float sum = 0.0f;
            for (size_t k = 0; k < innerdim; ++k)
            {
#pragma HLS pipeline
#pragma HLS loop_tripcount min=2704 max=2704
                sum += A[inneridx + k] * B[k * outcols + j];
            }
            C[outrowidx + j] += sum;
            C[outrowidx + j] += d[j];
        }
    }
}
//
void k2c_matmul(float *C, const float *A, const float *B, const size_t outrows, const size_t outcols, const size_t innerdim)
{

    // make sure output is empty
    // memset(C, 0, outrows*outcols*sizeof(C[0]));
    for (size_t foo = 0; foo < outrows * outcols; ++foo) {
#pragma HLS loop_tripcount min=27040 max=27040
#pragma HLS pipeline
#pragma HLS unroll factor=4
        C[foo] = 0;
    }

    // Perform matrix multiplication
    for (size_t j = 0; j < outcols; ++j) {
#pragma HLS loop_tripcount min=10 max=10
        for (size_t i = 0; i < outrows; ++i) {
#pragma HLS loop_tripcount min=2704 max=2704
            const size_t outrowidx = i * outcols;
            const size_t inneridx = i * innerdim;

            float sum = 0; // Temporary variable to accumulate the sum
            for (size_t k = 0; k < innerdim; ++k) {
#pragma HLS loop_tripcount min=2704 max=2704
#pragma HLS pipeline II=1
                sum += A[inneridx + k] * B[k * outcols + j];
            }
            C[outrowidx + j] = sum; // Write the accumulated sum to C
        }
    }
}

void k2c_dot(k2c_tensor* C, const k2c_tensor* A, const k2c_tensor* B, const size_t * axesA,
             const size_t * axesB, const size_t naxes, const int normalize, float * fwork,
             float* C_array, float* A_array, float* B_array) {

    size_t permA[K2C_MAX_NDIM];
    size_t permB[K2C_MAX_NDIM];
    size_t prod_axesA = 1;
    size_t prod_axesB = 1;
    size_t free_axesA, free_axesB;
    size_t freeA[K2C_MAX_NDIM];
    size_t freeB[K2C_MAX_NDIM];
    size_t count;
    int isin;
    size_t newshpA[K2C_MAX_NDIM];
    size_t newshpB[K2C_MAX_NDIM];
    const size_t ndimA = A->ndim;
    const size_t ndimB = B->ndim;
    float *reshapeA = &fwork[0];   // temp working storage
    float *reshapeB = &fwork[A->numel];
    size_t Asub[K2C_MAX_NDIM];
    size_t Bsub[K2C_MAX_NDIM];
    // find which axes are free (ie, not being summed over)
    count=0;
    for (size_t i=0; i<ndimA; ++i) {
#pragma HLS unroll factor=2
#pragma HLS loop_tripcount min=5 max=5
        isin = 0;
        for (size_t j=0; j<naxes; ++j) {
#pragma HLS loop_tripcount min=1 max=1
            if (i==axesA[j]) {
                isin=1;
            }
        }
        if (!isin) {
            freeA[count] = i;
            ++count;
        }
    }
    count=0;
    for (size_t k=0; k<ndimB; ++k) {
#pragma HLS unroll
#pragma HLS loop_tripcount min=5 max=5
        isin = 0;
        for (size_t l=0; l<naxes; ++l) {
#pragma HLS loop_tripcount min=1 max=1
            if (k==axesB[l]) {
                isin=1;
            }
        }
        if (!isin) {
            freeB[count] = k;
            ++count;
        }
    }

    // number of elements in inner dimension
    for (size_t m=0; m < naxes; ++m) {
#pragma HLS loop_tripcount min=1 max=1
        prod_axesA *= A->shape[axesA[m]];
    }
    for (size_t n=0; n < naxes; ++n) {
#pragma HLS loop_tripcount min=1 max=1
        prod_axesB *= B->shape[axesB[n]];
    }
    // number of elements in free dimension
    free_axesA = A->numel/prod_axesA;
    free_axesB = B->numel/prod_axesB;
    // find permutation of axes to get into matmul shape
    for (size_t o=0; o<ndimA-naxes; ++o) {
#pragma HLS unroll factor=2
#pragma HLS loop_tripcount min=4 max=4
        permA[o] = freeA[o];
    }
    for (size_t p=ndimA-naxes, q=0; p<ndimA; ++p, ++q) {
#pragma HLS loop_tripcount min=1 max=1
        permA[p] = axesA[q];
    }
    for (size_t r=0; r<naxes; ++r) {
#pragma HLS loop_tripcount min=1 max=1
        permB[r] = axesB[r];
    }
    for (size_t s=naxes, t=0; s<ndimB; ++s, ++t) {
#pragma HLS loop_tripcount min=5 max=5
#pragma HLS unroll
        permB[s] = freeB[t];
    }



    for (size_t u=0; u<ndimA; ++u) {
#pragma HLS loop_tripcount min=5 max=5
#pragma HLS unroll factor=2
        newshpA[u] = A->shape[permA[u]];
    }
    for (size_t v=0; v<ndimB; ++v) {
#pragma HLS loop_tripcount min=5 max=5
#pragma HLS unroll
        newshpB[v] = B->shape[permB[v]];
    }

    // reshape arrays
    for (size_t w=0; w<A->numel; ++w) {
#pragma HLS loop_tripcount min=2704 max=2704
        k2c_idx2sub(w,Asub,A->shape,ndimA);
        for (size_t x=0; x<ndimA; ++x) {
#pragma HLS loop_tripcount min=5 max=5
#pragma HLS unroll factor=2
            Bsub[x] = Asub[permA[x]];
        }
        size_t bidx = k2c_sub2idx(Bsub,newshpA,ndimA);
        reshapeA[bidx] = A_array[w];
    }

    for (size_t y=0; y<B->numel; ++y) {
#pragma HLS loop_tripcount min=27040 max=27040
        k2c_idx2sub(y,Bsub,B->shape,ndimB);
        for (size_t z=0; z<ndimB; ++z) {
#pragma HLS loop_tripcount min=5 max=5
#pragma HLS unroll
            Asub[z] = Bsub[permB[z]];
        }
        size_t bidx = k2c_sub2idx(Asub,newshpB,ndimB);
        reshapeB[bidx] = B_array[y];
    }


    if (normalize) {

        float sum;
        float inorm;
        for (size_t a=0; a<free_axesA; ++a) {
#pragma HLS loop_tripcount min=2704 max=2704
            sum = 0;
            for (size_t b=0; b<prod_axesA; ++b) {
#pragma HLS loop_tripcount min=1 max=1
                sum += reshapeA[a*prod_axesA + b]*reshapeA[a*prod_axesA + b];
            }
            inorm = 1.0f/sqrtf(sum);
            for (size_t c=0; c<prod_axesA; ++c) {
#pragma HLS loop_tripcount min=1 max=1
                reshapeA[a*prod_axesA + c] *= inorm;
            }
        }
        for (size_t d=0; d<free_axesB; ++d) {
#pragma HLS loop_tripcount min=27040 max=27040
            sum = 0;
            for (size_t e=0; e<prod_axesB; ++e) {
#pragma HLS loop_tripcount min=1 max=1
                sum += reshapeB[d + free_axesB*e]*reshapeB[d + free_axesB*e];
            }
            inorm = 1.0f/sqrtf(sum);
            for (size_t f=0; f<prod_axesB; ++f) {
#pragma HLS loop_tripcount min=1 max=1
                reshapeB[d + free_axesB*f] *= inorm;
            }
        }
    }

    k2c_matmul(C_array, reshapeA, reshapeB, free_axesA,
               free_axesB, prod_axesA);
}


void k2c_bias_add(k2c_tensor* A, const k2c_tensor* b, float* A_array, float* b_array) {
#pragma HLS inline
    for (size_t i=0; i<A->numel; i+=b->numel) {
#pragma HLS loop_tripcount min=676 max=676
#pragma HLS pipeline rewind
        for (size_t j=0; j<b->numel; ++j) {
#pragma HLS unroll
#pragma HLS loop_tripcount min=16 max=16
            A_array[i+j] += b_array[j];
        }
    }
}
