#include "dot.h"
#include <stddef.h>
#include <math.h>

/**
 * Converts subscripts to linear indices in row major order.
 *
 * :param sub: array[ndim] subscript to convert.
 * :param shape: array[ndim] shape of array being indexed.
 * :param ndim: number of dimensions of array being indexed.
 * :return: linear index in row major order.
 */
size_t k2c_sub2idx(const size_t * sub, const size_t * shape, const size_t ndim) {

    size_t idx = 0;
    size_t temp = 0;
    for (size_t i=0; i<ndim; ++i) {
#pragma HLS loop_tripcount min=5 max=5
        temp = sub[i];
        for (size_t j=ndim-1; j>i; --j) {
#pragma HLS loop_tripcount min=5 max=5
            temp *= shape[j];
        }
        idx += temp;
    }
    return idx;
}

void k2c_matmul(float C[10*15], const float A[10*20*30], const float B[30*20*15], const size_t outrows,
                const size_t outcols, const size_t innerdim) {

    // make sure output is empty
    // memset(C, 0, outrows*outcols*sizeof(C[0]));
        for (size_t foo =0; foo < outrows*outcols; ++foo) {
#pragma HLS loop_tripcount min=150 max=150
        C[foo] = 0;
    }

    for (size_t i = 0 ; i < outrows; ++i) {
#pragma HLS loop_tripcount min=6000 max=6000
        const size_t outrowidx = i*outcols;
        const size_t inneridx = i*innerdim;
        for (size_t k = 0; k < innerdim; ++k) {
#pragma HLS loop_tripcount min=1 max=1
            for (size_t j = 0;  j < outcols; ++j) {
#pragma HLS loop_tripcount min=9000 max=9000
                C[outrowidx+j] += A[inneridx+k] * B[k*outcols+j];
            }
        }
    }
}

/**
 * Converts linear indices to subscripts in row major order.
 *
 * :param idx: linear index in row major order.
 * :param sub: array[ndim] output subscript.
 * :param shape: array[ndim] shape of array being indexed.
 * :param ndim: number of dimensions of array being indexed.
 */
void k2c_idx2sub(const size_t idx, size_t * sub, const size_t * shape, const size_t ndim) {

    size_t idx2 = idx;
    for (int i=ndim-1; i>=0; --i) {
#pragma HLS loop_tripcount min=5 max=5
        sub[i] = idx2%shape[i];
        idx2 /= shape[i];
    }
}


/**
 * Dot product (tensor contraction) between 2 tensors. C=A*B
 *
 * :param C: output tensor.
 * :param A: input tensor 1.
 * :param B: input tensor 2.
 * :param axesA: array[naxes] of axes of A being contracted.
 * :param axesB: array[naxes] of axes of B being contracted.
 * :param naxes: number of axes being contracted from each input.
 * :param normalize: (0,1) whether to L2-normalize samples along the dot product axis before taking the dot product. If set to 1, then the output of the dot product is the cosine proximity between the two samples.
 * :param fwork: array of working space, size(fwork) = size(A) + size(B)
 */
void k2c_dot(float C_array[10*15],
            size_t C_ndim,
            size_t C_numel,
            size_t C_shape[5], 

            const float A_array[10*20*30],
            const size_t A_ndim,
            const size_t A_numel,
            const size_t A_shape[5], 

            const float B_array[30*20*15],
            const size_t B_ndim,
            const size_t B_numel,
            const size_t B_shape[5], 

            const size_t axesA[2],
            const size_t axesB[2], 
            const size_t naxes, 
            const int normalize, 
            float fwork[10*20*30 + 30*20*15]) {

    size_t permA[K2C_MAX_NDIM];
    size_t permB[K2C_MAX_NDIM];
    size_t prod_axesA = 1;
    size_t prod_axesB = 1;
    size_t free_axesA, free_axesB;
    size_t freeA[K2C_MAX_NDIM];
    size_t freeB[K2C_MAX_NDIM];
    size_t count_freeA;
    size_t count_freeB;
    int isin_axesA;
    int isin_axesB;
    size_t newshpA[K2C_MAX_NDIM];
    size_t newshpB[K2C_MAX_NDIM];
    const size_t ndimA = A_ndim;
    const size_t ndimB = B_ndim;
    float *reshapeA = &fwork[0];   // temp working storage
    float *reshapeB = &fwork[A_numel];
    size_t Asub[K2C_MAX_NDIM];
    size_t Bsub[K2C_MAX_NDIM];

    // Find which axes are free (ie, not being summed over)
    count_freeA = 0;
    for (size_t ia = 0; ia < ndimA; ++ia) {
#pragma HLS loop_tripcount min=5 max=5
        isin_axesA = 0;
        for (size_t ja = 0; ja < naxes; ++ja) {
#pragma HLS loop_tripcount min=2 max=2
            if (ia == axesA[ja]) {
                isin_axesA = 1;
            }
        }
        if (!isin_axesA) {
            freeA[count_freeA] = ia;
            ++count_freeA;
        }
    }

    count_freeB = 0;
    for (size_t ib = 0; ib < ndimB; ++ib) {
#pragma HLS loop_tripcount min=5 max=5
        isin_axesB = 0;
        for (size_t jb = 0; jb < naxes; ++jb) {
#pragma HLS loop_tripcount min=2 max=2
            if (ib == axesB[jb]) {
                isin_axesB = 1;
            }
        }
        if (!isin_axesB) {
            freeB[count_freeB] = ib;
            ++count_freeB;
        }
    }

    // Number of elements in inner dimension
    for (size_t iaxA = 0; iaxA < naxes; ++iaxA) {
#pragma HLS loop_tripcount min=2 max=2
        prod_axesA *= A_shape[axesA[iaxA]];
    }
    for (size_t iaxB = 0; iaxB < naxes; ++iaxB) {
#pragma HLS loop_tripcount min=2 max=2
        prod_axesB *= B_shape[axesB[iaxB]];
    }

    // Number of elements in free dimension
    free_axesA = A_numel / prod_axesA;
    free_axesB = B_numel / prod_axesB;

    // Find permutation of axes to get into matmul shape
    for (size_t ifreeA = 0; ifreeA < ndimA - naxes; ++ifreeA) {
#pragma HLS loop_tripcount min=3 max=3
        permA[ifreeA] = freeA[ifreeA];
    }
    for (size_t ipermA = ndimA - naxes, jaxesA = 0; ipermA < ndimA; ++ipermA, ++jaxesA) {
#pragma HLS loop_tripcount min=2 max=2
        permA[ipermA] = axesA[jaxesA];
    }

    for (size_t ipermB = 0; ipermB < naxes; ++ipermB) {
#pragma HLS loop_tripcount min=2 max=2
        permB[ipermB] = axesB[ipermB];
    }
    for (size_t ifreeB = naxes, jfreeB = 0; ifreeB < ndimB; ++ifreeB, ++jfreeB) {
#pragma HLS loop_tripcount min=3 max=3
        permB[ifreeB] = freeB[jfreeB];
    }

    for (size_t inewshpA = 0; inewshpA < ndimA; ++inewshpA) {
#pragma HLS loop_tripcount min=5 max=5
        newshpA[inewshpA] = A_shape[permA[inewshpA]];
    }
    for (size_t inewshpB = 0; inewshpB < ndimB; ++inewshpB) {
#pragma HLS loop_tripcount min=5 max=5
        newshpB[inewshpB] = B_shape[permB[inewshpB]];
    }

    // Reshape arrays
    for (size_t iA = 0; iA < A_numel; ++iA) {
#pragma HLS loop_tripcount min=6000 max=6000
        k2c_idx2sub(iA, Asub, A_shape, ndimA);
        for (size_t jA = 0; jA < ndimA; ++jA) {
#pragma HLS loop_tripcount min=5 max=5
            Bsub[jA] = Asub[permA[jA]];
        }
        size_t bidxA = k2c_sub2idx(Bsub, newshpA, ndimA);
        reshapeA[bidxA] = A_array[iA];
    }

    for (size_t iB = 0; iB < B_numel; ++iB) {
#pragma HLS loop_tripcount min=9000 max=9000
        k2c_idx2sub(iB, Bsub, B_shape, ndimB);
        for (size_t jB = 0; jB < ndimB; ++jB) {
#pragma HLS loop_tripcount min=5 max=5
            Asub[jB] = Bsub[permB[jB]];
        }
        size_t bidxB = k2c_sub2idx(Asub, newshpB, ndimB);
        reshapeB[bidxB] = B_array[iB];
    }

    if (normalize) {
        float sum_norm;
        float inorm_factor;
        for (size_t ifreeA_norm = 0; ifreeA_norm < free_axesA; ++ifreeA_norm) {
#pragma HLS loop_tripcount min=6000 max=6000
            sum_norm = 0;
            for (size_t iprodA = 0; iprodA < prod_axesA; ++iprodA) {
#pragma HLS loop_tripcount min=1 max=1
                sum_norm += reshapeA[ifreeA_norm * prod_axesA + iprodA] * reshapeA[ifreeA_norm * prod_axesA + iprodA];
            }
            inorm_factor = 1.0f / sqrtf(sum_norm);
            for (size_t iprodA = 0; iprodA < prod_axesA; ++iprodA) {
#pragma HLS loop_tripcount min=1 max=1
                reshapeA[ifreeA_norm * prod_axesA + iprodA] *= inorm_factor;
            }
        }

        for (size_t ifreeB_norm = 0; ifreeB_norm < free_axesB; ++ifreeB_norm) {
#pragma HLS loop_tripcount min=9000 max=9000
            sum_norm = 0;
            for (size_t iprodB = 0; iprodB < prod_axesB; ++iprodB) {
#pragma HLS loop_tripcount min=1 max=1
                sum_norm += reshapeB[ifreeB_norm + free_axesB * iprodB] * reshapeB[ifreeB_norm + free_axesB * iprodB];
            }
            inorm_factor = 1.0f / sqrtf(sum_norm);
            for (size_t iprodB = 0; iprodB < prod_axesB; ++iprodB) {
#pragma HLS loop_tripcount min=1 max=1
                reshapeB[ifreeB_norm + free_axesB * iprodB] *= inorm_factor;
            }
        }
    }

    k2c_matmul(C_array, reshapeA, reshapeB, free_axesA, free_axesB, prod_axesA);
}
