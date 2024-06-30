#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include <string.h>
#include "dot.h"

void reference_k2c_dot(float* C_array,
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

int compare_arrays(const float *a, const float *b, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > 1e-6) {
            return 0; // arrays are not equal
        }
    }
    return 1; // arrays are equal
}

void print_array(const float *array, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        printf("%f ", array[i]);
    }
    printf("\n");
}

void generate_random_array(float *array, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        array[i] = ((float)rand() / (float)(RAND_MAX)) * 2.0f - 1.0f; // random values between -1 and 1
    }
}

int main() {
    srand(0); // Seed for reproducibility

    // Test case
    const size_t A_ndim = 3;
    const size_t B_ndim = 3;
    const size_t naxes = 2;
    const size_t A_shape[3] = {10, 20, 30};
    const size_t B_shape[3] = {30, 20, 15};
    const size_t axesA[2] = {1, 2};
    const size_t axesB[2] = {0, 1};
    const size_t C_ndim = 2;
    const size_t C_shape[2] = {10, 15};
    const size_t A_numel = 10 * 20 * 30;
    const size_t B_numel = 30 * 20 * 15;
    const size_t C_numel = 10 * 15;
    const int normalize = 0;

    float *A_array = (float*)malloc(A_numel * sizeof(float));
    float *B_array = (float*)malloc(B_numel * sizeof(float));
    float *C_array = (float*)malloc(C_numel * sizeof(float));
    float *C_ref_array = (float*)malloc(C_numel * sizeof(float));
    float *fwork = (float*)malloc((A_numel + B_numel) * sizeof(float));

    generate_random_array(A_array, A_numel);
    generate_random_array(B_array, B_numel);
    generate_random_array(fwork, A_numel+B_numel);

    // Call the k2c_dot function
    k2c_dot(C_array, C_ndim, C_numel, (size_t *)C_shape,
            A_array, A_ndim, A_numel, (size_t *)A_shape,
            B_array, B_ndim, B_numel, (size_t *)B_shape,
            axesA, axesB, naxes, normalize, fwork);

    // Call the reference function
    reference_k2c_dot(C_ref_array, C_ndim, C_numel, (size_t *)C_shape,
                      A_array, A_ndim, A_numel, (size_t *)A_shape,
                      B_array, B_ndim, B_numel, (size_t *)B_shape,
                      axesA, axesB, naxes, normalize, fwork);

    // Print the result
    printf("Computed C:\n");
    print_array(C_array, C_numel);
    printf("Reference C:\n");
    print_array(C_ref_array, C_numel);

    // Compare with expected output
    if (compare_arrays(C_array, C_ref_array, C_numel)) {
        printf("Test passed!\n");
    } else {
        printf("Test failed.\n");
    }

    free(A_array);
    free(B_array);
    free(C_array);
    free(C_ref_array);
    free(fwork);

    return 0;
}

void reference_k2c_dot(float* C_array,
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
                      float * fwork) {

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
    const size_t ndimA = A_ndim;
    const size_t ndimB = B_ndim;
    float *reshapeA = &fwork[0];   // temp working storage
    float *reshapeB = &fwork[A_numel];
    size_t Asub[K2C_MAX_NDIM];
    size_t Bsub[K2C_MAX_NDIM];
    // find which axes are free (ie, not being summed over)
    count=0;
    for (size_t i=0; i<ndimA; ++i) {
        isin = 0;
        for (size_t j=0; j<naxes; ++j) {
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
    for (size_t i=0; i<ndimB; ++i) {
        isin = 0;
        for (size_t j=0; j<naxes; ++j) {
            if (i==axesB[j]) {
                isin=1;
            }
        }
        if (!isin) {
            freeB[count] = i;
            ++count;
        }
    }

    // number of elements in inner dimension
    for (size_t i=0; i < naxes; ++i) {
        prod_axesA *= A_shape[axesA[i]];
    }
    for (size_t i=0; i < naxes; ++i) {
        prod_axesB *= B_shape[axesB[i]];
    }
    // number of elements in free dimension
    free_axesA = A_numel/prod_axesA;
    free_axesB = B_numel/prod_axesB;
    // find permutation of axes to get into matmul shape
    for (size_t i=0; i<ndimA-naxes; ++i) {
        permA[i] = freeA[i];
    }
    for (size_t i=ndimA-naxes, j=0; i<ndimA; ++i, ++j) {
        permA[i] = axesA[j];
    }
    for (size_t i=0; i<naxes; ++i) {
        permB[i] = axesB[i];
    }
    for (size_t i=naxes, j=0; i<ndimB; ++i, ++j) {
        permB[i] = freeB[j];
    }



    for (size_t i=0; i<ndimA; ++i) {
        newshpA[i] = A_shape[permA[i]];
    }
    for (size_t i=0; i<ndimB; ++i) {
        newshpB[i] = B_shape[permB[i]];
    }

    // reshape arrays
    for (size_t i=0; i<A_numel; ++i) {
        k2c_idx2sub(i,Asub,A_shape,ndimA);
        for (size_t j=0; j<ndimA; ++j) {
            Bsub[j] = Asub[permA[j]];
        }
        size_t bidx = k2c_sub2idx(Bsub,newshpA,ndimA);
        reshapeA[bidx] = A_array[i];
    }

    for (size_t i=0; i<B_numel; ++i) {
        k2c_idx2sub(i,Bsub,B_shape,ndimB);
        for (size_t j=0; j<ndimB; ++j) {
            Asub[j] = Bsub[permB[j]];
        }
        size_t bidx = k2c_sub2idx(Asub,newshpB,ndimB);
        reshapeB[bidx] = B_array[i];
    }


    if (normalize) {

        float sum;
        float inorm;
        for (size_t i=0; i<free_axesA; ++i) {
            sum = 0;
            for (size_t j=0; j<prod_axesA; ++j) {
                sum += reshapeA[i*prod_axesA + j]*reshapeA[i*prod_axesA + j];
            }
            inorm = 1.0f/sqrtf(sum);
            for (size_t j=0; j<prod_axesA; ++j) {
                reshapeA[i*prod_axesA + j] *= inorm;
            }
        }
        for (size_t i=0; i<free_axesB; ++i) {
            sum = 0;
            for (size_t j=0; j<prod_axesB; ++j) {
                sum += reshapeB[i + free_axesB*j]*reshapeB[i + free_axesB*j];
            }
            inorm = 1.0f/sqrtf(sum);
            for (size_t j=0; j<prod_axesB; ++j) {
                reshapeB[i + free_axesB*j] *= inorm;
            }
        }
    }

    k2c_matmul(C_array, reshapeA, reshapeB, free_axesA,
               free_axesB, prod_axesA);
}
