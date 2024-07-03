#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "matmul.h"

// Utility function to print an array
void print_array(const char* name, const float* array, size_t rows, size_t cols) {
    printf("%s = [\n", name);
    for (size_t i = 0; i < rows; ++i) {
        printf("  [");
        for (size_t j = 0; j < cols; ++j) {
            printf(" %f", array[i * cols + j]);
            if (j < cols - 1) printf(",");
        }
        printf(" ]\n");
    }
    printf("]\n");
}

// Utility function to compare two arrays
int compare_arrays(const float* A, const float* B, size_t size, float epsilon) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(A[i] - B[i]) > epsilon) {
            return 0; // Arrays are not equal
        }
    }
    return 1; // Arrays are equal
}

// Reference implementation of the affine matrix multiplication
void reference_matmul(float * C, const float * A, const float * B,
                             const size_t outrows, const size_t outcols, const size_t innerdim) {

    for (size_t foo =0; foo < outrows*outcols; ++foo) {
        C[foo] = 0;
    }

    for (size_t i = 0 ; i < outrows; ++i) {
        const size_t outrowidx = i*outcols;
        const size_t inneridx = i*innerdim;
        for (size_t k = 0; k < innerdim; ++k) {
            for (size_t j = 0;  j < outcols; ++j) {
                C[outrowidx+j] += A[inneridx+k] * B[k*outcols+j];
            }
        }
    }
}

// Test function
int test_k2c_matmul() {
    // Define matrix sizes (example sizes typical for NN layers)
    const size_t outrows = 64;
    const size_t outcols = 128;
    const size_t innerdim = 256;
    const float epsilon = 1e-5; // Tolerance for floating point comparison

    // Allocate memory for matrices
    float* A = (float*)malloc(outrows * innerdim * sizeof(float));
    float* B = (float*)malloc(innerdim * outcols * sizeof(float));
    float* C = (float*)malloc(outrows * outcols * sizeof(float));
    float* C_ref = (float*)malloc(outrows * outcols * sizeof(float));

    // Initialize matrices with random values
    for (size_t i = 0; i < outrows * innerdim; ++i) {
        A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Random values between -1 and 1
    }
    for (size_t i = 0; i < innerdim * outcols; ++i) {
        B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Random values between -1 and 1
    }

    // Perform the affine matrix multiplication
    k2c_matmul(C, A, B, outrows, outcols, innerdim);
    reference_matmul(C_ref, A, B, outrows, outcols, innerdim);

    // Compare the resulting matrices
    int result = compare_arrays(C, C_ref, outrows * outcols, epsilon);

    // Print the result of the test
    if (result) {
        printf("Test passed!\n");
    } else {
        printf("Test failed!\n");
        print_array("C", C, outrows, outcols);
        print_array("C_ref", C_ref, outrows, outcols);
    }

    // Free allocated memory
    free(A);
    free(B);
    free(C);
    free(C_ref);

    return result;
}

int main() {
    // Seed for random number generation
    srand(time(NULL));

    // Run the test function
    int result = test_k2c_matmul();

    return result ? 0 : 1;
}