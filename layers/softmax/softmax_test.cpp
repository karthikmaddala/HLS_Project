#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "softmax.h"

// Utility function to print an array (for debugging purposes)
void print_array(const float * array, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        printf("%.6f ", array[i]);
    }
    printf("\n");
}

// Function to compute softmax manually for verification
void compute_softmax(const float *input, float *output, size_t size) {
    float xmax = input[0];
    for (size_t i = 1; i < size; ++i) {
        if (input[i] > xmax) {
            xmax = input[i];
        }
    }
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        output[i] = expf(input[i] - xmax);
        sum += output[i];
    }
    for (size_t i = 0; i < size; ++i) {
        output[i] /= sum;
    }
}

// Main function for test bench
int main() {
    // Define the size of the array to simulate a large neural network input
    const size_t array_size = 10000; // Adjust this size as needed

    // Allocate memory for the input and expected output arrays
    float *input_array = (float *)malloc(array_size * sizeof(float));
    float *expected_output_array = (float *)malloc(array_size * sizeof(float));
    float *output_array = (float *)malloc(array_size * sizeof(float));

    // Seed the random number generator
    srand((unsigned int)time(NULL));

    // Initialize the input array with random values
    for (size_t i = 0; i < array_size; ++i) {
        input_array[i] = (float)rand() / (float)(RAND_MAX / 20.0f) - 10.0f; // Random values between -10 and 10
        output_array[i] = input_array[i]; // Copy input to output array for in-place processing
    }

    // Compute the expected output using the manual softmax function
    compute_softmax(input_array, expected_output_array, array_size);

    // Print input array (optional)
    printf("Input array:\n");
    print_array(input_array, array_size);

    // Call the softmax function
    k2c_softmax_func(output_array, array_size);

    // Print output array (optional)
    printf("\nOutput array:\n");
    print_array(output_array, array_size);

    // Verify the output
    int test_passed = 1;
    for (size_t i = 0; i < array_size; ++i) {
        if (fabs(output_array[i] - expected_output_array[i]) > 1e-6) {
            test_passed = 0;
            printf("Mismatch at index %zu: expected %.6f, got %.6f\n", i, expected_output_array[i], output_array[i]);
        }
    }

    if (test_passed) {
        printf("\nTest PASSED\n");
    } else {
        printf("\nTest FAILED\n");
    }

    // Free allocated memory
    free(input_array);
    free(expected_output_array);
    free(output_array);

    return 0;
}
