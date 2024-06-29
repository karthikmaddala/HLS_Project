#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "relu.h"


// Utility function to print an array (for debugging purposes)
void print_array(const float * array, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        printf("%.2f ", array[i]);
    }
    printf("\n");
}

// Main function for test bench
int main() {
    // Define the size of the array to simulate a large neural network input
    const size_t array_size = 10000; // Adjust this size as needed

    // Allocate memory for the input and expected output arrays
    float *input_array = (float *)malloc(array_size * sizeof(float));
    float *expected_output_array = (float *)malloc(array_size * sizeof(float));

    // Seed the random number generator
    srand((unsigned int)time(NULL));

    // Initialize the input array with random values and expected output array
    for (size_t i = 0; i < array_size; ++i) {
        input_array[i] = (float)rand() / (float)(RAND_MAX / 20.0f) - 10.0f; // Random values between -10 and 10
        expected_output_array[i] = input_array[i] > 0.0f ? input_array[i] : 0.0f; // ReLU activation
    }

    // Print input array (optional)
    printf("Input array:\n");
    print_array(input_array, array_size);

    // Call the ReLU function
    k2c_relu_func(input_array, array_size);

    // Print output array (optional)
    printf("\nOutput array:\n");
    print_array(input_array, array_size);

    // Verify the output
    int test_passed = 1;
    for (size_t i = 0; i < array_size; ++i) {
        if (input_array[i] != expected_output_array[i]) {
            test_passed = 0;
            printf("Mismatch at index %zu: expected %.2f, got %.2f\n", i, expected_output_array[i], input_array[i]);
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

    return 0;
}
