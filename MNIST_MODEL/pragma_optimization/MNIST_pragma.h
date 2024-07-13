#pragma once 
#include <stddef.h>
#define K2C_MAX_NDIM 5
#define MAX_ARRAY_SIZE 21632

struct k2c_tensor {
    float array[MAX_ARRAY_SIZE]; // Maximum expected number of elements
    size_t ndim;
    size_t numel;
    size_t shape[K2C_MAX_NDIM];
};
typedef struct k2c_tensor k2c_tensor;

// Activations
void k2c_relu_func(float * x, const size_t size);
void k2c_softmax_func(float * x, const size_t size);
typedef void k2c_activationType(float * x, const size_t size);
extern k2c_activationType * k2c_relu;
extern k2c_activationType * k2c_softmax;


// Advanced Activations
void k2c_LeakyReLU(float * x, const size_t size, const float alpha);
void k2c_PReLU(float * x, const size_t size, const float * alpha);
void k2c_ELU(float * x, const size_t size, const float alpha);
void k2c_ThresholdedReLU(float * x, const size_t size, const float theta);
void k2c_ReLU(float * x, const size_t size, const float max_value, const float negative_slope,
              const float threshold);

// Convolutions
void k2c_conv2d(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* kernel,
                const k2c_tensor* bias, const size_t * stride, const size_t * dilation);

// Core Layers
void k2c_dense(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* kernel,
               const k2c_tensor* bias, float * fwork);
void k2c_flatten(k2c_tensor *output, const k2c_tensor* input);

// Helper functions
void k2c_matmul(float * C, const float * A, const float * B, const size_t outrows,
                const size_t outcols, const size_t innerdim);
void k2c_affine_matmul(float * C, const float * A, const float * B, const float * d,
                       const size_t outrows,const size_t outcols, const size_t innerdim);
size_t k2c_sub2idx(const size_t * sub, const size_t * shape, const size_t ndim);
void k2c_idx2sub(const size_t idx, size_t * sub, const size_t * shape, const size_t ndim);
void k2c_dot(k2c_tensor* C, const k2c_tensor* A, const k2c_tensor* B, const size_t * axesA,
             const size_t * axesB, const size_t naxes, const int normalize, float * fwork);
void k2c_bias_add(k2c_tensor* A, const k2c_tensor* b);

// Pooling layers
void k2c_maxpool2d(k2c_tensor* output, const k2c_tensor* input, const size_t * pool_size,
                   const size_t * stride);


void MNIST(k2c_tensor* input_1_input, k2c_tensor* dense_output); 
void MNIST_initialize(); 
void MNIST_terminate(); 
