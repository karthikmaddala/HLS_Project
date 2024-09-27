#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "HAR_MODEL.h" 
#define RELU_FLAG 0
#define SOFTMAX_FLAG 1

k2c_tensor conv2d_1_output;
k2c_tensor conv2d_1_kernel;
k2c_tensor conv2d_1_bias;
k2c_tensor max_pooling2d_1_output;
k2c_tensor flatten_1_output;
k2c_tensor dense_1_output;
k2c_tensor dense_1_kernel;
k2c_tensor dense_1_bias;
k2c_tensor dense_2_output;
k2c_tensor dense_2_kernel;
k2c_tensor dense_2_bias;
k2c_tensor dense_3_kernel;
k2c_tensor dense_3_bias;

void HAR_MODEL(k2c_tensor* conv2d_1_input_input, k2c_tensor* dense_3_output) { 

size_t conv2d_1_stride[2] = {1,1}; 
size_t conv2d_1_dilation[2] = {1,1}; 
float conv2d_1_output_array[22784] = {0}; 

for (size_t i = 0; i < 720896; i++) {
    conv2d_1_output.array[i] = conv2d_1_output_array[i];
}
conv2d_1_output.ndim = 3;
conv2d_1_output.numel = 22784;
conv2d_1_output.shape[0] = 89;
conv2d_1_output.shape[1] = 2;
conv2d_1_output.shape[2] = 128;
conv2d_1_output.shape[3] = 1;
conv2d_1_output.shape[4] = 1;

float conv2d_1_kernel_array[512] = {
//some data here
}; 

for (size_t i0 = 0; i0 < 512; i0++) {
    conv2d_1_kernel.array[i0] = conv2d_1_kernel_array[i0];
}
conv2d_1_kernel.ndim = 4;
conv2d_1_kernel.numel = 512;
conv2d_1_kernel.shape[0] = 2;
conv2d_1_kernel.shape[1] = 2;
conv2d_1_kernel.shape[2] = 1;
conv2d_1_kernel.shape[3] = 128;
conv2d_1_kernel.shape[4] = 1;

float conv2d_1_bias_array[128] = {
//some data here
}; 

for (size_t i1 = 0; i1 < 128; i1++) {
    conv2d_1_bias.array[i1] = conv2d_1_bias_array[i1];
}
conv2d_1_bias.ndim = 1;
conv2d_1_bias.numel = 128;
conv2d_1_bias.shape[0] = 128;
conv2d_1_bias.shape[1] = 1;
conv2d_1_bias.shape[2] = 1;
conv2d_1_bias.shape[3] = 1;
conv2d_1_bias.shape[4] = 1;
 
size_t max_pooling2d_1_stride[2] = {2,2}; 
size_t max_pooling2d_1_pool_size[2] = {2,2}; 
float max_pooling2d_1_output_array[5632] = {0}; 

for (size_t i2 = 0; i2 < 5632; i2++) {
    max_pooling2d_1_output.array[i2] = max_pooling2d_1_output_array[i2];
}
max_pooling2d_1_output.ndim = 3;
max_pooling2d_1_output.numel = 5632;
max_pooling2d_1_output.shape[0] = 44;
max_pooling2d_1_output.shape[1] = 1;
max_pooling2d_1_output.shape[2] = 128;
max_pooling2d_1_output.shape[3] = 1;
max_pooling2d_1_output.shape[4] = 1;

float flatten_1_output_array[5632] = {0}; 

for (size_t i3 = 0; i3 < 5632; i3++) {
    flatten_1_output.array[i3] = flatten_1_output_array[i3];
}
flatten_1_output.ndim = 1;
flatten_1_output.numel = 5632;
flatten_1_output.shape[0] = 5632;
flatten_1_output.shape[1] = 1;
flatten_1_output.shape[2] = 1;
flatten_1_output.shape[3] = 1;
flatten_1_output.shape[4] = 1;

float dense_1_output_array[128] = {0}; 
 
for (size_t i4 = 0; i4 < 128; i4++) {
    dense_1_output.array[i4] = dense_1_output_array[i4];
}
dense_1_output.ndim = 1;
dense_1_output.numel = 128;
dense_1_output.shape[0] = 128;
dense_1_output.shape[1] = 1;
dense_1_output.shape[2] = 1;
dense_1_output.shape[3] = 1;
dense_1_output.shape[4] = 1;

float dense_1_kernel_array[720896] = {
//some data here

}; 

for (size_t i5 = 0; i5 < 720896; i5++) {
    dense_1_kernel.array[i5] = dense_1_kernel_array[i5];
}
dense_1_kernel.ndim = 2;
dense_1_kernel.numel = 720896;
dense_1_kernel.shape[0] = 5632;
dense_1_kernel.shape[1] = 128;
dense_1_kernel.shape[2] = 1;
dense_1_kernel.shape[3] = 1;
dense_1_kernel.shape[4] = 1;

float dense_1_bias_array[128] = {
//some data here
}; 

for (size_t i6 = 0; i6 < 128; i6++) {
    dense_1_bias.array[i6] = dense_1_bias_array[i6];
}
dense_1_bias.ndim = 1;
dense_1_bias.numel = 128;
dense_1_bias.shape[0] = 128;
dense_1_bias.shape[1] = 1;
dense_1_bias.shape[2] = 1;
dense_1_bias.shape[3] = 1;
dense_1_bias.shape[4] = 1;

float dense_1_fwork[726528] = {0}; 
 
float dense_2_output_array[128] = {0}; 

for (size_t i7 = 0; i7 < 128; i7++) {
    dense_2_output.array[i7] = dense_2_output_array[i7];
}
dense_2_output.ndim = 1;
dense_2_output.numel = 128;
dense_2_output.shape[0] = 128;
dense_2_output.shape[1] = 1;
dense_2_output.shape[2] = 1;
dense_2_output.shape[3] = 1;
dense_2_output.shape[4] = 1;

float dense_2_kernel_array[16384] = {
//some data here
}; 
 
for (size_t i8 = 0; i8 < 16384; i8++) {
    dense_2_kernel.array[i8] = dense_2_kernel_array[i8];
}
dense_2_kernel.ndim = 2;
dense_2_kernel.numel = 16384;
dense_2_kernel.shape[0] = 128;
dense_2_kernel.shape[1] = 128;
dense_2_kernel.shape[2] = 1;
dense_2_kernel.shape[3] = 1;
dense_2_kernel.shape[4] = 1;

float dense_2_bias_array[128] = {
//some dataa here
}; 

for (size_t i9 = 0; i9 < 128; i9++) {
    dense_2_bias.array[i9] = dense_2_bias_array[i9];
}
dense_2_bias.ndim = 1;
dense_2_bias.numel = 128;
dense_2_bias.shape[0] = 128;
dense_2_bias.shape[1] = 1;
dense_2_bias.shape[2] = 1;
dense_2_bias.shape[3] = 1;
dense_2_bias.shape[4] = 1;

float dense_2_fwork[16512] = {0}; 

float dense_3_kernel_array[768] = {
//some data here
}; 

for (size_t i10 = 0; i10 < 768; i10++) {
    dense_3_kernel.array[i10] = dense_3_kernel_array[i10];
}
dense_3_kernel.ndim = 2;
dense_3_kernel.numel = 768;
dense_3_kernel.shape[0] = 128;
dense_3_kernel.shape[1] = 6;
dense_3_kernel.shape[2] = 1;
dense_3_kernel.shape[3] = 1;
dense_3_kernel.shape[4] = 1;

float dense_3_bias_array[6] = {
+2.54780829e-01f,-7.55986810e-01f,+2.28018850e-01f,-1.79226279e-01f,+5.60112178e-01f,
-4.32871372e-01f,}; 
 
for (size_t i11 = 0; i11 < 6; i11++) {
    dense_3_bias.array[i11] = dense_3_bias_array[i11];
}
dense_3_bias.ndim = 1;
dense_3_bias.numel = 6;
dense_3_bias.shape[0] = 6;
dense_3_bias.shape[1] = 1;
dense_3_bias.shape[2] = 1;
dense_3_bias.shape[3] = 1;
dense_3_bias.shape[4] = 1;

float dense_3_fwork[896] = {0}; 
 
k2c_conv2d(&conv2d_1_output,conv2d_1_input_input,&conv2d_1_kernel, 
	&conv2d_1_bias,conv2d_1_stride,conv2d_1_dilation,RELU_FLAG);

k2c_maxpool2d(&max_pooling2d_1_output,&conv2d_1_output,max_pooling2d_1_pool_size, 
	max_pooling2d_1_stride); 

k2c_tensor dropout_1_output; 
dropout_1_output.ndim = max_pooling2d_1_output.ndim; // copy data into output struct 
dropout_1_output.numel = max_pooling2d_1_output.numel; 
 
for (size_t i14 = 0; i14 < K2C_MAX_NDIM; i14++) {
    dropout_1_output.shape[i14] = max_pooling2d_1_output.shape[i14];
}


for (size_t i12 = 0; i12 < MAX_ARRAY_SIZE; ++i12) {
    dropout_1_output.array[i12] = max_pooling2d_1_output.array[i12];
}

k2c_flatten(&flatten_1_output,&dropout_1_output); 
k2c_dense(&dense_1_output,&flatten_1_output,&dense_1_kernel, 
	&dense_1_bias,RELU_FLAG,dense_1_fwork); 
k2c_dense(&dense_2_output,&dense_1_output,&dense_2_kernel, 
	&dense_2_bias,RELU_FLAG,dense_2_fwork); 


k2c_dense(dense_3_output,&dense_2_output,&dense_3_kernel, 
	&dense_3_bias,SOFTMAX_FLAG,dense_3_fwork); 

 } 

void HAR_MODEL_initialize() { 

} 

void HAR_MODEL_terminate() { 

} 

void k2c_relu_func(float * x, const size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (x[i] <= 0.0f) {
            x[i] = 0.0f;
        }
    }
}

void k2c_softmax_func(float * x, const size_t size) {
    float xmax = x[0];
    float sum = 0;
    for (size_t i = 0; i < size; ++i) {
        if (x[i] > xmax) {
            xmax = x[i];
        }
    }
    for (size_t i1 = 0; i1 < size; ++i1) {
        x[i1] = expf(x[i1] - xmax);
    }
    for (size_t i2 = 0; i2 < size; ++i2) {
        sum += x[i2];
    }
    sum = 1.0f / sum;
    for (size_t i3 = 0; i3 < size; ++i3) {
        x[i3] = x[i3] * sum;
    }
}

void k2c_matmul(float * C, const float * A, const float * B, const size_t outrows, const size_t outcols, const size_t innerdim) {
    for (size_t i12 = 0; i12 < outrows * outcols; i12++) {
        C[i12] = 0;
    }
    for (size_t i = 0; i < outrows; ++i) {
        const size_t outrowidx = i * outcols;
        const size_t inneridx = i * innerdim;
        for (size_t k = 0; k < innerdim; ++k) {
            for (size_t j = 0; j < outcols; ++j) {
                C[outrowidx + j] += A[inneridx + k] * B[k * outcols + j];
            }
        }
    }
}

void k2c_affine_matmul(float * C, const float * A, const float * B, const float * d, const size_t outrows, const size_t outcols, const size_t innerdim) {
    for (size_t i13 = 0; i13 < outrows * outcols; i13++) {
        C[i13] = 0;
    }
    for (size_t i = 0; i < outrows; ++i) {
        const size_t outrowidx = i * outcols;
        const size_t inneridx = i * innerdim;
        for (size_t j = 0; j < outcols; ++j) {
            for (size_t k = 0; k < innerdim; ++k) {
                C[outrowidx + j] += A[inneridx + k] * B[k * outcols + j];
            }
            C[outrowidx + j] += d[j];
        }
    }
}

size_t k2c_sub2idx(const size_t * sub, const size_t * shape, const size_t ndim) {
    size_t idx = 0;
    size_t temp = 0;
    for (size_t i = 0; i < ndim; ++i) {
        temp = sub[i];
        for (size_t j = ndim - 1; j > i; --j) {
            temp *= shape[j];
        }
        idx += temp;
    }
    return idx;
}

void k2c_idx2sub(const size_t idx, size_t * sub, const size_t * shape, const size_t ndim) {
    size_t idx2 = idx;
    for (int i = ndim - 1; i >= 0; --i) {
        sub[i] = idx2 % shape[i];
        idx2 /= shape[i];
    }
}

void k2c_dot(k2c_tensor* C, const k2c_tensor* A, const k2c_tensor* B, const size_t * axesA, const size_t * axesB, const size_t naxes, const int normalize, float * fwork) {
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
    count = 0;
    for (size_t a1 = 0; a1 < ndimA; ++a1) {
        isin = 0;
        for (size_t j1 = 0; j1 < naxes; ++j1) {
            if (a1 == axesA[j1]) {
                isin = 1;
            }
        }
        if (!isin) {
            freeA[count] = a1;
            ++count;
        }
    }
    count = 0;
    for (size_t a2 = 0; a2 < ndimB; ++a2) {
        #pragma HLS loop_tripcount min=1 max=5
        isin = 0;
        for (size_t j2 = 0; j2 < naxes; ++j2) {
            #pragma HLS loop_tripcount min=1 max=5
            if (a2 == axesB[j2]) {
                isin = 1;
            }
        }
        if (!isin) {
            freeB[count] = a2;
            ++count;
        }
    }
    for (size_t a3 = 0; a3 < naxes; ++a3) {
        prod_axesA *= A->shape[axesA[a3]];
    }
    for (size_t a4 = 0; a4 < naxes; ++a4) {
        prod_axesB *= B->shape[axesB[a4]];
    }
    free_axesA = A->numel / prod_axesA;
    free_axesB = B->numel / prod_axesB;
    for (size_t a5 = 0; a5 < ndimA - naxes; ++a5) {
        permA[a5] = freeA[a5];
    }
    for (size_t a6 = ndimA - naxes, j3 = 0; a6 < ndimA; ++a6, ++j3) {
        permA[a6] = axesA[j3];
    }
    for (size_t a7 = 0; a7 < naxes; ++a7) {
        permB[a7] = axesB[a7];
    }
    for (size_t a8 = naxes, j4 = 0; a8 < ndimB; ++a8, ++j4) {
        permB[a8] = freeB[j4];
    }
    for (size_t a9 = 0; a9 < ndimA; ++a9) {
        newshpA[a9] = A->shape[permA[a9]];
    }
    for (size_t a10 = 0; a10 < ndimB; ++a10) {
        newshpB[a10] = B->shape[permB[a10]];
    }
    for (size_t a11 = 0; a11 < A->numel; ++a11) {
        k2c_idx2sub(a11, Asub, A->shape, ndimA);
        for (size_t j5 = 0; j5 < ndimA; ++j5) {
            Bsub[j5] = Asub[permA[j5]];
        }
        size_t bidx = k2c_sub2idx(Bsub, newshpA, ndimA);
        reshapeA[bidx] = A->array[a11];
    }
    for (size_t a12 = 0; a12 < B->numel; ++a12) {
        k2c_idx2sub(a12, Bsub, B->shape, ndimB);
        for (size_t j6 = 0; j6 < ndimB; ++j6) {
            Asub[j6] = Bsub[permB[j6]];
        }
        size_t bidx = k2c_sub2idx(Asub, newshpB, ndimB);
        reshapeB[bidx] = B->array[a12];
    }
    if (normalize) {
        float sum;
        float inorm;
        for (size_t a13 = 0; a13 < free_axesA; ++a13) {
            sum = 0;
            for (size_t j7 = 0; j7 < prod_axesA; ++j7) {
                sum += reshapeA[a13 * prod_axesA + j7] * reshapeA[a13 * prod_axesA + j7];
            }
            inorm = 1.0f / sqrtf(sum);
            for (size_t j8 = 0; j8 < prod_axesA; ++j8) {
                reshapeA[a13 * prod_axesA + j8] *= inorm;
            }
        }
        for (size_t a14 = 0; a14 < free_axesB; ++a14) {
            sum = 0;
            for (size_t j9 = 0; j9 < prod_axesB; ++j9) {
                sum += reshapeB[a14 + free_axesB * j9] * reshapeB[a14 + free_axesB * j9];
            }
            inorm = 1.0f / sqrtf(sum);
            for (size_t j10 = 0; j10 < prod_axesB; ++j10) {
                reshapeB[a14 + free_axesB * j10] *= inorm;
            }
        }
    }
    k2c_matmul(C->array, reshapeA, reshapeB, free_axesA, free_axesB, prod_axesA);
}

void k2c_bias_add(k2c_tensor* A, const k2c_tensor* b) {
    for (size_t a15 = 0; a15 < A->numel; a15 += b->numel) {
        for (size_t j11 = 0; j11 < b->numel; ++j11) {
            A->array[a15 + j11] += b->array[j11];
        }
    }
}

void k2c_maxpool2d(k2c_tensor* output, const k2c_tensor* input, const size_t * pool_size, const size_t * stride) {
    const size_t channels = input->shape[2];
    for (size_t i = 0; i < channels; ++i) {
        for (size_t j = 0, k = 0; j < output->shape[1] * channels; j += channels, k += channels * stride[1]) {
            for (size_t l = 0, m = 0; l < output->numel; l += channels * output->shape[1], m += channels * input->shape[1] * stride[0]) {
                output->array[l + j + i] = input->array[m + k + i];
                for (size_t n = 0; n < pool_size[1] * channels; n += channels) {
                    for (size_t p = 0; p < pool_size[0] * channels * input->shape[1]; p += channels * input->shape[1]) {
                        if (output->array[l + j + i] < input->array[m + k + i + n + p]) {
                            output->array[l + j + i] = input->array[m + k + i + n + p];
                        }
                    }
                }
            }
        }
    }
}

void k2c_flatten(k2c_tensor *output, const k2c_tensor* input) {
    for (size_t i15 = 0; i15 < input->numel; i15++) {
        output->array[i15] = input->array[i15];
    }
    for (size_t i = 0; i < input->ndim; ++i) {
        output->shape[i] = 1;
    }
    output->shape[0] = input->numel;
    output->numel = input->numel;
    output->ndim = 1;
}

void k2c_dense(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* kernel, const k2c_tensor* bias, int activation_flag, float * fwork) {
    if (input->ndim <= 2) {
        size_t outrows;
        if (input->ndim > 1) {
            outrows = input->shape[0];
        } else {
            outrows = 1;
        }
        const size_t outcols = kernel->shape[1];
        const size_t innerdim = kernel->shape[0];
        const size_t outsize = outrows * outcols;
        k2c_affine_matmul(output->array, input->array, kernel->array, bias->array, outrows, outcols, innerdim);
        if (activation_flag == RELU_FLAG) {
            k2c_relu_func(output->array, outsize);
        } else if (activation_flag == SOFTMAX_FLAG) {
            k2c_softmax_func(output->array, outsize);
        }
    } else {
        const size_t axesA[1] = {input->ndim - 1};
        const size_t axesB[1] = {0};
        const size_t naxes = 1;
        const int normalize = 0;
        k2c_dot(output, input, kernel, axesA, axesB, naxes, normalize, fwork);
        k2c_bias_add(output, bias);
        if (activation_flag == RELU_FLAG) {
            k2c_relu_func(output->array, output->numel);
        } else if (activation_flag == SOFTMAX_FLAG) {
            k2c_softmax_func(output->array, output->numel);
        }
    }
}

void k2c_conv2d(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* kernel, const k2c_tensor* bias, const size_t * stride, const size_t * dilation, int activation_flag) {
    for (size_t i = 0; i < output->numel; ++i) {
        output->array[i] = 0;
    }
    const size_t out_rows = output->shape[0];
    const size_t out_cols = output->shape[1];
    const size_t out_channels = output->shape[2];
    const size_t in_channels = input->shape[2];
    for (size_t x0 = 0; x0 < out_rows; ++x0) {
        for (size_t x1 = 0; x1 < out_cols; ++x1) {
            for (size_t z0 = 0; z0 < kernel->shape[0]; ++z0) {
                for (size_t z1 = 0; z1 < kernel->shape[1]; ++z1) {
                    for (size_t q = 0; q < in_channels; ++q) {
                        for (size_t k = 0; k < out_channels; ++k) {
                            output->array[x0 * (output->shape[2] * output->shape[1])
                                          + x1 * (output->shape[2]) + k] +=
                                kernel->array[z0 * (kernel->shape[3] * kernel->shape[2] * kernel->shape[1])
                                              + z1 * (kernel->shape[3] * kernel->shape[2])
                                              + q * (kernel->shape[3]) + k] *
                                input->array[(x0 * stride[0]
                                              + dilation[0] * z0) * (input->shape[2] * input->shape[1])
                                             + (x1 * stride[1] + dilation[1] * z1) * (input->shape[2]) + q];
                        }
                    }
                }
            }
        }
    }
    k2c_bias_add(output, bias);
    if (activation_flag == RELU_FLAG) {
        k2c_relu_func(output->array, output->numel);
    } else if (activation_flag == SOFTMAX_FLAG) {
        k2c_softmax_func(output->array, output->numel);
    }
}
