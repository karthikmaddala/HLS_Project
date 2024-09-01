#include <math.h> 
 #include <string.h> 
#include "./include/k2c_include.h" 
#include "./include/k2c_tensor_include.h" 
#define RELU 0
#define SOFTMAX 1

k2c_tensor conv2d_41_output = {0};
k2c_tensor conv2d_41_padded_input = {0};
k2c_tensor conv2d_41_kernel = {0};
k2c_tensor conv2d_41_bias = {0};
k2c_tensor max_pooling2d_33_output = {0};
k2c_tensor conv2d_42_output = {0};
k2c_tensor conv2d_42_padded_input = {0};
k2c_tensor conv2d_42_kernel = {0};
k2c_tensor conv2d_42_bias = {0};
k2c_tensor max_pooling2d_34_output = {0};
k2c_tensor conv2d_43_output = {0};
k2c_tensor conv2d_43_padded_input = {0};
k2c_tensor conv2d_43_kernel = {0};
k2c_tensor conv2d_43_bias = {0};
k2c_tensor max_pooling2d_35_output = {0};
k2c_tensor flatten_10_output = {0};
k2c_tensor dense_29_output = {0};
k2c_tensor dense_29_kernel = {0};
k2c_tensor dense_29_bias = {0};
k2c_tensor dense_30_output = {0};
k2c_tensor dense_30_kernel = {0};
k2c_tensor dense_30_bias = {0};
k2c_tensor dense_31_kernel = {0};
k2c_tensor dense_31_bias = {0};

void k2c_pad2d(k2c_tensor* output, const k2c_tensor* input, const float fill, const size_t * pad) {
#pragma HLS inline
    const size_t in_height = input->shape[0];
    const size_t in_width = input->shape[1];
    const size_t in_channels = input->shape[2];
    const size_t pad_top = pad[0];
    const size_t pad_left = pad[2];
    const size_t pad_right = pad[3];

    // set output array to fill value
    if (fabs(fill) < 1e-6) {
        // fill is ~zero, use manual set to zero
        for (size_t i1 = 0; i1 < output->numel; ++i1) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=10404
			#pragma HLS PIPELINE II=1
			#pragma HLS UNROLL factor=2

            output->array[i1] = 0;
        }
    } else {
        for (size_t i1 = 0; i1 < output->numel; ++i1) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=10404
			#pragma HLS PIPELINE II=1
			#pragma HLS UNROLL factor=2
            output->array[i1] = fill;
        }
    }

    // manually copy the old array in the middle
    size_t offset = in_channels * (pad_left + pad_right + in_width) * pad_top +
                    in_channels * pad_left;
    const size_t num = in_channels * in_width;
    const size_t step = num + in_channels * (pad_left + pad_right);
    for (size_t i2 = 0; i2 < in_height; ++i2) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=102
		#pragma HLS PIPELINE II=1
		#pragma HLS UNROLL factor=2
        for (size_t j1 = 0; j1 < num; ++j1) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=102
			#pragma HLS PIPELINE II=1
			#pragma HLS UNROLL factor=2
            output->array[offset + j1] = input->array[i2 * num + j1];
        }
        offset += step;
    }
}



void k2c_maxpool2d(k2c_tensor* output, const k2c_tensor* input, const size_t * pool_size, const size_t * stride) {
#pragma HLS inline
    const size_t channels = input->shape[2];
    const size_t output_height = output->shape[1];
    const size_t input_height = input->shape[1];
    const size_t pool_height = pool_size[0];
    const size_t pool_width = pool_size[1];
    const size_t stride_height = stride[0];
    const size_t stride_width = stride[1];

    for (size_t i3 = 0; i3 < channels; ++i3) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=6
        #pragma HLS PIPELINE II=1
        for (size_t j2 = 0, k = 0; j2 < output_height * channels; j2 += channels, k += channels * stride_width) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=150
            #pragma HLS PIPELINE II=1
            for (size_t l = 0, m = 0; l < output->numel; l += channels * output_height, m += channels * input_height * stride_height) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=3750
                #pragma HLS PIPELINE II=1
                size_t out_idx = l + j2 + i3;
                output->array[out_idx] = input->array[m + k + i3];

                for (size_t n = 0; n < pool_width * channels; n += channels) {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=6
                    #pragma HLS UNROLL factor=2
                    for (size_t p = 0; p < pool_height * channels * input_height; p += channels * input_height) {
                        #pragma HLS LOOP_TRIPCOUNT min=1 max=6
                        #pragma HLS UNROLL factor=2
                        size_t in_idx = m + k + i3 + n + p;
                        if (output->array[out_idx] < input->array[in_idx]) {
                            output->array[out_idx] = input->array[in_idx];
                        }
                    }
                }
            }
        }
    }
}


void k2c_flatten(k2c_tensor *output, const k2c_tensor* input) {
#pragma HLS inline
    // Partitioning the arrays for better performance

    // Pipeline the loop for copying elements
    for (size_t i4 = 0; i4 < input->numel; ++i4) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=16
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=2
        output->array[i4] = input->array[i4];
    }

    // Set the shape for the output tensor
    for (size_t i5 = 1; i5 < input->ndim; ++i5) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=3
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=2
        output->shape[i5] = 1;
    }
    output->shape[0] = input->numel;
    output->numel = input->numel;
    output->ndim = 1;
}


void k2c_matmul(float * C, const float * A, const float * B, const size_t outrows, const size_t outcols, const size_t innerdim) {
#pragma HLS inline
    // Initialize the output matrix C to zero
    for (size_t i6 = 0; i6 < outrows * outcols; ++i6) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=840
		#pragma HLS pipeline II=1
		#pragma HLS unroll factor=2
        C[i6] = 0;
    }

    // Loop interchange for better memory access
    for (size_t i7 = 0; i7 < outrows; ++i7) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=84
		#pragma HLS pipeline II=1
		#pragma HLS unroll factor=2
        const size_t outrowidx = i7 * outcols;
        const size_t inneridx = i7 * innerdim;
        for (size_t j3 = 0; j3 < outcols; ++j3) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=10
			#pragma HLS pipeline II=1
			#pragma HLS unroll factor=2
            for (size_t k = 0; k < innerdim; ++k) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=84
				#pragma HLS pipeline II=1
				#pragma HLS unroll factor=2
                C[outrowidx + j3] += A[inneridx + k] * B[k * outcols + j3];
            }
        }
    }
}

void k2c_bias_add(k2c_tensor* A, const k2c_tensor* b) {
#pragma HLS inline
    for (size_t i8 = 0; i8 < A->numel; i8 += b->numel) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=16
		#pragma HLS pipeline II=1
		#pragma HLS unroll factor=2
        for (size_t j4 = 0; j4 < b->numel; ++j4) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=6
			#pragma HLS pipeline II=1
			#pragma HLS unroll factor=2
            A->array[i8 + j4] += b->array[j4];
        }
    }
}

void k2c_affine_matmul(float * C, const float * A, const float * B, const float * d, const size_t outrows, const size_t outcols, const size_t innerdim) {
#pragma HLS inline
    for (size_t i9 = 0; i9 < outrows * outcols; ++i9) {
		#pragma HLS pipeline II=1
		#pragma HLS unroll factor=2
        #pragma HLS LOOP_TRIPCOUNT min=1 max=840
        C[i9] = 0;
    }

    for (size_t i10 = 0; i10 < outrows; ++i10) {
		#pragma HLS pipeline II=1
		#pragma HLS unroll factor=2
        #pragma HLS LOOP_TRIPCOUNT min=1 max=84
        const size_t outrowidx = i10 * outcols;
        const size_t inneridx = i10 * innerdim;
        for (size_t j5 = 0; j5 < outcols; ++j5) {
			#pragma HLS pipeline II=1
			#pragma HLS unroll factor=2
            #pragma HLS LOOP_TRIPCOUNT min=1 max=10
            for (size_t k = 0; k < innerdim; ++k) {
				#pragma HLS pipeline II=1
				#pragma HLS unroll factor=2
                #pragma HLS LOOP_TRIPCOUNT min=1 max=84
                C[outrowidx + j5] += A[inneridx + k] * B[k * outcols + j5];
            }
            C[outrowidx + j5] += d[j5];
        }
    }
}

void k2c_relu_func(float * x, const size_t size) {
#pragma HLS inline
    for (size_t i11 = 0; i11 < size; ++i11) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=15000
		#pragma HLS PIPELINE II=1
		#pragma HLS UNROLL factor=2
        if (x[i11] <= 0.0f) {
            x[i11] = 0.0f;
        }
    }
}

void k2c_softmax_func(float * x, const size_t size) {
#pragma HLS inline
    float xmax = x[0];
    float sum = 0;

    // First loop to find xmax
    for (size_t i12 = 0; i12 < size; ++i12) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=10
		#pragma HLS PIPELINE II=1
		#pragma HLS UNROLL factor=2
        if (x[i12] > xmax) {
            xmax = x[i12];
        }
    }

    // Merged loop to calculate expf and sum
    for (size_t i13 = 0; i13 < size; ++i13) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=10
		#pragma HLS PIPELINE II=1
		#pragma HLS UNROLL factor=2
        x[i13] = expf(x[i13] - xmax);
        sum += x[i13];
    }

    // Normalize the output values
    sum = 1.0f / sum;
    for (size_t i15 = 0; i15 < size; ++i15) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=10
		#pragma HLS PIPELINE II=1
		#pragma HLS UNROLL factor=2
        x[i15] = x[i15] * sum;
    }
}

void k2c_dot(k2c_tensor* C, const k2c_tensor* A, const k2c_tensor* B, const size_t * axesA, const size_t * axesB, const size_t naxes, const int normalize, float * fwork) {
#pragma HLS inline
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
    count = 0;
    for (size_t i16 = 0; i16 < ndimA; ++i16) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=3
		#pragma HLS PIPELINE II=1
        isin = 0;
        for (size_t j6 = 0; j6 < naxes; ++j6) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=1
			#pragma HLS PIPELINE II=1
            if (i16 == axesA[j6]) {
                isin = 1;
            }
        }
        if (!isin) {
            freeA[count] = i16;
            ++count;
        }
    }
    count = 0;
    for (size_t i17 = 0; i17 < ndimB; ++i17) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=2
		#pragma HLS PIPELINE II=1
        isin = 0;
        for (size_t j7 = 0; j7 < naxes; ++j7) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=1
			#pragma HLS PIPELINE II=1
            if (i17 == axesB[j7]) {
                isin = 1;
            }
        }
        if (!isin) {
            freeB[count] = i17;
            ++count;
        }
    }

    // number of elements in inner dimension
    for (size_t i18 = 0; i18 < naxes; ++i18) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1
		#pragma HLS PIPELINE II=1
        prod_axesA *= A->shape[axesA[i18]];
    }
    for (size_t i19 = 0; i19 < naxes; ++i19) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1
		#pragma HLS PIPELINE II=1
        prod_axesB *= B->shape[axesB[i19]];
    }
    // number of elements in free dimension
    free_axesA = A->numel / prod_axesA;
    free_axesB = B->numel / prod_axesB;
    // find permutation of axes to get into matmul shape
    for (size_t i20 = 0; i20 < ndimA - naxes; ++i20) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=2
		#pragma HLS PIPELINE II=1
        permA[i20] = freeA[i20];
    }
    for (size_t i21 = ndimA - naxes, j8 = 0; i21 < ndimA; ++i21, ++j8) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1
		#pragma HLS PIPELINE II=1
        permA[i21] = axesA[j8];
    }
    for (size_t i22 = 0; i22 < naxes; ++i22) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1
		#pragma HLS PIPELINE II=1
        permB[i22] = axesB[i22];
    }
    for (size_t i23 = naxes, j9 = 0; i23 < ndimB; ++i23, ++j9) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1
		#pragma HLS PIPELINE II=1
        permB[i23] = freeB[j9];
    }

    for (size_t i24 = 0; i24 < ndimA; ++i24) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=3
		#pragma HLS PIPELINE II=1
        newshpA[i24] = A->shape[permA[i24]];
    }
    for (size_t i25 = 0; i25 < ndimB; ++i25) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=2
		#pragma HLS PIPELINE II=1
        newshpB[i25] = B->shape[permB[i25]];
    }

    // reshape arrays
    for (size_t i26 = 0; i26 < A->numel; ++i26) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=16
		#pragma HLS PIPELINE II=1
        k2c_idx2sub(i26, Asub, A->shape, ndimA);
        for (size_t j10 = 0; j10 < ndimA; ++j10) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=3
			#pragma HLS PIPELINE II=1
            Bsub[j10] = Asub[permA[j10]];
        }
        size_t bidx = k2c_sub2idx(Bsub, newshpA, ndimA);
        reshapeA[bidx] = A->array[i26];
    }

    for (size_t i27 = 0; i27 < B->numel; ++i27) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=84
		#pragma HLS PIPELINE II=1
        k2c_idx2sub(i27, Bsub, B->shape, ndimB);
        for (size_t j11 = 0; j11 < ndimB; ++j11) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=2
			#pragma HLS PIPELINE II=1
            Asub[j11] = Bsub[permB[j11]];
        }
        size_t bidx = k2c_sub2idx(Asub, newshpB, ndimB);
        reshapeB[bidx] = B->array[i27];
    }

    if (normalize) {
        float sum;
        float inorm;
        for (size_t i28 = 0; i28 < free_axesA; ++i28) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
			#pragma HLS PIPELINE II=1
            sum = 0;
            for (size_t j12 = 0; j12 < prod_axesA; ++j12) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=84
				#pragma HLS PIPELINE II=1
                sum += reshapeA[i28 * prod_axesA + j12] * reshapeA[i28 * prod_axesA + j12];
            }
            inorm = 1.0f / sqrtf(sum);
            for (size_t j13 = 0; j13 < prod_axesA; ++j13) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=84
				#pragma HLS PIPELINE II=1
                reshapeA[i28 * prod_axesA + j13] *= inorm;
            }
        }
        for (size_t i29 = 0; i29 < free_axesB; ++i29) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=10
			#pragma HLS PIPELINE II=1
            sum = 0;
            for (size_t j14 = 0; j14 < prod_axesB; ++j14) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=3
				#pragma HLS PIPELINE II=1
                sum += reshapeB[i29 + free_axesB * j14] * reshapeB[i29 + free_axesB * j14];
            }
            inorm = 1.0f / sqrtf(sum);
            for (size_t j15 = 0; j15 < prod_axesB; ++j15) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=3
				#pragma HLS PIPELINE II=1
                reshapeB[i29 + free_axesB * j15] *= inorm;
            }
        }
    }

    k2c_matmul(C->array, reshapeA, reshapeB, free_axesA, free_axesB, prod_axesA);
}

void k2c_idx2sub(const size_t idx, size_t * sub, const size_t * shape, const size_t ndim) {
#pragma HLS inline
    size_t idx2 = idx;
    for (int i30 = ndim - 1; i30 >= 0; --i30) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=3
		#pragma HLS PIPELINE II=1
		#pragma HLS UNROLL factor=2

        sub[i30] = idx2 % shape[i30];
        idx2 /= shape[i30];
    }
}

size_t k2c_sub2idx(const size_t * sub, const size_t * shape, const size_t ndim) {
#pragma HLS inline
    size_t idx = 0;
    size_t multiplier = 1;
    for (int i31 = ndim - 1; i31 >= 0; --i31) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=3
		#pragma HLS PIPELINE II=1
		#pragma HLS UNROLL factor=2

        idx += sub[i31] * multiplier;
        multiplier *= shape[i31];
    }
    return idx;
}


void k2c_conv2d(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* kernel, const k2c_tensor* bias, const size_t * stride, const size_t * dilation, int activation) {
#pragma HLS inline
    // Initialize output array to zero
    for (size_t i32 = 0; i32 < output->numel; ++i32) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=15000
        #pragma HLS PIPELINE II=1
        output->array[i32] = 0;
    }

    const size_t out_rows = output->shape[0];
    const size_t out_cols = output->shape[1];
    const size_t out_channels = output->shape[2];
    const size_t in_channels = input->shape[2];

    // Perform convolution
    for (size_t x0 = 0; x0 < out_rows; ++x0) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=50
        for (size_t x1 = 0; x1 < out_cols; ++x1) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=50
            for (size_t z0 = 0; z0 < kernel->shape[0]; ++z0) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=3
                for (size_t z1 = 0; z1 < kernel->shape[1]; ++z1) {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=3
                    for (size_t q = 0; q < in_channels; ++q) {
                        #pragma HLS LOOP_TRIPCOUNT min=1 max=1
                        for (size_t k = 0; k < out_channels; ++k) {
                            #pragma HLS LOOP_TRIPCOUNT min=1 max=6
                            #pragma HLS PIPELINE II=1
                            #pragma HLS UNROLL factor=2
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

    // Add bias and apply activation function
    k2c_bias_add(output, bias);
    if (activation == RELU) {
        k2c_relu_func(output->array, output->numel);
    }
}


void k2c_dense(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* kernel, const k2c_tensor* bias, int activation, float * fwork) {
#pragma HLS inline
    size_t outrows = (input->ndim > 1) ? input->shape[0] : 1;
    const size_t outcols = kernel->shape[1];
    const size_t innerdim = kernel->shape[0];
    const size_t outsize = outrows * outcols;

    if (input->ndim <= 2) {
        k2c_affine_matmul(output->array, input->array, kernel->array, bias->array, outrows, outcols, innerdim);
    } else {
        const size_t axesA[1] = {input->ndim - 1};
        const size_t axesB[1] = {0};
        const size_t naxes = 1;
        const int normalize = 0;

        k2c_dot(output, input, kernel, axesA, axesB, naxes, normalize, fwork);
        k2c_bias_add(output, bias);
    }

    if (activation == RELU) {
        k2c_relu_func(output->array, output->numel);
    } else if (activation == SOFTMAX) {
        k2c_softmax_func(output->array, output->numel);
    }
}




void finalcovid(k2c_tensor* input_11_input, k2c_tensor* dense_31_output) { 

size_t conv2d_41_stride[2] = {2,2}; 
size_t conv2d_41_dilation[2] = {1,1}; 
float conv2d_41_output_array[15000] = {0}; 
//k2c_tensor conv2d_41_output = {&conv2d_41_output_array[0],3,15000,{50,50, 6, 1, 1}};

// Copy the contents of conv2d_41_output_array into conv2d_41_output.array
for (size_t i123 = 0; i123 < 15000; ++i123) {
    conv2d_41_output.array[i123] = conv2d_41_output_array[i123];
}

// Set the other members of conv2d_41_output
conv2d_41_output.ndim = 3;
conv2d_41_output.numel = 15000;
conv2d_41_output.shape[0] = 50;
conv2d_41_output.shape[1] = 50;
conv2d_41_output.shape[2] = 6;
conv2d_41_output.shape[3] = 1;
conv2d_41_output.shape[4] = 1;


float conv2d_41_padded_input_array[10404] = {0}; 
//k2c_tensor conv2d_41_padded_input = {&conv2d_41_padded_input_array[0],3,10404,{102,102,  1,  1,  1}};
// Copy the contents of conv2d_41_padded_input_array into conv2d_41_padded_input.array
for (size_t i122 = 0; i122 < 10404; ++i122) {
    conv2d_41_padded_input.array[i122] = conv2d_41_padded_input_array[i122];
}

// Set the other members of conv2d_41_padded_input
conv2d_41_padded_input.ndim = 3;
conv2d_41_padded_input.numel = 10404;
conv2d_41_padded_input.shape[0] = 102;
conv2d_41_padded_input.shape[1] = 102;
conv2d_41_padded_input.shape[2] = 1;
conv2d_41_padded_input.shape[3] = 1;
conv2d_41_padded_input.shape[4] = 1;

size_t conv2d_41_pad[4] = {1,1,1,1}; 
float conv2d_41_fill = 0.0f; 
float conv2d_41_kernel_array[54] = {
-2.55067516e-02f,+5.83739765e-02f,-8.95136371e-02f,-3.43439311e-01f,+1.53045252e-01f,
-7.88822323e-02f,+8.00363719e-02f,+7.87166208e-02f,+3.01350921e-01f,-3.48612607e-01f,
+1.76965773e-01f,+5.61222143e-04f,+3.50030035e-01f,+3.87555450e-01f,+9.47314948e-02f,
+4.86701680e-03f,+7.24235922e-02f,+2.67726243e-01f,-1.26612440e-01f,-9.42768604e-02f,
+9.59738567e-02f,+3.18638422e-02f,-5.51007390e-02f,-2.42772534e-01f,-1.19435832e-01f,
+3.24220955e-01f,+1.58793882e-01f,-3.55597705e-01f,+1.44641742e-01f,-1.50206760e-02f,
-2.07593128e-01f,+2.53390431e-01f,-1.17699457e-02f,-1.13293856e-01f,-5.07774949e-02f,
-1.42860726e-01f,-1.63017899e-01f,+2.06529185e-01f,+1.18194610e-01f,+1.55012175e-01f,
+3.14348608e-01f,-1.56049192e-01f,+6.49732277e-02f,+4.94595543e-02f,+3.15250993e-01f,
+1.86340973e-01f,+2.81690747e-01f,+8.77765268e-02f,+7.18520358e-02f,+2.78305173e-01f,
+1.58508286e-01f,+4.49127518e-02f,+3.68793428e-01f,+9.84197408e-02f,}; 
//k2c_tensor conv2d_41_kernel = {&conv2d_41_kernel_array[0],4,54,{3,3,1,6,1}};
// Copy the contents of conv2d_41_kernel_array into conv2d_41_kernel.array
for (size_t i121 = 0; i121 < 54; ++i121) {
    conv2d_41_kernel.array[i121] = conv2d_41_kernel_array[i121];
}

// Set the other members of conv2d_41_kernel
conv2d_41_kernel.ndim = 4;
conv2d_41_kernel.numel = 54;
conv2d_41_kernel.shape[0] = 3;
conv2d_41_kernel.shape[1] = 3;
conv2d_41_kernel.shape[2] = 1;
conv2d_41_kernel.shape[3] = 6;
conv2d_41_kernel.shape[4] = 1;
float conv2d_41_bias_array[6] = {
+5.68718500e-02f,+6.49347436e-03f,+7.91293979e-02f,+5.48061635e-03f,-5.74038886e-02f,
+1.09323479e-01f,}; 
//k2c_tensor conv2d_41_bias = {&conv2d_41_bias_array[0],1,6,{6,1,1,1,1}}; 
// Copy the contents of conv2d_41_bias_array into conv2d_41_bias.array
for (size_t i120 = 0; i120 < 6; ++i120) {
    conv2d_41_bias.array[i120] = conv2d_41_bias_array[i120];
}

// Set the other members of conv2d_41_bias
conv2d_41_bias.ndim = 1;
conv2d_41_bias.numel = 6;
conv2d_41_bias.shape[0] = 6;
conv2d_41_bias.shape[1] = 1;
conv2d_41_bias.shape[2] = 1;
conv2d_41_bias.shape[3] = 1;
conv2d_41_bias.shape[4] = 1;

 
size_t max_pooling2d_33_stride[2] = {2,2}; 
size_t max_pooling2d_33_pool_size[2] = {2,2}; 
float max_pooling2d_33_output_array[3750] = {0}; 
//k2c_tensor max_pooling2d_33_output = {&max_pooling2d_33_output_array[0],3,3750,{25,25, 6, 1, 1}};
// Copy the contents of max_pooling2d_33_output_array into max_pooling2d_33_output.array
for (size_t i119 = 0; i119 < 3750; ++i119) {
    max_pooling2d_33_output.array[i119] = max_pooling2d_33_output_array[i119];
}

// Set the other members of max_pooling2d_33_output
max_pooling2d_33_output.ndim = 3;
max_pooling2d_33_output.numel = 3750;
max_pooling2d_33_output.shape[0] = 25;
max_pooling2d_33_output.shape[1] = 25;
max_pooling2d_33_output.shape[2] = 6;
max_pooling2d_33_output.shape[3] = 1;
max_pooling2d_33_output.shape[4] = 1; 


size_t conv2d_42_stride[2] = {2,2}; 
size_t conv2d_42_dilation[2] = {1,1}; 
float conv2d_42_output_array[1014] = {0}; 
//k2c_tensor conv2d_42_output = {&conv2d_42_output_array[0],3,1014,{13,13, 6, 1, 1}}; 
// Copy the contents of conv2d_42_output_array into conv2d_42_output.array
for (size_t i118 = 0; i118 < 1014; ++i118) {
    conv2d_42_output.array[i118]= conv2d_42_output_array[i118];
}

// Set the other members of conv2d_42_output
conv2d_42_output.ndim = 3;
conv2d_42_output.numel = 1014;
conv2d_42_output.shape[0] = 13;
conv2d_42_output.shape[1] = 13;
conv2d_42_output.shape[2] = 6;
conv2d_42_output.shape[3] = 1;
conv2d_42_output.shape[4] = 1;
float conv2d_42_padded_input_array[4374] = {0}; 
//k2c_tensor conv2d_42_padded_input = {&conv2d_42_padded_input_array[0],3,4374,{27,27, 6, 1, 1}};
// Copy the contents of conv2d_42_padded_input_array into conv2d_42_padded_input.array
for (size_t i117 = 0; i117 < 4374; ++i117) {
    conv2d_42_padded_input.array[i117] = conv2d_42_padded_input_array[i117];
}

// Set the other members of conv2d_42_padded_input
conv2d_42_padded_input.ndim = 3;
conv2d_42_padded_input.numel = 4374;
conv2d_42_padded_input.shape[0] = 27;
conv2d_42_padded_input.shape[1] = 27;
conv2d_42_padded_input.shape[2] = 6;
conv2d_42_padded_input.shape[3] = 1;
conv2d_42_padded_input.shape[4] = 1; 
size_t conv2d_42_pad[4] = {1,1,1,1}; 
float conv2d_42_fill = 0.0f; 
float conv2d_42_kernel_array[324] = {
+1.44030219e-02f,-1.10098340e-01f,-1.35100126e-01f,-2.43304037e-02f,+1.09268695e-01f,
+3.68055522e-01f,-2.14568838e-01f,+1.32384434e-01f,-1.66369304e-01f,+7.26704746e-02f,
-2.96563447e-01f,-1.22372456e-01f,-7.54395127e-02f,+1.33536950e-01f,+6.46443339e-03f,
+1.85374506e-02f,-1.46087065e-01f,-1.99016348e-01f,-1.52509183e-01f,+7.10294694e-02f,
-1.15595020e-01f,+1.15307875e-01f,-1.29172340e-01f,-1.06911577e-01f,+1.27014637e-01f,
-2.06047654e-01f,-2.15341121e-01f,-6.11191317e-02f,-2.11647555e-01f,+1.20488435e-01f,
+1.31008342e-01f,-8.11097175e-02f,+1.85340658e-01f,+4.54625301e-02f,+1.04101986e-01f,
+2.43421197e-01f,-1.58573389e-01f,-2.53103878e-02f,+2.16623582e-03f,-1.19545884e-01f,
-1.89218625e-01f,+1.67439923e-01f,+2.17839152e-01f,+2.97334135e-01f,+7.29196239e-03f,
+1.28661944e-02f,-1.41163439e-01f,+6.07195497e-02f,-1.92813694e-01f,-6.79905564e-02f,
-1.92119926e-01f,+1.73515007e-01f,+1.91901490e-01f,+1.97994500e-01f,+2.63524465e-02f,
-1.41106918e-01f,-1.07919060e-01f,+1.84745550e-01f,+1.48473352e-01f,+8.85170624e-02f,
-1.19004831e-01f,+1.63293451e-01f,-8.23752657e-02f,+2.32263938e-01f,-1.82172552e-01f,
-1.10267118e-01f,+2.97450274e-01f,+2.74936855e-01f,+1.73596337e-01f,-2.34688893e-01f,
+2.35356279e-02f,-6.43034503e-02f,-1.70985967e-01f,+1.27171040e-01f,+2.77581006e-01f,
-1.16763003e-01f,+1.15003129e-02f,+1.07904568e-01f,-8.16050023e-02f,-6.97137639e-02f,
+2.35843286e-01f,-5.32237031e-02f,+4.56041284e-02f,+2.58013815e-01f,-1.42085373e-01f,
-4.74750027e-02f,+3.05330813e-01f,+1.44875705e-01f,-1.34491906e-01f,+1.86508775e-01f,
+1.95772618e-01f,+9.27010030e-02f,+1.54019758e-01f,-1.31126136e-01f,+1.24496795e-01f,
-4.70096692e-02f,+1.10670187e-01f,-1.67376399e-01f,+1.93814631e-04f,-6.42773742e-03f,
+2.21184477e-01f,+1.40318349e-01f,-8.45170841e-02f,+1.35701895e-01f,+9.71237272e-02f,
+5.60006611e-02f,+7.04929903e-02f,-1.17496386e-01f,+1.65399700e-01f,+2.32863992e-01f,
-1.93109512e-01f,-3.52016419e-01f,+1.79704666e-01f,-1.11249462e-01f,+1.77362710e-01f,
-1.72837302e-01f,-9.67746675e-02f,+2.93891460e-01f,-2.62860149e-01f,-3.03641677e-01f,
-2.09146619e-01f,+9.32304561e-02f,-1.64881036e-01f,+2.98779547e-01f,-8.56357887e-02f,
-1.22815646e-01f,-6.71731383e-02f,+2.34713182e-01f,-1.06127299e-01f,+2.86019862e-01f,
+1.91313382e-02f,-1.60762556e-02f,+8.34808871e-02f,+1.96149141e-01f,-4.32147160e-02f,
+1.38818890e-01f,-2.98421402e-02f,-2.53853768e-01f,+1.14050195e-01f,+2.64542758e-01f,
+2.26704672e-01f,-1.77878678e-01f,-2.35059662e-04f,+2.39832383e-02f,+2.21783087e-01f,
-6.02281950e-02f,+1.04984380e-01f,+8.90195444e-02f,-2.26933714e-02f,+1.01300031e-01f,
+1.56472072e-01f,+2.34139696e-01f,-2.27292374e-01f,+2.14475751e-01f,+3.36146384e-01f,
-1.52544677e-01f,+1.62667856e-01f,-1.69037521e-01f,-3.12513828e-01f,+1.23918355e-01f,
+2.70857751e-01f,-4.12686020e-02f,-1.37783125e-01f,-1.28507808e-01f,+8.38354975e-02f,
+7.46776909e-02f,+5.57740070e-02f,+1.74643189e-01f,-5.46497814e-02f,+6.67317882e-02f,
-5.78299761e-02f,-1.19687647e-01f,+3.01386148e-01f,-2.75482852e-02f,-1.49849758e-01f,
+6.99896459e-03f,+1.76951200e-01f,-1.84017271e-01f,-2.64100581e-01f,-1.21988878e-01f,
-1.88884825e-01f,+1.29562378e-01f,+3.83394629e-01f,-3.02808076e-01f,-1.91668093e-01f,
+6.71924427e-02f,-1.43553063e-01f,-1.20964192e-01f,+1.20422289e-01f,-1.39173746e-01f,
+1.91308990e-01f,+1.86323687e-01f,+1.49654448e-01f,-1.72231689e-01f,-1.23253778e-01f,
+2.74838358e-01f,+2.75274038e-01f,+3.20217274e-02f,+8.40279311e-02f,-8.42970610e-02f,
-1.77121416e-01f,+1.08063780e-01f,-2.87719190e-01f,-6.33543059e-02f,+9.28824469e-02f,
-3.01981461e-03f,-1.03425518e-01f,+1.45700067e-01f,+2.77749151e-01f,+1.33671120e-01f,
-3.93704548e-02f,+1.95337340e-01f,+3.64303261e-01f,-1.74833849e-01f,-2.22361162e-02f,
+1.46240383e-01f,+5.73737621e-02f,-6.51263893e-02f,+2.99470931e-01f,-4.52651531e-02f,
-5.32001890e-02f,+2.23400131e-01f,-1.46875262e-01f,-1.73310284e-02f,-1.85399100e-01f,
+3.49741817e-01f,+6.48869425e-02f,-2.27986604e-01f,-4.00868058e-02f,+1.41798809e-01f,
+2.81922400e-01f,+1.06549323e-01f,+1.34870768e-01f,-1.38032049e-01f,-6.91589266e-02f,
+1.06250323e-01f,-8.19061548e-02f,+8.69684368e-02f,+2.38210112e-01f,+1.55583516e-01f,
-2.55822629e-01f,-3.01644485e-02f,-9.80606228e-02f,+1.98753476e-01f,+2.51959234e-01f,
+1.51376147e-02f,-1.86995313e-01f,-8.56370199e-03f,+3.60153764e-01f,+4.87646200e-02f,
-1.59884393e-01f,+9.93617624e-03f,+1.91845074e-01f,+8.75073597e-02f,-1.59549545e-02f,
-1.99295610e-01f,-4.10983562e-02f,-2.89779995e-02f,+2.55612254e-01f,+1.38912797e-01f,
-1.58744127e-01f,-1.24353440e-02f,+1.80119097e-01f,-5.35152927e-02f,-2.07824018e-02f,
+5.11752591e-02f,+9.30985287e-02f,+1.21503964e-01f,+7.39592314e-02f,-7.36781210e-02f,
-1.35748118e-01f,+2.15932176e-01f,+1.73540086e-01f,+7.31689334e-02f,+2.07778960e-02f,
-1.86120858e-04f,+1.65359914e-01f,+2.19102819e-02f,+6.91945013e-03f,+1.14670828e-01f,
+3.54233325e-01f,+1.79202586e-01f,+2.79940367e-01f,+2.25770876e-01f,+2.00147703e-01f,
+7.87381753e-02f,-3.98103781e-02f,+2.02200621e-01f,-1.88938498e-01f,-2.13722393e-01f,
+3.41623962e-01f,-6.42971694e-02f,-3.68445098e-01f,+1.28021449e-01f,+1.47681937e-01f,
+1.27737001e-01f,+2.04082370e-01f,+2.70666689e-01f,+2.74058878e-01f,+2.76693672e-01f,
-6.82351068e-02f,-6.45407289e-02f,+1.65173605e-01f,-8.14147219e-02f,+3.13070208e-01f,
+6.54871240e-02f,-1.61839873e-01f,-1.08713955e-01f,-5.58083206e-02f,-1.19443901e-01f,
+2.37566754e-01f,+8.18376243e-02f,+1.45964767e-03f,-6.48437291e-02f,-1.70648962e-01f,
+2.29239970e-01f,+2.31820326e-02f,+1.54207632e-01f,+1.43793404e-01f,+1.62345693e-01f,
+3.39900821e-01f,-9.89583060e-02f,-2.02955276e-01f,+2.59749532e-01f,}; 
//k2c_tensor conv2d_42_kernel = {&conv2d_42_kernel_array[0],4,324,{3,3,6,6,1}};
// Copy the contents of conv2d_42_kernel_array into conv2d_42_kernel.array
for (size_t i116 = 0; i116 < 324; ++i116) {  // Use the actual number of iterations required
    conv2d_42_kernel.array[i116] = conv2d_42_kernel_array[i116];
}

// Set the other members of conv2d_42_kernel
conv2d_42_kernel.ndim = 4;
conv2d_42_kernel.numel = 324;
conv2d_42_kernel.shape[0] = 3;
conv2d_42_kernel.shape[1] = 3;
conv2d_42_kernel.shape[2] = 6;
conv2d_42_kernel.shape[3] = 6;
conv2d_42_kernel.shape[4] = 1;


float conv2d_42_bias_array[6] = {
-4.41676900e-02f,-3.27363163e-02f,+1.20726675e-01f,+7.59290010e-02f,+8.46707225e-02f,
-1.65348686e-02f,}; 
//k2c_tensor conv2d_42_bias = {&conv2d_42_bias_array[0],1,6,{6,1,1,1,1}};
// Copy the contents of conv2d_42_bias_array into conv2d_42_bias.array
for (size_t i115 = 0; i115 < 6; ++i115) {  // Use the actual number of iterations required
    conv2d_42_bias.array[i115] = conv2d_42_bias_array[i115];
}

// Set the other members of conv2d_42_bias
conv2d_42_bias.ndim = 1;
conv2d_42_bias.numel = 6;
conv2d_42_bias.shape[0] = 6;
conv2d_42_bias.shape[1] = 1;
conv2d_42_bias.shape[2] = 1;
conv2d_42_bias.shape[3] = 1;
conv2d_42_bias.shape[4] = 1;
 
size_t max_pooling2d_34_stride[2] = {2,2}; 
size_t max_pooling2d_34_pool_size[2] = {2,2}; 
float max_pooling2d_34_output_array[216] = {0}; 
//k2c_tensor max_pooling2d_34_output = {&max_pooling2d_34_output_array[0],3,216,{6,6,6,1,1}}; 
// Copy the contents of max_pooling2d_34_output_array into max_pooling2d_34_output.array
for (size_t i114 = 0; i114 < 216; ++i114) {  // Use the actual number of iterations required
    max_pooling2d_34_output.array[i114] = max_pooling2d_34_output_array[i114];
}

// Set the other members of max_pooling2d_34_output
max_pooling2d_34_output.ndim = 3;
max_pooling2d_34_output.numel = 216;
max_pooling2d_34_output.shape[0] = 6;
max_pooling2d_34_output.shape[1] = 6;
max_pooling2d_34_output.shape[2] = 6;
max_pooling2d_34_output.shape[3] = 1;
max_pooling2d_34_output.shape[4] = 1;

size_t conv2d_43_stride[2] = {2,2}; 
size_t conv2d_43_dilation[2] = {1,1}; 
float conv2d_43_output_array[144] = {0}; 
//k2c_tensor conv2d_43_output = {&conv2d_43_output_array[0],3,144,{ 3, 3,16, 1, 1}};
// Copy the contents of conv2d_43_output_array into conv2d_43_output.array
for (size_t i113 = 0; i113 < 144; ++i113) {  // Use the actual number of iterations required
    conv2d_43_output.array[i113] = conv2d_43_output_array[i113];
}

// Set the other members of conv2d_43_output
conv2d_43_output.ndim = 3;
conv2d_43_output.numel = 144;
conv2d_43_output.shape[0] = 3;
conv2d_43_output.shape[1] = 3;
conv2d_43_output.shape[2] = 16;
conv2d_43_output.shape[3] = 1;
conv2d_43_output.shape[4] = 1; 
float conv2d_43_padded_input_array[384] = {0}; 
//k2c_tensor conv2d_43_padded_input = {&conv2d_43_padded_input_array[0],3,384,{8,8,6,1,1}};
// Copy the contents of conv2d_43_padded_input_array into conv2d_43_padded_input.array
for (size_t i112 = 0; i112 < 384; ++i112) {  // Use the actual number of iterations required
    conv2d_43_padded_input.array[i112] = conv2d_43_padded_input_array[i112];
}

// Set the other members of conv2d_43_padded_input
conv2d_43_padded_input.ndim = 3;
conv2d_43_padded_input.numel = 384;
conv2d_43_padded_input.shape[0] = 8;
conv2d_43_padded_input.shape[1] = 8;
conv2d_43_padded_input.shape[2] = 6;
conv2d_43_padded_input.shape[3] = 1;
conv2d_43_padded_input.shape[4] = 1;

size_t conv2d_43_pad[4] = {1,1,1,1}; 
float conv2d_43_fill = 0.0f; 
float conv2d_43_kernel_array[864] = {
-5.01349792e-02f,+6.41878471e-02f,-2.39880309e-01f,+1.53452307e-01f,+3.87640186e-02f,
+1.65755510e-01f,+1.67982727e-01f,+2.47984335e-01f,-1.09916389e-01f,-4.43557575e-02f,
-1.88522235e-01f,-4.85189855e-02f,+8.26046020e-02f,-8.46815482e-02f,-1.82563499e-01f,
-2.93204356e-02f,-6.03719940e-03f,+2.72888392e-02f,+1.26621991e-01f,-7.97298849e-02f,
+1.51324332e-01f,-8.49317908e-02f,-1.51950434e-01f,+1.28364310e-01f,+1.50082365e-01f,
+9.42708924e-02f,-1.07309394e-01f,+5.29333353e-02f,-1.78005263e-01f,+2.15219408e-01f,
+8.67435485e-02f,+6.15767390e-02f,-1.28011584e-01f,+4.29669470e-02f,-1.88846737e-01f,
-4.19009924e-02f,+1.98883668e-01f,+2.13783160e-01f,-1.54797480e-01f,+7.18345791e-02f,
-3.81251685e-02f,+2.94817477e-01f,+4.62412983e-02f,+6.13182783e-03f,+1.58310562e-01f,
-1.32517189e-01f,-2.52613962e-01f,+4.69885133e-02f,+5.57586551e-02f,-3.31762359e-02f,
+8.13717395e-02f,+1.28659934e-01f,-7.40572065e-02f,+7.75547996e-02f,-5.83649389e-02f,
+2.40935877e-01f,-7.23433793e-02f,+1.73862174e-01f,+1.71285316e-01f,-1.28990576e-01f,
+1.08879164e-01f,+1.02038726e-01f,+1.24460950e-01f,+1.10020936e-02f,+3.12452093e-02f,
+1.53756708e-01f,-4.30533402e-02f,-5.52482232e-02f,+1.17123134e-01f,-4.36986564e-03f,
+1.59074187e-01f,-9.83425751e-02f,+8.98630470e-02f,+3.85394916e-02f,+1.21434264e-01f,
+2.31304020e-02f,+4.40480150e-02f,+1.80798873e-01f,-1.77455232e-01f,+1.52254879e-01f,
-7.53763542e-02f,+4.25219312e-02f,-1.72778279e-01f,+1.63810194e-01f,-2.65085585e-02f,
+4.68311366e-04f,-2.04134583e-01f,+7.44387358e-02f,+5.93615696e-02f,+1.42130896e-01f,
+2.12104488e-02f,-1.18054613e-01f,+1.87028691e-01f,-1.87296346e-01f,-2.47091711e-01f,
+3.75056490e-02f,-2.26292282e-01f,+5.02050221e-02f,-2.54319347e-02f,-1.36098027e-01f,
+1.68727919e-01f,+2.69981660e-02f,+1.64975137e-01f,-1.38294563e-01f,-6.08782694e-02f,
-6.90656900e-03f,-4.72516753e-02f,+1.71601862e-01f,-4.70007695e-02f,+2.27330521e-01f,
-2.54550934e-01f,-2.30928302e-01f,+1.17643336e-02f,-2.21491620e-01f,-6.72289133e-02f,
+5.51894456e-02f,+4.48166765e-02f,+2.00235844e-01f,+1.51295081e-01f,-1.47121074e-02f,
+2.23225936e-01f,+1.44890055e-01f,-1.26925409e-01f,-2.26855278e-02f,+1.84958190e-01f,
-7.76833147e-02f,-1.48052514e-01f,-1.69411123e-01f,+2.42804676e-01f,-8.00898392e-03f,
-6.19037971e-02f,+4.52763736e-02f,+1.19518466e-01f,+1.29554451e-01f,-9.98144411e-03f,
+6.58877268e-02f,-1.42831787e-01f,+5.62627101e-03f,-1.23367384e-02f,+6.02033436e-02f,
-2.27555901e-01f,-2.93779105e-01f,-1.35420635e-01f,+3.02485645e-01f,+7.50593841e-02f,
-3.30148824e-02f,-8.06720778e-02f,-1.19827844e-01f,+1.25282764e-01f,+6.94096088e-02f,
+1.26279652e-01f,-9.05147046e-02f,+2.12956652e-01f,+4.11325991e-02f,-1.27041757e-01f,
-1.64883345e-01f,-6.17247559e-02f,-3.47663723e-02f,-2.51207035e-03f,-1.99904561e-01f,
+9.40305088e-03f,-2.22791180e-01f,-1.23174516e-02f,-1.03448011e-01f,+4.18063886e-02f,
+4.87105548e-02f,+1.20875143e-01f,+7.82091692e-02f,-1.14636973e-01f,+1.46879137e-01f,
+1.91884004e-02f,+9.53464806e-02f,-8.49429443e-02f,+1.40968710e-01f,-8.39210972e-02f,
-1.35807112e-01f,+3.00457757e-02f,-2.26918206e-01f,-7.82773569e-02f,+1.11399382e-01f,
+2.38183048e-02f,+2.51271985e-02f,+7.80992769e-03f,-2.59893775e-01f,+1.67326376e-01f,
+1.29150376e-01f,+1.80424049e-01f,+1.22383565e-01f,-1.06511526e-01f,-1.12694569e-01f,
-7.67459571e-02f,-1.54050782e-01f,-2.27486148e-01f,-7.33149126e-02f,-1.44025266e-01f,
-1.73451155e-02f,-1.68140791e-02f,+9.77306627e-03f,+2.03944907e-01f,-1.00935251e-01f,
+2.05438480e-01f,+7.67480209e-02f,-1.38114348e-01f,-1.13832355e-01f,+1.69594735e-01f,
+5.64846508e-02f,-2.25868285e-01f,+3.39359492e-02f,-1.37639776e-01f,-1.69555977e-01f,
+1.12773918e-01f,-3.78624201e-02f,-6.34128824e-02f,-9.97633114e-03f,+2.72087306e-01f,
-1.81596622e-01f,-4.58888449e-02f,-7.75193721e-02f,-7.98664615e-02f,-8.32379982e-02f,
+1.75660461e-01f,+2.40520686e-01f,+2.18481243e-01f,-4.59828414e-02f,-1.26579508e-01f,
+1.74437821e-01f,+1.46210283e-01f,+7.80871511e-02f,-2.54641250e-02f,+5.56358276e-03f,
-1.36035770e-01f,+3.39910060e-01f,+7.63554722e-02f,+6.28473610e-02f,+1.45059349e-02f,
+4.96705025e-02f,+3.79767967e-03f,-4.87662293e-02f,-4.23279442e-02f,+3.32667947e-01f,
+2.16125280e-01f,-6.08701296e-02f,+1.78809594e-02f,-1.08070359e-01f,+1.71776280e-01f,
+1.23040251e-01f,-9.51243564e-02f,+6.21136948e-02f,-3.54629643e-02f,-1.21101923e-01f,
+2.18325593e-02f,-1.02663845e-01f,-9.77468938e-02f,+8.09211563e-03f,+1.11029990e-01f,
+1.17054451e-02f,+1.21243142e-01f,+7.13494867e-02f,-4.96486649e-02f,-4.52804863e-02f,
+1.52116880e-01f,+1.73027501e-01f,+1.82003334e-01f,+2.95514017e-02f,+1.00612573e-01f,
-4.48003970e-02f,-1.53461948e-01f,-3.85869890e-02f,+2.34997064e-01f,+1.15228087e-01f,
+9.58741531e-02f,+9.43289027e-02f,-1.01508386e-01f,-1.88929200e-01f,-9.87441689e-02f,
+1.42063648e-01f,-1.04452096e-01f,+9.12493169e-02f,-1.21940061e-01f,+1.01521567e-01f,
+7.19410852e-02f,+1.31628677e-01f,+1.10691279e-01f,+9.80561972e-03f,+2.23371491e-01f,
-1.36062771e-01f,-5.53449541e-02f,-6.65507615e-02f,-2.77067032e-02f,-2.11369157e-01f,
-1.44611895e-02f,-6.39244542e-02f,+1.26027092e-01f,-1.69725642e-01f,-9.06904042e-03f,
+2.53056288e-01f,+1.85185056e-02f,-1.18638635e-01f,+2.27906480e-01f,+1.29665494e-01f,
-1.31097093e-01f,-4.91693988e-03f,+7.48881102e-02f,-2.15935875e-02f,-2.14406429e-03f,
+1.36673003e-01f,-2.55297404e-02f,-1.45222709e-01f,+2.08978996e-01f,+1.51041195e-01f,
+5.04466891e-02f,-8.89042318e-02f,+9.67364535e-02f,+1.43468961e-01f,-6.90761507e-02f,
-1.09529048e-02f,+1.78088829e-01f,+6.71149194e-02f,+6.95890337e-02f,-1.78941756e-01f,
-2.41162449e-01f,-2.60233968e-01f,-1.72613889e-01f,-9.98195335e-02f,+1.65770248e-01f,
+2.98937619e-01f,-8.01771786e-03f,-1.24142773e-01f,+8.40540156e-02f,+1.81537583e-01f,
-1.78547442e-01f,+1.07140839e-03f,+1.50201246e-01f,+9.71552506e-02f,+3.62582244e-02f,
-2.48994574e-01f,+5.91275003e-03f,-7.49319792e-02f,+1.67947322e-01f,+1.49250895e-01f,
-1.97047800e-01f,-7.33184686e-04f,+8.38994384e-02f,+5.22484891e-02f,-7.60169774e-02f,
+5.11173978e-02f,-9.16192308e-02f,-7.65764564e-02f,-1.08153671e-01f,-1.42883763e-01f,
+1.97619393e-01f,+2.24146351e-01f,-5.70542328e-02f,+8.20623413e-02f,-1.26285255e-01f,
-1.39292330e-01f,-5.49776033e-02f,-1.22109190e-01f,-6.58331364e-02f,+1.48778692e-01f,
-1.13627739e-01f,-1.28442228e-01f,+2.05958530e-01f,-1.11728176e-01f,+7.20852464e-02f,
+1.59878731e-01f,+4.62003089e-02f,+1.09367475e-01f,+1.20039985e-01f,-1.02167204e-01f,
-1.25476331e-01f,-1.58128336e-01f,-4.20484692e-03f,+5.20953685e-02f,-7.79518709e-02f,
+2.80520111e-01f,-9.86564010e-02f,-1.22358948e-01f,-6.41867220e-02f,-1.70838818e-01f,
+4.13254872e-02f,+1.73925199e-02f,-7.88359728e-04f,+1.44407108e-01f,-1.84382766e-01f,
-1.60585970e-01f,-2.07597256e-01f,-3.45491320e-02f,+7.74385780e-02f,+2.69274805e-02f,
+1.98854208e-01f,-5.05289733e-02f,+7.26065710e-02f,-9.46732014e-02f,-1.82170182e-01f,
+1.59362108e-02f,+3.71716022e-02f,+2.57438552e-02f,-7.42991865e-02f,-1.62584066e-01f,
-4.29882854e-02f,-1.23714723e-01f,+7.29420781e-02f,+6.25362098e-02f,-9.49045792e-02f,
-3.90465260e-02f,+2.13837922e-01f,-1.14583403e-01f,+2.04641685e-01f,+3.76255549e-02f,
+8.11163411e-02f,+8.35980475e-03f,+2.08376408e-01f,-1.56569965e-02f,-1.45801008e-01f,
-1.74133480e-01f,+2.48524785e-01f,+1.55430045e-02f,+2.05082133e-01f,-1.65653065e-01f,
-2.02756241e-01f,+1.53338043e-02f,+3.96028347e-02f,+5.59262931e-02f,-5.69364168e-02f,
+2.71039099e-01f,+2.67663717e-01f,-1.14679635e-02f,-1.17794797e-01f,-8.29783604e-02f,
+2.04118237e-01f,+2.09475175e-01f,-9.67895091e-02f,-1.36053011e-01f,+1.34760179e-02f,
-1.69628412e-01f,+7.60395229e-02f,+1.31400406e-01f,+3.12439147e-02f,-1.83728058e-02f,
-9.27229077e-02f,-2.39581317e-02f,-1.60914455e-02f,-1.65091768e-01f,+5.86977974e-02f,
-3.96091305e-02f,+9.50369909e-02f,-1.51789457e-01f,-4.05964777e-02f,+4.38036807e-02f,
+1.50139555e-01f,-6.90108165e-02f,+2.97293067e-02f,+7.90530518e-02f,+2.85757277e-02f,
+2.48874798e-02f,-8.41624662e-02f,-1.40366584e-01f,+2.20093299e-02f,-1.61848247e-01f,
-3.73063958e-03f,+1.27086103e-01f,-1.11066736e-02f,+1.09078333e-01f,+9.20791999e-02f,
-3.32974344e-02f,+4.81359586e-02f,+1.09033436e-02f,+1.02981217e-01f,+2.17017978e-01f,
-8.55875015e-03f,-2.13047236e-01f,+1.33295134e-01f,+1.81732982e-01f,+2.39244401e-02f,
+2.82775611e-02f,+9.10148546e-02f,-5.40883616e-02f,-8.09702128e-02f,-2.06005767e-01f,
+7.38112554e-02f,-3.61883305e-02f,+1.31557003e-01f,-1.49412468e-01f,+8.13378319e-02f,
+3.25413458e-02f,+2.66746044e-01f,+3.30978981e-03f,+2.01499552e-01f,-8.05767253e-02f,
+4.64951135e-02f,+4.61273193e-02f,+1.56652480e-01f,+1.20522700e-01f,+2.48964448e-02f,
-1.73128378e-02f,-6.75556660e-02f,-7.01593086e-02f,-7.98990298e-03f,+9.92763937e-02f,
-5.26624061e-02f,-6.86868152e-04f,+9.03815702e-02f,-7.89428204e-02f,+4.05643135e-02f,
-1.11623496e-01f,+8.24743956e-02f,-6.74520284e-02f,+5.59533387e-02f,+2.19316274e-01f,
+1.64971471e-01f,+9.78951007e-02f,-6.49256483e-02f,+9.99817625e-02f,+1.84890851e-01f,
-1.93488598e-02f,-7.45850801e-02f,-2.38244966e-01f,+1.03184260e-01f,+3.38275850e-01f,
-1.43905610e-01f,+1.31589502e-01f,+3.54129933e-02f,+1.59605443e-01f,-2.57789522e-01f,
-2.01563045e-01f,-2.09133089e-01f,+1.38567537e-01f,-2.50230413e-02f,+4.17893268e-02f,
+4.63415608e-02f,-1.57428756e-01f,-1.52338082e-02f,+1.23783611e-01f,+1.92854434e-01f,
+8.44884962e-02f,+1.43451998e-02f,-1.96587861e-01f,+1.38305113e-01f,+1.05951875e-01f,
-8.15193877e-02f,+3.37636024e-02f,-2.97095366e-02f,+1.12790897e-01f,+3.61222029e-02f,
+1.61645934e-01f,+1.82088077e-01f,-8.25226009e-02f,+1.99165642e-01f,+1.07424796e-01f,
+1.15690336e-01f,+2.76235938e-01f,+4.71151918e-02f,-9.28406343e-02f,+2.08309874e-01f,
-1.70544729e-01f,+4.97028884e-03f,-6.78041345e-03f,+3.68989781e-02f,+7.48590603e-02f,
-1.26980782e-01f,+1.88304193e-03f,-8.83304253e-02f,+5.25471866e-02f,-2.92303953e-02f,
+5.58598414e-02f,+3.49217504e-02f,+2.05334306e-01f,+1.33770540e-01f,+1.83268622e-01f,
+4.77638096e-02f,-9.99981239e-02f,+1.59015447e-01f,-4.93971445e-02f,-1.48242772e-01f,
+2.57859956e-02f,-3.57208662e-02f,-1.60975114e-01f,+4.88208272e-02f,+8.26771557e-03f,
-1.53141633e-01f,-2.51080811e-01f,+1.03603490e-01f,+2.63963372e-01f,-1.85911030e-01f,
+1.67765319e-01f,+1.63676336e-01f,+8.26356113e-02f,+7.20079467e-02f,+5.27110845e-02f,
-1.63264766e-01f,+9.60656255e-02f,+5.52048907e-02f,-2.58985125e-02f,+1.15323707e-01f,
-7.10263848e-03f,-1.59483522e-01f,-1.95426613e-01f,-1.90391578e-02f,+9.14871246e-02f,
+1.78973511e-01f,+1.05277345e-01f,-1.59498960e-01f,-8.29871893e-02f,+7.49243237e-03f,
-2.77736131e-02f,+8.08092654e-02f,-1.36125848e-01f,-4.29790393e-02f,-1.48680538e-01f,
-1.90815672e-01f,-1.06996506e-01f,+1.82956576e-01f,-1.68086383e-02f,-5.23360400e-03f,
-1.02136722e-02f,+8.67657214e-02f,-2.74703801e-02f,-1.32263139e-01f,+4.57607210e-02f,
+2.40391672e-01f,-7.37270117e-02f,-5.92184328e-02f,-7.42463768e-02f,+3.25056538e-02f,
+2.45183766e-01f,-7.69288167e-02f,+1.43640548e-01f,+6.48330525e-03f,-1.99976176e-01f,
+6.46798462e-02f,-1.39729425e-01f,-1.86821088e-01f,+1.66441977e-01f,+1.29382347e-03f,
+7.60752559e-02f,-2.05797389e-01f,+6.97546033e-03f,+1.75894693e-01f,-6.14147410e-02f,
-7.91239142e-02f,+6.08471110e-02f,-2.50427052e-02f,-1.26542002e-01f,-3.00021991e-02f,
+5.35893962e-02f,-4.26518768e-02f,+7.58583993e-02f,-2.10290030e-01f,-6.17685430e-02f,
-5.62108234e-02f,-5.17249554e-02f,-1.24787629e-01f,+1.49517909e-01f,+3.06435674e-02f,
-5.48770986e-02f,+1.75974429e-01f,-1.88517764e-01f,-1.14580251e-01f,+5.96284866e-02f,
+1.71712831e-01f,+1.60183936e-01f,+2.43491262e-01f,+2.92727768e-01f,-4.34967950e-02f,
+1.99192777e-01f,-7.08838343e-04f,-1.18449599e-01f,+7.48564079e-02f,+1.61784083e-01f,
-1.50033563e-01f,+2.11006671e-01f,+9.23418161e-03f,-1.53693020e-01f,-1.28259510e-01f,
+1.66484922e-01f,+1.99712604e-01f,+1.63247615e-01f,+9.41619128e-02f,+6.82239532e-02f,
-1.71924278e-01f,-9.58791599e-02f,+4.50168699e-02f,-9.52596962e-03f,-1.11927956e-01f,
+6.29960075e-02f,+1.04738571e-01f,+4.71992306e-02f,-1.13260649e-01f,-1.23625748e-01f,
+7.51323327e-02f,-1.08912751e-01f,-3.18278112e-02f,+2.48208687e-01f,+1.41614705e-01f,
-1.84908226e-01f,+1.71085298e-01f,+1.72585547e-01f,+9.68843922e-02f,+2.08767205e-02f,
+1.65475398e-01f,-1.07975103e-01f,-1.08165249e-01f,-6.03816565e-03f,-1.98435321e-01f,
-2.10406445e-02f,+5.28344736e-02f,+1.19779170e-01f,-1.99189112e-01f,-2.59542502e-02f,
-2.16847613e-01f,+2.52814621e-01f,+4.35455702e-02f,+3.68543155e-02f,+1.53330237e-01f,
+2.26805061e-02f,+1.77727446e-01f,-2.58106500e-01f,-1.30854230e-02f,+2.72266772e-02f,
+2.29985088e-01f,+1.93584591e-01f,-4.84234886e-03f,-1.46186039e-01f,-3.93972779e-03f,
-7.84220770e-02f,+9.41081643e-02f,+1.45649552e-01f,+2.03042421e-02f,-8.37254450e-02f,
+1.48500800e-01f,+2.31359899e-03f,-5.03054708e-02f,-5.86349592e-02f,+6.02847077e-02f,
+2.71958522e-02f,+2.23613530e-01f,+6.05362607e-03f,-8.16767663e-02f,+1.29429609e-01f,
-3.68296243e-02f,-9.61354822e-02f,-1.10567875e-01f,+1.00711018e-01f,+3.13408114e-02f,
+8.92591998e-02f,+1.32028922e-01f,-9.43035334e-02f,-9.00128707e-02f,+2.01251149e-01f,
+1.82547897e-01f,-4.11034226e-02f,+1.13956638e-01f,-3.40922140e-02f,+1.11389466e-01f,
+1.04136139e-01f,-6.64417818e-02f,+1.36979923e-01f,-6.90359473e-02f,-1.42688841e-01f,
+4.25762869e-02f,+9.21968222e-02f,+1.29243687e-01f,-1.09287195e-01f,+1.85660020e-01f,
-9.12188888e-02f,+4.61282954e-02f,+3.40045430e-02f,-1.69666827e-01f,-9.03917849e-02f,
-9.79438424e-02f,+9.63703096e-02f,+3.62446997e-03f,+5.82290627e-03f,-6.26471862e-02f,
+2.36087739e-01f,+1.62524194e-01f,-1.00131102e-01f,+6.52310997e-02f,-1.36653960e-01f,
+2.60640293e-01f,+2.59312405e-03f,+9.62374359e-02f,-7.40111247e-02f,+1.16386168e-01f,
-3.68477292e-02f,+5.29907644e-02f,+9.33555663e-02f,+2.08490565e-02f,-1.15460314e-01f,
-1.24868065e-01f,-5.48332930e-02f,-5.37523404e-02f,+9.91278812e-02f,-7.75439665e-02f,
+1.48982048e-01f,+1.43649489e-01f,-1.03212141e-01f,+4.70752977e-02f,-7.90927634e-02f,
+6.50078245e-03f,-1.79612473e-01f,-1.97187662e-01f,-1.56985909e-01f,-3.73500548e-02f,
+1.09171458e-01f,+1.98915809e-01f,+1.09910835e-02f,-1.90699846e-02f,+1.92109510e-01f,
+2.97430158e-02f,-1.64343297e-01f,-1.33091748e-01f,-1.74523834e-02f,-1.45644218e-01f,
-7.83147439e-02f,+1.89255446e-01f,+2.50022739e-01f,+2.36682072e-01f,-2.40686536e-03f,
+3.91104817e-02f,-1.38289437e-01f,+5.15105277e-02f,-5.52391671e-02f,+9.92746428e-02f,
+4.22881730e-03f,+1.88989654e-01f,-1.59564734e-01f,+9.38383043e-02f,-2.88210716e-02f,
+9.92489606e-02f,+1.94879517e-01f,+2.12585390e-01f,+3.96524742e-02f,-1.82312820e-02f,
-4.47314978e-03f,+2.38321833e-02f,+1.02995522e-01f,-1.49666443e-01f,+2.37945691e-01f,
+2.37136886e-01f,-1.11426651e-01f,+7.73852840e-02f,-1.17414132e-01f,-4.65562083e-02f,
-1.37153119e-01f,+1.56131282e-01f,-4.84468117e-02f,+9.06210914e-02f,+5.49265072e-02f,
-1.99747607e-01f,-7.56806880e-02f,+2.80039817e-01f,+2.36715585e-01f,+2.41138428e-01f,
+4.05225903e-02f,+2.66946286e-01f,-4.89426926e-02f,+9.02789831e-02f,-8.94797221e-02f,
-5.08679301e-02f,+2.20151275e-01f,-2.34327540e-01f,-1.46506622e-01f,}; 
//k2c_tensor conv2d_43_kernel = {&conv2d_43_kernel_array[0],4,864,{ 3, 3, 6,16, 1}};
// Copy the contents of conv2d_43_kernel_array into conv2d_43_kernel.array
for (size_t i111 = 0; i111 < 864; ++i111) {  // Use the actual number of iterations required
    conv2d_43_kernel.array[i111] = conv2d_43_kernel_array[i111];
}

// Set the other members of conv2d_43_kernel
conv2d_43_kernel.ndim = 4;
conv2d_43_kernel.numel = 864;
conv2d_43_kernel.shape[0] = 3;
conv2d_43_kernel.shape[1] = 3;
conv2d_43_kernel.shape[2] = 6;
conv2d_43_kernel.shape[3] = 16;
conv2d_43_kernel.shape[4] = 1; 
float conv2d_43_bias_array[16] = {
+8.14355090e-02f,+6.79318160e-02f,+7.03534111e-02f,+0.00000000e+00f,-5.66898733e-02f,
-5.82899563e-02f,-5.27003892e-02f,+7.99519867e-02f,-3.36624868e-02f,+7.82038942e-02f,
+7.66309649e-02f,+0.00000000e+00f,-3.57811488e-02f,-7.02783018e-02f,+3.92036550e-02f,
+6.97538480e-02f,}; 
//k2c_tensor conv2d_43_bias = {&conv2d_43_bias_array[0],1,16,{16, 1, 1, 1, 1}}; 
// Copy the contents of conv2d_43_bias_array into conv2d_43_bias.array
for (size_t i110 = 0; i110 < 16; ++i110) {  // Use the actual number of iterations required
    conv2d_43_bias.array[i110] = conv2d_43_bias_array[i110];
}

// Set the other members of conv2d_43_bias
conv2d_43_bias.ndim = 1;
conv2d_43_bias.numel = 16;
conv2d_43_bias.shape[0] = 16;
conv2d_43_bias.shape[1] = 1;
conv2d_43_bias.shape[2] = 1;
conv2d_43_bias.shape[3] = 1;
conv2d_43_bias.shape[4] = 1;
 
size_t max_pooling2d_35_stride[2] = {2,2}; 
size_t max_pooling2d_35_pool_size[2] = {2,2}; 
float max_pooling2d_35_output_array[16] = {0}; 
//k2c_tensor max_pooling2d_35_output = {&max_pooling2d_35_output_array[0],3,16,{ 1, 1,16, 1, 1}}; 
// Copy the contents of max_pooling2d_35_output_array into max_pooling2d_35_output.array
for (size_t i109 = 0; i109 < 16; ++i109) {  // Use the actual number of iterations required
    max_pooling2d_35_output.array[i109] = max_pooling2d_35_output_array[i109];
}

// Set the other members of max_pooling2d_35_output
max_pooling2d_35_output.ndim = 3;
max_pooling2d_35_output.numel = 16;
max_pooling2d_35_output.shape[0] = 1;
max_pooling2d_35_output.shape[1] = 1;
max_pooling2d_35_output.shape[2] = 16;
max_pooling2d_35_output.shape[3] = 1;
max_pooling2d_35_output.shape[4] = 1;


float flatten_10_output_array[16] = {0}; 
//k2c_tensor flatten_10_output = {&flatten_10_output_array[0],1,16,{16, 1, 1, 1, 1}};
// Copy the contents of flatten_10_output_array into flatten_10_output.array
for (size_t i108 = 0; i108 < 16; ++i108) {  // Use the actual number of iterations required
    flatten_10_output.array[i108] = flatten_10_output_array[i108];
}

// Set the other members of flatten_10_output
flatten_10_output.ndim = 1;
flatten_10_output.numel = 16;
flatten_10_output.shape[0] = 16;
flatten_10_output.shape[1] = 1;
flatten_10_output.shape[2] = 1;
flatten_10_output.shape[3] = 1;
flatten_10_output.shape[4] = 1;

float dense_29_output_array[84] = {0}; 
//k2c_tensor dense_29_output = {&dense_29_output_array[0],1,84,{84, 1, 1, 1, 1}};
// Copy the contents of dense_29_output_array into dense_29_output.array
for (size_t i107 = 0; i107 < 84; ++i107) {  // Use the actual number of iterations required
    dense_29_output.array[i107] = dense_29_output_array[i107];
}

// Set the other members of dense_29_output
dense_29_output.ndim = 1;
dense_29_output.numel = 84;
dense_29_output.shape[0] = 84;
dense_29_output.shape[1] = 1;
dense_29_output.shape[2] = 1;
dense_29_output.shape[3] = 1;
dense_29_output.shape[4] = 1;

float dense_29_kernel_array[1344] = {
+1.29233494e-01f,+5.34918811e-03f,-2.42792010e-01f,+7.78914765e-02f,+2.16923043e-01f,
+5.71983345e-02f,+1.31635293e-01f,+1.81421235e-01f,+1.25342295e-01f,-7.17501938e-02f,
-1.54874241e-02f,-1.25918478e-01f,+2.84864217e-01f,-1.77486494e-01f,+2.12785795e-01f,
+1.74357727e-01f,+1.52255893e-01f,+2.07683578e-01f,+2.59426355e-01f,+1.37833208e-01f,
+2.29548380e-01f,-2.80929264e-02f,+2.15513706e-01f,-1.03096887e-01f,+9.11874846e-02f,
-2.32712328e-01f,+2.90954381e-01f,+5.02197370e-02f,-1.95008554e-02f,+1.13218710e-01f,
+1.79779887e-01f,-1.32662252e-01f,-1.06329173e-01f,+2.10208327e-01f,+1.47374973e-01f,
-1.35264859e-01f,-9.68042761e-02f,+1.19248323e-01f,-1.17680855e-01f,-2.06641287e-01f,
+1.04927413e-01f,-9.18583572e-02f,-1.72976758e-02f,+7.84121528e-02f,-2.37733983e-02f,
-7.20980465e-02f,+1.32509917e-01f,+6.10136837e-02f,-2.05289081e-01f,-1.73821971e-01f,
-9.23638940e-02f,-2.39400402e-01f,-3.08125559e-02f,+1.01027630e-01f,+9.60919037e-02f,
-1.73658967e-01f,+2.03176618e-01f,+1.47518858e-01f,+6.09184131e-02f,+5.00259064e-02f,
+2.12979183e-01f,-1.59949362e-02f,+2.32171789e-01f,-1.86533630e-01f,-1.96793184e-01f,
-1.01495370e-01f,+2.03486189e-01f,+1.16323255e-01f,+2.13276267e-01f,+1.80320010e-01f,
+2.33226702e-01f,+2.05908194e-01f,+2.46119455e-01f,+1.26959309e-01f,+1.85313731e-01f,
-4.51766104e-02f,+1.72048323e-02f,+2.82667160e-01f,-7.90595934e-02f,-1.61197901e-01f,
-2.64782310e-01f,+9.86023247e-02f,-2.31835216e-01f,+5.20421714e-02f,+2.50402302e-01f,
-2.86964446e-01f,+2.67475843e-04f,+2.17451025e-02f,+2.50784338e-01f,-1.67402968e-01f,
+1.99280381e-01f,+3.37211430e-01f,-2.34882221e-01f,-7.30855763e-02f,+2.76942533e-02f,
-7.96574727e-02f,+2.01689318e-01f,-6.71008602e-02f,+9.51121897e-02f,-1.67522728e-02f,
+3.40811551e-01f,+1.43494770e-01f,+1.18312077e-03f,+1.20702319e-01f,-1.40098378e-01f,
+2.73517817e-01f,+1.04084268e-01f,-2.07969666e-01f,-8.63278657e-02f,-5.57004809e-02f,
+4.84210812e-02f,+1.95483714e-01f,+8.26354176e-02f,+2.85687029e-01f,-5.41844331e-02f,
-2.32935891e-01f,-1.72561049e-01f,-9.61612090e-02f,+1.00562219e-02f,-2.47164275e-02f,
-2.08149984e-01f,+1.26400650e-01f,-7.86590353e-02f,-1.84163272e-01f,+6.40862435e-02f,
+1.50564834e-01f,-7.69585297e-02f,+9.46681499e-02f,+2.73399770e-01f,+1.37971014e-01f,
-2.31599778e-01f,-1.58220828e-01f,-2.06117705e-01f,-9.04796571e-02f,+1.41711578e-01f,
-8.13987553e-02f,+2.99292386e-01f,-7.46407658e-02f,+3.82954814e-02f,-2.84579843e-02f,
-2.03344911e-01f,-1.88060626e-01f,+2.41857409e-01f,-6.84156567e-02f,+7.60090798e-02f,
+8.19828659e-02f,+5.66886365e-03f,-3.12236547e-02f,+2.07433552e-02f,+1.50422111e-01f,
+5.52860973e-03f,+2.90686160e-01f,-2.19229966e-01f,-1.63710132e-01f,-5.23029827e-02f,
+2.25288779e-01f,+3.37385647e-02f,-3.04059386e-01f,-3.63727403e-03f,+2.35626295e-01f,
+2.37627476e-01f,+2.52854377e-02f,-1.18544791e-02f,-1.01547122e-01f,+6.25378937e-02f,
+3.10401917e-01f,+9.18613523e-02f,-2.95644403e-02f,+2.96598613e-01f,-5.00255153e-02f,
+2.18604371e-01f,-2.75171518e-01f,+3.41499597e-01f,+3.02320719e-03f,-8.15621316e-02f,
+2.59293139e-01f,-2.64351487e-01f,+1.08894303e-01f,+1.85019717e-01f,+1.41997218e-01f,
+8.23618323e-02f,+2.24540949e-01f,+1.89712688e-01f,+1.65707693e-01f,+2.52268016e-01f,
-1.74320191e-01f,+8.42906237e-02f,+1.03370197e-01f,+7.87194520e-02f,+3.48513544e-01f,
-2.23196745e-01f,-1.67858481e-01f,-8.36382210e-02f,+2.60853618e-02f,+3.69829923e-01f,
-4.52485420e-02f,+9.12864283e-02f,+7.35518560e-02f,-1.10262729e-01f,+1.53818697e-01f,
+4.72599417e-02f,-1.41314000e-01f,-3.42093371e-02f,-1.96038067e-01f,-3.57667357e-01f,
-1.75324470e-01f,+9.00637358e-02f,-2.17762649e-01f,-5.95315024e-02f,-8.63253474e-02f,
-2.03827143e-01f,-1.29051790e-01f,+2.24950522e-01f,-2.03586593e-01f,-5.11299632e-03f,
+1.02058396e-01f,+3.04020401e-02f,-1.39338702e-01f,+8.47014636e-02f,+1.17708370e-01f,
+1.55233547e-01f,+4.69417311e-02f,+1.74598396e-01f,+2.36363515e-01f,-1.87232807e-01f,
-1.99775994e-02f,+3.21127892e-01f,+8.71540308e-02f,+2.16213778e-01f,-4.70651090e-02f,
+1.79580078e-01f,-2.36554086e-01f,+1.45489320e-01f,-1.46813661e-01f,-2.66421754e-02f,
+1.58317745e-01f,-3.15276146e-01f,-2.12308139e-01f,-1.57140598e-01f,+4.69216071e-02f,
-1.13267414e-01f,-1.26823500e-01f,-9.19480622e-02f,-1.70967311e-01f,+3.89046043e-01f,
+3.09277952e-01f,+1.87079906e-01f,-2.23215848e-01f,-3.73591244e-01f,+1.33681610e-01f,
+7.53561109e-02f,-9.01064277e-02f,-2.30521157e-01f,-2.13462710e-02f,-1.99815169e-01f,
-6.79260790e-02f,-2.19636381e-01f,+2.10959956e-01f,+2.14632675e-01f,+6.67690486e-02f,
+1.58195302e-01f,+1.16807774e-01f,-7.74290413e-02f,+1.88270822e-01f,-1.92260608e-01f,
-5.23808748e-02f,+2.19690099e-01f,-7.52363950e-02f,-5.68631142e-02f,-5.04226536e-02f,
+1.87825963e-01f,+1.48992524e-01f,+2.09158048e-01f,-1.25930101e-01f,+1.28322944e-01f,
-1.32224202e-01f,+1.07918605e-01f,+6.17515594e-02f,+2.59337276e-02f,+2.25822672e-01f,
-2.08808541e-01f,+1.34851232e-01f,-1.11364976e-01f,-2.45286822e-02f,-1.37598082e-01f,
-3.82550508e-02f,+2.22819999e-01f,+1.13619164e-01f,+1.04373589e-01f,-2.20717952e-01f,
-1.27794594e-01f,+7.77909011e-02f,-1.92514479e-01f,+5.54232299e-03f,-2.42107496e-01f,
-2.05037504e-01f,-2.43675724e-01f,+3.04967165e-03f,+1.07041493e-01f,-6.58334643e-02f,
-1.59177423e-01f,+2.08284095e-01f,+2.19679460e-01f,+1.25953272e-01f,-2.97392905e-02f,
+1.12503722e-01f,+1.57834277e-01f,+6.53843582e-03f,+1.63577899e-01f,-2.39347547e-01f,
-1.74376786e-01f,-7.04608858e-02f,+9.97639149e-02f,+2.82006115e-02f,-1.14922032e-01f,
-1.25639901e-01f,+1.56933069e-03f,-1.51492357e-01f,+1.09217599e-01f,-1.02240965e-01f,
+2.31575683e-01f,+1.01892188e-01f,+8.89457613e-02f,-1.88799456e-01f,-2.37663925e-01f,
-1.31424755e-01f,+2.22738907e-01f,+2.42782548e-01f,+1.57073572e-01f,+5.26540130e-02f,
+1.48838013e-02f,-3.43471318e-02f,-2.24241704e-01f,-2.43484974e-05f,+8.30756575e-02f,
+8.62182528e-02f,-1.87241748e-01f,+3.40565108e-02f,-1.87844023e-01f,+2.02616125e-01f,
-9.41965058e-02f,-7.63344914e-02f,+9.58225727e-02f,+2.25980237e-01f,+2.65865207e-01f,
+9.01231319e-02f,-6.25390932e-02f,-2.14042649e-01f,-1.48731321e-01f,+1.60535410e-01f,
+1.77606821e-01f,+1.39398918e-01f,+1.15067638e-01f,-2.91886687e-01f,-4.17968333e-02f,
-2.36520290e-01f,-1.91536844e-01f,+9.71278325e-02f,-1.47510454e-01f,-6.35230988e-02f,
+1.78068712e-01f,-1.08097479e-01f,+2.08196044e-01f,-1.51388839e-01f,+2.11213633e-01f,
-1.70026366e-02f,-7.06676021e-02f,-1.46735504e-01f,-5.83394766e-02f,+2.63042897e-01f,
-1.84096694e-01f,-1.55949518e-01f,+2.40496859e-01f,+2.96080023e-01f,+2.65433669e-01f,
+2.24076912e-01f,-6.62228316e-02f,-2.88912445e-01f,+2.69486994e-01f,-1.65726423e-01f,
-7.74435401e-02f,-2.25633428e-01f,-2.90166512e-02f,-1.23585083e-01f,-1.71769649e-01f,
-6.36047423e-02f,-2.14742869e-02f,-2.13407964e-01f,+2.17785120e-01f,+3.69997998e-03f,
+1.44526824e-01f,-1.96663350e-01f,-1.78989079e-02f,-3.32402140e-02f,+2.85129854e-03f,
+2.42390692e-01f,-1.80478036e-01f,+8.47364962e-03f,-1.84093893e-01f,-1.68412521e-01f,
+1.82576478e-03f,-1.26530856e-01f,-1.31630614e-01f,-2.27622882e-01f,-9.26853810e-03f,
+1.69229582e-02f,-5.10204732e-02f,+4.29796427e-02f,-2.62502521e-01f,+1.43263474e-01f,
+1.03494883e-01f,+9.89374965e-02f,+1.45950317e-01f,-1.04275234e-01f,-1.66962937e-01f,
-4.19874787e-02f,+5.45673631e-03f,-2.54148960e-01f,+1.02552876e-01f,+2.37119570e-01f,
-2.51022279e-01f,+2.87058741e-01f,-1.07090011e-01f,+7.27074407e-03f,+8.19432437e-02f,
+1.56160161e-01f,-2.30241984e-01f,+1.97880253e-01f,+2.92136848e-01f,-1.74363047e-01f,
-1.53953403e-01f,-6.01197965e-02f,+2.26046175e-01f,-1.15775362e-01f,+1.51551530e-01f,
-9.15484428e-02f,-2.03955844e-01f,-1.92338660e-01f,-1.51726633e-01f,+1.45172462e-01f,
+1.30144373e-01f,-2.07687303e-01f,-2.29503155e-01f,-3.47039104e-03f,-3.36776976e-03f,
-1.12173587e-01f,+2.78730746e-02f,-2.31603682e-01f,+8.33904296e-02f,+9.65690892e-03f,
+1.41268402e-01f,+5.89845255e-02f,-1.94053620e-01f,+1.36483446e-01f,+8.46779048e-02f,
+9.33063030e-02f,+3.05486798e-01f,-1.76282778e-01f,+6.69775531e-02f,-2.27134869e-01f,
+1.97096258e-01f,-2.53075629e-01f,+5.96086383e-02f,-2.56211996e-01f,+2.09060654e-01f,
+1.86202556e-01f,-2.12810218e-01f,-2.23153472e-01f,+3.89558077e-02f,-2.32055858e-01f,
-1.38131738e-01f,-9.86926258e-02f,+3.49975601e-02f,+2.10891366e-01f,-1.16044678e-01f,
-8.71455222e-02f,-9.24849659e-02f,-2.86591649e-02f,-1.50851965e-01f,+4.99837548e-02f,
+1.43054500e-01f,-1.28540933e-01f,-1.10131741e-01f,-3.62863839e-02f,-1.79363176e-01f,
-2.11202368e-01f,-4.64661084e-02f,+2.50156857e-02f,-2.34825653e-03f,+2.55858898e-01f,
-2.56151289e-01f,+7.10903406e-02f,-1.76811054e-01f,-1.29103393e-03f,-1.37494430e-02f,
-2.35208303e-01f,-1.13550171e-01f,-2.32175291e-01f,+1.43217832e-01f,-1.71628922e-01f,
+4.04100679e-02f,+3.77841666e-03f,-1.00813836e-01f,-1.47429675e-01f,-2.38738835e-01f,
+2.81098038e-01f,-2.29406521e-01f,+1.75373480e-01f,-1.57654479e-01f,-8.34064335e-02f,
-2.01338157e-01f,-2.24473268e-01f,+2.00560629e-01f,+9.92299616e-03f,-2.80000091e-01f,
-1.52802378e-01f,+1.32724285e-01f,+1.01358265e-01f,+1.43146962e-01f,-7.47611970e-02f,
-1.83962271e-01f,-2.40461394e-01f,-2.56585509e-01f,-9.44707394e-02f,+1.87722728e-01f,
-1.87025130e-01f,-2.60128915e-01f,-2.06597894e-01f,+2.06291731e-02f,-3.19488645e-02f,
+1.41852170e-01f,-9.17487890e-02f,+6.22258559e-02f,-1.18601516e-01f,+1.83637530e-01f,
+1.44508854e-01f,-1.72392964e-01f,+8.94939750e-02f,-1.42147854e-01f,+2.75773704e-02f,
+3.13236922e-01f,+1.31925736e-02f,+5.57602532e-02f,+1.19501799e-02f,+2.37463027e-01f,
+2.66662389e-02f,+2.28609741e-01f,-7.27856457e-02f,-2.02488765e-01f,-2.09119394e-01f,
-1.73102826e-01f,+1.45850331e-02f,+1.26551449e-01f,-1.44087642e-01f,-9.80662256e-02f,
-7.42564350e-02f,+1.08003490e-01f,-1.32311106e-01f,+4.87304851e-02f,-8.63421708e-02f,
-1.63956910e-01f,+2.97474116e-02f,+2.35716254e-02f,+2.58409977e-01f,-2.99500585e-01f,
-8.81109387e-02f,-2.40905732e-01f,-2.20430672e-01f,-1.06814891e-01f,+2.31985077e-01f,
-1.53154309e-03f,-1.44675031e-01f,+2.73770541e-01f,+2.41375476e-01f,+2.08519757e-01f,
-3.55835915e-01f,-1.59764782e-01f,+1.45408183e-01f,+1.58653408e-01f,-5.36703616e-02f,
-1.78098641e-02f,+1.56656682e-01f,+9.14739072e-02f,-8.52583200e-02f,+2.18912899e-01f,
+9.32236835e-02f,-2.01855153e-02f,+1.14707336e-01f,+2.33898312e-01f,+1.47575900e-01f,
-1.63562477e-01f,-7.44236186e-02f,-1.27415791e-01f,-1.40376776e-01f,-1.77320555e-01f,
-2.52693444e-02f,-1.18028671e-01f,-9.68226343e-02f,+9.61132199e-02f,+9.37050022e-03f,
-1.36371091e-01f,+2.08791256e-01f,+2.12158728e-03f,-1.66552007e-01f,+1.40250951e-01f,
+3.03464919e-01f,-1.64631475e-02f,-1.72312677e-01f,-7.70598650e-03f,+1.44729599e-01f,
+2.86676168e-01f,-2.21832454e-01f,-2.04872072e-01f,+1.03839383e-01f,+1.85655177e-01f,
+2.45554168e-02f,-7.28375316e-02f,+1.06519617e-01f,+1.60186574e-01f,-1.49551764e-01f,
-1.12217456e-01f,+1.90069124e-01f,+1.79862022e-01f,+2.33977154e-01f,+1.93103239e-01f,
-2.14714691e-01f,+2.30425999e-01f,+1.07743219e-01f,-2.67436564e-01f,+1.41485482e-01f,
+2.88044363e-01f,+4.69145738e-02f,-1.73660312e-02f,-8.24966878e-02f,+7.19236210e-02f,
-1.22459710e-01f,+1.61308780e-01f,+1.62470981e-01f,+5.29337674e-02f,-2.82631069e-02f,
+1.28405750e-01f,+1.98850811e-01f,+2.62602717e-01f,-9.74697173e-02f,+4.17531002e-03f,
-2.02358037e-01f,+2.03081176e-01f,-8.78259018e-02f,+1.21360667e-01f,-1.66487172e-01f,
-1.64093286e-01f,-7.76366591e-02f,-7.28611946e-02f,-1.38617054e-01f,-1.67818069e-02f,
-2.73490436e-02f,+1.21645734e-01f,+1.32259309e-01f,+2.42996857e-01f,-1.54675141e-01f,
+2.44326726e-01f,+1.44969061e-01f,-2.01176982e-02f,+7.45054632e-02f,-8.68042111e-02f,
+2.05778375e-01f,-1.95325434e-01f,-1.91926032e-01f,+2.19530091e-01f,-1.19808808e-01f,
-1.98156774e-01f,-1.59366697e-01f,-5.73501922e-02f,+3.51622492e-01f,-1.00846887e-02f,
+2.13689238e-01f,+3.23717445e-02f,-2.40799077e-02f,+2.04627395e-01f,-1.67061418e-01f,
+1.84697926e-01f,+1.05505273e-01f,-1.25404269e-01f,-6.05625473e-02f,+3.39433439e-02f,
-2.08304584e-01f,+1.60738170e-01f,-3.11390311e-02f,-3.33157092e-01f,-3.34066778e-01f,
-1.05895540e-02f,+1.16402641e-01f,-7.75572360e-02f,+1.16787560e-01f,-1.81836318e-02f,
-1.27952844e-02f,+1.91736042e-01f,-1.41073495e-01f,+5.23944013e-02f,-2.95816988e-01f,
-6.41767183e-05f,-3.22794099e-03f,+7.51757920e-02f,-5.89474589e-02f,-1.66876644e-01f,
-6.31141011e-03f,-2.10868970e-01f,-2.22938225e-01f,-3.98002304e-02f,+2.41193831e-01f,
-1.42330691e-01f,-9.05083865e-02f,+2.42969424e-01f,-2.67639190e-01f,+1.02281652e-01f,
-1.82975888e-01f,-8.78740773e-02f,-1.15237735e-01f,+2.02342048e-01f,-2.11721778e-01f,
-2.15488687e-01f,+1.49565592e-01f,+1.72341913e-02f,-5.39856106e-02f,-2.49186471e-01f,
+3.10719967e-01f,-1.60826415e-01f,+5.51024526e-02f,-7.70390257e-02f,-7.53163546e-02f,
-9.38458070e-02f,+6.85971826e-02f,+7.91357458e-02f,-1.74362347e-01f,+2.69881636e-02f,
-2.11978391e-01f,+1.30478010e-01f,+8.38893503e-02f,-2.47847557e-01f,-2.65849829e-02f,
+2.96800524e-01f,+3.34283561e-01f,-1.14032276e-01f,+1.35563910e-01f,-1.18568093e-01f,
-1.44173086e-01f,-1.45524457e-01f,-1.85647756e-02f,+7.50733614e-02f,-3.36550891e-01f,
-2.94724144e-02f,+6.37767762e-02f,+2.41099045e-01f,-1.71285085e-02f,+1.33563325e-01f,
-1.63125992e-01f,-4.85430621e-02f,-2.58258909e-01f,+1.14480749e-01f,+5.10099046e-02f,
-3.71582896e-01f,+4.12843823e-02f,-2.40501136e-01f,+7.76191726e-02f,-1.16050974e-01f,
+1.46893665e-01f,+2.04301581e-01f,-6.82151765e-02f,+1.45337477e-01f,-1.00364223e-01f,
+1.20390980e-02f,-2.73635536e-02f,+5.73527142e-02f,+3.82361084e-01f,+2.90951610e-01f,
-1.67396948e-01f,+1.88689545e-01f,+3.80146913e-02f,-4.09790613e-02f,+1.74051419e-01f,
+1.37615174e-01f,+8.03941935e-02f,+3.11564896e-02f,-1.87929794e-01f,-2.06103787e-01f,
+2.23259926e-01f,+3.11238706e-01f,+2.34552503e-01f,-6.02830350e-02f,-2.82695174e-01f,
+5.61786965e-02f,+3.04938495e-01f,+2.77191084e-02f,-4.47670557e-03f,+6.13762364e-02f,
+1.79957822e-01f,-1.26768962e-01f,-9.02262554e-02f,-4.13515121e-02f,+1.81158170e-01f,
-2.86666840e-01f,-1.63161933e-01f,-7.42094591e-02f,-9.89677459e-02f,+3.42848673e-02f,
-2.04625309e-01f,-1.34649798e-01f,-3.91338021e-02f,+2.77562231e-01f,+2.33377665e-01f,
+2.68330038e-01f,-5.47180027e-02f,-1.09128803e-01f,-2.24263147e-01f,+7.31328651e-02f,
-8.43169764e-02f,+2.46707872e-01f,+1.11016616e-01f,+1.80379227e-01f,-2.15035826e-01f,
+6.81705326e-02f,-2.84828544e-02f,-1.86521083e-01f,-2.70406693e-01f,-4.14942689e-02f,
-2.19393805e-01f,+3.14537644e-01f,+1.60842955e-01f,+2.23145038e-01f,-3.56777370e-01f,
+1.23527877e-01f,+1.24095723e-01f,+1.74173921e-01f,+4.76492085e-02f,-9.97264683e-02f,
-6.92330301e-03f,-3.86089422e-02f,+1.86840713e-01f,+2.61672288e-02f,+1.13513008e-01f,
+1.14724435e-01f,+1.19559010e-02f,+2.15591148e-01f,+3.83225735e-04f,+1.96355253e-01f,
-5.24796322e-02f,+1.02613293e-01f,+1.79304808e-01f,+2.28437379e-01f,-1.19152561e-01f,
-2.47943867e-02f,+1.78023219e-01f,-2.58516762e-02f,-1.39339045e-02f,+2.97369838e-01f,
-1.59579217e-01f,-5.20528257e-02f,+1.69841439e-01f,-4.68230620e-03f,-2.24025309e-01f,
-2.03823179e-01f,+1.74305961e-01f,-1.91161279e-02f,-8.88435692e-02f,-2.78232396e-02f,
-1.12654522e-01f,+6.63238168e-02f,+2.79354930e-01f,+4.27831002e-02f,-7.62781873e-02f,
-2.45150045e-01f,+1.73648492e-01f,+1.83052316e-01f,-2.10488573e-01f,+2.85798669e-01f,
+2.11638764e-01f,-2.32835025e-01f,-8.17968994e-02f,-1.75826296e-01f,-1.89753354e-01f,
+1.15638778e-01f,+3.22589241e-02f,-7.24690557e-02f,-2.30904073e-02f,+5.39559405e-03f,
+7.73301646e-02f,-3.91805321e-02f,-1.09640941e-01f,-8.13834071e-02f,-1.39442310e-01f,
-1.01641312e-01f,-1.67193860e-02f,+2.87070423e-01f,+2.23417059e-01f,+1.60555001e-02f,
+6.55091256e-02f,+5.98574132e-02f,-4.19624299e-02f,+1.67502135e-01f,-5.03572524e-02f,
+1.63199678e-01f,-5.46571612e-02f,-1.17067188e-01f,-1.94150686e-01f,+3.60959321e-02f,
-2.23888159e-01f,+3.29179801e-02f,+2.13835731e-01f,-1.27062008e-01f,-1.27020016e-01f,
+4.58584689e-02f,+1.47906885e-01f,-1.81544037e-03f,+3.72838005e-02f,-1.95756592e-02f,
+2.30571777e-02f,+1.82996457e-03f,-1.12863386e-03f,+1.41012341e-01f,-1.79227501e-01f,
+1.83286205e-01f,+2.53529936e-01f,-2.42116898e-02f,-2.47912556e-02f,-2.91635692e-02f,
-1.74200922e-01f,+1.00061700e-01f,-2.10157692e-01f,-9.98513997e-02f,-2.19541639e-02f,
+4.17939872e-02f,+1.04303315e-01f,+1.62869677e-01f,-1.57679319e-02f,+1.12300679e-01f,
-7.97557682e-02f,-1.48731008e-01f,+1.73309043e-01f,-1.32475138e-01f,-2.12916121e-01f,
-2.36515701e-03f,+1.15186289e-01f,+4.88789529e-02f,+1.58146605e-01f,-8.58632177e-02f,
-2.43376657e-01f,-1.69423670e-01f,-1.49242654e-01f,-9.92204994e-02f,+2.20738098e-01f,
+4.73357737e-03f,-6.37880564e-02f,+1.34015337e-01f,-2.07532257e-01f,-9.90482867e-02f,
+1.06264070e-01f,-1.96607873e-01f,-2.27759182e-02f,+1.75446555e-01f,-1.95152313e-01f,
+2.41017118e-01f,+1.97792009e-01f,+1.62835106e-01f,-1.81074828e-01f,+5.81028312e-02f,
-1.11813262e-01f,+1.61158189e-01f,+1.89218238e-01f,-1.97627082e-01f,-1.21900819e-01f,
+1.91582099e-01f,-2.25214779e-01f,+8.31619054e-02f,+7.66973943e-02f,+8.56377333e-02f,
-1.99443102e-03f,+2.13540599e-01f,-1.93364501e-01f,-1.62451774e-01f,-7.19481111e-02f,
+1.57868281e-01f,-8.62926245e-04f,-1.15023762e-01f,-6.27801269e-02f,+2.16805935e-03f,
+2.52399594e-02f,+1.86307684e-01f,-5.71464747e-02f,-5.98641336e-02f,+1.92143768e-02f,
-9.43839550e-02f,-1.68451816e-02f,-5.68430871e-02f,+2.46040523e-03f,-8.94007087e-02f,
-1.26030475e-01f,-1.72066167e-01f,-1.61774725e-01f,+2.23577768e-02f,+5.32394797e-02f,
-1.03573024e-01f,-2.71829963e-03f,+8.73791724e-02f,+2.13301107e-01f,+8.57500583e-02f,
+2.41117373e-01f,-1.78186059e-01f,+9.70058888e-02f,-3.27594966e-01f,+2.12878481e-01f,
-1.01250619e-01f,-6.60720095e-02f,+1.27862349e-01f,+2.40278646e-01f,+2.32443288e-01f,
-9.80609059e-02f,+3.33752126e-01f,+2.01494649e-01f,-2.99788285e-02f,-8.40009283e-03f,
+6.47608414e-02f,-3.09773803e-01f,-2.35860974e-01f,+2.00641945e-01f,+7.43918866e-02f,
+2.70139705e-02f,-2.90389135e-02f,+5.28350249e-02f,-7.32037425e-03f,-3.27338189e-01f,
+9.55201089e-02f,-1.38400212e-01f,+2.73083717e-01f,-1.75926134e-01f,-2.85896242e-01f,
-7.74319023e-02f,+2.53744841e-01f,-2.37030629e-02f,+1.31784245e-01f,+1.20455928e-01f,
-2.11172462e-01f,+3.46236795e-01f,-1.99175790e-01f,+2.71975826e-02f,+1.30435660e-01f,
+2.28250667e-01f,+2.43153408e-01f,-1.43454075e-02f,+3.22278678e-01f,-4.02628258e-02f,
+1.76240191e-01f,+1.26102373e-01f,+8.16942230e-02f,-1.38784111e-01f,+4.49082479e-02f,
+2.07683593e-02f,-2.25393102e-01f,+1.86203197e-01f,-2.44790703e-01f,-1.18922219e-01f,
-2.77849823e-01f,+3.73914540e-01f,-2.33511820e-01f,+1.54333308e-01f,+1.67079285e-01f,
-1.53971106e-01f,+8.72913864e-04f,+3.35172057e-01f,+8.31711739e-02f,+1.88628748e-01f,
+4.64638025e-02f,+1.72295198e-01f,-3.07883322e-02f,+2.35490218e-01f,-2.22652890e-02f,
+4.09542490e-03f,+1.27215251e-01f,+1.14024438e-01f,-1.21687829e-01f,-1.79316044e-01f,
+7.17542022e-02f,-7.05271140e-02f,-2.99403459e-01f,+3.82317603e-03f,-1.29117910e-02f,
-1.55134454e-01f,+9.25847515e-02f,-1.11405805e-01f,-3.69245112e-02f,-3.55375469e-01f,
+1.64606616e-01f,-2.07657218e-02f,-2.47246936e-01f,+9.63244736e-02f,+1.47917554e-01f,
+2.70739168e-01f,-1.79114014e-01f,-2.87520494e-02f,-4.12167655e-03f,+1.08649835e-01f,
+4.19875011e-02f,-6.88299388e-02f,-3.33739109e-02f,-2.61944402e-02f,+2.03442365e-01f,
-9.87428576e-02f,-7.40790218e-02f,-6.08871877e-02f,-9.83914286e-02f,-1.25838235e-01f,
+7.65814111e-02f,+1.18973389e-01f,-1.59742564e-01f,-1.92523628e-01f,+1.05949946e-01f,
+1.23881236e-01f,+2.91335702e-01f,-1.86265454e-01f,-1.74141511e-01f,+9.99438688e-02f,
+8.07400569e-02f,-1.12643376e-01f,+3.30019481e-02f,-1.08176954e-01f,+1.55476645e-01f,
-4.28616032e-02f,-1.22924238e-01f,-2.38979489e-01f,+6.63935393e-02f,+7.38683045e-02f,
+8.32368284e-02f,+1.47858515e-01f,+2.80386567e-01f,-3.02368224e-01f,-7.02528805e-02f,
-1.18092380e-01f,+7.68331811e-02f,-7.37053230e-02f,-1.27954453e-01f,-2.36211270e-01f,
+1.65644109e-01f,-2.07560807e-01f,+1.04171857e-01f,-9.64161158e-02f,-2.99290512e-02f,
-1.76073685e-02f,-2.01444224e-01f,+2.31861487e-01f,-3.05745564e-02f,-7.98089206e-02f,
+5.75398989e-02f,+1.16429366e-01f,-3.15157324e-01f,-8.00988674e-02f,+2.16150001e-01f,
+8.82772654e-02f,+1.50750563e-01f,+1.43324986e-01f,+8.50899592e-02f,-2.27385107e-02f,
+1.90928712e-01f,+2.11614028e-01f,+7.94187840e-03f,+1.39057338e-02f,-5.39128762e-03f,
+2.46995851e-01f,+1.03660047e-01f,-1.28873467e-01f,-1.34126004e-02f,+3.87959592e-02f,
-4.56323363e-02f,-2.61276960e-02f,+3.20324190e-02f,-1.79318026e-01f,-1.56241477e-01f,
-1.80476218e-01f,+2.73288459e-01f,-2.03015581e-01f,-1.18036225e-01f,-5.28997853e-02f,
+1.65916383e-01f,+6.47296533e-02f,-2.45556861e-01f,+3.28882903e-01f,+1.85016677e-01f,
+1.33159176e-01f,-1.74966574e-01f,-4.04074369e-03f,+2.14892790e-01f,+2.42813811e-01f,
+3.23499858e-01f,-4.80134636e-02f,-8.20265338e-02f,+2.19019994e-01f,+7.00173229e-02f,
+6.36954159e-02f,-9.58815217e-02f,-8.91920701e-02f,+1.57545373e-01f,+2.44547576e-02f,
+2.88013637e-01f,+3.62263471e-02f,+3.84738669e-02f,+1.84142992e-01f,-3.06547116e-02f,
+1.83710366e-01f,-1.37445897e-01f,+5.04016019e-02f,+1.20146587e-01f,+1.93410814e-01f,
-1.21688664e-01f,-5.77816591e-02f,-1.60037261e-02f,+2.17683494e-01f,+2.30847066e-03f,
+5.00225574e-02f,+7.55562261e-02f,-9.08739045e-02f,+2.33680923e-02f,-1.11777820e-01f,
+2.16789529e-01f,+1.70762300e-01f,+1.90445945e-01f,+1.19537517e-01f,-1.88291132e-01f,
+2.13007405e-01f,-1.62801653e-01f,+3.32722217e-02f,+1.66051447e-01f,-2.11899742e-01f,
-2.04519480e-02f,-1.56667918e-01f,+1.66470818e-02f,-2.38369524e-01f,-1.15894720e-01f,
-1.40879512e-01f,-1.50106266e-01f,+1.96043149e-01f,-2.04101205e-02f,+2.21131071e-01f,
-1.77984402e-01f,-9.07546729e-02f,+2.30203420e-01f,+1.12185217e-01f,+4.72716093e-02f,
+1.11188609e-02f,+1.93857446e-01f,+3.21848392e-01f,-8.29186961e-02f,-1.42936036e-01f,
+5.43125756e-02f,-2.11893126e-01f,+2.64539391e-01f,+1.69653371e-01f,+8.81414264e-02f,
-2.73109972e-02f,-9.63566974e-02f,+1.88160881e-01f,+1.58065841e-01f,+5.61668724e-02f,
+7.44781867e-02f,-3.13409299e-01f,+9.92306024e-02f,-1.68942884e-01f,-1.40605243e-02f,
-1.60218418e-01f,+3.13364039e-03f,-8.23444948e-02f,+1.50692984e-01f,+3.27933878e-02f,
+1.67695284e-01f,+1.41428232e-01f,-1.17401801e-01f,-5.38564734e-02f,+1.13935657e-01f,
+3.17452699e-02f,+3.21967900e-01f,+3.19755614e-01f,+1.55678108e-01f,+1.88787058e-01f,
-1.03138179e-01f,+1.31688684e-01f,+4.21154909e-02f,+8.67635757e-02f,-3.65672447e-02f,
+1.93467155e-01f,-9.86504480e-02f,+7.27920681e-02f,-2.25715756e-01f,+1.95777386e-01f,
-1.49981037e-01f,+5.57424054e-02f,-1.05402827e-01f,-3.37491520e-02f,+2.61636883e-01f,
+3.51236075e-01f,-9.15964022e-02f,+1.06752448e-01f,-2.23929778e-01f,-1.77171230e-01f,
-3.38038653e-01f,+1.42821640e-01f,+3.18533301e-01f,+1.53830618e-01f,-1.74584597e-01f,
+5.03195524e-02f,-1.32438526e-01f,+2.23569229e-01f,+1.65683508e-01f,-4.18112874e-02f,
+1.51367262e-01f,+1.44545928e-01f,+9.84294415e-02f,-2.13795938e-02f,+1.72968879e-01f,
-4.99507785e-02f,+3.47547047e-02f,-1.82671145e-01f,+3.24222982e-01f,+1.36740610e-01f,
+2.95491695e-01f,-1.14147753e-01f,-5.86916208e-02f,-1.19137958e-01f,+1.17569312e-01f,
-1.76553652e-01f,+1.82902403e-02f,+1.28995135e-01f,+1.76661788e-03f,-2.86540210e-01f,
-5.39323203e-02f,+1.21916123e-01f,+2.24915341e-01f,-2.39766374e-01f,+2.29769230e-01f,
-1.16930529e-01f,-1.29143581e-01f,+2.52590068e-02f,-2.26078629e-01f,-1.64519697e-02f,
-1.54328495e-01f,+1.27342775e-01f,-1.60580501e-01f,+1.53385326e-01f,}; 
//k2c_tensor dense_29_kernel = {&dense_29_kernel_array[0],2,1344,{16,84, 1, 1, 1}};
// Copy the contents of dense_29_kernel_array into dense_29_kernel.array
for (size_t i106 = 0; i106 < 1344; ++i106) {  // Use the actual number of iterations required
    dense_29_kernel.array[i106] = dense_29_kernel_array[i106];
}

// Set the other members of dense_29_kernel
dense_29_kernel.ndim = 2;
dense_29_kernel.numel = 1344;
dense_29_kernel.shape[0] = 16;
dense_29_kernel.shape[1] = 84;
dense_29_kernel.shape[2] = 1;
dense_29_kernel.shape[3] = 1;
dense_29_kernel.shape[4] = 1;

float dense_29_bias_array[84] = {
+7.23362789e-02f,-1.51304621e-02f,+0.00000000e+00f,-4.33967970e-02f,-2.09296104e-02f,
+4.14130744e-04f,-5.14096441e-03f,+6.54436052e-02f,-3.73467803e-02f,+0.00000000e+00f,
-2.51079462e-02f,-1.47894491e-02f,+5.43449186e-02f,+3.22723091e-02f,+6.76576048e-02f,
+0.00000000e+00f,+7.04570040e-02f,+7.37796053e-02f,+1.88138112e-02f,+1.60878559e-03f,
+0.00000000e+00f,+7.88340047e-02f,+1.98482238e-02f,+0.00000000e+00f,+3.27368490e-02f,
+0.00000000e+00f,+7.34687001e-02f,+3.85765620e-02f,-6.00136593e-02f,+2.80915219e-02f,
-1.76996570e-02f,-1.22507457e-02f,+0.00000000e+00f,-4.86962050e-02f,+8.23981017e-02f,
+4.06646393e-02f,-5.43849953e-02f,-2.99617499e-02f,-7.37016974e-03f,+0.00000000e+00f,
-5.94299510e-02f,-4.63197678e-02f,+8.49095285e-02f,-3.91129106e-02f,+4.55734506e-02f,
-1.49870506e-02f,-6.73630135e-03f,+0.00000000e+00f,-1.72655880e-02f,+0.00000000e+00f,
+0.00000000e+00f,+0.00000000e+00f,+9.12586153e-02f,+4.41597635e-03f,+4.31156121e-02f,
+0.00000000e+00f,-5.42388204e-03f,+0.00000000e+00f,+6.44350499e-02f,+5.50622419e-02f,
+6.31967932e-02f,+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,
+0.00000000e+00f,+4.13889587e-02f,+4.96529974e-02f,-2.34188605e-02f,-5.20317145e-02f,
+5.64589947e-02f,+3.03188842e-02f,+5.49136735e-02f,-7.64700919e-02f,+6.01231046e-02f,
+0.00000000e+00f,+3.27748396e-02f,+6.23458587e-02f,-1.08757047e-02f,+0.00000000e+00f,
+1.56841446e-02f,+6.11811318e-02f,+0.00000000e+00f,+0.00000000e+00f,}; 
//k2c_tensor dense_29_bias = {&dense_29_bias_array[0],1,84,{84, 1, 1, 1, 1}};

// Copy the contents of dense_29_bias_array into dense_29_bias.array
for (size_t i105 = 0; i105 < 84; ++i105) {  // Use the actual number of iterations required
    dense_29_bias.array[i105] = dense_29_bias_array[i105];
}

// Set the other members of dense_29_bias
dense_29_bias.ndim = 1;
dense_29_bias.numel = 84;
dense_29_bias.shape[0] = 84;
dense_29_bias.shape[1] = 1;
dense_29_bias.shape[2] = 1;
dense_29_bias.shape[3] = 1;
dense_29_bias.shape[4] = 1; 
float dense_29_fwork[1360] = {0}; 

 
float dense_30_output_array[10] = {0}; 
//k2c_tensor dense_30_output = {&dense_30_output_array[0],1,10,{10, 1, 1, 1, 1}};
// Copy the contents of dense_30_output_array into dense_30_output.array
for (size_t i104 = 0; i104 < 10; ++i104) {  // Use the actual number of iterations required
    dense_30_output.array[i104] = dense_30_output_array[i104];
}

// Set the other members of dense_30_output
dense_30_output.ndim = 1;
dense_30_output.numel = 10;
dense_30_output.shape[0] = 10;
dense_30_output.shape[1] = 1;
dense_30_output.shape[2] = 1;
dense_30_output.shape[3] = 1;
dense_30_output.shape[4] = 1;

float dense_30_kernel_array[840] = {
+5.17356917e-02f,+3.52095455e-01f,-2.28201151e-01f,+7.59738833e-02f,+1.38261199e-01f,
+2.39035964e-01f,-2.38269374e-01f,-1.69111118e-01f,+2.23405659e-01f,-1.96847972e-03f,
-2.33973473e-01f,-2.82057166e-01f,-1.37375325e-01f,+3.74729857e-02f,-2.39921004e-01f,
-1.08378027e-02f,+3.88200283e-02f,+3.65265906e-01f,-8.79596770e-02f,+2.03017354e-01f,
+2.20917195e-01f,+4.88815606e-02f,+2.94183791e-02f,+1.68357104e-01f,-2.60320306e-03f,
-3.34111303e-02f,+5.20533919e-02f,-5.75745553e-02f,+9.43527222e-02f,+4.13769484e-03f,
+1.89622462e-01f,-3.05331141e-01f,+2.69473195e-02f,-3.01202107e-02f,-2.30267942e-01f,
-5.51573038e-02f,-7.46291876e-03f,+1.70074161e-02f,+2.33210605e-02f,+1.52172580e-01f,
+2.67625898e-01f,+9.63220298e-02f,+6.05089962e-02f,-1.15251206e-01f,+1.20631114e-01f,
+2.28307489e-02f,-1.01595864e-01f,-1.20206026e-03f,-3.36356342e-01f,-1.26733899e-01f,
-2.04508841e-01f,-1.89411521e-01f,-1.10460341e-01f,-2.19015330e-02f,-1.25838563e-01f,
-2.23418951e-01f,+2.05333620e-01f,-6.23895526e-02f,-5.60046434e-02f,+2.20091969e-01f,
-3.66567113e-02f,-2.69957297e-02f,+9.76108015e-02f,-2.16085702e-01f,-5.44730425e-02f,
-2.30536476e-01f,+1.98751450e-01f,-4.87215631e-02f,-8.67113322e-02f,+1.95328653e-01f,
+3.02036166e-01f,+1.23166643e-01f,+5.88392615e-02f,+7.70943537e-02f,+2.01631516e-01f,
+1.90569118e-01f,+2.35051811e-02f,+4.02551033e-02f,+1.23073541e-01f,+1.07555233e-01f,
-1.38683751e-01f,-2.54841775e-01f,+1.21419936e-01f,-2.19007641e-01f,-4.08440530e-02f,
-1.77689999e-01f,+6.43373728e-02f,+2.93863207e-01f,-2.24652351e-03f,-1.12547792e-01f,
-1.36427045e-01f,+7.68724084e-02f,-2.13963971e-01f,+4.52862978e-02f,-2.45994806e-01f,
-2.23141626e-01f,-3.93958241e-02f,+2.47455984e-01f,-2.34792829e-01f,+7.05389678e-02f,
+1.18120216e-01f,-1.18772075e-01f,+1.26459479e-01f,+1.35256857e-01f,-2.27242753e-01f,
+2.53289729e-01f,-1.94395900e-01f,-1.72148705e-01f,-1.71283670e-02f,-1.87673748e-01f,
-1.95727110e-01f,-1.49857566e-01f,+2.06229091e-01f,-1.23493239e-01f,+2.51244992e-01f,
-8.61080084e-03f,+7.59983063e-03f,+1.66082978e-01f,-1.24033183e-01f,-4.23540473e-02f,
+3.09744805e-01f,+1.41358241e-01f,-1.37014389e-02f,+1.09811999e-01f,+2.34501600e-01f,
-7.45095313e-02f,+9.18062329e-02f,+2.81893224e-01f,+1.29425600e-01f,+2.07600519e-01f,
+1.61493018e-01f,+3.00356179e-01f,+4.19046283e-02f,-2.08130822e-01f,+2.19102025e-01f,
-1.18420750e-01f,-1.87402457e-01f,-1.16975635e-01f,-8.98800120e-02f,+1.39959127e-01f,
+1.73245538e-02f,+1.69491097e-01f,+2.02270448e-02f,+1.21918738e-01f,-1.05100140e-01f,
+1.21135563e-01f,-2.36375153e-01f,+3.05426478e-01f,-9.62228030e-02f,-2.37487867e-01f,
-1.83656111e-01f,-8.89431685e-02f,-1.72722727e-01f,+1.55550569e-01f,-4.19579297e-02f,
+4.36885655e-02f,+6.29065335e-02f,+2.52541333e-01f,+1.53210849e-01f,-4.18380648e-02f,
-1.27464682e-01f,+1.45485193e-01f,+3.72762680e-02f,+2.14991122e-01f,-1.92768335e-01f,
+3.19998562e-01f,-1.86431557e-02f,-2.84060597e-01f,+1.39357690e-02f,+1.77889735e-01f,
-1.86008006e-01f,+1.39789596e-01f,+2.45095015e-01f,-1.06150664e-01f,+1.08098872e-01f,
+3.44306350e-01f,+4.05266881e-02f,-2.67349899e-01f,+3.56700063e-01f,+8.04589316e-02f,
+3.04204747e-02f,+2.37602592e-01f,+2.49865949e-02f,+3.82388569e-02f,-2.09825918e-01f,
-4.91303131e-02f,+2.34195054e-01f,-1.57471612e-01f,-1.35400500e-02f,+1.24322586e-01f,
-6.60727546e-03f,+1.29393980e-01f,+1.28778398e-01f,-2.41737306e-01f,-1.31018683e-01f,
+2.63451904e-01f,-1.29784346e-01f,-1.86962023e-01f,-2.16704071e-01f,-6.98386133e-02f,
-1.56107217e-01f,-2.28150129e-01f,-8.06582123e-02f,-1.29595980e-01f,-1.31404966e-01f,
+8.15566182e-02f,+1.21707976e-01f,-1.53265059e-01f,-4.71875072e-02f,+2.96082795e-02f,
+1.47610098e-01f,+2.36420974e-01f,-2.45657191e-01f,+8.54923427e-02f,+1.29121691e-01f,
+2.34250546e-01f,-2.15537146e-01f,-1.39031172e-01f,+2.16871545e-01f,-6.14791028e-02f,
-1.57991394e-01f,+3.37227024e-02f,+1.88361824e-01f,+1.29369318e-01f,+1.34060442e-01f,
+3.03160191e-01f,+1.44513637e-01f,-1.25699088e-01f,+3.06340754e-01f,-9.56557542e-02f,
+2.36677915e-01f,+1.62280202e-02f,-2.25586087e-01f,-5.89303970e-02f,-4.28117663e-02f,
+8.89223814e-02f,-1.44934148e-01f,-1.14716813e-01f,-2.27749631e-01f,-2.44168416e-01f,
-1.79313906e-02f,+5.25440052e-02f,-8.72995257e-02f,+1.15254313e-01f,-1.53133154e-01f,
-2.08283424e-01f,-5.20235747e-02f,+3.06338549e-01f,-1.54665127e-01f,+1.69682711e-01f,
-2.26581722e-01f,-1.84328884e-01f,-2.20102519e-02f,-1.04228929e-01f,+1.18343592e-01f,
-2.35657036e-01f,+1.90116167e-02f,-6.60836697e-04f,+2.49886245e-01f,+4.87726927e-03f,
+2.13991404e-01f,+1.90478399e-01f,-2.31452838e-01f,+1.03929974e-02f,-2.09107354e-01f,
+5.76701798e-02f,+1.30515069e-01f,+1.06216781e-01f,+1.24381006e-01f,-1.78206086e-01f,
+1.58213675e-01f,+3.41387928e-01f,-4.45549190e-03f,-1.74584836e-01f,-7.76924640e-02f,
+5.84148765e-02f,-6.05691671e-02f,+3.16478424e-02f,-1.74600258e-01f,+1.43676624e-01f,
+1.97787821e-01f,-3.57350469e-01f,-1.08584672e-01f,-1.71845276e-02f,+5.83453402e-02f,
-1.82001874e-01f,-1.09129861e-01f,+1.22124031e-01f,-2.10797578e-01f,-6.17818609e-02f,
+2.06131950e-01f,+7.26937577e-02f,-1.38432398e-01f,+4.12009023e-02f,-1.52353451e-01f,
+3.42737108e-01f,-2.23152339e-02f,-1.22311279e-01f,-3.07480842e-02f,-9.13294330e-02f,
-1.86000898e-01f,-1.93215385e-01f,-1.34275496e-01f,+6.09980151e-02f,+1.32788390e-01f,
-3.36777568e-01f,-1.97473273e-01f,-1.71852130e-02f,+3.19339871e-01f,+1.64586425e-01f,
+2.26838097e-01f,+1.06337361e-01f,+6.78145885e-02f,-1.61360353e-02f,+2.14823604e-01f,
-1.93877310e-01f,-1.58527493e-03f,+3.22983339e-02f,-5.28028943e-02f,+2.08613694e-01f,
+1.29623502e-01f,-1.06137007e-01f,-4.76036221e-02f,-2.20937073e-01f,+1.20646030e-01f,
+2.50756115e-01f,+6.02362454e-02f,+2.47304380e-01f,-1.17277429e-01f,+1.94797188e-01f,
+7.89736733e-02f,-2.86670119e-01f,-2.24345431e-01f,+1.73403174e-01f,-5.16692922e-02f,
-6.90104514e-02f,-1.48842990e-01f,+2.18911558e-01f,-7.33947530e-02f,+2.49221874e-03f,
-7.14853033e-02f,+3.47221464e-01f,-6.30075485e-02f,-1.11665115e-01f,+1.17499880e-01f,
-1.28689602e-01f,-3.98699939e-02f,-6.73030838e-02f,+3.04143041e-01f,+7.05829561e-02f,
-3.32286477e-01f,-1.01823127e-02f,+1.50925398e-01f,-1.06957413e-01f,-2.33099982e-01f,
+3.54675740e-01f,-2.23335519e-01f,-3.02608967e-01f,+7.36673325e-02f,-1.12485982e-01f,
-2.63299406e-01f,-2.54606694e-01f,+5.44128120e-02f,-2.60954320e-01f,+1.14558823e-01f,
-1.86987445e-01f,+1.67297006e-01f,+1.01085268e-01f,-2.09132358e-02f,-1.12929136e-01f,
+2.20622018e-01f,-2.19716266e-01f,+1.31214023e-01f,-1.80203736e-01f,-4.75895256e-02f,
+1.06326886e-01f,+3.83035839e-02f,+4.46416140e-02f,-4.06989716e-02f,-1.20186806e-01f,
-2.29977846e-01f,+4.40631174e-02f,-1.10754162e-01f,-1.76638469e-01f,+5.08411080e-02f,
-2.77037680e-01f,-1.92187250e-01f,+2.03780100e-01f,-5.95107637e-02f,-2.00825006e-01f,
-2.22715735e-03f,-1.90048590e-01f,+4.33574319e-02f,+8.51638317e-02f,+5.94514310e-02f,
+2.46356755e-01f,-9.29446667e-02f,+2.16166049e-01f,-1.65671557e-01f,-1.28452063e-01f,
+1.76062018e-01f,-3.25820804e-01f,-1.62851408e-01f,+1.48948476e-01f,+1.18674420e-01f,
-8.41354579e-02f,-1.54435918e-01f,-4.56533171e-02f,-4.03614603e-02f,+5.78697436e-02f,
-2.64874045e-02f,-4.24775146e-02f,+2.18167126e-01f,+9.43835676e-02f,-5.15572876e-02f,
-1.59735247e-01f,-1.40498161e-01f,+1.68570146e-01f,+1.21216781e-01f,+3.95646691e-02f,
-9.73815098e-02f,+3.89111321e-03f,+1.54140055e-01f,+1.43626750e-01f,+1.49110079e-01f,
+1.72038108e-01f,-2.75880545e-02f,+2.45920047e-01f,+2.98904598e-01f,-1.51957527e-01f,
+1.07138976e-01f,-9.10675824e-02f,-9.85630006e-02f,-1.93310395e-01f,-1.86403632e-01f,
+2.28563458e-01f,-2.40376890e-01f,+5.68216806e-03f,-1.66003153e-01f,-2.48717636e-01f,
+2.62884200e-01f,+2.53484637e-01f,-7.86830485e-03f,+9.16138366e-02f,+1.57777965e-01f,
+3.12993601e-02f,-2.89231092e-02f,+1.39288098e-01f,-3.00130725e-01f,-1.93919897e-01f,
+1.69519842e-01f,-5.01237214e-02f,-2.52404213e-01f,-1.56586081e-01f,-1.55993715e-01f,
+1.84038326e-01f,+2.35658407e-01f,+1.23685047e-01f,+2.29244754e-01f,-3.63069922e-02f,
+5.41132577e-02f,-2.07751185e-01f,+1.50640488e-01f,-9.00375247e-02f,+1.48767471e-01f,
-1.62362859e-01f,+1.05952621e-01f,+2.74582058e-02f,-2.39395052e-01f,+1.16936890e-02f,
+2.31948048e-01f,+7.31206536e-02f,+3.63569558e-02f,-3.64785641e-02f,+7.77990222e-02f,
-1.79884642e-01f,+3.28626931e-02f,+1.43851399e-01f,+1.06280237e-01f,+1.97579890e-01f,
+1.03194125e-01f,+4.22917269e-02f,+2.01158643e-01f,-2.35840812e-01f,+8.25517178e-02f,
+5.04126474e-02f,-2.06234008e-02f,+1.90306157e-01f,+2.50046641e-01f,+1.79418981e-01f,
+7.04187155e-03f,+2.59239972e-02f,-1.00904301e-01f,-4.85453308e-02f,+2.45697200e-01f,
-1.51257113e-01f,+1.90748721e-01f,+1.35719866e-01f,-1.62451148e-01f,+9.64462757e-02f,
-7.19278753e-02f,-1.38122246e-01f,+2.21841186e-01f,+2.48842806e-01f,+2.22565174e-01f,
+2.26071715e-01f,-4.06039059e-02f,-1.85042486e-01f,+1.01956546e-01f,+2.10631460e-01f,
-2.42697105e-01f,+2.51802891e-01f,-3.64575982e-02f,-2.93068588e-03f,+7.04309344e-03f,
+3.20997536e-02f,+2.25895047e-01f,+1.66637868e-01f,-1.03121385e-01f,+1.19829595e-01f,
+8.93857181e-02f,+2.01691017e-01f,-1.36330068e-01f,-2.04810157e-01f,-9.91484523e-02f,
+1.23851322e-01f,-1.18166313e-01f,+1.63527697e-01f,+2.28481382e-01f,+1.01690777e-01f,
-2.28836238e-01f,-3.64658162e-02f,+1.74577057e-01f,+2.50434764e-02f,-2.11900920e-01f,
-1.13502163e-02f,-1.71059325e-01f,+2.90279150e-01f,+4.81435098e-02f,+8.98365527e-02f,
+1.37872294e-01f,+2.73599237e-01f,+3.27458978e-03f,-2.34517083e-01f,+8.62413198e-02f,
-3.05910576e-02f,-1.33882150e-01f,-2.70625293e-01f,+1.90626726e-01f,+2.39283349e-02f,
+2.18603432e-01f,-1.45473972e-01f,+2.24734366e-01f,-8.00539851e-02f,+3.51986289e-03f,
-1.28123283e-01f,+3.30823660e-03f,+4.94346023e-03f,-1.68252051e-01f,+4.39503491e-02f,
-1.90000281e-01f,+2.13592350e-01f,+1.89051330e-02f,-1.86863884e-01f,-2.40940094e-01f,
+8.32597632e-03f,+1.99515313e-01f,+9.78319049e-02f,-2.00035751e-01f,-2.35267326e-01f,
-1.14817053e-01f,+1.95881426e-01f,+7.63646364e-02f,-1.97563976e-01f,+1.29991680e-01f,
+1.78608149e-01f,+1.45902932e-02f,+1.84852332e-01f,-2.51355827e-01f,+8.79448950e-02f,
-1.41053483e-01f,+2.97112882e-01f,-2.12843180e-01f,-3.04497872e-02f,+1.14270575e-01f,
+3.40512335e-01f,+8.51253867e-02f,+1.62288472e-01f,-4.38166708e-02f,+9.02628005e-02f,
+1.22151867e-01f,-2.12191239e-01f,-1.38974339e-01f,+2.05062926e-02f,+5.26770949e-03f,
-6.18491173e-02f,+1.46737903e-01f,+4.80283126e-02f,+2.33977139e-01f,+3.49384849e-03f,
+1.27216876e-01f,+3.24630439e-01f,+4.42481935e-02f,-1.88385531e-01f,-3.38910744e-02f,
+1.87466487e-01f,-1.35327250e-01f,-2.25473031e-01f,+3.51180822e-01f,+1.23722926e-01f,
+6.67947829e-02f,-7.67093003e-02f,+4.99685109e-02f,-4.10997570e-02f,+1.66585743e-02f,
-1.79423183e-01f,+6.32225871e-02f,-6.71769232e-02f,+1.71394467e-01f,+2.15547442e-01f,
+2.40873009e-01f,-2.01710910e-02f,+1.98672324e-01f,+9.34579372e-02f,+2.47184098e-01f,
-1.63757294e-01f,-2.16537237e-01f,-3.91066074e-03f,+2.18217909e-01f,-2.18956888e-01f,
+1.19992048e-01f,-2.15575993e-01f,+1.24413759e-01f,+5.21138608e-02f,+2.27787584e-01f,
-1.86426461e-01f,-1.80298090e-03f,-1.58019990e-01f,+3.30650210e-02f,-2.33534098e-01f,
+1.53393149e-02f,-8.53014588e-02f,+1.58126414e-01f,+1.82053655e-01f,+7.40829706e-02f,
-3.92445624e-02f,+9.63990390e-02f,-8.39570016e-02f,-2.27681398e-02f,+1.49323553e-01f,
+1.33387148e-02f,+1.23223245e-01f,+4.55113351e-02f,-1.53269276e-01f,-1.85554847e-01f,
-2.49665380e-01f,-2.16257676e-01f,+1.70956552e-02f,+7.08559155e-02f,+2.08401173e-01f,
+1.23594590e-01f,+3.36685658e-01f,-3.28664780e-02f,+1.74912307e-02f,+3.12248301e-02f,
-1.24037996e-01f,+1.68882668e-01f,+5.33732437e-02f,-9.50606391e-02f,+1.58907935e-01f,
+1.50692627e-01f,+3.53006989e-01f,+2.51851350e-01f,+3.94408554e-02f,-1.36071280e-01f,
-1.63306564e-03f,+2.33051181e-03f,+7.07779378e-02f,-1.65726826e-01f,-7.49825686e-02f,
-1.80712089e-01f,-1.71105444e-01f,-2.25855887e-01f,+1.35408610e-01f,-9.00060833e-02f,
-3.71640995e-02f,+2.24400789e-01f,+3.04620564e-01f,-7.87933171e-02f,-3.20638232e-02f,
+2.32345968e-01f,-1.61878616e-01f,-3.08856368e-03f,-7.46935531e-02f,+6.79116175e-02f,
-3.10273260e-01f,+1.08503163e-01f,+1.33101776e-01f,-1.69651404e-01f,-1.86372951e-01f,
-7.33921826e-02f,+3.10862482e-01f,+1.99215710e-01f,-2.37635285e-01f,+3.69118527e-02f,
-1.39782578e-01f,-2.34693557e-01f,+1.51745826e-01f,+3.17144454e-01f,+2.25811377e-01f,
+1.75586827e-02f,+2.23050147e-01f,-1.57393172e-01f,+6.34579137e-02f,-1.77182645e-01f,
+1.13392614e-01f,+1.56887680e-01f,+4.12938930e-02f,-1.81949794e-01f,+1.39095515e-01f,
-8.39802921e-02f,+4.94375117e-02f,-1.01706237e-02f,+8.25968385e-02f,-1.02362111e-01f,
+2.88829714e-01f,+5.76236844e-03f,-3.40657741e-01f,-1.81803536e-02f,-2.44140759e-01f,
+3.38870175e-02f,-2.87341267e-01f,+8.48412514e-04f,-2.24786341e-01f,+1.34876430e-01f,
-2.27757215e-01f,-1.90460786e-01f,-3.11728846e-02f,-1.41457126e-01f,+6.78620785e-02f,
+8.86202033e-04f,+2.25226972e-02f,-1.70569181e-01f,-2.33689994e-01f,+2.65127867e-02f,
+2.51420289e-01f,-2.30857417e-01f,-5.74572831e-02f,+2.60685951e-01f,-2.61599594e-03f,
+4.63619828e-03f,+1.11491382e-02f,-1.44150600e-01f,+1.56376928e-01f,-3.00952345e-02f,
+2.36813426e-01f,+6.50590062e-02f,-1.23943612e-01f,-4.28348333e-02f,+3.08488309e-02f,
+2.87096351e-01f,+1.93605900e-01f,-1.27463654e-01f,-1.60931461e-02f,-2.23380402e-01f,
-6.21083528e-02f,+1.16906315e-01f,-3.41656879e-02f,-6.82026818e-02f,-1.32077172e-01f,
+1.70793518e-01f,+3.06280404e-01f,-1.73375547e-01f,+4.39686105e-02f,-2.50646800e-01f,
+1.30528718e-01f,+6.73673451e-02f,-2.35728938e-02f,+2.08779573e-01f,-5.78470975e-02f,
-7.51720145e-02f,+1.74745172e-02f,+1.43795609e-02f,+2.15100735e-01f,-7.48780817e-02f,
-1.56008929e-01f,-6.76462054e-03f,+1.14357555e-02f,+2.33078435e-01f,+2.42627844e-01f,
-2.24161521e-01f,-6.67703450e-02f,-8.98880213e-02f,-2.09108338e-01f,+3.99776399e-02f,
-2.27156550e-01f,-6.63391203e-02f,+5.32222390e-02f,+1.67550296e-01f,+1.74369872e-01f,
-1.93290412e-01f,-2.05716491e-02f,-3.93849760e-02f,-1.07455142e-01f,+1.06614701e-01f,
-2.55057096e-01f,+1.68781459e-01f,+1.67231366e-01f,+2.51212925e-01f,+2.97602359e-02f,
+3.62955391e-01f,+2.78574109e-01f,+1.41211748e-02f,+1.59142882e-01f,-1.86805755e-01f,
+7.51323700e-02f,+4.44886684e-02f,-2.80297369e-01f,+1.28667830e-02f,-2.22644553e-01f,
-3.97013873e-02f,-1.12395287e-01f,-1.07148230e-01f,-2.49085665e-01f,-2.91297138e-02f,
+8.58702064e-02f,-4.24554199e-02f,+4.10425365e-02f,+2.11740196e-01f,+4.81992364e-02f,
+3.55349183e-02f,+1.65454060e-01f,-5.67207187e-02f,+2.28196561e-01f,-8.54113251e-02f,
-6.38689250e-02f,+1.35588676e-01f,+4.66169119e-02f,+8.11051726e-02f,+1.00088298e-01f,
}; 
//k2c_tensor dense_30_kernel = {&dense_30_kernel_array[0],2,840,{84,10, 1, 1, 1}};
// Copy the contents of dense_30_kernel_array into dense_30_kernel.array
for (size_t i103 = 0; i103 < 840; ++i103) {  // Use the actual number of iterations required
    dense_30_kernel.array[i103] = dense_30_kernel_array[i103];
}

// Set the other members of dense_30_kernel
dense_30_kernel.ndim = 2;
dense_30_kernel.numel = 840;
dense_30_kernel.shape[0] = 84;
dense_30_kernel.shape[1] = 10;
dense_30_kernel.shape[2] = 1;
dense_30_kernel.shape[3] = 1;
dense_30_kernel.shape[4] = 1; 
float dense_30_bias_array[10] = {
+2.17461735e-02f,+8.71752724e-02f,+0.00000000e+00f,-1.17719881e-02f,-9.02325555e-05f,
+6.87118247e-02f,+0.00000000e+00f,+8.06451589e-03f,+6.04715087e-02f,-1.76114049e-02f,
}; 
//k2c_tensor dense_30_bias = {&dense_30_bias_array[0],1,10,{10, 1, 1, 1, 1}};
// Copy the contents of dense_30_bias_array into dense_30_bias.array
for (size_t i102 = 0; i102 < 10; ++i102) {  // Use the actual number of iterations required
    dense_30_bias.array[i102] = dense_30_bias_array[i102];
}

// Set the other members of dense_30_bias
dense_30_bias.ndim = 1;
dense_30_bias.numel = 10;
dense_30_bias.shape[0] = 10;
dense_30_bias.shape[1] = 1;
dense_30_bias.shape[2] = 1;
dense_30_bias.shape[3] = 1;
dense_30_bias.shape[4] = 1; 
float dense_30_fwork[924] = {0}; 

 
float dense_31_kernel_array[30] = {
-1.41397342e-01f,-7.28822589e-01f,+2.28530336e-02f,+6.35224104e-01f,-5.81701875e-01f,
-2.77927130e-01f,-2.32074261e-02f,-5.27413011e-01f,-4.95534062e-01f,-2.86368191e-01f,
-3.45055317e-03f,-4.36109900e-01f,+5.17399669e-01f,-1.76445767e-01f,-6.58225954e-01f,
-2.33326048e-01f,-6.53282821e-01f,-6.03079855e-01f,-3.37403506e-01f,-3.83547008e-01f,
+3.34733725e-01f,-3.66716199e-02f,+4.99288112e-01f,+6.10676229e-01f,+2.45202422e-01f,
+4.67688054e-01f,-2.15757966e-01f,+2.48080760e-01f,-5.32527387e-01f,-4.38909739e-01f,
}; 
//k2c_tensor dense_31_kernel = {&dense_31_kernel_array[0],2,30,{10, 3, 1, 1, 1}};
// Copy the contents of dense_31_kernel_array into dense_31_kernel.array
for (size_t i101 = 0; i101 < 30; ++i101) {  // Use the actual number of iterations required
    dense_31_kernel.array[i101] = dense_31_kernel_array[i101];
}

// Set the other members of dense_31_kernel
dense_31_kernel.ndim = 2;
dense_31_kernel.numel = 30;
dense_31_kernel.shape[0] = 10;
dense_31_kernel.shape[1] = 3;
dense_31_kernel.shape[2] = 1;
dense_31_kernel.shape[3] = 1;
dense_31_kernel.shape[4] = 1;

float dense_31_bias_array[3] = {
+2.88988948e-02f,+2.98769604e-02f,-8.76061693e-02f,}; 
//k2c_tensor dense_31_bias = {&dense_31_bias_array[0],1,3,{3,1,1,1,1}};

// Copy the contents of dense_31_bias_array into dense_31_bias.array
for (size_t i100 = 0; i100 < 3; ++i100) {  // Use the actual number of iterations required
    dense_31_bias.array[i100] = dense_31_bias_array[i100];
}

// Set the other members of dense_31_bias
dense_31_bias.ndim = 1;
dense_31_bias.numel = 3;
dense_31_bias.shape[0] = 3;
dense_31_bias.shape[1] = 1;
dense_31_bias.shape[2] = 1;
dense_31_bias.shape[3] = 1;
dense_31_bias.shape[4] = 1; 
float dense_31_fwork[40] = {0}; 

 
k2c_pad2d(&conv2d_41_padded_input,input_11_input,conv2d_41_fill, 
	conv2d_41_pad); 
k2c_conv2d(&conv2d_41_output,&conv2d_41_padded_input,&conv2d_41_kernel, 
	&conv2d_41_bias,conv2d_41_stride,conv2d_41_dilation,RELU);
k2c_maxpool2d(&max_pooling2d_33_output,&conv2d_41_output,max_pooling2d_33_pool_size, 
	max_pooling2d_33_stride); 
k2c_pad2d(&conv2d_42_padded_input,&max_pooling2d_33_output,conv2d_42_fill, 
	conv2d_42_pad); 
k2c_conv2d(&conv2d_42_output,&conv2d_42_padded_input,&conv2d_42_kernel, 
	&conv2d_42_bias,conv2d_42_stride,conv2d_42_dilation,RELU);
k2c_maxpool2d(&max_pooling2d_34_output,&conv2d_42_output,max_pooling2d_34_pool_size, 
	max_pooling2d_34_stride); 
k2c_pad2d(&conv2d_43_padded_input,&max_pooling2d_34_output,conv2d_43_fill, 
	conv2d_43_pad); 
k2c_conv2d(&conv2d_43_output,&conv2d_43_padded_input,&conv2d_43_kernel, 
	&conv2d_43_bias,conv2d_43_stride,conv2d_43_dilation,RELU);
k2c_maxpool2d(&max_pooling2d_35_output,&conv2d_43_output,max_pooling2d_35_pool_size, 
	max_pooling2d_35_stride); 
k2c_flatten(&flatten_10_output,&max_pooling2d_35_output); 
k2c_dense(&dense_29_output,&flatten_10_output,&dense_29_kernel, 
	&dense_29_bias,RELU,dense_29_fwork);
k2c_dense(&dense_30_output,&dense_29_output,&dense_30_kernel, 
	&dense_30_bias,RELU,dense_30_fwork);
k2c_dense(dense_31_output,&dense_30_output,&dense_31_kernel, 
	&dense_31_bias,SOFTMAX,dense_31_fwork);

 } 

void finalcovid_initialize() { 

} 

void finalcovid_terminate() { 

} 

