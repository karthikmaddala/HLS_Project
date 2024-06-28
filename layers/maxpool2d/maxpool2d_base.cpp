#include <stddef.h>

void k2c_maxpool2d(float output_array[2704],
                size_t output_ndim,
                size_t output_numel,
                size_t output_shape[5],

                const float input_array[10816],
                const size_t input_ndim,
                const size_t input_numel,
                const size_t input_shape[5],

                const size_t pool_size[2],
                const size_t stride[2]) {


    const size_t channels = input_shape[2];
    // i,j,l output indices
    /// i, k, m input indices
    for (size_t i=0; i< channels; ++i) {
        for (size_t j=0, k=0; j<output_shape[1]*channels;
                j+=channels, k+=channels*stride[1]) {
            for (size_t l=0, m=0; l<output_numel; l+=channels*output_shape[1],
                    m+=channels*input_shape[1]*stride[0]) {
                output_array[l+j+i] = input_array[m+k+i];
                for (size_t n=0; n<pool_size[1]*channels; n+=channels) {
                    for (size_t p=0; p<pool_size[0]*channels*input_shape[1];
                            p+=channels*input_shape[1]) {
                        if (output_array[l+j+i] < input_array[m+k+i+n+p]) {
                            output_array[l+j+i] = input_array[m+k+i+n+p];
                        }
                    }
                }
            }
        }
    }
}