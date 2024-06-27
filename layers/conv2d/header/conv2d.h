#ifndef CONV2D_H_
#define CONV2D_H_

#include <stddef.h>

void k2c_conv2d(float*,
                size_t,
                size_t,
                size_t*,

                const float*,
                const size_t,
                const size_t,
                const size_t*,
                
                const float*,
                const size_t,
                const size_t ,
                const size_t*,


                const float*,
                const size_t,
                const size_t,
                const size_t*,

                const size_t*, 
                const size_t*);

#endif
