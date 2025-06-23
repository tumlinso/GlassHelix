#ifndef GLASSHELIX_GLASSETCH_ENCODE_CUH
#define GLASSHELIX_GLASSETCH_ENCODE_CUH

#include <cudnn.h>

#include "include/types.cuh"
#include "include/DeviceMatrix.cuh"

using namespace glasshelix::devicematrix;

namespace glasshelix::glassetch {
    __global__ void encode(Sparse<ui16, fp16>& __restrict__ transcriptome,
                           Dense<ui16, fp16>& __restrict__

    }

}

#endif //GLASSHELIX_GLASSETCH_ENCODE_CUH
