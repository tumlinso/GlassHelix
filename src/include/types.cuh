#ifndef GLASSHELIX_TYPES_HH
#define GLASSHELIX_TYPES_HH

#include <cstdint>

namespace glasshelix {
    using IndexT = int;

#ifdef __CUDACC__
#include <cuda_fp16.h>
    using fp16 = __half;
#else
    using fp16 = float; // float promotion for non-CUDA code
#endif

    using fp32 = float;
    using fp64 = double;
    using ui8 = uint8_t;
    using ui16 = uint16_t;
    using ui32 = uint32_t;
    using ui64 = uint64_t;
}

#endif //GLASSHELIX_TYPES_HH
