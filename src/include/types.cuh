#ifndef GLASSHELIX_TYPES_HH
#define GLASSHELIX_TYPES_HH

#include <cstdint>

namespace glasshelix {
    using fp32 = float;
    using fp64 = double;
    using ui8 = uint8_t;
    using ui16 = uint16_t;
    using ui32 = uint32_t;
    using ui64 = uint64_t;

    #ifdef __CUDACC__
        #include <cuda_fp16.h>
    using fp16 = __half;
    using Stream = cudaStream_t;
    #else
    using fp16 = float; // float promotion
    using Stream = void*; // placeholder for API compat
    #endif
}

#endif //GLASSHELIX_TYPES_HH
