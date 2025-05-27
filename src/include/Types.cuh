#ifndef GLASSHELIX_TYPES_CUH
#define GLASSHELIX_TYPES_CUH

#include <cuda_fp16.h>
#include <cstdint>

namespace glasshelix {
    using IndexT = int;
    using fp16 = __half;
    using fp32 = float;
    using fp64 = double;
    using ui16 = uint16_t;
    using ui32 = uint32_t;
    using ui64 = uint64_t;
}

#endif //GLASSHELIX_TYPES_CUH
