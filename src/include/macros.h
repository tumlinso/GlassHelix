#ifndef GLASSHELIX_MACROS_H
#define GLASSHELIX_MACROS_H

#if defined(__CUDACC__)
#define __inlRadical__ __host__ __device__ __forceinline__
#else
#define __inlRadical__ static inline
#endif

#endif //GLASSHELIX_MACROS_H
