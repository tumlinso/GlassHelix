#ifndef GLASSHELIX_SCATTER_CUH
#define GLASSHELIX_SCATTER_CUH

#include "include/Types.cuh"
#include "include/DeviceMatrix.cuh"
#include "include/DeviceMacros.cuh"

using namespace glasshelix;

using SparseMatrix  = devicematrix::SparseCOO<fp16>;
using DenseFp16  = devicematrix::Dense<fp16>;
using DenseFp32  = devicematrix::Dense<fp32>;

namespace glasshelix::glassetch {
    __global__ void scatterKernel(ui32 nnz,
                                  const IndexT* __restrict__ rowIdx,
                                  const IndexT* __restrict__ colIdx,
                                  const fp16* __restrict__ val,
                                  fp16* __restrict__ out,
                                  ui16 ld) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nnz) return;

        ui16 row = rowIdx[idx];
        ui16 col = colIdx[idx];
        fp32 value = __half2float(val[idx]);

        fp16* slot = &out[row * ld + col];
        atomicAdd(slot, __float2half(value));
    }

    __global__ void scatterDerivativeKernel(ui16 nnz,
                                            const IndexT* __restrict__ rowIdx,
                                            const IndexT* __restrict__ colIdx,
                                            const fp32* __restrict__ dX,
                                            fp32* __restrict__ dVal,
                                            ui16 ld) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nnz) return;

        ui16 row = rowIdx[idx];
        ui16 col = colIdx[idx];
        dVal[idx] = dX[row * ld + col];
    }

    inline void scatter(SparseMatrix &in,
                        DenseFp16 &out,
                        cudaStream_t stream = nullptr) {
        const unsigned int THREADS = 256;
        unsigned int blocks = (in.nnz + THREADS - 1) / THREADS;
        scatterKernel<<<blocks, THREADS, 0, stream>>>(in.nnz,
                                  in.rowIdx,
                                  in.colIdx,
                                  in.val,
                                  out.val,
                                  out.ld);
        GLASSHELIX_CHECK(cudaGetLastError());
    }

    inline void scatterDerivative(SparseMatrix &in,
                                  DenseFp32 &dX,
                                  fp32* dVal,
                                  cudaStream_t stream = nullptr) {
        const unsigned int THREADS = 256;
        unsigned int blocks = (in.nnz + THREADS - 1) / THREADS;
        scatterDerivativeKernel<<<blocks, THREADS, 0, stream>>>(in.nnz,
                                  in.rowIdx,
                                  in.colIdx,
                                  dX.val,
                                  dVal,
                                  dX.ld);
        GLASSHELIX_CHECK(cudaGetLastError());
    }
} // namespace glasshelix::glassetch

#endif //GLASSHELIX_SCATTER_CUH
