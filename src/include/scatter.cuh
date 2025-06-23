#ifndef GLASSHELIX_SCATTER_CUH
#define GLASSHELIX_SCATTER_CUH

#include "types.cuh"
#include "macros.cuh"
#include "matrix.cuh"

namespace glasshelix {

#ifdef __CUDACC__
// CUDA kernels: templated for any ValueT and IndexT
    template<typename ValueT, typename IndexT = int>
    __global__ void scatterKernel(size_t nnz,
                                  const IndexT* __restrict__ rowIdx,
                                  const IndexT* __restrict__ colIdx,
                                  const ValueT* __restrict__ val,
                                  ValueT* __restrict__ out,
                                  IndexT ld) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nnz) return;

        IndexT row = rowIdx[idx];
        IndexT col = colIdx[idx];
        ValueT value = val[idx];
        ValueT* slot = &out[row * ld + col];
        atomicAdd(slot, value);
    }

    template<typename ValueT, typename IndexT = int>
    __global__ void gatherKernel(size_t nnz,
                                 const IndexT* __restrict__ rowIdx,
                                 const IndexT* __restrict__ colIdx,
                                 const ValueT* __restrict__ in,
                                 ValueT* __restrict__ out,
                                 IndexT ld) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nnz) return;

        IndexT row = rowIdx[idx];
        IndexT col = colIdx[idx];
        out[idx] = in[row * ld + col];
    }
#endif // __CUDACC__

// Host API: scatter
    template<typename ValueT, typename IndexT = int>
    inline void scatter(matrix::sparse::coo<ValueT, IndexT>& in,
                        matrix::dense<ValueT, IndexT>& out,
                        Stream stream = nullptr,
                        unsigned int threads = 256) {
#ifdef __CUDACC__
        size_t blocks = (in.nnz + threads - 1) / threads;
        scatterKernel<ValueT, IndexT><<<blocks, threads, 0, stream>>>(
                in.nnz, in.rowIdx, in.colIdx, in.val, out.val, out.ld);
        check(cudaGetLastError());
#else
        #pragma omp parallel for
    for (size_t idx = 0; idx < in.nnz; ++idx) {
        IndexT row = in.rowIdx[idx];
        IndexT col = in.colIdx[idx];
        #pragma omp atomic
        out.val[row * out.ld + col] += in.val[idx];
    }
#endif
    }

// Host API: gather
    template<typename ValueT, typename IndexT = int>
    inline void gather(matrix::sparse::coo<ValueT, IndexT>& in,
                       matrix::dense<ValueT, IndexT>& mat,
                       ValueT* out,
                       Stream stream = nullptr,
                       unsigned int threads = 256) {
#ifdef __CUDACC__
        size_t blocks = (in.nnz + threads - 1) / threads;
        gatherKernel<ValueT, IndexT><<<blocks, threads, 0, stream>>>(
                in.nnz, in.rowIdx, in.colIdx, mat.val, out, mat.ld);
        check(cudaGetLastError());
#else
        #pragma omp parallel for
    for (size_t idx = 0; idx < in.nnz; ++idx) {
        IndexT row = in.rowIdx[idx];
        IndexT col = in.colIdx[idx];
        out[idx] = mat.val[row * mat.ld + col];
    }
#endif
    }

} // namespace glasshelix

#endif // GLASSHELIX_SCATTER_CUH