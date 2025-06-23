#ifndef GLASSHELIX_MATRIX_CUH
#define GLASSHELIX_MATRIX_CUH

#include "macros.cuh"
#include "types.cuh"

namespace glasshelix::matrix {
    template<typename ValueT>
    struct Dense {
        ValueT* val;
        IndexT rows, cols, ld;

        radical ValueT& operator()(IndexT r, IndexT c) { return val[r * ld + c]; }
        radical const ValueT& operator()(IndexT r, IndexT c) const { return val[r * ld + c]; }
        radical ValueT* address(IndexT r, IndexT c) { return &val[r * ld + c]; }
    };

    template<typename ValueT>
    struct SparseCSR {
        IndexT* rowPtr; // length = rows+1
        IndexT* colIdx; // length = nnz
        ValueT* val;    // length = nnz
        IndexT  rows, cols, nnz;
    };

    template<typename ValueT>
    struct SparseCOO {
        IndexT* rowIdx; // length = nnz
        IndexT* colIdx; // length = nnz
        ValueT* val;    // length = nnz
        IndexT  nnz;
    };
}

#endif //GLASSHELIX_MATRIX_CUH
