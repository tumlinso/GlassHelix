#ifndef GLASSHELIX_MATRIX_CUH
#define GLASSHELIX_MATRIX_CUH

#include "macros.cuh"
#include "types.cuh"

namespace glasshelix::matrix {
    // Dense matrix in row-major order
    template<typename ValueT, typename IndexT = int>
    struct dense {
        ValueT* val;
        IndexT rows, cols, ld; // ld = leading dimension

        radical const ValueT *operator()(IndexT r, IndexT c) const { return val + r * ld + c; }
    };

    // all native sparse matrix formats
    namespace sparse {
        // Compressed Sparse Row
        template<typename ValueT, typename IndexT = int>
        struct csr {
            IndexT* rowPtr; // length = rows+1
            IndexT* colIdx; // length = nnz
            ValueT* val;    // length = nnz
            IndexT  rows, cols, nnz;

            radical const ValueT *operator()(IndexT r, IndexT c) const {
                for (IndexT i = rowPtr[r]; i < rowPtr[r + 1]; ++i) {
                    if (colIdx[i] == c) {
                        return val + i;
                    }
                }
                return nullptr; // Element not found
            }
        };

        // Compressed Sparse Column
        template<typename ValueT, typename IndexT = int>
        struct csc {
            IndexT* colPtr; // length = cols+1
            IndexT* rowIdx; // length = nnz
            ValueT* val;    // length = nnz
            IndexT  rows, cols, nnz;

            radical const ValueT *operator()(IndexT r, IndexT c) const {
                for (IndexT i = colPtr[c]; i < colPtr[c + 1]; ++i) {
                    if (rowIdx[i] == r) {
                        return val + i;
                    }
                }
                return nullptr; // Element not found
            }
        };

        // Coordinate List
        template<typename ValueT, typename IndexT = int>
        struct coo {
            IndexT* rowIdx; // length = nnz
            IndexT* colIdx; // length = nnz
            ValueT* val;    // length = nnz
            IndexT  nnz;

            radical const ValueT *operator()(IndexT r, IndexT c) const {
                for (IndexT i = 0; i < nnz; ++i) {
                    if (rowIdx[i] == r && colIdx[i] == c) {
                        return val + i;
                    }
                }
                return nullptr; // Element not found
            }
        };

        // Diagonal storage
        template<typename ValueT, typename IndexT = int>
        struct dia {
            IndexT* offsets; // length = num_diagonals
            ValueT* val;     // length = nnz
            IndexT  rows, cols, num_diagonals, nnz;

            radical const ValueT *operator()(IndexT r, IndexT c) const {
                for (IndexT i = 0; i < num_diagonals; ++i) {
                    if (offsets[i] == c - r) {
                        return val + i * rows + r; // Assuming a diagonal structure
                    }
                }
                return nullptr; // Element not found
            }
        };

        // ELLPACK format
        template<typename ValueT, typename IndexT = int>
        struct ell {
            IndexT* colIdx; // length = rows * max_nnz_per_row
            ValueT* val;    // length = rows * max_nnz_per_row
            IndexT  rows, cols, max_nnz_per_row;

            radical const ValueT *operator()(IndexT r, IndexT c) const {
                IndexT idx = r * max_nnz_per_row + c; // Assuming a fixed structure
                if (colIdx[idx] == c) {
                    return val + idx;
                }
                return nullptr; // Element not found
            }
        };
    }
}

#endif //GLASSHELIX_MATRIX_CUH
