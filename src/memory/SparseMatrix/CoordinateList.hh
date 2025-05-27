#ifndef GLASSHELIX_COOMATRIX_HH
#define GLASSHELIX_COOMATRIX_HH

#include "IRandomAccess.hh"
#include "CompressedSparseRow.hh"
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace glasshelix::SparseMatrix {

    template<typename IndexT, typename ValueT>
    class CoordinateList : public IRandomAccess<IndexT,ValueT> {
    public:
        CoordinateList(IndexT r = 0, IndexT c = 0)
                : _rows(r), _cols(c) {}

        // dimensions
        IndexT rows()   const noexcept override { return _rows; }
        IndexT cols()   const noexcept override { return _cols; }
        std::size_t nonZeros() const noexcept override { return _values.size(); }

        // test if entry exists
        bool contains(IndexT i, IndexT j) const noexcept override {
            return std::any_of(_row.begin(), _row.end(), [&](IndexT k){
                auto idx = &k - &_row[0];
                return _row[idx]==i && _col[idx]==j;
            });
        }

        // read (throws if not present)
        ValueT at(IndexT i, IndexT j) const override {
            for (std::size_t k = 0; k < _values.size(); ++k) {
                if (_row[k] == i && _col[k] == j)
                    return _values[k];
            }
            throw std::out_of_range("COO entry not found");
        }

        // write / insert
        ValueT& operator()(IndexT i, IndexT j) override {
            // search existing
            for (std::size_t k = 0; k < _values.size(); ++k) {
                if (_row[k] == i && _col[k] == j)
                    return _values[k];
            }
            // append new zero entry
            _row.push_back(i);
            _col.push_back(j);
            _values.push_back(ValueT{});
            return _values.back();
        }

        // erase entry if exists
        void erase(IndexT i, IndexT j) noexcept override {
            for (std::size_t k = 0; k < _values.size(); ++k) {
                if (_row[k] == i && _col[k] == j) {
                    _row.erase(_row.begin()   + k);
                    _col.erase(_col.begin()   + k);
                    _values.erase(_values.begin() + k);
                    return;
                }
            }
        }

        // remove all entries
        void clear() noexcept override {
            _row.clear();
            _col.clear();
            _values.clear();
        }

        // optional: reserve space if you know nnz in advance
        void reserve(std::size_t nnz) {
            _row.reserve(nnz);
            _col.reserve(nnz);
            _values.reserve(nnz);
        }

        // convert to CSR (simple two‐pass)
        CompressedSparseRow<IndexT,ValueT> toCSR() const {
            CompressedSparseRow<IndexT,ValueT> csr(_rows, _cols);
            // count per‐row
            std::vector<IndexT> counts(_rows, 0);
            for (auto i : _row) ++counts[i];

            // build rowPtr
            csr._rowPtr.resize(_rows+1);
            csr._rowPtr[0] = 0;
            for (IndexT i = 0; i < _rows; ++i)
                csr._rowPtr[i+1] = csr._rowPtr[i] + counts[i];

            // allocate
            csr._nnz = _values.size();
            csr._colIdx.resize(csr._nnz);
            csr._values.resize(csr._nnz);

            // fill (second pass)
            std::vector<IndexT> offset = csr._rowPtr;
            for (std::size_t k = 0; k < _values.size(); ++k) {
                auto i = _row[k], j = _col[k];
                auto pos = offset[i]++;
                csr._colIdx[pos] = j;
                csr._values[pos] = _values[k];
            }
            return csr;
        }

    private:
        IndexT _rows, _cols;
        std::vector<IndexT> _row, _col;
        std::vector<ValueT> _values;
    };

} // namespace glasshelix

#endif //GLASSHELIX_COOMATRIX_HH
