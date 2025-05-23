#ifndef GLASSHELIX_CSRMATRIX_HH
#define GLASSHELIX_CSRMATRIX_HH

#include "ISparseMatrix.hh"

#include <vector>
#include <stdexcept>
#include <fstream>

namespace glasshelix {

    template<typename IndexT, typename ValueT>
    class CooMatrix;

    template<typename IndexT, typename ValueT>
    class CsrMatrix : public ISparseMatrix<IndexT,ValueT> {
        friend class CooMatrix<IndexT,ValueT>;
    public:
        CsrMatrix(IndexT r=0, IndexT c=0)
                : _rows(r), _cols(c), _nnz(0)
        {
            _rowPtr.assign(r+1, 0);
        }

        // dimensions
        IndexT rows()   const noexcept override { return _rows; }
        IndexT cols()   const noexcept override { return _cols; }
        std::size_t nonZeros() const noexcept override { return _nnz; }

        bool contains(IndexT i, IndexT j) const noexcept override {
            for (auto idx = _rowPtr[i]; idx < _rowPtr[i+1]; ++idx)
                if (_colIdx[idx] == j) return true;
            return false;
        }

        ValueT at(IndexT i, IndexT j) const override {
            for (auto idx = _rowPtr[i]; idx < _rowPtr[i+1]; ++idx)
                if (_colIdx[idx] == j) return _values[idx];
            throw std::out_of_range("Entry not stored in CSR");
        }

        // this will insert a zero entry at end of row i if not found!
        ValueT& operator()(IndexT i, IndexT j) override {
            // naive append: for real use you'd want to keep sorted order,
            // or build a COO first and compress later
            for (auto idx = _rowPtr[i]; idx < _rowPtr[i+1]; ++idx)
                if (_colIdx[idx] == j) return _values[idx];

            // insert at end of row i
            auto insertPos = _rowPtr[i+1];
            _colIdx.insert(_colIdx.begin()+insertPos, j);
            _values.insert (_values.begin()+insertPos, ValueT{});
            ++_nnz;
            for (std::size_t k = i+1; k <= _rows; ++k)
                ++_rowPtr[k];
            return _values[insertPos];
        }

        void erase(IndexT i, IndexT j) noexcept override {
            for (auto idx = _rowPtr[i]; idx < _rowPtr[i+1]; ++idx) {
                if (_colIdx[idx] == j) {
                    _colIdx.erase(_colIdx.begin()+idx);
                    _values.erase(_values.begin()+idx);
                    --_nnz;
                    for (std::size_t k = i+1; k <= _rows; ++k)
                        --_rowPtr[k];
                    return;
                }
            }
        }

        void clear() noexcept override {
            _nnz = 0;
            _rowPtr.assign(_rows+1, 0);
            _colIdx.clear();
            _values.clear();
        }

        // load from file
        void loadFromFile(std::string const& file) {
            std::ifstream f(file, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot open " + file);
            f.read(reinterpret_cast<char*>(&_rows), sizeof(_rows));
            f.read(reinterpret_cast<char*>(&_cols), sizeof(_cols));
            f.read(reinterpret_cast<char*>(&_nnz ), sizeof(_nnz ));
            _rowPtr.resize(_rows+1);
            _colIdx.resize(_nnz);
            _values.resize(_nnz);
            f.read(reinterpret_cast<char*>(_rowPtr.data()), (_rows+1)*sizeof(IndexT));
            f.read(reinterpret_cast<char*>(_colIdx.data()), _nnz*sizeof(IndexT));
            f.read(reinterpret_cast<char*>(_values.data()), _nnz*sizeof(ValueT));
        }

    private:
        IndexT _rows, _cols;
        std::size_t _nnz;
        std::vector<IndexT> _rowPtr;
        std::vector<IndexT> _colIdx;
        std::vector<ValueT> _values;
    };

} // namespace glasshelix

#endif //GLASSHELIX_CSRMATRIX_HH
