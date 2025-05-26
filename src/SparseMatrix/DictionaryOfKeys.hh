#ifndef GLASSHELIX_DOKMATRIX_HH
#define GLASSHELIX_DOKMATRIX_HH

#include "IRandomAccess.hh"
#include <unordered_map>

namespace glasshelix::SparseMatrix {

    template <typename I, typename J>
    struct PairHash {
        std::size_t operator()(std::pair<I,J> const& p) const noexcept {
            return std::hash<I>()(p.first) ^ (std::hash<J>()(p.second) << 1);
        }
    };

    template<typename IndexT, typename ValueT>
    class DictionaryOfKeys : public IRandomAccess<IndexT,ValueT> {
    public:
        using Key = std::pair<IndexT,IndexT>;

        DictionaryOfKeys(IndexT r=0, IndexT c=0)
                : _rows(r), _cols(c) {}

        // dimensions
        IndexT rows()   const noexcept override { return _rows; }
        IndexT cols()   const noexcept override { return _cols; }
        std::size_t nonZeros() const noexcept override { return _data.size(); }

        // element access
        bool contains(IndexT i, IndexT j) const noexcept override {
            return _data.find({i,j}) != _data.end();
        }

        ValueT at(IndexT i, IndexT j) const override {
            return _data.at({i,j});  // will throw std::out_of_range
        }

        ValueT& operator()(IndexT i, IndexT j) override {
            return _data[{i,j}];
        }

        void erase(IndexT i, IndexT j) noexcept override {
            _data.erase({i,j});
        }

        void clear() noexcept override {
            _data.clear();
        }

        void resize(IndexT r, IndexT c) noexcept {
            _rows = r; _cols = c;
            _data.clear();
        }

    private:
        IndexT _rows, _cols;
        std::unordered_map<Key, ValueT, PairHash<IndexT,IndexT>> _data;
    };

} // namespace glasshelix

#endif //GLASSHELIX_DOKMATRIX_HH
