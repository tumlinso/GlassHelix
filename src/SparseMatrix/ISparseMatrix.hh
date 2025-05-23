#ifndef GLASSHELIX_ISPARSEMATRIX_HH
#define GLASSHELIX_ISPARSEMATRIX_HH

#include <cstddef>

namespace glasshelix {

    template<typename IndexT, typename ValueT>
    class ISparseMatrix {
    public:
        virtual ~ISparseMatrix() = default;

        virtual IndexT rows()   const noexcept = 0;
        virtual IndexT cols()   const noexcept = 0;
        virtual std::size_t nonZeros() const noexcept = 0;

        virtual bool contains(IndexT i, IndexT j) const noexcept = 0;
        virtual ValueT at(IndexT i, IndexT j) const;         // throws if absent
        virtual ValueT& operator()(IndexT i, IndexT j) = 0;   // inserts if needed
        virtual void erase(IndexT i, IndexT j) noexcept = 0;

        virtual void clear() noexcept = 0;
    };

} // namespace glasshelix

#endif //GLASSHELIX_ISPARSEMATRIX_HH
