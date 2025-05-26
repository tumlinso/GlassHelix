#ifndef GLASSHELIX_CSRDEVICE_HH
#define GLASSHELIX_CSRDEVICE_HH

#include <cuda_runtime.h>
#include <cstddef>
#include <utility>
#include <vector>
#include <stdexcept>

namespace glasshelix::SparseMatrix {

/// A GPU-resident CSR matrix with templated index and value types (e.g. uint32_t, __half)
    template<typename IndexT, typename ValueT>
    class CsrDevice {
    public:
        // Dimensions and nonzeros
        IndexT rows{};
        IndexT cols{};
        std::size_t nnz{};

        // Device pointers
        IndexT*  d_rowPtr{nullptr};  // length: rows+1
        IndexT*  d_colIdx{nullptr};  // length: nnz
        ValueT*  d_values{nullptr};  // length: nnz

        /// Default ctor: empty
        CsrDevice() noexcept = default;

        /// Move ctor
        CsrDevice(CsrDevice&& o) noexcept
                : rows(o.rows), cols(o.cols), nnz(o.nnz),
        d_rowPtr(o.d_rowPtr), d_colIdx(o.d_colIdx), d_values(o.d_values)
        {
            o.d_rowPtr = nullptr; o.d_colIdx = nullptr; o.d_values = nullptr;
            o.rows = o.cols = 0; o.nnz = 0;
        }

        /// Move assign
        CsrDevice& operator=(CsrDevice&& o) noexcept {
            if (this != &o) {
                free();
                rows    = o.rows;
                cols    = o.cols;
                nnz     = o.nnz;
                d_rowPtr = o.d_rowPtr;
                d_colIdx = o.d_colIdx;
                d_values = o.d_values;
                o.d_rowPtr = o.d_colIdx = o.d_values = nullptr;
                o.rows = o.cols = 0; o.nnz = 0;
            }
            return *this;
        }

        /// No copying
        CsrDevice(const CsrDevice&) = delete;
        CsrDevice& operator=(const CsrDevice&) = delete;

        /// Destructor: free device memory
        ~CsrDevice() {
            free();
        }

        /// Allocate and copy from host-side CSR arrays
        /// host_rowPtr size = rows+1, host_colIdx.size() = nnz, host_values.size() = nnz
        static CsrDevice fromHost(
                const std::vector<IndexT>& host_rowPtr,
                const std::vector<IndexT>& host_colIdx,
                const std::vector<ValueT>& host_values,
                cudaStream_t stream = nullptr
        ) {
            CsrDevice dev;
            dev.rows = static_cast<IndexT>(host_rowPtr.size() - 1);
            dev.cols = 0;
            dev.nnz  = host_colIdx.size();
            // optionally infer cols
            for (auto c : host_colIdx) if (c + 1 > dev.cols) dev.cols = c + 1;

            // Allocate
            cudaMallocAsync(&dev.d_rowPtr, sizeof(IndexT) * (dev.rows + 1), stream);
            cudaMallocAsync(&dev.d_colIdx, sizeof(IndexT) *  dev.nnz,      stream);
            cudaMallocAsync(&dev.d_values, sizeof(ValueT) *  dev.nnz,      stream);

            // Copy
            cudaMemcpyAsync(dev.d_rowPtr,
                            host_rowPtr.data(),
                            sizeof(IndexT)*(dev.rows + 1),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(dev.d_colIdx,
                            host_colIdx.data(),
                            sizeof(IndexT)*dev.nnz,
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(dev.d_values,
                            host_values.data(),
                            sizeof(ValueT)*dev.nnz,
                            cudaMemcpyHostToDevice, stream);
            return dev;
        }

    private:
        void free() noexcept {
            if (d_rowPtr) cudaFreeAsync(d_rowPtr, nullptr);
            if (d_colIdx) cudaFreeAsync(d_colIdx, nullptr);
            if (d_values) cudaFreeAsync(d_values, nullptr);
        }
    };

} // namespace glasshelix

#endif // GLASSHELIX_CSRDEVICE_HH
