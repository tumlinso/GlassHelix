#ifndef GLASSHELIX_BLOCK_HH
#define GLASSHELIX_BLOCK_HH

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <stdexcept>
#include <algorithm>
#include <omp.h>

template<typename T>
class Block {
    /* ────────────────────── user parameters ────────────────────── */
    std::string filename_;
    size_t      recordCount_, recordLen_, recsPerChunk_;

    /* ────────────────────── derived sizes ──────────────────────── */
    size_t totalElems_;    // recordCount_ * recordLen_
    size_t chunkElems_;    // recsPerChunk_ * recordLen_  (elements)
    size_t chunkBytes_;    // chunkElems_   * sizeof(T)   (bytes)
    size_t chunkCount_;    // ceil(recordCount_/recsPerChunk_)

    /* ────────────────────── storage & state ───────────────────── */
    std::vector<std::atomic<T*>> chunks_;         // nullptr until loaded
    std::atomic<size_t>          cursor_{0};      // for fetch_chunk()

    std::ifstream file_;      // single file handle
    std::mutex    io_mtx_;    // guards seekg/read on file_

    std::thread   prefetcher_;          // single look-ahead thread
    std::atomic<bool> shutting_down_{false};

    /* ─────────── low-level I/O: read one chunk from disk ───────── */
    T* allocate_and_read(size_t ci) {
        const size_t elems  = std::min(chunkElems_, totalElems_ - ci*chunkElems_);
        const size_t bytes  = elems * sizeof(T);

        T* buf = static_cast<T*>(std::malloc(bytes));
        if (!buf) throw std::bad_alloc();

        {
            std::lock_guard lk(io_mtx_);
            file_.seekg(static_cast<std::streamoff>(ci) * static_cast<std::streamoff>(chunkBytes_),
                        std::ios::beg);
            file_.read(reinterpret_cast<char*>(buf), bytes);
            if (!file_) {
                std::free(buf);
                throw std::runtime_error("I/O error reading chunk " + std::to_string(ci));
            }
        }
        return buf;
    }

    /* ───── ensure chunk ci is loaded; return pointer (thread-safe) ───── */
    T* loadChunk(size_t ci) {
        T* p = chunks_[ci].load(std::memory_order_acquire);
        if (p) return p;

        T* buf = allocate_and_read(ci);

        if (!chunks_[ci].compare_exchange_strong(p, buf,
                                                 std::memory_order_release,
                                                 std::memory_order_relaxed)) {
            std::free(buf);            // another thread won
            return p;
        }
        return buf;                    // we won
    }

    /* ───────────── look-ahead prefetch worker (one thread) ──────────── */
    void prefetch_worker(size_t ci) {
        try {
            loadChunk(ci);
        } catch (...) { /* swallow to avoid std::terminate; could log */ }
    }

public:
    /* ───────────────────── constructor ─────────────────────────── */
    Block(std::string filename,
          size_t recordLen,
          size_t recordCount,
          size_t recsPerChunk)
            : filename_(std::move(filename))
            , recordCount_(recordCount)
            , recordLen_(recordLen)
            , recsPerChunk_(recsPerChunk)
    {
        if (!recordLen_ || !recsPerChunk_)
            throw std::invalid_argument("recordLen and recsPerChunk must be > 0");

        totalElems_  = recordCount_ * recordLen_;
        chunkElems_  = recsPerChunk_ * recordLen_;
        chunkBytes_  = chunkElems_   * sizeof(T);
        chunkCount_  = (recordCount_ + recsPerChunk_ - 1) / recsPerChunk_;

        chunks_.resize(chunkCount_);
        for (auto &a : chunks_) a.store(nullptr, std::memory_order_relaxed);

        file_.open(filename_, std::ios::binary);
        if (!file_) throw std::runtime_error("Cannot open " + filename_);

        /* optional: warm first chunk */
        loadChunk(0);
    }

    /* ───────────────────── destructor ──────────────────────────── */
    ~Block() {
        shutting_down_ = true;
        if (prefetcher_.joinable()) prefetcher_.join();
        for (auto &a : chunks_)
            std::free(a.load(std::memory_order_relaxed));
    }

    /* ─── eager load all chunks in parallel (OpenMP) ─── */
    void prefetch_all(int nThreads = omp_get_max_threads()) {
        omp_set_num_threads(nThreads);
#pragma omp parallel for schedule(dynamic,1)
        for (size_t ci = 0; ci < chunkCount_; ++ci)
            loadChunk(ci);
    }

    /* ─── fetch next chunk pointer, start look-ahead of following ─── */
    T* fetch_chunk() {
        size_t ci = cursor_.fetch_add(1);
        if (ci >= chunkCount_)
            throw std::out_of_range("No more chunks to fetch");

        T* ptr = loadChunk(ci);

        size_t next = ci + 1;
        if (next < chunkCount_ && !shutting_down_) {
            if (prefetcher_.joinable()) prefetcher_.join();  // wait for previous
            prefetcher_ = std::thread(&Block::prefetch_worker, this, next);
        }
        return ptr;
    }

    /* ─── element access ─── */
    T& operator[](size_t idx) {
        if (idx >= totalElems_)
            throw std::out_of_range("Block::operator[] index");
        const size_t ci  = idx / chunkElems_;
        const size_t off = idx % chunkElems_;
        T* buf = loadChunk(ci);
        return buf[off];
    }

    /* ─── queries ─── */
    size_t size()            const noexcept { return totalElems_;  }
    size_t num_chunks()      const noexcept { return chunkCount_;  }
    size_t elems_per_chunk() const noexcept { return chunkElems_;  }
};

#endif // GLASSHELIX_BLOCK_HH