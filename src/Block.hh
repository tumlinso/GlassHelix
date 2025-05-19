#ifndef GLASSHELIX_BLOCK_HH
#define GLASSHELIX_BLOCK_HH

//-----------------------------------------------------------------------------
// Block.hh
// Provides the Block<T> class for lazy-loading and chunked access to large binary files.
//-----------------------------------------------------------------------------

//--- Includes ---
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <atomic>
#include <thread>
#include <stdexcept>
#include <algorithm>
#include <omp.h>

//-----------------------------------------------------------------------------
// Class: Block<T>
// Lazy-loading, chunked access to a large binary file of T records.
//-----------------------------------------------------------------------------
template<typename T>
class Block {
    // User parameters
    std::string filename_;
    size_t      recordCount_, recordLen_, recsPerChunk_;

    // Derived sizes
    size_t totalElems_;   // = recordCount_ * recordLen_
    size_t chunkElems_;   // = recsPerChunk_  * recordLen_  (elements)
    size_t chunkBytes_;   // = chunkElems_    * sizeof(T)   (bytes)
    size_t chunkCount_;   // = ceil(recordCount_/recsPerChunk_)

    // Storage & state
    std::atomic<T*> *chunks_;    // array of atomics, nullptr until loaded
    std::atomic<size_t> cursor_{0};

    // Prefetch state
    std::thread        prefetcher_;
    std::atomic<bool>  shutting_down_{false};

    //-----------------------------------------------------------------------------
    // Low-level file I/O
    //-----------------------------------------------------------------------------
    T* allocate_and_read(size_t ci) {
        // how many elements in this chunk? (last one may be smaller)
        size_t elems = std::min(chunkElems_, totalElems_ - ci * chunkElems_);
        size_t bytes = elems * sizeof(T);

        T* buf = static_cast<T*>(std::malloc(bytes));
        if (!buf) throw std::bad_alloc();

        std::ifstream f(filename_, std::ios::binary);
        if (!f) {
            std::free(buf);
            throw std::runtime_error("Cannot open " + filename_);
        }
        f.seekg(static_cast<std::streamoff>(ci) * static_cast<std::streamoff>(chunkBytes_),
                std::ios::beg);
        f.read(reinterpret_cast<char*>(buf), static_cast<std::streamsize>(bytes));
        if (!f) {
            std::free(buf);
            throw std::runtime_error("I/O error reading chunk " + std::to_string(ci));
        }
        return buf;
    }

    //-----------------------------------------------------------------------------
    // Chunk loading with single-instantiation guarantee
    //-----------------------------------------------------------------------------
    T* loadChunk(size_t ci) {
        T* p = chunks_[ci].load(std::memory_order_acquire);
        if (p) return p;

        T* buf = allocate_and_read(ci);
        // install buf if nobody else has
        if (!chunks_[ci].compare_exchange_strong(
                p, buf,
                std::memory_order_release,
                std::memory_order_relaxed))
        {
            // someone else won
            std::free(buf);
            return p;
        }
        return buf;
    }

    //-----------------------------------------------------------------------------
    // Background prefetch worker
    //-----------------------------------------------------------------------------
    void prefetch_worker(size_t ci) {
        if (shutting_down_) return;
        try {
            loadChunk(ci);
        } catch (...) {
            // swallow exceptions to avoid terminate;
            // you could log or store an exception_ptr here
        }
    }

    //-----------------------------------------------------------------------------
    // Public interface
    //-----------------------------------------------------------------------------
public:
    /// Constructor: initialize all state and warm first chunk
    Block(std::string filename,
          size_t recordLen,
          size_t recordCount,
          size_t recsPerChunk)
            : filename_(std::move(filename))
            , recordCount_(recordCount)
            , recordLen_( recordLen)
            , recsPerChunk_(recsPerChunk)
    {
        if (!recordLen_ || !recsPerChunk_)
            throw std::invalid_argument("recordLen/recsPerChunk must be >0");

        totalElems_ = recordCount_ * recordLen_;
        chunkElems_ = recsPerChunk_  * recordLen_;
        chunkBytes_ = chunkElems_     * sizeof(T);
        chunkCount_ = (recordCount_ + recsPerChunk_ - 1)/recsPerChunk_;

        // allocate raw arrays
        chunks_ = static_cast<std::atomic<T*>*>(
                std::malloc(sizeof(std::atomic<T*>) * chunkCount_));
        for (size_t i = 0; i < chunkCount_; ++i)
            new (&chunks_[i]) std::atomic<T*>(nullptr);  // placement new

        // optionally warm the first chunk
        loadChunk(0);
    }

    /// Destructor: shutdown prefetcher and free all memory
    ~Block() {
        shutting_down_ = true;
        if (prefetcher_.joinable())
            prefetcher_.join();

        // free all buffers
        for (size_t i = 0; i < chunkCount_; ++i) {
            T* p = chunks_[i].load(std::memory_order_relaxed);
            std::free(p);
            chunks_[i].~atomic();
        }
        std::free(chunks_);
    }

    /// Eagerly load all chunks in parallel via OpenMP
    void prefetch_all(int nThreads = omp_get_max_threads()) {
        omp_set_num_threads(nThreads);
#pragma omp parallel for schedule(dynamic,1) default(none) shared(this)
        for (size_t ci = 0; ci < chunkCount_; ++ci) {
            loadChunk(ci);
        }
    }

    /// Fetch next chunk (FIFO) with optional freeing of the previous chunk and lookahead prefetch
    T* fetch_chunk(bool freeLast = false) {
        size_t ci = cursor_.fetch_add(1);
        if (ci >= chunkCount_)
            throw std::out_of_range("No more chunks to fetch");

        T* ptr = loadChunk(ci);

        // free the last chunk if requested
        if (freeLast && ci > 0) {
            T* last = chunks_[ci - 1].exchange(nullptr);
            if (last)
                std::free(last);
        }

        // spawn or reuse the prefetcher for ci+1
        size_t next = ci + 1;
        if (next < chunkCount_ && !shutting_down_) {
            if (prefetcher_.joinable())
                prefetcher_.join();
            prefetcher_ = std::thread(&Block::prefetch_worker, this, next);
        }
        return ptr;
    }

    /// Random-access record: returns pointer to the start of the record at the given index
    T* operator[](size_t recordIdx) {
        if (recordIdx >= recordCount_)
            throw std::out_of_range("Block::operator[] recordIdx out of range");
        size_t ci = recordIdx / recsPerChunk_;             // which chunk
        size_t recOff = recordIdx % recsPerChunk_;         // record offset within chunk
        T* buf = loadChunk(ci);
        return buf + recOff * recordLen_;                  // pointer to record start
    }

    /// Const version of record-access operator[]
    const T* operator[](size_t recordIdx) const {
        if (recordIdx >= recordCount_)
            throw std::out_of_range("Block::operator[] recordIdx out of range");
        size_t ci = recordIdx / recsPerChunk_;
        size_t recOff = recordIdx % recsPerChunk_;
        // need const-cast to call loadChunk; or duplicate logic to a private const-load
        T* buf = const_cast<Block*>(this)->loadChunk(ci);
        return buf + recOff * recordLen_;
    }

    // Queries
    size_t size()            const noexcept { return totalElems_;   }
    size_t num_chunks()      const noexcept { return chunkCount_;   }
    size_t elems_per_chunk() const noexcept { return chunkElems_;   }
};

#endif // GLASSHELIX_BLOCK_HH