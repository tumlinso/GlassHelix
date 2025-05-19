// BlockTest.cpp

#include "../Block.hh"

#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <chrono>
#include <random>
#include <cstdio>    // for std::remove

using namespace std::chrono;

// type alias for convenience
using T = int32_t;

int main() {
    const std::string filename = "test_data.bin";

    try {
        // Test parameters
        const size_t recordCount     = 1'000'000;  // total number of records
        const size_t recordLen       = 1024;     // elements per record
        const size_t recordsPerChunk = 5000;     // grouping into chunks

        const size_t totalElements = recordCount * recordLen;

        // 1) Generate test .bin file
        {
            std::ofstream out(filename, std::ios::binary);
            assert(out && "Failed to open test file for writing");
            for (size_t i = 0; i < totalElements; ++i) {
                T value = static_cast<T>(i);
                out.write(reinterpret_cast<const char*>(&value), sizeof(T));
            }
        }
        std::cout << "Generated test file '" << filename
                  << "' with " << totalElements << " elements.\n";

        // 2) Construct Block
        Block<T> block(filename, recordLen, recordCount, recordsPerChunk);
        std::cout << "Block created: "
                  << " totalElems="     << block.size()
                  << " num_chunks="     << block.num_chunks()
                  << " elems_per_chunk="<< block.elems_per_chunk()
                  << "\n";

        // 3) Test random-access operator[]
        {
            std::mt19937_64 rng(12345);
            std::uniform_int_distribution<size_t> dist(0, totalElements - 1);

            const int trials = 1000;
            for (int t = 0; t < trials; ++t) {
                size_t idx = dist(rng);
                T      v   = block[idx];
                assert(v == static_cast<T>(idx));
            }
            std::cout << "operator[] random-access test passed ("
                      << trials << " trials).\n";
        }

        // 4) Time on-demand random accesses
        {
            auto start = steady_clock::now();
            for (size_t i = 0; i < totalElements; i += recordLen) {
                volatile T v = block[i];
                (void)v;
            }
            auto dur = duration_cast<milliseconds>(steady_clock::now() - start).count();
            std::cout << "Timed on-demand single-element access every "
                      << recordLen << " in " << dur << " ms\n";
        }

        // 5) Test fetch_chunk() + lookahead
        {
            size_t chunks     = block.num_chunks();
            size_t chunkElems = block.elems_per_chunk();
            for (size_t ci = 0; ci < chunks; ++ci) {
                T* data = block.fetch_chunk();
                size_t base = ci * chunkElems;
                assert(data[0] == static_cast<T>(base));
                size_t elemsThis = std::min(chunkElems, totalElements - base);
                assert(data[elemsThis - 1] == static_cast<T>(base + elemsThis - 1));
            }
            std::cout << "fetch_chunk() sequential test passed for "
                      << chunks << " chunks.\n";
        }

        // 6) Time fetch_chunk loop
        {
            Block<T> blk2(filename, recordLen, recordCount, recordsPerChunk);
            auto start = steady_clock::now();
            size_t chunks = blk2.num_chunks();
            for (size_t ci = 0; ci < chunks; ++ci) {
                volatile T* ptr = blk2.fetch_chunk();
                (void)ptr;
            }
            auto dur = duration_cast<milliseconds>(steady_clock::now() - start).count();
            std::cout << "Timed fetch_chunk for " << chunks
                      << " chunks in " << dur << " ms\n";
        }

        // 7) Test prefetch_all()
        {
            Block<T> blk3(filename, recordLen, recordCount, recordsPerChunk);
            auto start = steady_clock::now();
            blk3.prefetch_all();
            auto dur = duration_cast<milliseconds>(steady_clock::now() - start).count();
            std::cout << "prefetch_all() initiated in " << dur << " ms\n";

            // verify all chunks loaded
            for (size_t i = 0; i < totalElements; i += recordLen) {
                assert(blk3[i] == static_cast<T>(i));
            }
            std::cout << "prefetch_all() correctness test passed.\n";
        }

        std::cout << "All Block tests passed successfully.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        // fall through to cleanup
    }

    // Cleanup test file
    if (std::remove(filename.c_str()) == 0) {
        std::cout << "Cleaned up test file '" << filename << "'.\n";
    } else {
        std::perror("Error removing test file");
    }

    return 0;
}