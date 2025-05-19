// BlockPerformanceTest.cpp

#include "../Block.hh"

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>
#include <cstdio>
#include <sys/resource.h>
#include <unistd.h>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// Helper: get current RSS (resident set size) in kilobytes
static long getCurrentRSS() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;  // on Linux this is in KB
}

struct Result {
    size_t recordsPerChunk;
    double ctor_ms;
    long   rss_after_ctor_kb;
    double seq_scan_ms;
    double rand_access_ms;
    double fetch_chunk_ms;
    double prefetch_ms;
    long   rss_after_prefetch_kb;
};

int main() {
    // Test parameters — tweak these for your machine & file‐size needs:
    const string filename   = "block_perf_test.bin";
    const size_t recordLen  = 2048;    // elements per record
    const size_t recordCount= 100000;  // total records → ~819 MB file
    const size_t totalElems = recordLen * recordCount;

    // Different chunk‐sizes to test (in records per chunk):
    vector<size_t> chunkSizes = {1, 8, 32, 128, 512, 1024, 2048, 4096};

    cout << fixed << setprecision(1);
    cout << "\n=== TestBlockPerformance ===\n"
         << "File: " << filename
         << " (" << (totalElems * sizeof(int32_t))/(1024*1024) << " MiB)\n"
         << "recordLen="  << recordLen
         << ", recordCount=" << recordCount << "\n\n";

    // 1) Generate the binary file
    {
        ofstream out(filename, ios::binary);
        assert(out && "Failed to open test file for writing");
        int32_t v = 0;
        for (size_t i = 0; i < totalElems; ++i) {
            out.write(reinterpret_cast<char*>(&v), sizeof(v));
            ++v;
        }
    }
    cout << "Generated test file with " << totalElems << " ints.\n";

    // Prepare RNG for random‐access tests
    mt19937_64 rng(20250516);
    uniform_int_distribution<size_t> dist(0, totalElems - 1);

    // Container for results
    vector<Result> results;
    results.reserve(chunkSizes.size());

    // Baseline memory before any Block
    long rss_before = getCurrentRSS();
    cout << "RSS before any Block: " << rss_before << " KB\n\n";

    // Loop over chunk‐sizes
    for (size_t rpc : chunkSizes) {
        cout << "--- Testing recordsPerChunk = " << rpc << " ---\n";

        // 2) Construct Block and measure time + memory
        auto t0 = steady_clock::now();
        Block<int32_t> blk(filename, recordLen, recordCount, rpc);
        auto t1 = steady_clock::now();
        long rss_after_ctor = getCurrentRSS();
        double ctor_ms = duration_cast<duration<double, milli>>(t1 - t0).count();
        cout << "  ctor: " << ctor_ms << " ms, RSS: " << rss_after_ctor << " KB\n";

        // 3) Sequential scan: sum all first‐elements of each record
        int64_t checksum = 0;
        auto t2 = steady_clock::now();
        for (size_t i = 0; i < totalElems; i += recordLen) {
            checksum += blk[i];
        }
        auto t3 = steady_clock::now();
        double seq_ms = duration_cast<duration<double, milli>>(t3 - t2).count();
        cout << "  seq_scan: " << seq_ms << " ms"
             << "  (checksum=" << checksum << ")\n";

        // 4) Random accesses: M samples
        const int M = 100000;
        int64_t rcsum = 0;
        auto t4 = steady_clock::now();
        for (int i = 0; i < M; ++i) {
            size_t idx = dist(rng);
            int32_t v = blk[idx];
            assert(v == static_cast<int32_t>(idx));
            rcsum += v;
        }
        auto t5 = steady_clock::now();
        double rand_ms = duration_cast<duration<double, milli>>(t5 - t4).count();
        cout << "  rand_access(" << M << "): " << rand_ms
             << " ms  (checksum=" << rcsum << ")\n";

        // 5) fetch_chunk loop: walk chunk by chunk
        size_t chunks = blk.num_chunks();
        int64_t fc_sum = 0;
        auto t6 = steady_clock::now();
        for (size_t ci = 0; ci < chunks; ++ci) {
            int32_t* data = blk.fetch_chunk();
            size_t base = ci * blk.elems_per_chunk();
            // sum first & last of this chunk
            fc_sum += data[0];
            size_t len = min(blk.elems_per_chunk(), totalElems - base);
            fc_sum += data[len - 1];
        }
        auto t7 = steady_clock::now();
        double fetch_ms = duration_cast<duration<double, milli>>(t7 - t6).count();
        cout << "  fetch_chunk (" << chunks << "): " << fetch_ms
             << " ms  (sum=" << fc_sum << ")\n";

        // 6) prefetch_all()
        auto t8 = steady_clock::now();
        blk.prefetch_all();
        auto t9 = steady_clock::now();
        long rss_after_prefetch = getCurrentRSS();
        double prefetch_ms = duration_cast<duration<double, milli>>(t9 - t8).count();
        cout << "  prefetch_all: " << prefetch_ms
             << " ms, RSS: " << rss_after_prefetch << " KB\n\n";

        // Store result
        results.push_back({
                                  rpc,
                                  ctor_ms,
                                  rss_after_ctor,
                                  seq_ms,
                                  rand_ms,
                                  fetch_ms,
                                  prefetch_ms,
                                  rss_after_prefetch
                          });
    }

    // 7) Cleanup test file
    if (remove(filename.c_str()) == 0) {
        cout << "Removed test file '" << filename << "'.\n";
    } else {
        perror("Error removing test file");
    }

    // 8) Summary table
    cout << "\n=== Performance Summary ===\n"
         << setw(15) << "recordsPerChunk"
         << setw(10) << "ctor(ms)"
         << setw(12) << "RSS_ctor(KB)"
         << setw(12) << "seq(ms)"
         << setw(12) << "rand(ms)"
         << setw(12) << "fetch(ms)"
         << setw(12) << "prefetch(ms)"
         << setw(15) << "RSS_prefetch(KB)"
         << "\n";

    for (auto& r : results) {
        cout << setw(15) << r.recordsPerChunk
             << setw(10) << r.ctor_ms
             << setw(12) << r.rss_after_ctor_kb
             << setw(12) << r.seq_scan_ms
             << setw(12) << r.rand_access_ms
             << setw(12) << r.fetch_chunk_ms
             << setw(12) << r.prefetch_ms
             << setw(15) << r.rss_after_prefetch_kb
             << "\n";
    }

    cout << "\nAll Block performance tests passed and cleaned up.\n";
    return 0;
}