// SequenceTest.cpp
// ----------------------------------------------------------------------------
// Build with:
//   clang++ -std=c++17 -stdlib=libc++ -O2 SequenceTest.cpp -o SequenceTest
// ----------------------------------------------------------------------------

#include <iostream>
#include <string>
#include <random>
#include <cassert>
#include <chrono>

#include "../../Sequence.hh"
using Clock = std::chrono::steady_clock;

int main() {
    // test parameters
    const size_t N = 5'000'000;       // sequence length
    const int    M = 100'000;         // random-access trials

    // prepare random nucleotide string
    std::string bases = "ACGT";
    std::mt19937_64 rng(20250517);
    std::uniform_int_distribution<int> dist4(0, 3);

    std::string reference;
    reference.reserve(N);
    for (size_t i = 0; i < N; ++i)
        reference.push_back(bases[dist4(rng)]);

    // 1) Construct Sequence<> and time it
    auto t0 = Clock::now();
    Sequence<> seq(reference);
    auto t1 = Clock::now();
    auto ms_ctor = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Constructor: built Sequence<>(" << N << ") in " << ms_ctor << " ms\n";

    // 2) Round-trip toString
    auto t2 = Clock::now();
    std::string roundtrip = Sequence<>::toString(seq);
    auto t3 = Clock::now();
    assert(roundtrip == reference);
    std::cout << "toString round-trip OK ("
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << " ms)\n";

    // 3) Random-access profiling
    std::uniform_int_distribution<size_t> distN(0, N - 1);
    size_t mismatches = 0;
    auto t4 = Clock::now();
    for (int i = 0; i < M; ++i) {
        size_t pos = distN(rng);
        char c = Sequence<>::toChar(seq(pos));
        if (c != reference[pos]) ++mismatches;
    }
    auto t5 = Clock::now();
    auto ms_rand = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();
    std::cout << "Random access (" << M << " ops): " << ms_rand << " ms"
              << ", mismatches=" << mismatches << "\n";

    // 4) Sequential scan profiling
    uint64_t checksum = 0;
    auto t6 = Clock::now();
    for (size_t i = 0; i < N; ++i) {
        checksum += seq(i);
    }
    auto t7 = Clock::now();
    auto ms_seq = std::chrono::duration_cast<std::chrono::milliseconds>(t7 - t6).count();
    std::cout << "Sequential scan (" << N << " bases): " << ms_seq << " ms"
              << ", checksum=" << checksum << "\n";

    // 5) buffer() and unitCount()
    size_t units = seq.unitCount();
    std::cout << "unitCount() = " << units
              << ", buffer() non-null: " << (seq.buffer() != nullptr) << "\n";

    // 6) Test setter operator()
    Sequence<> copy = seq;  // copy constructor from Feature inherited
    for (int i = 0; i < 1000; ++i) {
        size_t pos = distN(rng);
        seq(pos, Sequence<>::A);
        assert(seq(pos) == Sequence<>::A);
    }
    std::cout << "Base setter operator() OK\n";

    // 7) Test subsequence get and set
    Feature feat(100, 200, Feature::Strand::UNDEFINED);
    Sequence<> window = seq[feat];
    assert(window.length() == feat.length());
    std::string winStr = Sequence<>::toString(window);
    assert(winStr == reference.substr(feat.start, feat.length()));
    std::cout << "Subsequence get OK\n";

    // set a subsequence to all 'C'
    Sequence<> allC(std::string(feat.length(), 'C'));
    seq[feat] = allC;
    std::string after = Sequence<>::toString(seq[feat]);
    assert(after == std::string(feat.length(), 'C'));
    std::cout << "Subsequence set OK\n";

    std::cout << "All Sequence tests passed.\n";
    return 0;
}