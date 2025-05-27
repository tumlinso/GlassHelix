// DictionaryTest.cpp
#include "src/memory/Dictionary.hh"

#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <filesystem>

using Clock = std::chrono::high_resolution_clock;
using ms    = std::chrono::duration<double, std::milli>;

// --- Utility: generate N random lowercase strings of given length
std::vector<std::string> generate_random_strings(std::size_t n, std::size_t length) {
    std::vector<std::string> v;
    v.reserve(n);
    // libc++ only allows certain integer types here, so use int and cast to char
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> dist('a','z');
    for (std::size_t i = 0; i < n; ++i) {
        std::string s;
        s.reserve(length);
        for (std::size_t j = 0; j < length; ++j)
            s.push_back(static_cast<char>(dist(rng)));
        v.push_back(std::move(s));
    }
    return v;
}

// --- Test Dictionary<std::string, size_t> (integral U → uses vector reverse)
void test_string_int(std::size_t n) {
    std::cout << "\n--- Dictionary<std::string, size_t> with " << n << " entries ---\n";
    auto keys = generate_random_strings(n, 12);
    Dictionary<std::string, size_t> dict;
    dict.reserve(n);

    // 1) measure dict.generate()
    {
        auto t0 = Clock::now();
        for (auto& k : keys) dict.generate(k);
        auto t1 = Clock::now();
        std::cout << "generate():             " << ms(t1-t0).count() << " ms\n";
    }

    // 2) lookup by key
    {
        std::mt19937_64 rng(1234);
        std::uniform_int_distribution<std::size_t> pick(0, n-1);
        auto t0 = Clock::now();
        std::size_t hits = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const auto& k = keys[pick(rng)];
            if (dict.contains(k)) {
                volatile auto id = dict.get(k);
                ++hits;
            }
        }
        auto t1 = Clock::now();
        std::cout << "lookup by key:          " << ms(t1-t0).count()
                  << " ms (hits=" << hits << ")\n";
    }

    // 3) reverse lookup by value
    {
        std::mt19937_64 rng(1234);
        std::uniform_int_distribution<std::size_t> pick(0, n-1);
        auto t0 = Clock::now();
        std::size_t hits = 0;
        for (std::size_t i = 0; i < n; ++i) {
            auto id = pick(rng);
            if (dict.contains(id)) {
                volatile auto k = dict.get(id);
                ++hits;
            }
        }
        auto t1 = Clock::now();
        std::cout << "reverse lookup by val:  " << ms(t1-t0).count()
                  << " ms (hits=" << hits << ")\n";
    }

    // 4) compose: build a Dictionary<int, string> and compose → Dictionary<int, size_t>
    {
        Dictionary<int, std::string> back;
        back.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            back.emplace(int(i), keys[i]);

        auto t0 = Clock::now();
        auto composed = dict.compose(back);  // V=int, T=string
        auto t1 = Clock::now();
        assert(composed.size() == n);
        std::cout << "compose(back):          " << ms(t1-t0).count()
                  << " ms (size=" << composed.size() << ")\n";
    }
}

// --- Test Dictionary<int, std::string> (non-integral U → uses unordered_map reverse)
void test_int_string(std::size_t n) {
    std::cout << "\n--- Dictionary<int, std::string> with " << n << " entries ---\n";
    auto values = generate_random_strings(n, 12);
    Dictionary<int, std::string> dict;
    dict.reserve(n);

    // 1) insertion via emplace()
    {
        auto t0 = Clock::now();
        for (std::size_t i = 0; i < n; ++i)
            dict.emplace(int(i), values[i]);
        auto t1 = Clock::now();
        std::cout << "emplace():               " << ms(t1-t0).count() << " ms\n";
    }

    // 2) lookup by key
    {
        std::mt19937_64 rng(1234);
        std::uniform_int_distribution<int> pick(0, int(n-1));
        auto t0 = Clock::now();
        std::size_t hits = 0;
        for (std::size_t i = 0; i < n; ++i) {
            int k = pick(rng);
            if (dict.contains(k)) {
                volatile auto v = dict.get(k);
                ++hits;
            }
        }
        auto t1 = Clock::now();
        std::cout << "lookup by key:          " << ms(t1-t0).count()
                  << " ms (hits=" << hits << ")\n";
    }

    // 3) reverse lookup by value
    {
        std::mt19937_64 rng(1234);
        std::uniform_int_distribution<std::size_t> pick(0, n-1);
        auto t0 = Clock::now();
        std::size_t hits = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const auto& v = values[pick(rng)];
            if (dict.contains(v)) {
                volatile auto k = dict.get(v);
                ++hits;
            }
        }
        auto t1 = Clock::now();
        std::cout << "reverse lookup by val:  " << ms(t1-t0).count()
                  << " ms (hits=" << hits << ")\n";
    }

    // 4) compose: build Dictionary<std::string, double> and compose → Dictionary<int, double>
    {
        Dictionary<std::string, double> next;
        next.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            next.emplace(values[i], double(i) * 0.5);

        auto t0 = Clock::now();
        auto composed = dict.compose(next);
        auto t1 = Clock::now();
        assert(composed.size() == n);
        std::cout << "compose(next):          " << ms(t1-t0).count()
                  << " ms (size=" << composed.size() << ")\n";
    }
}

// --- Test readFromFile (both with and without header) ---
void test_readFromFile(std::size_t n) {
    std::cout << "\n--- readFromFile with " << n << " entries ---\n";
    // generate Dataset
    auto keys = generate_random_strings(n, 12);
    std::string file_no_hdr = "dict_no_header_" + std::to_string(n) + ".csv";
    std::string file_hdr    = "dict_with_header_" + std::to_string(n) + ".csv";

    // write no-header
    {
        std::ofstream ofs(file_no_hdr);
        for (std::size_t i = 0; i < n; ++i)
            ofs << keys[i] << "," << i << "\n";
    }
    // write with header
    {
        std::ofstream ofs(file_hdr);
        ofs << "word,id\n";
        for (std::size_t i = 0; i < n; ++i)
            ofs << keys[i] << "," << i << "\n";
    }

    // read no header
    {
        Dictionary<std::string, size_t> dict;
        auto t0 = Clock::now();
        dict.readFromFile(file_no_hdr, false, n);
        auto t1 = Clock::now();
        assert(dict.size() == n);
        std::cout << "readFromFile(no header):  " << ms(t1-t0).count() << " ms\n";
    }
    // read with header skip
    {
        Dictionary<std::string, size_t> dict;
        auto t0 = Clock::now();
        dict.readFromFile(file_hdr, true, n);
        auto t1 = Clock::now();
        assert(dict.size() == n);
        std::cout << "readFromFile(skip header): " << ms(t1-t0).count() << " ms\n";
    }

    // ---- CLEANUP ----
    std::filesystem::remove(file_no_hdr);
    std::filesystem::remove(file_hdr);
}

int main() {
    std::vector<std::size_t> sizes = {1'000, 10'000, 100'000, 1'000'000};
    for (auto n : sizes) {
        test_string_int(n);
        test_int_string(n);
        test_readFromFile(n);
    }
    return 0;
}