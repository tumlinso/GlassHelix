#ifndef GLASSHELIX_CELL_HH
#define GLASSHELIX_CELL_HH

#include <cstddef>
#include <string>
#include <vector>

#include "Dictionary.hh"

template<typename T, std::size_t length>
struct Cell {
    T* tokens = nullptr;         ///< pointer into the underlying block buffer
    std::size_t offset = 0;      ///< absolute offset (in tokens) from start of block

    inline T& operator[](std::size_t idx) const noexcept { return tokens[idx]; }
    static constexpr std::size_t size() noexcept { return length; }
    inline T* begin() const noexcept { return tokens; }
    inline T* end() const noexcept { return tokens + length; }

    std::vector<std::string> transcripts(const Dictionary<std::string, T>& dict) const {
        return dict.translate(tokens, length);
    }
};

#endif //GLASSHELIX_CELL_HH
