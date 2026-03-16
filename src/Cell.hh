#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "types.cuh"

// sparse csr
template<typename T, std::size_t length>
class Cell {
private:

public:
    T *tokens = nullptr;         ///< pointer into the underlying buffer
    std::size_t offset = 0;      ///< absolute offset (in tokens) from start of block

    Cell(T *tokens, std::size_t offset) : tokens(tokens), offset(offset) {}

    inline T &operator[](std::size_t idx) const noexcept { return tokens[idx]; }

    static constexpr std::size_t size() noexcept { return length; }

    inline T *begin() const noexcept { return tokens; }

    inline T *end() const noexcept { return tokens + length; }

    std::vector<std::string> transcripts(const Dictionary<std::string, T> &dict) const {
        return dict.translate(tokens, length);
    }
};

#endif //GLASSHELIX_CELL_HH
