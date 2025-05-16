#ifndef GLASSHELIX_CELL_HH
#define GLASSHELIX_CELL_HH

#include <iostream>
#include <vector>
#include <string>

template<typename T>
struct Cell {
    static inline const unsigned long length;
    T *tokens;

    Cell() = default;
};

#include "Cell.inl"

#endif //GLASSHELIX_CELL_HH
