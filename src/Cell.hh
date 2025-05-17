#ifndef GLASSHELIX_CELL_HH
#define GLASSHELIX_CELL_HH

#include <iostream>
#include <vector>
#include <string>

template<typename T, unsigned long length>
struct Cell {
    T *tokens = nullptr;

    Cell() { tokens = new T[length]; }
};

#endif //GLASSHELIX_CELL_HH
