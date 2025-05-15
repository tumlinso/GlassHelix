#ifndef GLASSHELIX_CELL_HH
#define GLASSHELIX_CELL_HH

#include <iostream>
#include <vector>
#include <string>

static const unsigned long length = 1000;

template<typename TokenType>
class Cell {
public:
    Cell() = default;
    TokenType *tokens;
};

#include "Cell.inl"

#endif //GLASSHELIX_CELL_HH
