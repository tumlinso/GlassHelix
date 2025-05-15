#ifndef GLASSHELIX_CELL_HH
#define GLASSHELIX_CELL_HH

#include <iostream>
#include <vector>
#include <string>

static const unsigned long length = 1000;

template<typename TranscriptTokenType>
class Cell {
public:
    Cell() = default;
    TranscriptTokenType *tokens;

    std::vector<std::string> toTranscripts;
};

#include "Cell.inl"

#endif //GLASSHELIX_CELL_HH
