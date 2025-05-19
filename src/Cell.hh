#ifndef GLASSHELIX_CELL_HH
#define GLASSHELIX_CELL_HH

#include <iostream>
#include <vector>
#include <string>

template<typename T, unsigned long length>
struct Cell {
    T *transcripts = nullptr;
    T *chromatin = nullptr;
    unsigned long transcriptOffset = 0;
    unsigned long chromatinOffset = 0;
};

#endif //GLASSHELIX_CELL_HH
