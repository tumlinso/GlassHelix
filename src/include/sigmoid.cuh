#ifndef GLASSHELIX_SIGMOID_CUH
#define GLASSHELIX_SIGMOID_CUH

#include "macros.cuh"

radical double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

#endif //GLASSHELIX_SIGMOID_CUH
