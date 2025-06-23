#ifndef GLASSHELIX_SIGMOID_CUH
#define GLASSHELIX_SIGMOID_CUH

#include "macros.cuh"

radical double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
radical void sigmoid(double *in, double *out, unsigned int len) {
    for (unsigned int i = 0; i < len; ++i) {
        out[i] = sigmoid(in[i]);
    }
}

#endif //GLASSHELIX_SIGMOID_CUH
