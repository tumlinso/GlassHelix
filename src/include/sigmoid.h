#ifndef GLASSHELIX_SIGMOID_H
#define GLASSHELIX_SIGMOID_H

#include "macros.h"
#include <math.h>

__inlRadical__ double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
__inlRadical__ void sigmoid(double *in, double *out, unsigned int len) {
    for (unsigned int i = 0; i < len; ++i) {
        out[i] = sigmoid(in[i]);
    }
}

#endif //GLASSHELIX_SIGMOID_H
