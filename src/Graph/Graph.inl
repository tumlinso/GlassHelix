#ifndef GLASSHELIX_GRAPH_INL
#define GLASSHELIX_GRAPH_INL

#include <cmath>

static inline const double ACCEPTABLE_TOLERANCE = 1e-6;
static inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

#endif //GLASSHELIX_GRAPH_INL
