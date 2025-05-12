export module Graph;

// Trimmed down version of Graph.h/Graph.cpp/Graph.inl from original in project GraphMap

import <cstddef>;
import <cmath>;
import <vector>;

export struct Link;

static inline const double ACCEPTABLE_TOLERANCE = 1e-6;
static inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

export struct Node {
    std::vector<Link *> out;
    const double bias;
    double value;
    Node(double bias, double value)
            : bias(bias), value(value) {}
    virtual void link(Node &sink, double bias, double value);
};

export struct Link : Node {
    Node &source, &sink;
    explicit Link(Node &source, Node &sink, double bias, double value)
            : Node(bias, value), source(source), sink(sink) {}
};

export class Graph : public std::vector<Node> {
    virtual void link(Node &source, Node &sink, double bias, double value);
};
