#ifndef GLASSHELIX_GRAPH_HH
#define GLASSHELIX_GRAPH_HH

#include <vector>
#include <string>

struct Link;

struct Node {
    std::vector<Link *> out;
    const double bias;
    double value;
    Node(double bias, double value)
            : bias(bias), value(value) {}
    virtual void link(Node &sink, double bias, double value);
};

struct Link : Node {
    Node &source, &sink;
    explicit Link(Node &source, Node &sink, double bias, double value)
            : Node(bias, value), source(source), sink(sink) {}
};

class Graph : public std::vector<Node> {
    virtual void link(Node &source, Node &sink, double bias, double value);
};

#include "Graph.inl"

#endif //GLASSHELIX_GRAPH_HH
