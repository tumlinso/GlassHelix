#ifndef GLASSHELIX_GRAPH_HH
#define GLASSHELIX_GRAPH_HH

#include <vector>
#include <list>
#include <cmath>

static inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

struct Node {
public:
    double bias, value;
};

struct Link {
    double strength;
    Node *sink, *source;
};

class Graph {
public:
    std::vector<Node> nodes;
    std::list<Link> links;

    Graph() = default;

    Graph(size_t numNodes) {
        nodes.resize(numNodes);
        for (size_t i = 0; i < numNodes; ++i) {
            nodes[i].bias = 0.0;
            nodes[i].value = 0.0;
        }
    }

    void addLink(size_t sourceIndex, size_t sinkIndex, double strength) {
        if (sourceIndex >= nodes.size() || sinkIndex >= nodes.size()) {
            throw std::out_of_range("Node index out of range");
        }
        Link link{strength, &nodes[sinkIndex], &nodes[sourceIndex]};
        links.push_back(link);
    }

    void propagate() {
        for (auto& link : links) {
            double input = link.source->value + link.source->bias;
            link.sink->value += sigmoid(input) * link.strength;
        }
    }

};

#endif //GLASSHELIX_GRAPH_HH
