export module Graph;

import <cstddef>;
import <cmath>;
import <algorithm>;

export class Node;
export class Graph;

inline static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

export class Link {
public:
    Link *nextIn, *nextOut;
    Node *source, *sink;
    double weight;

    explicit Link(Node *source, Node *sink, double weight = 0.0);
    ~Link();

    inline bool operator==(const Link &other) const;
};

export class Node {
public:
    Graph *graph;
    Node *next = nullptr;
    Link *in = nullptr, *out = nullptr;

    const long id;
    double bias, value;

    explicit Node(Graph *graph, double bias = -5.0, double value = 0.0, double decay = 0.0);
    ~Node();

    [[nodiscard]] inline double activation(double delta = 0.0) const;
    inline Link *operator[](unsigned long sinkID) const;
    inline Link *operator()(unsigned long sourceID) const;
    inline bool operator==(const Node &other) const;

    [[maybe_unused]] [[nodiscard]] inline unsigned long size() const;
};

export class Graph {
public:
    Node *head;
    long nextID;

    Graph() : head(nullptr), nextID(0) {}
    ~Graph();

    inline Node *operator[](unsigned long id) const;
    inline Link *operator()(unsigned long sourceID, unsigned long sinkID) const;
    inline bool operator==(const Graph &other) const;

    [[nodiscard]] inline unsigned long size() const;
};

// Inline definitions
static inline const double ACCEPTABLE_TOLERANCE = 1e-6;

// for class Link:
inline bool Link::operator==(const Link &other) const {
    return source->id == other.source->id && sink->id == other.sink->id
           && std::abs(weight - other.weight) < ACCEPTABLE_TOLERANCE;
}

inline Link *Node::operator[](unsigned long sinkID) const {
    Link *l = out;
    while (l) {
        if (l->sink->id == sinkID) return l;
        l = l->nextOut;
    }
    return nullptr;
}

inline Link *Node::operator()(unsigned long sourceID) const {
    Link *l = in;
    while (l) {
        if (l->source->id == sourceID) return l;
        l = l->nextIn;
    }
    return nullptr;
}

// for class Node:
inline bool Node::operator==(const Node &other) const {
    return id == other.id && std::abs(bias - other.bias) < ACCEPTABLE_TOLERANCE;
}

inline double Node::activation(double delta) const {
    double sum = delta;
    Link *l = in;
    while (l) {
        sum += l->source->value * l->weight;
        l = l->nextIn;
    }
    sum += bias;
    return sigmoid(sum);
}

[[maybe_unused]] inline unsigned long Node::size() const {
    long count = 0;
    for (Link *l = out; l; l = l->nextOut) ++count;
    return count;
}

// for class Graph:
inline Node *Graph::operator[](unsigned long id) const {
    Node *n = head;
    while (n) {
        if (n->id == id) return n;
        n = n->next;
    }
    return nullptr;
}

inline Link *Graph::operator()(unsigned long sourceID, unsigned long sinkID) const {
    Node *source = this->operator[](sourceID);
    Link *l = source->out;
    while (l) {
        if (l->sink->id == sinkID) return l;
        l = l->nextOut;
    }
    return nullptr;
}

inline bool Graph::operator==(const Graph &other) const {
    Node *n = head, *m = other.head;
    while (n && m) {
        // check if nodes are equal
        if (!((*n) == (*m))) return false;
        n = n->next;
        m = m->next;
    }
    return n == nullptr && m == nullptr;
}

[[maybe_unused]] inline unsigned long Graph::size() const {
    long count = 0;
    for (Node *n = head; n; n = n->next) ++count;
    return count;
}

// Constructors and destructors
Link::Link(Node *source, Node *sink, double weight)
        : weight(weight), source(source), sink(sink), nextIn(nullptr), nextOut(nullptr) {
    [[unlikely]] if (source == nullptr || sink == nullptr) abort(); // Invalid source or sink
    [[unlikely]] if ((*source)[sink->id] != nullptr) abort(); // Link already exists
    // Insert into source's outgoing links
    nextOut = source->out;
    source->out = this;

    // Insert into sink's incoming links
    nextIn = sink->in;
    sink->in = this;
}

Link::~Link() {
    // Remove from source->out list
    if (source->out == this) {
        source->out = nextOut;
    } else {
        for (Link *l = source->out; l != nullptr; l = l->nextOut) {
            if (l->nextOut == this) {
                l->nextOut = nextOut;
                break;
            }
        }
    }

    // Remove from sink->in list
    if (sink->in == this) {
        sink->in = nextIn;
    } else {
        for (Link *l = sink->in; l != nullptr; l = l->nextIn) {
            if (l->nextIn == this) {
                l->nextIn = nextIn;
                break;
            }
        }
    }
}

Node::Node(Graph *graph, double bias, double value, double decay)
        : graph(graph), bias(bias), id(graph->nextID++),
        in(nullptr), out(nullptr), next(nullptr), value(value) {
    next = graph->head;
    graph->head = this;

    this->value = std::clamp(value, 0.0, 1.0); // value is clamped to [0, 1]
    new Link(this, this, std::clamp(decay, 0.0, 1.0)); // all nodes have a self link with weight = decay rate, clamped to [0, 1]
}

Node::~Node() {
    // Delete all incoming links
    while (in) {
        Link *l = in;
        in = in->nextIn;
        delete l;
    }

    // Delete all outgoing links
    while (out) {
        Link *l = out;
        out = out->nextOut;
        delete l;
    }

    // Remove from graph's node list
    if (graph->head == this) {
        graph->head = next;
    } else {
        for (Node *n = graph->head; n != nullptr; n = n->next) {
            if (n->next == this) {
                n->next = next;
                break;
            }
        }
    }
}

Graph::~Graph() {
    for (Node *n = head; n != nullptr; n = head) {
        head = head->next;
        delete n;
    }
}
