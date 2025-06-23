#ifndef GLASSHELIX_NETWORK_HH
#define GLASSHELIX_NETWORK_HH

#include <stdexcept>
#include <memory>

namespace glasshelix::network {
    struct Node { double *bias, *value; };
    struct Link { double *strength; Node sink, source; };

    struct Network {
        unsigned int nNodes, nLinks;
        double *act, *val, *bis; // Node components
        double *str; // link strengths
        unsigned int *src, *snk; // link source and sink indices
        void (*activateFx)(double in, double *out);
        inline void activateFn(double *in, double *out, unsigned int len) {
            for (unsigned int i = 0; i < len; ++i) { activateFx(in[i], &out[i]); }}

        void allocate();
        static void copy(const Network *src, Network *dst, unsigned int offset = 0);
        static Network *compose(const Network *n1, const Network *n2,
                                unsigned int srcN1, unsigned int snkN2, double lnStr);

        Network(unsigned int nNodes, unsigned int nLinks,
                void (*actFn)(double in, double *out));
        Network(const Network &other);
        ~Network();

        inline void step();
        inline Node getNode(unsigned int idx);
        inline Link getLink(unsigned int idx);
    };

    class Graph {
    private:
        struct _Node : Node { _Node *next; double _bis, _val; };
        struct _Link : Link { _Link *next; double _str; unsigned int isrc, isnk; };

        _Node *nodes = nullptr;
        _Link *links = nullptr;
        void (*activateFx)(double in, double *out);
    public:
        Graph(void (*actFx)(double in, double *out) = nullptr)
                : activateFx(actFx) {}
        ~Graph();

        Node &getNode(unsigned int idx);
        Node *addNode(double bias = 0.0, double value = 0.0);
        void removeNode(Node *node);

        Link &getLink(unsigned int idx);
        Link *addLink(Node *src, Node *snk, double strength = 1.0);
        void removeLink(Link *link);

        void concatenate(Graph *other, Node *srcNode, Node *snkNode, double linkStrength = 1.0);
        Network compile();
    };

    inline void Network::step() {
        std::memset(act, 0, nNodes * sizeof(double));
        for (unsigned int i = 0; i < nLinks; ++i) {
            unsigned int isrc = src[i];
            unsigned int isnk = snk[i];
            act[isnk] += val[isrc] * str[i];
        }
        activateFn(act, val, nNodes);
    }

    inline Node Network::getNode(unsigned int idx) {
        if (idx >= nNodes) { throw std::out_of_range("Node index out of range"); }
        return Node{&bis[idx], &val[idx]};
    }

    inline Link Network::getLink(unsigned int idx) {
        if (idx >= nLinks) { throw std::out_of_range("Link index out of range"); }
        return Link{&str[idx], Node{&bis[src[idx]], &val[src[idx]]} ,
                    Node{&bis[snk[idx]], &val[snk[idx]]}};
    }

} // namespace glasshelix::network

#endif //GLASSHELIX_NETWORK_HH
