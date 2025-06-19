#ifndef GLASSHELIX_NETWORK_HH
#define GLASSHELIX_NETWORK_HH

#include <stdexcept>
#include <memory>
#include "src/include/sigmoid.h"

class Network;

struct Node {
    Network *net;
    unsigned int idx;
    inline double &pls();
    inline double &val();
};

struct Link {
    Network *net;
    unsigned int idx, src, snk;
    inline double &str();
};

class Network {
    friend struct Node;
    friend struct Link;
protected:
    unsigned int nNodes, nLinks;
    double          *act, *val, *pls; // node activations, values, biases
    double          *str; // link strengths
    unsigned int    *src, *snk; // link sources and sinks
    void (*actFn)(double *in, double *out, unsigned int len);

    inline void allocate() {
        act = new double        [nNodes]();
        val = new double        [nNodes]();
        pls = new double        [nNodes]();
        str = new double        [nLinks]();
        src = new unsigned int  [nLinks]();
        snk = new unsigned int  [nLinks]();
    }

    static void copy(const Network *src, Network *dst, unsigned int offset = 0) {
        if (src->nNodes + offset > dst->nNodes || src->nLinks > dst->nLinks) {
            throw std::out_of_range("Source network exceeds destination capacity"); }
        std::memcpy(dst->act + offset, src->act, src->nNodes * sizeof(double));
        std::memcpy(dst->val + offset, src->val, src->nNodes * sizeof(double));
        std::memcpy(dst->pls + offset, src->pls, src->nNodes * sizeof(double));
        std::memcpy(dst->str + offset, src->str, src->nLinks * sizeof(double));
        std::memcpy(dst->src + offset, src->src, src->nLinks * sizeof(unsigned int));
        std::memcpy(dst->snk + offset, src->snk, src->nLinks * sizeof(unsigned int));
    }

    static Network *concat(Network *n1,
                           Network *n2,
                           unsigned int srcN1,
                           unsigned int snkN2,
                           double lnStr) {
        if (n1->actFn != n2->actFn) {
            throw std::invalid_argument("Networks must have the same activation function"); }
        auto result = new Network(n1->nNodes + n2->nNodes, n1->nLinks + n2->nLinks + 1, n1->actFn);
        copy(n1, result);
        copy(n2, result, n1->nNodes);
        result->str[result->nLinks] = lnStr;
        result->src[result->nLinks] = srcN1;
        result->snk[result->nLinks] = n1->nNodes + snkN2;
        return result;
    }
public:
    Network(unsigned int nNodes,
            unsigned int nLinks,
            void (*actFn)(double *in, double *out, unsigned int len) = sigmoid)
        : nNodes(nNodes), nLinks(nLinks), actFn(actFn) {
        allocate();
    }

    Network(const Network &other)
        : nNodes(other.nNodes), nLinks(other.nLinks), actFn(other.actFn) {
        allocate();
        copy(&other, this);
    }

    ~Network() {
        delete[] act;
        delete[] val;
        delete[] pls;
        delete[] src;
        delete[] snk;
        delete[] str;
    }

    inline void update() {
        std::memset(act, 0, nNodes * sizeof(double));
        for (unsigned int i = 0; i < nLinks; ++i) {
            unsigned int isrc = src[i];
            unsigned int isnk = snk[i];
            act[isnk] += val[isrc] * str[i];
        }
        actFn(act, val, nNodes);
    }

    Node getNode(unsigned int idx) {
        if (idx >= nNodes) { throw std::out_of_range("Node index out of range"); }
        return Node{this, idx};
    }

    Link getLink(unsigned int idx) {
        if (idx >= nLinks) { throw std::out_of_range("Link index out of range"); }
        return Link{this, idx, src[idx], snk[idx]};
    }
};

double &Node::pls() { return *(net->pls + idx); }
double &Node::val() { return *(net->val + idx); }
double &Link::str() { return *(net->str + idx); }

#endif //GLASSHELIX_NETWORK_HH
