#ifndef GLASSHELIX_NETWORKLOGIC_HH
#define GLASSHELIX_NETWORKLOGIC_HH

#include "Network.hh"
#include "src/include/sigmoid.h"

namespace glasshelix::network::logic {
    inline Network buildANDGate() {
        // AND gate network with 2 inputs and 1 output
        Network andGate(3, 2, sigmoid);
        andGate.bis[2] = -15.0; // bias output, logic low until activation from both inputs
        // links: input1 → output, input2 → output
        // one active -> activation output = -5 -> sigmoid makes value near 0
        // both active -> activation output = 5 -> sigmoid makes value near 1
        andGate.str[0] = 10.0; // input1 to output
        andGate.str[1] = 10.0; // input2 to output
        return andGate;
    }

    inline Network buildORGate() {
        // OR gate network with 2 inputs and 1 output
        Network orGate(3, 2, sigmoid);
        orGate.pls[2] = -5.0; // bias output, logic low until activation from either input
        // links: input1 → output, input2 → output
        // either or both active -> activation output = 5/15 -> value near 1
        // neither active -> activation output = -5 -> value near 0
        orGate.str[0] = 10.0; // input1 to output
        orGate.str[1] = 10.0; // input2 to output
        return orGate;
    }

    inline Network buildNANDGate() {
        // NAND gate network with 2 inputs and 1 output
        Network nandGate(3, 2, sigmoid);
        nandGate.pls[2] = 15.0; // bias output, logic high until activation from both inputs
        // links: input1 → output, input2 → output
        // both active -> activation output = -5 -> sigmoid makes value near 0
        // either or both active -> activation output = 5 -> sigmoid makes value near 1
        nandGate.str[0] = -10.0; // input1 to output
        nandGate.str[1] = -10.0; // input2 to output
        return nandGate;
    }




} // namespace glasshelix::network::logic

#endif //GLASSHELIX_NETWORKLOGIC_HH
