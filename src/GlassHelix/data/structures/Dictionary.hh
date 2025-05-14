#ifndef GLASSHELIX_DICTIONARY_HH
#define GLASSHELIX_DICTIONARY_HH

#include <string>
#include "src/GlassHelix/data/structures/BidirectionalMap.hh"

template<typename T>
class Dictionary : virtual public IDMap<T> {
public:
    explicit Dictionary(std::string filename = "") {

    }
};

#endif //GLASSHELIX_DICTIONARY_HH
