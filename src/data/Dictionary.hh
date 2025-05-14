#ifndef GLASSHELIX_DICTIONARY_HH
#define GLASSHELIX_DICTIONARY_HH

#include <string>
#include "BidirectionalMap.hh"

// explicit use case for id dictionary
template <typename T>
class Dictionary : virtual public BidirectionalMap<T, unsigned long> {
public:
    explicit Dictionary() : BidirectionalMap<T, unsigned long>() {}

    unsigned long generate(T t) {
        if (this->contains(t)) {
            return this->forward[t];
        } else {
            unsigned long id = this->size();
            this->emplace(t, id);
            return id;
        }
    }
};

#endif //GLASSHELIX_DICTIONARY_HH
