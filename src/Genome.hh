#ifndef GLASSHELIX_GENOME_HH
#define GLASSHELIX_GENOME_HH

#include "Sequence.hh"
#include "Feature.hh"
#include "Dictionary.hh"

// should be the default parent for most sequences
template<typename Word, typename T>
class Genome : public Sequence<Word> {
private:
    Dictionary<Sequence<Word>, std::string> *sequenceNameDictionary;
    Dictionary<Sequence<Word>, T> *sequenceTokenDictionary;

public:

};


#endif //GLASSHELIX_GENOME_HH
