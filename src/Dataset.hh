#ifndef GLASSHELIX_DATASET_HH
#define GLASSHELIX_DATASET_HH

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

#include "Dictionary.hh"
#include "Block.hh"
#include "Cell.hh"

class Dataset {
private:
    std::string pathTranscriptDictionary;
    std::string pathTranscriptData;
    std::string pathChromatinDictionary;
    std::string pathChromatinData;

    unsigned long maxTranscriptRecords = 0;
    unsigned long maxChromatinRecords = 0;
public:
    Dictionary<std::string, unsigned short> transcriptDictionary;
    Block<unsigned short> transcriptBlock;

    Dictionary<std::string, unsigned short> chromatinDictionary;
    Block<unsigned short> chromatinBlock;


};


#endif //GLASSHELIX_DATASET_HH
