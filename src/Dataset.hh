#ifndef GLASSHELIX_DATASET_HH
#define GLASSHELIX_DATASET_HH

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "Dictionary.hh"
#include "Block.hh"
#include "Cell.hh"

template<typename T, size_t recordLength = 2048>
class Dataset {
private:
    std::string pathTranscriptDictionary;
    std::string pathTranscriptData;

    unsigned long maxTranscripts;
public:
    Dictionary<std::string, T> transcriptDictionary;
    Block<T> transcriptBlock;

    Dataset(std::string pathTokenDict, std::string pathTokenData, unsigned long numCells)
        : pathTranscriptDictionary(std::move(pathTokenDict)), pathTranscriptData(std::move(pathTokenData)), maxTranscripts(maxTranscripts) {
        transcriptDictionary.readFromFile(pathTranscriptDictionary);
        transcriptBlock = Block<T>(pathTranscriptData, recordLength, numCells);
    }


};

// for test: 15,229,000 records, 2048 length, unsigned short, max token 25425
// path to binary file: /home/tumlinson/GlassHelix/data/dataset.bin
// path to dictionary file: /home/tumlinson/GlassHelix/data/token_dictionary.csv

#endif //GLASSHELIX_DATASET_HH
