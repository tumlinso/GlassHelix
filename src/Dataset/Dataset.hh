#ifndef GLASSHELIX_DATASET_HH
#define GLASSHELIX_DATASET_HH

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

#include "Dictionary.hh"
#include "Block.hh"

template<typename T>
class Dataset {
public:
    Dictionary<std::string, T> dictionary;
    Block<T> block;

    Dataset(std::string pathTokenDict, std::string pathData) {
        readTokenDictionary(pathTokenDict);
        readData(pathData);
    }
};


#endif //GLASSHELIX_DATASET_HH
