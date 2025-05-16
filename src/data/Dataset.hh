#ifndef GLASSHELIX_DATASET_HH
#define GLASSHELIX_DATASET_HH

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

#include "Dictionary.hh"

template<typename T>
class Dataset {
private:
    void readTokenDictionary(std::string filename);
    void readData(std::string filename);
public:
    Dictionary<std::string, unsigned short> tokenDict;

    Dataset(std::string pathTokenDict, std::string pathData) {
        readTokenDictionary(pathTokenDict);
        readData(pathData);
    }
};


#endif //GLASSHELIX_DATASET_HH
