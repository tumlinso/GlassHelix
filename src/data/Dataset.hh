#ifndef GLASSHELIX_DATASET_HH
#define GLASSHELIX_DATASET_HH

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

#include "Dictionary.hh"

static inline const unsigned long l3cacheMB = 16;
static inline constexpr unsigned long chunkSize = l3cacheMB * 1024 * 1024;
static inline const maxToken = 25425;

template<typename T>
class Dataset {
private:
    std::string pathTokenDict;
    std::string pathData;

    void readTokenDictionary(std::string filename);
    void readData(std::string filename);
public:
    Dictionary<std::string, unsigned short> tokenDict;

    Dataset(std::string pathTokenDict, std::string pathData) : pathTokenDict(pathTokenDict), pathData(pathData) {
        readTokenDictionary(pathTokenDict);
        readData(pathData);
    }
};


#endif //GLASSHELIX_DATASET_HH
