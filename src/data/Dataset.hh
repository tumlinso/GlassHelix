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

class Dataset {
private:
    void readTokenDictionary(std::string filename);
public:
    Dictionary<std::string> Tokens;

    Dataset(std::string PathTokenDict);
};


#endif //GLASSHELIX_DATASET_HH
