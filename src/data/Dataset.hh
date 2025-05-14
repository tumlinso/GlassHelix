#ifndef GLASSHELIX_DATASET_HH
#define GLASSHELIX_DATASET_HH

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <iostream>
#include <string>

#include "Dictionary.hh"

class Dataset {
private:
    void readTokenDictionary(std::string filename);
public:
    Dictionary<std::string> Tokens;

    Dataset(std::string PathTokenDict);

};


#endif //GLASSHELIX_DATASET_HH
