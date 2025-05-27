#ifndef GLASSHELIX_DATASET_HH
#define GLASSHELIX_DATASET_HH

#include <c++/14/string>

#include "Dictionary.hh"
#include "AsynchronousArray.hh"
#include "src/Cell.hh"
namespace glasshelix {


    template<typename T, size_t recordLength = 2048>
    class Dataset {
    private:
        unsigned long maxTranscripts;
    public:
        Dictionary<std::string, T> transcriptDictionary;
        AsynchronousArray<T> transcriptBlock;

        Dataset(std::string pathTokenDict, std::string pathTokenData, unsigned long numCells)
                : maxTranscripts(maxTranscripts),
                  transcriptDictionary(pathTokenDict),
                  transcriptBlock(pathTokenData, recordLength, numCells, 100'000) {}

        Cell<T, recordLength> operator[](unsigned long index) {
            if (index >= transcriptBlock.size()) {
                throw std::out_of_range("Index out of range");
            }
            return Cell<T, recordLength>(transcriptBlock[index], index);
        }
    };

// for test: 15,229,000 records, 2048 length, unsigned short, max token 25425
// path to binary file: /home/tumlinson/GlassHelix/data/dataset.bin
// path to dictionary file: /home/tumlinson/GlassHelix/data/token_dictionary.csv

}

#endif //GLASSHELIX_DATASET_HH
