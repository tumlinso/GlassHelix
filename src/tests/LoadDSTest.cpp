#include "../memory/Dataset.hh"
#include <iostream>

int main() {
    Dataset<unsigned short, 2048> ds("/home/tumlinson/GlassHelix/data/token_dictionary.csv",
                                     "/home/tumlinson/GlassHelix/data/dataset.bin",
                                     15'229'000);

    for (int i = 0; i < 10; ++i) {
        auto cell = ds[i];
        auto transcripts = cell.transcripts(ds.transcriptDictionary);
        std::cout << "Cell " << i << " first 5: ";
        // print first 5 transcripts
        for (int j = 0; j < 5; ++j) {
            std::cout << transcripts[j] << " ";
        }
        std::cout << std::endl;
        std::cout << "Cell " << i << " last 5: ";
        for (int j = 0; j < 5; ++j) {
            std::cout << transcripts[transcripts.size() - 1 - j] << " ";
        }
        std::cout << std::endl;
    }
}