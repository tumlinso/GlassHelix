#include "../Dataset.hh"
#include "../Cell.hh"
#include "../Dictionary.hh"
#include "../Block.hh"
#include <iostream>

int main() {
    Dataset<unsigned short, 2048> ds("/home/tumlinson/GlassHelix/data/dataset.bin",
                                     "/home/tumlinson/GlassHelix/data/dataset.bin",
                                     15'229'000);

    for (int i = 0; i < 10; ++i) {
        auto cell = ds[i];



    }


}