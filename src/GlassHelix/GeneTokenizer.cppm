// Created by Gavin Tumlinson on 4/27/25.
// All rights reserved.

export module GeneTokenizer;

import Tokenizer;

export struct GeneTokenizer : public Tokenizer {
private:
    auto rankGenes();
    auto tokenizeCell();

public:
    auto tokenize();
    auto createDataset();

};
