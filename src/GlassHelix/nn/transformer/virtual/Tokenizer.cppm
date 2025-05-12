export module Tokenizer;

import <string>;
import <unordered_map>;

export struct Tokenizer {

    auto rankGenes ();
    auto tokenize_cell()

};

struct dictionary {
    std::unordered_map<std::string, int> word_to_index;
    std::unordered_map<int, std::string> index_to_word;
};

