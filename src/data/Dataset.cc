#include "Dataset.hh"

Dataset::Dataset(std::string pathTokenDict) :
    pathTokenDict(pathTokenDict) {
    readTokenDictionary(pathTokenDict);

}

void Dataset::readTokenDictionary(std::string filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Error opening Token Dictionary File: " + filename);

    std::string line;
    std::string token, index_string;
    while (std::getline(file, line)) {
        std::istringstream iss(line);

        if (!std::getline(iss, token, ',')) continue;
        if (!std::getline(iss, index_string, ',')) continue;

        unsigned long index = std::stoi(index_string);
        this->Tokens.emplace(token, index);
    }
    file.close();
}