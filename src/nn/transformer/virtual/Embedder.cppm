module;
#include <torch/torch.h>

export module Embedder;

export struct Embedder : public torch::nn::Module {
    torch::nn::Embedding Embedding{nullptr};
    torch::nn::Linear Unembedding{nullptr};

    explicit Embedder(int64_t vocabSize, int64_t embeddingSize);
    torch::Tensor forward(const torch::Tensor& input);
};

Embedder::Embedder(int64_t vocabSize, int64_t embeddingSize) {
    Embedding = register_module("Embedding", torch::nn::Embedding(vocabSize, embeddingSize));
    Unembedding = register_module("Unembedding", torch::nn::Linear(embeddingSize, vocabSize));
}

torch::Tensor Embedder::forward(const torch::Tensor& input) {
    auto x = Embedding->forward(input);
    return Unembedding->forward(x);
}