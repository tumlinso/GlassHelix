export module Attention;

import <cstddef>;

export
template<typename T>
class Attention {
public:
    T Query{nullptr}, Key{nullptr}, Value{nullptr}, Output{nullptr};
    torch::Tensor queries, keys, values, weights;

    explicit Attention(int64_t embeddingDim);

    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& mask = {});
    torch::Tensor scaledDotProduct(const torch::Tensor &K, const torch::Tensor &Q,
                                   const torch::Tensor &V, const torch::Tensor &mask = {});
};

Attention::Attention(int64_t embeddingDim) {
    Query = register_module("Query", torch::nn::Linear(embeddingDim, embeddingDim));
    Key = register_module("Key", torch::nn::Linear(embeddingDim, embeddingDim));
    Value = register_module("Value", torch::nn::Linear(embeddingDim, embeddingDim));
    Output = register_module("Output", torch::nn::Linear(embeddingDim, embeddingDim));
}

torch::Tensor Attention::forward(const torch::Tensor& input, const torch::Tensor& mask) {
    queries = Query->forward(input);
    keys = Key->forward(input);
    values = Value->forward(input);

    weights = scaledDotProduct(keys, queries, values, mask);
    return Output->forward(weights);
}

torch::Tensor Attention::scaledDotProduct(const torch::Tensor &K, const torch::Tensor &Q,
                                           const torch::Tensor &V, const torch::Tensor &mask) {
    auto d_k = K.size(-1);
    auto scores = torch::matmul(Q, K.transpose(-2, -1)) / std::sqrt(d_k);

    if (mask.defined()) scores += mask;

    this->weights = torch::softmax(scores, -1);
    return torch::matmul(weights, V);
}
