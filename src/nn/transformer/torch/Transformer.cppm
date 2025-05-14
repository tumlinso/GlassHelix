module;
#include <torch/torch.h>

export module Transformer;
import Attention;

export class Transformer : public torch::nn::Module {
public:
    int64_t dVocab, dEmbeddings, nHeads;
    torch::nn::ModuleHolder<Attention> SelfAttention{nullptr};


    explicit Transformer(int64_t dVocab, int64_t dEmbeddings, int64_t nHeads,
                         const std::array<int64_t, 2>& dFeedForward);

    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& mask = {});


};

