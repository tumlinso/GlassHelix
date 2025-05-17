#ifndef MGS_TRANSFORMER_HH
#define MGS_TRANSFORMER_HH

#include <torch/torch.h>
#include <iostream>
#include <array>

using namespace torch;

namespace Transformer {
    class EmbeddingLayer : public nn::Module {
    public:
        Tensor weights;
        explicit EmbeddingLayer(int64_t vocab_size, int64_t embedding_dim);
        Tensor forward(const Tensor& input) const;
    };

    class Attention : public nn::Module { // single head, init mult
    public:
        nn::Linear Query{nullptr}, Key{nullptr}, Value{nullptr}, Output{nullptr};
        Tensor queries, keys, values, weights;
        const int64_t embeddingDim;

        explicit Attention(int64_t embeddingDim);

        Tensor forward(const Tensor& input, const Tensor& mask = {});
        Tensor scaledDotProduct(const Tensor &K, const Tensor &Q, const Tensor &V, const Tensor &mask = {});
    };

    class FeedForward : public nn::Module {
    public:
        nn::Sequential layers;
        explicit FeedForward(int64_t dEmbeddings, const std::array<int64_t, 2>& dFeedForward);
        inline Tensor forward(const Tensor& input) { return layers->forward(input);}
    };


    class TorchTransformer : public nn::Module {
    public:
        int64_t dVocab, dEmbeddings, nHeads;
        nn::ModuleHolder<EmbeddingLayer> Embedding{nullptr};
        nn::ModuleHolder<Attention> SelfAttention{nullptr};
        nn::ModuleHolder<FeedForward> FFN{nullptr};
        nn::LayerNorm Normalizer1{nullptr}, Normalizer2{nullptr};
        nn::Linear Unembedding{nullptr};

        explicit TorchTransformer(int64_t dVocab, int64_t dEmbeddings, int64_t nHeads,
                                  const std::array<int64_t, 2>& dFeedForward);

        Tensor forward(const Tensor& input, const Tensor& mask = {});

        void trainModel(Tensor &input, Tensor &target, optim::Optimizer& optimizer, int epochs = 100);
        void evalModel(Tensor &input, Tensor &target);
    };
}


#endif //MGS_TRANSFORMER_HH
