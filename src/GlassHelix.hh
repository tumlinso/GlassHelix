#ifndef GLASSHELIX_GLASSHELIX_HH
#define GLASSHELIX_GLASSHELIX_HH
#include <cmath>

#include <cstddef>
#include <iostream>
#include <torch/torch.h>

#include "Dataset.hh"

template<typename T, size_t recordLength = 2048>
class GlassHelix : public torch::nn::Module {
public:
    GlassHelix(int64_t vocab_size,
               int64_t embed_dim,
               int64_t nhead,
               int64_t nhid,
               int64_t nlayers,
               double dropout = 0.1)
      : embed_dim_(embed_dim)
    {
        // Embedding layer
        embedding = register_module("embedding", torch::nn::Embedding(vocab_size, embed_dim));

        // Transformer encoder layer and encoder
        auto encoder_layer = torch::nn::TransformerEncoderLayer(
            torch::nn::TransformerEncoderLayerOptions(embed_dim, nhead)
                .dim_feedforward(nhid)
                .dropout(dropout)
        );
        transformer = register_module("transformer", torch::nn::TransformerEncoder(encoder_layer, nlayers));

        // Final linear decoder
        decoder = register_module("decoder", torch::nn::Linear(embed_dim, vocab_size));
    }

    // Forward expects input shape [batch, seq_len]
    torch::Tensor forward(torch::Tensor src) {
        // src -> [batch, seq_len, embed_dim]
        auto x = embedding(src) * std::sqrt(static_cast<double>(embed_dim_));
        // transformer expects [seq_len, batch, embed_dim]
        x = x.transpose(0, 1);
        x = transformer->forward(x);
        // back to [batch, seq_len, embed_dim]
        x = x.transpose(0, 1);
        // decode
        return decoder(x);
    }

private:
    int64_t embed_dim_;
    torch::nn::Embedding embedding{nullptr};
    torch::nn::TransformerEncoder transformer{nullptr};
    torch::nn::Linear decoder{nullptr};
};


#endif //GLASSHELIX_GLASSHELIX_HH
