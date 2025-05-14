#include "TorchTransformer.hh"

using namespace Transformer;

namespace Transformer {

    EmbeddingLayer::EmbeddingLayer(int64_t vocabSize, int64_t embeddingSize)
            : weights(torch::randn({vocabSize, embeddingSize})) {
        register_parameter("weights", weights);
    }

    Tensor EmbeddingLayer::forward(const Tensor& input) const {
        return embedding(weights, input);
    }

    Attention::Attention(int64_t embeddingDim)
            : embeddingDim(embeddingDim),
              Query(torch::nn::Linear(embeddingDim, embeddingDim)),
              Key(torch::nn::Linear(embeddingDim, embeddingDim)),
              Value(torch::nn::Linear(embeddingDim, embeddingDim)),
              Output(torch::nn::Linear(embeddingDim, embeddingDim)) {
        register_module("Query", Query);
        register_module("Key", Key);
        register_module("Value", Value);
        register_module("Output", Output);
    }

    Tensor Attention::scaledDotProduct(const at::Tensor &K, const at::Tensor &Q, const at::Tensor &V,
                                       const at::Tensor &mask) {
        auto scores = Q.matmul(K.transpose(-2, -1)) / std::sqrt(embeddingDim);
        if (mask.defined()) scores.masked_fill_(mask, -1e9);

        this->weights = softmax(scores, -1);
        return weights.matmul(V);
    }

    Tensor Attention::forward(const at::Tensor &input, const at::Tensor &mask) {
        queries = Query->forward(input);
        keys = Key->forward(input);
        values = Value->forward(input);

        queries = queries.view({-1, input.size(1), embeddingDim});
        keys = keys.view({-1, input.size(1), embeddingDim});
        values = values.view({-1, input.size(1), embeddingDim});

        auto attentionWeights = scaledDotProduct(keys, queries, values, mask);
        attentionWeights = attentionWeights.transpose(1, 2).contiguous().view({input.size(0), -1, embeddingDim});
        return Output->forward(attentionWeights);
    }

    FeedForward::FeedForward(int64_t dEmbeddings, const std::array<int64_t, 2>& dFeedForward) {
        // Input layer
        layers->push_back(nn::Linear(dEmbeddings, dFeedForward[0]));
        layers->push_back(nn::ReLU());

        // Hidden layers
        for (int i = 0; i < dFeedForward[1] - 1; ++i) {
            layers->push_back(nn::Linear(dFeedForward[0], dFeedForward[0]));
            layers->push_back(nn::ReLU());
        }

        // Output layer
        layers->push_back(nn::Linear(dFeedForward[0], dEmbeddings));

        // Register layers
        register_module("layers", layers);
    }

    TorchTransformer::TorchTransformer(int64_t dVocab, int64_t dEmbeddings, int64_t nHeads,
                                       const std::array<int64_t, 2>& dFeedForward)
            : dVocab(dVocab), dEmbeddings(dEmbeddings), nHeads(nHeads),
              Embedding(dVocab, dEmbeddings),
              SelfAttention(dEmbeddings, nHeads),
              FFN(dEmbeddings, dFeedForward),
              Normalizer1(nn::LayerNorm(nn::LayerNormOptions({dEmbeddings}))),
              Normalizer2(nn::LayerNorm(nn::LayerNormOptions({dEmbeddings}))),
              Unembedding(nn::Linear(dEmbeddings, dVocab)) {
        register_module("Embedding", Embedding);
        register_module("SelfAttention", SelfAttention);
        register_module("Normalizer1", Normalizer1);
        register_module("FFN", FFN);
        register_module("Normalizer2", Normalizer2);
        register_module("Unembedding", Unembedding);
    }

    Tensor TorchTransformer::forward(const Tensor& input, const Tensor& mask) {
        // Embed tokens into dense vectors
        auto x = Embedding->forward(input) * std::sqrt(static_cast<float>(Embedding->weights.size(1)));

        // Self-Attention with residual connection
        auto attn_output = SelfAttention->forward(x, mask);
        x = Normalizer1->forward(x + attn_output);

        // Feed-Forward Network with residual connection
        auto ffn_output = FFN->forward(x);
        x = Normalizer2->forward(x + ffn_output);

        // Unembed to logits
        return Unembedding->forward(x);
    }

    void TorchTransformer::trainModel(Tensor &input, Tensor &target, optim::Optimizer &optimizer, int epochs) {
        this->train(); // Set the model to training mode

        // Ensure input and target are on the same device as the model
        auto model_device = this->parameters()[0].device();
        input = input.to(model_device);
        target = target.to(model_device);

        auto criterion = nn::CrossEntropyLoss();

        for (int epoch = 0; epoch < epochs; ++epoch) {
            optimizer.zero_grad();
            auto output = forward(input); // Shape: [batch_size, seq_len, dVocab]

            auto reshaped_output = output.view({-1, dVocab}); // Shape: [batch_size * seq_len, dVocab]
            auto reshaped_target = target.view(-1);          // Shape: [batch_size * seq_len]

            auto loss = criterion(reshaped_output, reshaped_target);

            loss.backward();
            torch::nn::utils::clip_grad_norm_(this->parameters(), 1.0); // Gradient clipping
            optimizer.step();

            // Log progress
            std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "], Loss: " << loss.item<float>() << std::endl;
        }
    }

    void TorchTransformer::evalModel(Tensor &input, Tensor &target) {
        this->eval();

        auto device = this->parameters()[0].device();
        input = input.to(device);
        target = target.to(device);

        torch::NoGradGuard no_grad;
        auto output = forward(input); // Shape: [batch_size, seq_len, dVocab]

        auto reshaped_output = output.view({-1, dVocab}); // Shape: [batch_size * seq_len, dVocab]
        auto reshaped_target = target.view(-1);   // Shape: [batch_size * seq_len]

        auto criterion = torch::nn::CrossEntropyLoss();
        auto loss = criterion(reshaped_output, reshaped_target);

        std::cout << "Evaluation Loss: " << loss.item<float>() << std::endl;
    }

} // Transformer