#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>
#include <omp.h>

/// This structure contains the shape specifications for the Transformer model
struct LLamaConfig {
    int dim;        // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers;   // number of layers
    int n_heads;    // number of query heads
    int vocab_size; // vocabulary size
    int seq_len;    // max sequence length

    [[nodiscard]] int head_size() const {
        return dim / n_heads;
    }
};

/// This structure contains the weights of a single transformer block/layer
struct LLamaLayer {
    std::vector<float> rms_attention;    // (dim,)
    std::vector<float> rms_feed_forward; // (dim,)

    std::vector<float> query_weight_matrix; // (dim, dim)
    std::vector<float> key_weight_matrix;   // (dim, dim)
    std::vector<float> value_weight_matrix; // (dim, dim)
    std::vector<float> out_weight_matrix;   // (dim, dim)

    std::vector<float> feed_forward_w1; // (dim, hidden_dim)
    std::vector<float> feed_forward_w2; // (hidden_dim, dim)
    std::vector<float> feed_forward_w3; // (dim, hidden_dim)
};

/// This structure contains the weights of the entire model
struct LLamaParameters {
    // token embedding table
    std::vector<float> TokenEmbeddingMatrix; // (vocab_size, dim)
    std::vector<float> TokenOutputMatrix;    // (vocab_size, dim)
    std::vector<float> RmsFinal;             // (dim,)
    std::vector<LLamaLayer> LayerWeights;
};

/// Special type to indicate a token id.
enum class token_t : int {
    BOS = 1, // beginning of sequence
    EOS = 2  // end of sequence
};

// Function to be implemented
void llm(LLamaConfig config, LLamaParameters params, const std::vector<token_t> &tokens, std::vector<float> &logits);

// Utility functions:
// You may use these, but are allowed to provide your own alternatives

namespace utils {

/// Normalize the vector in `x`, and scale each coordinate by the corresponding weight.
/// The result is stored in `out`.
/// `x`, `out`, and `weight` should all be arrays of length `size`.
inline void rmsnorm(float *out, const float *x, const float *weights, int size) {
    int op; // number of operations in the loop in parallel
    if (size % 4 == 0){op = 4;}
    else if (size % 2 == 0){op = 2;}
    else {op = 1;}

    float ss = (float)std::inner_product(x, x + size, x, 0.0) / (float)size;
    float scale = 1.0f / std::sqrt(ss + 1e-5f);

    if (op == 4) {
        #pragma omp parallel for
        for (int i = 0; i < size; i=i+4) {
            out[i] = x[i] * weights[i] * scale;
            out[i+1] = x[i+1] * weights[i+1] * scale;
            out[i+2] = x[i+2] * weights[i+2] * scale;
            out[i+3] = x[i+3] * weights[i+3] * scale;
        }
    }
    else if (op == 2) {
        #pragma omp parallel for
        for (int i = 0; i < size; i=i+2) {
            out[i] = x[i] * weights[i] * scale;
            out[i+1] = x[i+1] * weights[i+1] * scale;
        }
    }
    else {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            out[i] = x[i] * weights[i] * scale;
        }
    }
}

/// Turn un-normalized scores `x` into a probability distribution. In-place operation.
inline void softmax(float *x, int size) {
    float max = *std::max_element(x, x + size);
    // Subtract the maximum score. This does not affect the result mathematically,
    // but prevents overflows in floating-point representations.
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max);
    }

    float sum = (float)std::accumulate(x, x + size, 0.0);
    for (int i = 0; i < size; ++i) {
        x[i] = x[i] / sum;
    }
}

/// activation function used in LLama
inline float silu(float x) {
    return x / (1.f + std::exp(-x));
}

/// activation function used in LLama
inline void swiglu(float *out, const float *a, const float *b, int size) {
    for (int i = 0; i < size; ++i) {
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        // elementwise multiply with w3(x)
        out[i] = silu(a[i]) * b[i];
    }
}

/// Multiply the complex number pointed to by `target` with
/// real part `fcr` and imaginary part `fci`.
inline void rotate(float *target, float fcr, float fci) {
    target[0] = target[0] * fcr - target[1] * fci;
    target[1] = target[0] * fci + target[1] * fcr;
}

/// ROtary Position Encoding. See https://arxiv.org/abs/2104.09864 for motivation.
inline void rope(const LLamaConfig &config, float *query, float *key, int pos) {
    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    for (int i = 0; i < config.dim; i += 2) {
        int head_dim = i % config.head_size();
        float freq = 1.0f / std::pow(10000.0f, head_dim / (float)config.head_size());
        float val = pos * freq;
        float fcr = std::cos(val);
        float fci = std::sin(val);
        // rotate query vector
        rotate(query + i, fcr, fci);
        // rotate key vector
        rotate(key + i, fcr, fci);
    }
}

/// Given attention scores `attention` and stored values `values_cache`, this calculates
/// the interpolation which is the output of the attention processing, in `result`.
inline void lookup_with_attention(const LLamaConfig &config,
                                  const float *attention, float *result, int pos,
                                  const float *value_cache) {
    // initialize result to zero
    std::fill(result, result + config.head_size(), 0.f);

    int timesteps = pos + 1;
    int head_size = config.head_size();

    int t_op;
    if (timesteps % 4 == 0) {t_op = 4;}
    else if (timesteps % 2 == 0) {t_op = 2;}
    else {t_op = 1;}

    if (t_op == 4) {
        for (int t = 0; t < timesteps; t=t+4) {
            float a0 = attention[t];
            float a1 = attention[t+1];
            float a2 = attention[t+2];
            float a3 = attention[t+3];
            const float *v0 = value_cache + t * config.dim;
            const float *v1 = value_cache + (t+1) * config.dim;
            const float *v2 = value_cache + (t+2) * config.dim;
            const float *v3 = value_cache + (t+3) * config.dim;

            for (int i = 0; i < head_size; ++i) {
                result[i] += a0*v0[i] + a1*v1[i] + a2*v2[i] + a3*v3[i];
            }
        }
    }
    else if (t_op == 2) {
        for (int t = 0; t < timesteps; t=t+2) {
            float a0 = attention[t];
            float a1 = attention[t+1];
            const float *v0 = value_cache + t * config.dim;
            const float *v1 = value_cache + (t+1) * config.dim;

            for (int i = 0; i < head_size; ++i) {
                result[i] += a0*v0[i] + a1*v1[i];
            }
        }
    }
    else {
        for (int t = 0; t < timesteps; ++t) {
            float a0 = attention[t];
            const float *v0 = value_cache + t * config.dim;

            for (int i = 0; i < head_size; ++i) {
                result[i] += a0*v0[i];
            }
        }
    }
}

inline void calculate_attention(const LLamaConfig &config,
                                float *attention, const float *query, int pos,
                                const float *key_cache) {
    float norm = 1.f / sqrtf(config.head_size());

    int head_size = config.head_size();
    int op;
    if (head_size % 4 == 0) {op = 4;}
    else if (head_size % 2 == 0) {op = 2;}
    else {op = 1;}

    // iterate over all timesteps, including the current one
    if (op == 4) {
        for (int t = 0; t <= pos; ++t) {
            const float *key = key_cache + t * config.dim;
            double score = 0.0;

            for (int i = 0; i < head_size; i=i+4) {
                score += query[i] * key[i];
                score += query[i+1] * key[i+1];
                score += query[i+2] * key[i+2];
                score += query[i+3] * key[i+3];
            }

            attention[t] = (float)score * norm;
        }
    }
    else if (op == 2) {
        for (int t = 0; t <= pos; ++t) {
            const float *key = key_cache + t * config.dim;
            double score = 0.0;

            for (int i = 0; i < head_size; i=i+2) {
                score += query[i] * key[i];
                score += query[i+1] * key[i+1];
            }

            attention[t] = (float)score * norm;
        }
    }
    else {
        for (int t = 0; t <= pos; ++t) {
            const float *key = key_cache + t * config.dim;
            double score = 0.0;

            for (int i = 0; i < head_size; ++i) {
                score += query[i] * key[i];
            }

            attention[t] = (float)score * norm;
        }
    }

    // softmax the scores to get attention weights, from 0..pos inclusively
    softmax(attention, pos + 1);
}

} // namespace utils
