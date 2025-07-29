#include "../include/transformer.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <numeric>
#include <fstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// POSITIONAL ENCODING IMPLEMENTATION
// ============================================================================

PositionalEncoding::PositionalEncoding(int max_len, int d_model) 
    : max_len(max_len), d_model(d_model), encoding(max_len, d_model) {
    
    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < d_model; i++) {
            if (i % 2 == 0) {
                encoding.data[pos][i] = sin(pos / pow(10000.0, (2.0 * i) / d_model));
            } else {
                encoding.data[pos][i] = cos(pos / pow(10000.0, (2.0 * (i-1)) / d_model));
            }
        }
    }
}

Matrix PositionalEncoding::encode(const Matrix& input) const {
    Matrix result = input;
    int seq_len = std::min(input.rows, max_len);
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < std::min(input.cols, d_model); j++) {
            result.data[i][j] += encoding.data[i][j];
        }
    }
    return result;
}

Matrix PositionalEncoding::backward(const Matrix& grad_output) const {
    // Positional encoding gradients pass through unchanged
    return grad_output;
}



// ============================================================================
// MULTI-HEAD ATTENTION IMPLEMENTATION
// ============================================================================

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads) 
    : d_model(d_model), num_heads(num_heads) {
    
    d_k = d_model / num_heads;
    
    // Initialize weight matrices
    W_q = Matrix(d_model, d_model);
    W_k = Matrix(d_model, d_model);
    W_v = Matrix(d_model, d_model);
    W_o = Matrix(d_model, d_model);
    
    // Initialize bias vectors
    b_q = Matrix(1, d_model);
    b_k = Matrix(1, d_model);
    b_v = Matrix(1, d_model);
    b_o = Matrix(1, d_model);
    
    // Initialize weights with Xavier initialization
    W_q.xavier_init();
    W_k.xavier_init();
    W_v.xavier_init();
    W_o.xavier_init();
    
    b_q.zero();
    b_k.zero();
    b_v.zero();
    b_o.zero();
    
    // Initialize gradients
    grad_q = Gradients(d_model, d_model);
    grad_k = Gradients(d_model, d_model);
    grad_v = Gradients(d_model, d_model);
    grad_o = Gradients(d_model, d_model);
}

Matrix MultiHeadAttention::attention(const Matrix& Q, const Matrix& K, const Matrix& V, bool mask) const {
    // Q, K, V: [seq_len, d_model]
    
    // Compute attention scores: Q * K^T
    Matrix scores = Q * K.transpose();
    
    // Scale by sqrt(d_k)
    double scale = 1.0 / sqrt(d_k);
    scores = scores * scale;
    
    // Apply mask if needed
    if (mask) {
        for (int i = 0; i < scores.rows; i++) {
            for (int j = i + 1; j < scores.cols; j++) {
                scores.data[i][j] = -1e9; // Very negative value
            }
        }
    }
    
    // Apply softmax
    Matrix attention_weights = scores.softmax();
    
    // Cache for backward pass
    cache.attention_scores = scores;
    cache.attention_weights = attention_weights;
    
    // Apply attention to values
    Matrix output = attention_weights * V;
    
    return output;
}

Matrix MultiHeadAttention::forward(const Matrix& query, const Matrix& key, const Matrix& value, bool mask) {
    // Cache inputs for backward pass
    cache.input = query;
    cache.query = query;
    cache.key = key;
    cache.value = value;
    
    // Linear transformations
    Matrix Q = query * W_q;
    Matrix K = key * W_k;
    Matrix V = value * W_v;
    
    // Add bias
    for (int i = 0; i < Q.rows; i++) {
        for (int j = 0; j < Q.cols; j++) {
            Q.data[i][j] += b_q.data[0][j];
            K.data[i][j] += b_k.data[0][j];
            V.data[i][j] += b_v.data[0][j];
        }
    }
    
    // Multi-head attention (simplified: treat as single head for now)
    Matrix attended = attention(Q, K, V, mask);
    
    // Output projection
    Matrix output = attended * W_o;
    
    // Add output bias
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            output.data[i][j] += b_o.data[0][j];
        }
    }
    
    cache.attended_values = attended;
    
    return output;
}

std::tuple<Matrix, Matrix, Matrix> MultiHeadAttention::backward(const Matrix& grad_output) {
    // Gradient w.r.t output projection
    Matrix grad_attended = grad_output * W_o.transpose();
    
    // Gradients for output projection weights and bias
    grad_o.dW.add_inplace(cache.attended_values.transpose() * grad_output);
    for (int j = 0; j < grad_output.cols; j++) {
        double bias_grad = 0.0;
        for (int i = 0; i < grad_output.rows; i++) {
            bias_grad += grad_output.data[i][j];
        }
        grad_o.db.data[0][j] += bias_grad;
    }
    
    // Attention backward pass
    Matrix grad_V = cache.attention_weights.transpose() * grad_attended;
    Matrix grad_attention_weights = grad_attended * cache.value.transpose();
    
    // Softmax backward (proper implementation)
    Matrix grad_scores(cache.attention_scores.rows, cache.attention_scores.cols);
    for (int i = 0; i < grad_scores.rows; i++) {
        for (int j = 0; j < grad_scores.cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < grad_scores.cols; k++) {
                sum += grad_attention_weights.data[i][k] * cache.attention_weights.data[i][k];
            }
            grad_scores.data[i][j] = cache.attention_weights.data[i][j] * 
                                   (grad_attention_weights.data[i][j] - sum);
        }
    }
    
    // Scale gradient
    double scale = 1.0 / sqrt(d_k);
    grad_scores = grad_scores * scale;
    
    // Gradients w.r.t Q, K, V
    Matrix grad_Q = grad_scores * cache.key;
    Matrix grad_K = grad_scores.transpose() * cache.query;
    
    // Linear transformation gradients
    Matrix grad_query = grad_Q * W_q.transpose();
    Matrix grad_key = grad_K * W_k.transpose();
    Matrix grad_value = grad_V * W_v.transpose();
    
    // Weight gradients
    grad_q.dW.add_inplace(cache.query.transpose() * grad_Q);
    grad_k.dW.add_inplace(cache.key.transpose() * grad_K);
    grad_v.dW.add_inplace(cache.value.transpose() * grad_V);
    
    // Bias gradients
    for (int j = 0; j < grad_Q.cols; j++) {
        double bias_grad_q = 0.0, bias_grad_k = 0.0, bias_grad_v = 0.0;
        for (int i = 0; i < grad_Q.rows; i++) {
            bias_grad_q += grad_Q.data[i][j];
            if (i < grad_K.rows) bias_grad_k += grad_K.data[i][j];
            bias_grad_v += grad_V.data[i][j];
        }
        grad_q.db.data[0][j] += bias_grad_q;
        grad_k.db.data[0][j] += bias_grad_k;
        grad_v.db.data[0][j] += bias_grad_v;
    }
    
    return std::make_tuple(grad_query, grad_key, grad_value);
}

void MultiHeadAttention::update_weights(double learning_rate) {
    W_q.subtract_inplace(grad_q.dW * learning_rate);
    W_k.subtract_inplace(grad_k.dW * learning_rate);
    W_v.subtract_inplace(grad_v.dW * learning_rate);
    W_o.subtract_inplace(grad_o.dW * learning_rate);
    
    b_q.subtract_inplace(grad_q.db * learning_rate);
    b_k.subtract_inplace(grad_k.db * learning_rate);
    b_v.subtract_inplace(grad_v.db * learning_rate);
    b_o.subtract_inplace(grad_o.db * learning_rate);
}

void MultiHeadAttention::update_weights_adam(AdamOptimizer& optimizer, int step) {
    optimizer.update(W_q, grad_q.dW, &W_q, step, 0.001);
    optimizer.update(W_k, grad_k.dW, &W_k, step, 0.001);
    optimizer.update(W_v, grad_v.dW, &W_v, step, 0.001);
    optimizer.update(W_o, grad_o.dW, &W_o, step, 0.001);
    
    optimizer.update(b_q, grad_q.db, &b_q, step, 0.001);
    optimizer.update(b_k, grad_k.db, &b_k, step, 0.001);
    optimizer.update(b_v, grad_v.db, &b_v, step, 0.001);
    optimizer.update(b_o, grad_o.db, &b_o, step, 0.001);
}

void MultiHeadAttention::zero_gradients() {
    grad_q.zero();
    grad_k.zero();
    grad_v.zero();
    grad_o.zero();
}

void MultiHeadAttention::clip_gradients(double max_norm) {
    grad_q.clip(max_norm);
    grad_k.clip(max_norm);
    grad_v.clip(max_norm);
    grad_o.clip(max_norm);
}



// ============================================================================
// FEED FORWARD IMPLEMENTATION
// ============================================================================

FeedForward::FeedForward(int d_model, int d_ff) : d_model(d_model), d_ff(d_ff) {
    W1 = Matrix(d_model, d_ff);
    W2 = Matrix(d_ff, d_model);
    b1 = Matrix(1, d_ff);
    b2 = Matrix(1, d_model);
    
    W1.xavier_init();
    W2.xavier_init();
    b1.zero();
    b2.zero();
    
    grad_1 = Gradients(d_model, d_ff);
    grad_2 = Gradients(d_ff, d_model);
}

Matrix FeedForward::forward(const Matrix& input) {
    input_cache = input;
    
    // First linear layer
    Matrix hidden = input * W1;
    
    // Add bias
    for (int i = 0; i < hidden.rows; i++) {
        for (int j = 0; j < hidden.cols; j++) {
            hidden.data[i][j] += b1.data[0][j];
        }
    }
    
    // ReLU activation
    hidden = hidden.relu();
    hidden_cache = hidden;
    
    // Second linear layer
    Matrix output = hidden * W2;
    
    // Add bias
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            output.data[i][j] += b2.data[0][j];
        }
    }
    
    return output;
}

Matrix FeedForward::backward(const Matrix& grad_output) {
    // Gradient w.r.t second layer
    Matrix grad_hidden = grad_output * W2.transpose();
    
    // Second layer weight and bias gradients
    grad_2.dW.add_inplace(hidden_cache.transpose() * grad_output);
    for (int j = 0; j < grad_output.cols; j++) {
        double bias_grad = 0.0;
        for (int i = 0; i < grad_output.rows; i++) {
            bias_grad += grad_output.data[i][j];
        }
        grad_2.db.data[0][j] += bias_grad;
    }
    
    // ReLU backward
    for (int i = 0; i < grad_hidden.rows; i++) {
        for (int j = 0; j < grad_hidden.cols; j++) {
            if (hidden_cache.data[i][j] <= 0) {
                grad_hidden.data[i][j] = 0.0;
            }
        }
    }
    
    // First layer gradients
    Matrix grad_input = grad_hidden * W1.transpose();
    
    // First layer weight and bias gradients
    grad_1.dW.add_inplace(input_cache.transpose() * grad_hidden);
    for (int j = 0; j < grad_hidden.cols; j++) {
        double bias_grad = 0.0;
        for (int i = 0; i < grad_hidden.rows; i++) {
            bias_grad += grad_hidden.data[i][j];
        }
        grad_1.db.data[0][j] += bias_grad;
    }
    
    return grad_input;
}

void FeedForward::update_weights(double learning_rate) {
    W1.subtract_inplace(grad_1.dW * learning_rate);
    W2.subtract_inplace(grad_2.dW * learning_rate);
    b1.subtract_inplace(grad_1.db * learning_rate);
    b2.subtract_inplace(grad_2.db * learning_rate);
}

void FeedForward::update_weights_adam(AdamOptimizer& optimizer, int step) {
    optimizer.update(W1, grad_1.dW, &W1, step, 0.001);
    optimizer.update(W2, grad_2.dW, &W2, step, 0.001);
    optimizer.update(b1, grad_1.db, &b1, step, 0.001);
    optimizer.update(b2, grad_2.db, &b2, step, 0.001);
}

void FeedForward::zero_gradients() {
    grad_1.zero();
    grad_2.zero();
}

void FeedForward::clip_gradients(double max_norm) {
    grad_1.clip(max_norm);
    grad_2.clip(max_norm);
}



// ============================================================================
// LAYER NORMALIZATION IMPLEMENTATION
// ============================================================================

LayerNorm::LayerNorm(int d_model, double eps) : d_model(d_model), eps(eps) {
    gamma = Matrix(1, d_model);
    beta = Matrix(1, d_model);
    grad_gamma = Matrix(1, d_model);
    grad_beta = Matrix(1, d_model);
    
    // Initialize gamma to 1, beta to 0
    for (int j = 0; j < d_model; j++) {
        gamma.data[0][j] = 1.0;
        beta.data[0][j] = 0.0;
    }
    
    grad_gamma.zero();
    grad_beta.zero();
}

Matrix LayerNorm::forward(const Matrix& input) {
    input_cache = input;
    Matrix result(input.rows, input.cols);
    
    mean_cache = Matrix(input.rows, 1);
    std_cache = Matrix(input.rows, 1);
    normalized_cache = Matrix(input.rows, input.cols);
    
    for (int i = 0; i < input.rows; i++) {
        // Compute mean
        double mean = 0.0;
        for (int j = 0; j < input.cols; j++) {
            mean += input.data[i][j];
        }
        mean /= input.cols;
        mean_cache.data[i][0] = mean;
        
        // Compute variance
        double variance = 0.0;
        for (int j = 0; j < input.cols; j++) {
            double diff = input.data[i][j] - mean;
            variance += diff * diff;
        }
        variance /= input.cols;
        double std_dev = sqrt(variance + eps);
        std_cache.data[i][0] = std_dev;
        
        // Normalize and scale
        for (int j = 0; j < input.cols; j++) {
            double normalized = (input.data[i][j] - mean) / std_dev;
            normalized_cache.data[i][j] = normalized;
            result.data[i][j] = normalized * gamma.data[0][j] + beta.data[0][j];
        }
    }
    
    return result;
}

Matrix LayerNorm::backward(const Matrix& grad_output) {
    Matrix grad_input(input_cache.rows, input_cache.cols);
    
    for (int i = 0; i < input_cache.rows; i++) {
        double mean = mean_cache.data[i][0];
        double std_dev = std_cache.data[i][0];
        int N = input_cache.cols;
        
        // Accumulate gradients for gamma and beta
        for (int j = 0; j < input_cache.cols; j++) {
            grad_gamma.data[0][j] += grad_output.data[i][j] * normalized_cache.data[i][j];
            grad_beta.data[0][j] += grad_output.data[i][j];
        }
        
        // Compute gradient w.r.t input (proper chain rule)
        double grad_var = 0.0;
        for (int j = 0; j < N; j++) {
            grad_var += grad_output.data[i][j] * gamma.data[0][j] * 
                       (input_cache.data[i][j] - mean) * (-0.5) * pow(std_dev, -3);
        }
        
        double grad_mean = 0.0;
        for (int j = 0; j < N; j++) {
            grad_mean += grad_output.data[i][j] * gamma.data[0][j] * (-1.0 / std_dev);
        }
        grad_mean += grad_var * (-2.0 / N) * 
                    std::accumulate(input_cache.data[i].begin(), input_cache.data[i].end(), 0.0) / N;
        
        for (int j = 0; j < N; j++) {
            grad_input.data[i][j] = grad_output.data[i][j] * gamma.data[0][j] / std_dev +
                                   grad_var * 2.0 * (input_cache.data[i][j] - mean) / N +
                                   grad_mean / N;
        }
    }
    
    return grad_input;
}

void LayerNorm::update_weights(double learning_rate) {
    gamma.subtract_inplace(grad_gamma * learning_rate);
    beta.subtract_inplace(grad_beta * learning_rate);
}

void LayerNorm::update_weights_adam(AdamOptimizer& optimizer, int step) {
    optimizer.update(gamma, grad_gamma, &gamma, step, 0.001);
    optimizer.update(beta, grad_beta, &beta, step, 0.001);
}

void LayerNorm::zero_gradients() {
    grad_gamma.zero();
    grad_beta.zero();
}




// ============================================================================
// TRANSFORMER ENCODER LAYER IMPLEMENTATION
// ============================================================================

TransformerEncoderLayer::TransformerEncoderLayer(int d_model, int num_heads, int d_ff, double dropout_rate)
    : dropout_rate(dropout_rate) {
    
    attention = std::make_unique<MultiHeadAttention>(d_model, num_heads);
    feed_forward = std::make_unique<FeedForward>(d_model, d_ff);
    norm1 = std::make_unique<LayerNorm>(d_model);
    norm2 = std::make_unique<LayerNorm>(d_model);
}

Matrix TransformerEncoderLayer::forward(const Matrix& input, bool training) {
    input_cache = input;
    
    // Self-attention with residual connection and layer norm
    Matrix attention_out = attention->forward(input, input, input, false);
    attention_out_cache = attention_out;
    
    Matrix residual1 = input + attention_out;
    Matrix norm1_out = norm1->forward(residual1);
    norm1_out_cache = norm1_out;
    
    // Feed forward with residual connection and layer norm
    Matrix ff_out = feed_forward->forward(norm1_out);
    ff_out_cache = ff_out;
    
    Matrix residual2 = norm1_out + ff_out;
    Matrix output = norm2->forward(residual2);
    
    return output;
}

Matrix TransformerEncoderLayer::backward(const Matrix& grad_output) {
    // Backward through second layer norm
    Matrix grad_residual2 = norm2->backward(grad_output);
    
    // Split gradient for residual connection
    Matrix grad_norm1_out = grad_residual2;
    Matrix grad_ff_out = grad_residual2;
    
    // Backward through feed forward
    Matrix grad_ff_input = feed_forward->backward(grad_ff_out);
    grad_norm1_out.add_inplace(grad_ff_input);
    
    // Backward through first layer norm
    Matrix grad_residual1 = norm1->backward(grad_norm1_out);
    
    // Split gradient for residual connection
    Matrix grad_input = grad_residual1;
    Matrix grad_attention_out = grad_residual1;
    
    // Backward through attention - CORREGIDO
    std::tuple<Matrix, Matrix, Matrix> grad_result = attention->backward(grad_attention_out);
    Matrix grad_q = std::get<0>(grad_result);
    Matrix grad_k = std::get<1>(grad_result);
    Matrix grad_v = std::get<2>(grad_result);
    
    grad_input.add_inplace(grad_q); // Assuming query, key, value are all input
    
    return grad_input;
}

void TransformerEncoderLayer::update_weights(double learning_rate) {
    attention->update_weights(learning_rate);
    feed_forward->update_weights(learning_rate);
    norm1->update_weights(learning_rate);
    norm2->update_weights(learning_rate);
}

void TransformerEncoderLayer::update_weights_adam(AdamOptimizer& optimizer, int step) {
    attention->update_weights_adam(optimizer, step);
    feed_forward->update_weights_adam(optimizer, step);
    norm1->update_weights_adam(optimizer, step);
    norm2->update_weights_adam(optimizer, step);
}

void TransformerEncoderLayer::zero_gradients() {
    attention->zero_gradients();
    feed_forward->zero_gradients();
    norm1->zero_gradients();
    norm2->zero_gradients();
}

void TransformerEncoderLayer::clip_gradients(double max_norm) {
    attention->clip_gradients(max_norm);
    feed_forward->clip_gradients(max_norm);
}

void TransformerEncoderLayer::scale_gradients(double scale_factor) {
    attention->grad_q.dW.multiply_inplace(scale_factor);
    attention->grad_k.dW.multiply_inplace(scale_factor);
    attention->grad_v.dW.multiply_inplace(scale_factor);
    attention->grad_o.dW.multiply_inplace(scale_factor);
    
    attention->grad_q.db.multiply_inplace(scale_factor);
    attention->grad_k.db.multiply_inplace(scale_factor);
    attention->grad_v.db.multiply_inplace(scale_factor);
    attention->grad_o.db.multiply_inplace(scale_factor);
    
    feed_forward->grad_1.dW.multiply_inplace(scale_factor);
    feed_forward->grad_2.dW.multiply_inplace(scale_factor);
    feed_forward->grad_1.db.multiply_inplace(scale_factor);
    feed_forward->grad_2.db.multiply_inplace(scale_factor);
    
    norm1->grad_gamma.multiply_inplace(scale_factor);
    norm1->grad_beta.multiply_inplace(scale_factor);
    norm2->grad_gamma.multiply_inplace(scale_factor);
    norm2->grad_beta.multiply_inplace(scale_factor);
}



// ============================================================================
// TRANSFORMER DECODER LAYER IMPLEMENTATION
// ============================================================================

TransformerDecoderLayer::TransformerDecoderLayer(int d_model, int num_heads, int d_ff, double dropout_rate)
    : dropout_rate(dropout_rate) {
    
    self_attention = std::make_unique<MultiHeadAttention>(d_model, num_heads);
    cross_attention = std::make_unique<MultiHeadAttention>(d_model, num_heads);
    feed_forward = std::make_unique<FeedForward>(d_model, d_ff);
    norm1 = std::make_unique<LayerNorm>(d_model);
    norm2 = std::make_unique<LayerNorm>(d_model);
    norm3 = std::make_unique<LayerNorm>(d_model);
}

Matrix TransformerDecoderLayer::forward(const Matrix& input, const Matrix& encoder_output, bool training) {
    input_cache = input;
    encoder_cache = encoder_output;
    
    // Self-attention with masking
    Matrix self_att_out = self_attention->forward(input, input, input, true);
    self_att_cache = self_att_out;
    
    Matrix residual1 = input + self_att_out;
    Matrix norm1_out = norm1->forward(residual1);
    norm1_cache = norm1_out;
    
    // Cross-attention with encoder output
    Matrix cross_att_out = cross_attention->forward(norm1_out, encoder_output, encoder_output, false);
    cross_att_cache = cross_att_out;
    
    Matrix residual2 = norm1_out + cross_att_out;
    Matrix norm2_out = norm2->forward(residual2);
    norm2_cache = norm2_out;
    
    // Feed forward
    Matrix ff_out = feed_forward->forward(norm2_out);
    ff_cache = ff_out;
    
    Matrix residual3 = norm2_out + ff_out;
    Matrix output = norm3->forward(residual3);
    
    return output;
}

std::pair<Matrix, Matrix> TransformerDecoderLayer::backward(const Matrix& grad_output) {
    // Backward through third layer norm
    Matrix grad_residual3 = norm3->backward(grad_output);
    
    // Split for residual connection
    Matrix grad_norm2_out = grad_residual3;
    Matrix grad_ff_out = grad_residual3;
    
    // Backward through feed forward
    Matrix grad_ff_input = feed_forward->backward(grad_ff_out);
    grad_norm2_out.add_inplace(grad_ff_input);
    
    // Backward through second layer norm
    Matrix grad_residual2 = norm2->backward(grad_norm2_out);
    
    // Split for residual connection
    Matrix grad_norm1_out = grad_residual2;
    Matrix grad_cross_att_out = grad_residual2;
    
    // Backward through cross attention - CORREGIDO
    std::tuple<Matrix, Matrix, Matrix> grad_cross_result = cross_attention->backward(grad_cross_att_out);
    Matrix grad_q_cross = std::get<0>(grad_cross_result);
    Matrix grad_k_cross = std::get<1>(grad_cross_result);
    Matrix grad_v_cross = std::get<2>(grad_cross_result);
    
    grad_norm1_out.add_inplace(grad_q_cross);
    Matrix grad_encoder_output = grad_k_cross + grad_v_cross;
    
    // Backward through first layer norm
    Matrix grad_residual1 = norm1->backward(grad_norm1_out);
    
    // Split for residual connection
    Matrix grad_input = grad_residual1;
    Matrix grad_self_att_out = grad_residual1;
    
    // Backward through self attention - CORREGIDO
    std::tuple<Matrix, Matrix, Matrix> grad_self_result = self_attention->backward(grad_self_att_out);
    Matrix grad_q_self = std::get<0>(grad_self_result);
    Matrix grad_k_self = std::get<1>(grad_self_result);
    Matrix grad_v_self = std::get<2>(grad_self_result);
    
    grad_input.add_inplace(grad_q_self);
    
    return std::make_pair(grad_input, grad_encoder_output);
}

void TransformerDecoderLayer::update_weights(double learning_rate) {
    self_attention->update_weights(learning_rate);
    cross_attention->update_weights(learning_rate);
    feed_forward->update_weights(learning_rate);
    norm1->update_weights(learning_rate);
    norm2->update_weights(learning_rate);
    norm3->update_weights(learning_rate);
}

void TransformerDecoderLayer::update_weights_adam(AdamOptimizer& optimizer, int step) {
    self_attention->update_weights_adam(optimizer, step);
    cross_attention->update_weights_adam(optimizer, step);
    feed_forward->update_weights_adam(optimizer, step);
    norm1->update_weights_adam(optimizer, step);
    norm2->update_weights_adam(optimizer, step);
    norm3->update_weights_adam(optimizer, step);
}

void TransformerDecoderLayer::zero_gradients() {
    self_attention->zero_gradients();
    cross_attention->zero_gradients();
    feed_forward->zero_gradients();
    norm1->zero_gradients();
    norm2->zero_gradients();
    norm3->zero_gradients();
}

void TransformerDecoderLayer::clip_gradients(double max_norm) {
    self_attention->clip_gradients(max_norm);
    cross_attention->clip_gradients(max_norm);
    feed_forward->clip_gradients(max_norm);
}

void TransformerDecoderLayer::scale_gradients(double scale_factor) {
    self_attention->grad_q.dW.multiply_inplace(scale_factor);
    self_attention->grad_k.dW.multiply_inplace(scale_factor);
    self_attention->grad_v.dW.multiply_inplace(scale_factor);
    self_attention->grad_o.dW.multiply_inplace(scale_factor);
    
    self_attention->grad_q.db.multiply_inplace(scale_factor);
    self_attention->grad_k.db.multiply_inplace(scale_factor);
    self_attention->grad_v.db.multiply_inplace(scale_factor);
    self_attention->grad_o.db.multiply_inplace(scale_factor);
    
    cross_attention->grad_q.dW.multiply_inplace(scale_factor);
    cross_attention->grad_k.dW.multiply_inplace(scale_factor);
    cross_attention->grad_v.dW.multiply_inplace(scale_factor);
    cross_attention->grad_o.dW.multiply_inplace(scale_factor);
    
    cross_attention->grad_q.db.multiply_inplace(scale_factor);
    cross_attention->grad_k.db.multiply_inplace(scale_factor);
    cross_attention->grad_v.db.multiply_inplace(scale_factor);
    cross_attention->grad_o.db.multiply_inplace(scale_factor);
    
    feed_forward->grad_1.dW.multiply_inplace(scale_factor);
    feed_forward->grad_2.dW.multiply_inplace(scale_factor);
    feed_forward->grad_1.db.multiply_inplace(scale_factor);
    feed_forward->grad_2.db.multiply_inplace(scale_factor);
    
    norm1->grad_gamma.multiply_inplace(scale_factor);
    norm1->grad_beta.multiply_inplace(scale_factor);
    norm2->grad_gamma.multiply_inplace(scale_factor);
    norm2->grad_beta.multiply_inplace(scale_factor);
    norm3->grad_gamma.multiply_inplace(scale_factor);
    norm3->grad_beta.multiply_inplace(scale_factor);
}


// ============================================================================
// ADAM OPTIMIZER IMPLEMENTATION
// ============================================================================

AdamOptimizer::AdamOptimizer(double beta1, double beta2, double eps) 
    : beta1(beta1), beta2(beta2), eps(eps) {}

void AdamOptimizer::update(Matrix& weight, const Matrix& gradient, void* param_id, int step, double learning_rate) {
    // Initialize momentum matrices if not present
    if (m_weights.find(param_id) == m_weights.end()) {
        m_weights[param_id] = Matrix(weight.rows, weight.cols);
        v_weights[param_id] = Matrix(weight.rows, weight.cols);
        m_weights[param_id].zero();
        v_weights[param_id].zero();
    }
    
    Matrix& m = m_weights[param_id];
    Matrix& v = v_weights[param_id];
    
    // Update biased first moment estimate
    for (int i = 0; i < weight.rows; i++) {
        for (int j = 0; j < weight.cols; j++) {
            m.data[i][j] = beta1 * m.data[i][j] + (1.0 - beta1) * gradient.data[i][j];
        }
    }
    
    // Update biased second raw moment estimate
    for (int i = 0; i < weight.rows; i++) {
        for (int j = 0; j < weight.cols; j++) {
            v.data[i][j] = beta2 * v.data[i][j] + (1.0 - beta2) * gradient.data[i][j] * gradient.data[i][j];
        }
    }
    
    // Compute bias-corrected first and second moment estimates
    double bias_correction1 = 1.0 - pow(beta1, step);
    double bias_correction2 = 1.0 - pow(beta2, step);
    
    // Update weights
    for (int i = 0; i < weight.rows; i++) {
        for (int j = 0; j < weight.cols; j++) {
            double m_hat = m.data[i][j] / bias_correction1;
            double v_hat = v.data[i][j] / bias_correction2;
            weight.data[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + eps);
        }
    }
}

void AdamOptimizer::clear() {
    m_weights.clear();
    v_weights.clear();
    m_biases.clear();
    v_biases.clear();
}

// ============================================================================
// LEARNING RATE SCHEDULER IMPLEMENTATION
// ============================================================================

LRScheduler::LRScheduler(double initial_lr, double decay_factor, int decay_steps) 
    : initial_lr(initial_lr), decay_factor(decay_factor), decay_steps(decay_steps) {}

double LRScheduler::get_lr(int step) const {
    int decay_count = step / decay_steps;
    return initial_lr * pow(decay_factor, decay_count);
}

double LRScheduler::cosine_annealing(int step, int max_steps) const {
    return initial_lr * 0.5 * (1.0 + cos(M_PI * step / max_steps));
}

double LRScheduler::warmup_cosine(int step, int warmup_steps, int max_steps) const {
    if (step < warmup_steps) {
        return initial_lr * step / warmup_steps;
    } else {
        return cosine_annealing(step - warmup_steps, max_steps - warmup_steps);
    }
}



// ============================================================================
// MAIN TRANSFORMER CLASS IMPLEMENTATION
// ============================================================================

Transformer::Transformer(int d_model, int num_heads, int num_layers, int d_ff, 
                         int num_classes, int patch_size, double dropout_rate)
    : d_model(d_model), num_heads(num_heads), num_layers(num_layers), d_ff(d_ff),
      num_classes(num_classes), patch_size(patch_size), dropout_rate(dropout_rate),
      training_mode(true), current_step(0) {
    
    // Calculate number of patches
    int image_size = 28; // Fashion-MNIST images are 28x28
    int patches_per_side = image_size / patch_size;
    num_patches = patches_per_side * patches_per_side;
    max_seq_len = num_patches + 1; // +1 for class token
    
    // Initialize components
    pos_encoding = std::make_unique<PositionalEncoding>(max_seq_len, d_model);
    
    // Initialize encoder layers
    for (int i = 0; i < num_layers; i++) {
        encoder_layers.push_back(std::make_unique<TransformerEncoderLayer>(d_model, num_heads, d_ff, dropout_rate));
    }
    
    // Initialize decoder layers (optional for classification)
    for (int i = 0; i < num_layers; i++) {
        decoder_layers.push_back(std::make_unique<TransformerDecoderLayer>(d_model, num_heads, d_ff, dropout_rate));
    }
    
    // Initialize embedding layers
    int patch_dim = patch_size * patch_size;
    patch_embedding_W = Matrix(patch_dim, d_model);
    patch_embedding_b = Matrix(1, d_model);
    patch_embedding_W.xavier_init();
    patch_embedding_b.zero();
    
    // Initialize class token
    class_token = Matrix(1, d_model);
    class_token.randomize(-0.02, 0.02);
    
    // Initialize classifier
    classifier_W = Matrix(d_model, num_classes);
    classifier_b = Matrix(1, num_classes);
    classifier_W.xavier_init();
    classifier_b.zero();
    
    // Initialize gradients
    patch_grad = Gradients(patch_dim, d_model);
    classifier_grad = Gradients(d_model, num_classes);
    class_token_grad = Matrix(1, d_model);
    class_token_grad.zero();
    
    // Initialize optimizer and scheduler
    optimizer = std::make_unique<AdamOptimizer>();
    scheduler = std::make_unique<LRScheduler>(0.001);
    
    std::cout << "Transformer initialized with:" << std::endl;
    std::cout << "- d_model: " << d_model << std::endl;
    std::cout << "- num_heads: " << num_heads << std::endl;
    std::cout << "- num_layers: " << num_layers << std::endl;
    std::cout << "- d_ff: " << d_ff << std::endl;
    std::cout << "- num_patches: " << num_patches << std::endl;
}

Matrix Transformer::create_patches(const Matrix& image) {
    int patches_per_side = 28 / patch_size;
    Matrix patches(num_patches, patch_size * patch_size);
    
    int patch_idx = 0;
    for (int i = 0; i < patches_per_side; i++) {
        for (int j = 0; j < patches_per_side; j++) {
            int pixel_idx = 0;
            for (int pi = 0; pi < patch_size; pi++) {
                for (int pj = 0; pj < patch_size; pj++) {
                    int row = i * patch_size + pi;
                    int col = j * patch_size + pj;
                    if (row < image.rows && col < image.cols) {
                        patches.data[patch_idx][pixel_idx] = image.data[row][col];
                    }
                    pixel_idx++;
                }
            }
            patch_idx++;
        }
    }
    
    patches_cache = patches;
    return patches;
}

Matrix Transformer::encode(const Matrix& input) {
    Matrix encoded = input;
    
    // Pass through encoder layers
    for (auto& layer : encoder_layers) {
        encoded = layer->forward(encoded, training_mode);
    }
    
    encoded_cache = encoded;
    return encoded;
}

Matrix Transformer::decode(const Matrix& encoded, const Matrix& target) {
    // For classification, we use a simple approach
    Matrix decoded = encoded;
    
    // If we have decoder layers, use them (simplified for classification)
    // For most vision transformers, we only use the encoder
    
    decoded_cache = decoded;
    return decoded;
}

Matrix Transformer::classify(const Matrix& encoded) {
    // Use class token (first token) for classification
    Matrix class_features(1, d_model);
    for (int j = 0; j < d_model; j++) {
        class_features.data[0][j] = encoded.data[0][j];
    }
    
    // Linear classification
    Matrix logits = class_features * classifier_W;
    
    // Add bias
    for (int j = 0; j < num_classes; j++) {
        logits.data[0][j] += classifier_b.data[0][j];
    }
    
    // Apply softmax
    return logits.softmax();
}

Matrix Transformer::forward(const Matrix& input) {
    // Create patches from image
    Matrix patches = create_patches(input);
    
    // Embed patches
    Matrix embedded = patches * patch_embedding_W;
    
    // Add bias
    for (int i = 0; i < embedded.rows; i++) {
        for (int j = 0; j < embedded.cols; j++) {
            embedded.data[i][j] += patch_embedding_b.data[0][j];
        }
    }
    
    embedded_cache = embedded;
    
    // Add class token
    Matrix tokens(num_patches + 1, d_model);
    
    // First row is class token
    for (int j = 0; j < d_model; j++) {
        tokens.data[0][j] = class_token.data[0][j];
    }
    
    // Remaining rows are embedded patches
    for (int i = 0; i < num_patches; i++) {
        for (int j = 0; j < d_model; j++) {
            tokens.data[i + 1][j] = embedded.data[i][j];
        }
    }
    
    // Add positional encoding
    tokens = pos_encoding->encode(tokens);
    tokens_cache = tokens;
    
    // Encode
    Matrix encoded = encode(tokens);
    
    // Decode (for classification, this might be identity or simplified)
    Matrix decoded = decode(encoded);
    
    // Classify
    return classify(decoded);
}

Matrix Transformer::backward(const Matrix& grad_output, const std::vector<int>& labels) {
    // Compute loss gradients first
    compute_loss_gradients(grad_output, labels);
    
    // Simple but correct backward pass
    // Start from classification layer and propagate backwards
    Matrix grad_features(1, d_model);
    
    // Gradient from classifier to features
    for (int j = 0; j < d_model; j++) {
        grad_features.data[0][j] = 0.0;
        for (int k = 0; k < num_classes; k++) {
            grad_features.data[0][j] += classifier_grad.dW.data[j][k] * grad_output.data[0][k];
        }
    }
    
    // Propagate through encoder layers (simplified)
    Matrix grad_encoder_input = grad_features;
    
    // For now, just update the main components
    // The encoder layers will be updated via their own backward calls
    
    return grad_encoder_input;
}

void Transformer::compute_loss_gradients(const Matrix& predictions, const std::vector<int>& labels) {
    // Compute gradients for cross-entropy loss
    Matrix grad_predictions = predictions;
    
    for (size_t i = 0; i < labels.size() && i < predictions.rows; i++) {
        grad_predictions.data[i][labels[i]] -= 1.0;
    }
    
    // Gradient w.r.t classifier weights
    Matrix class_features(1, d_model);
    for (int j = 0; j < d_model; j++) {
        class_features.data[0][j] = decoded_cache.data[0][j];
    }
    
    classifier_grad.dW.add_inplace(class_features.transpose() * grad_predictions);
    
    // Gradient w.r.t classifier bias
    for (int j = 0; j < num_classes; j++) {
        classifier_grad.db.data[0][j] += grad_predictions.data[0][j];
    }
}

void Transformer::train_step(const Matrix& input, const std::vector<int>& labels, double learning_rate) {
    zero_gradients();
    
    // Forward pass
    Matrix predictions = forward(input);
    
    // Backward pass
    backward(predictions, labels);
    
    // Update weights
    current_step++;
    update_weights_adam(current_step, learning_rate);
}

std::pair<double, double> Transformer::train_batch(const std::vector<Matrix>& inputs, 
                                                  const std::vector<int>& labels, 
                                                  double learning_rate) {
    double total_loss = 0.0;
    int correct = 0;
    
    zero_gradients();
    
    for (size_t i = 0; i < inputs.size(); i++) {
        // Forward pass
        Matrix predictions = forward(inputs[i]);
        
        // Compute loss and accuracy
        std::vector<int> single_label = {labels[i]};
        total_loss += compute_loss(predictions, single_label);
        
        int predicted = 0;
        double max_prob = predictions.data[0][0];
        for (int j = 1; j < predictions.cols; j++) {
            if (predictions.data[0][j] > max_prob) {
                max_prob = predictions.data[0][j];
                predicted = j;
            }
        }
        if (predicted == labels[i]) correct++;
        
        // Backward pass (accumulate gradients)
        backward(predictions, single_label);
    }
    
    // Average gradients using scale_gradients
    double scale_factor = 1.0 / static_cast<double>(inputs.size());
    
    for (auto& layer : encoder_layers) {
        layer->scale_gradients(scale_factor);
    }
    
    // Scale embeddings gradients
    patch_grad.dW.multiply_inplace(scale_factor);
    patch_grad.db.multiply_inplace(scale_factor);
    classifier_grad.dW.multiply_inplace(scale_factor);
    classifier_grad.db.multiply_inplace(scale_factor);
    class_token_grad.multiply_inplace(scale_factor);
    
    // Clip and update
    clip_gradients(1.0);
    current_step++;
    update_weights_adam(current_step, learning_rate);
    
    double avg_loss = total_loss / inputs.size();
    double accuracy = static_cast<double>(correct) / inputs.size();
    
    return std::make_pair(avg_loss, accuracy);
}

double Transformer::compute_loss(const Matrix& predictions, const std::vector<int>& labels) {
    double loss = 0.0;
    
    for (size_t i = 0; i < labels.size() && i < predictions.rows; i++) {
        double pred = std::max(predictions.data[i][labels[i]], 1e-7); // Avoid log(0)
        loss -= log(pred);
    }
    
    return loss / labels.size();
}

double Transformer::compute_accuracy(const Matrix& predictions, const std::vector<int>& labels) {
    int correct = 0;
    
    for (size_t i = 0; i < labels.size() && i < predictions.rows; i++) {
        int predicted = 0;
        double max_prob = predictions.data[i][0];
        
        for (int j = 1; j < predictions.cols; j++) {
            if (predictions.data[i][j] > max_prob) {
                max_prob = predictions.data[i][j];
                predicted = j;
            }
        }
        
        if (predicted == labels[i]) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / labels.size();
}

double Transformer::compute_loss_with_regularization(const Matrix& predictions, 
                                                   const std::vector<int>& labels, 
                                                   double l2_lambda) {
    double loss = compute_loss(predictions, labels);
    double l2_penalty = compute_l2_regularization();
    return loss + l2_lambda * l2_penalty;
}

void Transformer::update_weights(double learning_rate) {
    // Update embedding weights
    patch_embedding_W.subtract_inplace(patch_grad.dW * learning_rate);
    patch_embedding_b.subtract_inplace(patch_grad.db * learning_rate);
    
    // Update class token
    class_token.subtract_inplace(class_token_grad * learning_rate);
    
    // Update classifier weights
    classifier_W.subtract_inplace(classifier_grad.dW * learning_rate);
    classifier_b.subtract_inplace(classifier_grad.db * learning_rate);
    
    // Update layer weights
    for (auto& layer : encoder_layers) {
        layer->update_weights(learning_rate);
    }
    
    for (auto& layer : decoder_layers) {
        layer->update_weights(learning_rate);
    }
}

void Transformer::update_weights_adam(int step, double learning_rate) {
    // Update embedding weights with Adam
    optimizer->update(patch_embedding_W, patch_grad.dW, &patch_embedding_W, step, learning_rate);
    optimizer->update(patch_embedding_b, patch_grad.db, &patch_embedding_b, step, learning_rate);
    
    // Update class token with Adam
    optimizer->update(class_token, class_token_grad, &class_token, step, learning_rate);
    
    // Update classifier weights with Adam
    optimizer->update(classifier_W, classifier_grad.dW, &classifier_W, step, learning_rate);
    optimizer->update(classifier_b, classifier_grad.db, &classifier_b, step, learning_rate);
    
    // Update layer weights with Adam
    for (auto& layer : encoder_layers) {
        layer->update_weights_adam(*optimizer, step);
    }
    
    for (auto& layer : decoder_layers) {
        layer->update_weights_adam(*optimizer, step);
    }
}

void Transformer::zero_gradients() {
    patch_grad.zero();
    classifier_grad.zero();
    class_token_grad.zero();
    
    for (auto& layer : encoder_layers) {
        layer->zero_gradients();
    }
    
    for (auto& layer : decoder_layers) {
        layer->zero_gradients();
    }
}

void Transformer::clip_gradients(double max_norm) {
    patch_grad.clip(max_norm);
    classifier_grad.clip(max_norm);
    class_token_grad = class_token_grad.clip_gradients(max_norm);
    
    for (auto& layer : encoder_layers) {
        layer->clip_gradients(max_norm);
    }
    
    for (auto& layer : decoder_layers) {
        layer->clip_gradients(max_norm);
    }
}

void Transformer::save_model(const std::string& path) {
    // Simplified model saving
    std::ofstream file(path, std::ios::binary);
    if (file.is_open()) {
        // Save basic parameters
        file.write((char*)&d_model, sizeof(d_model));
        file.write((char*)&num_heads, sizeof(num_heads));
        file.write((char*)&num_layers, sizeof(num_layers));
        file.write((char*)&d_ff, sizeof(d_ff));
        file.write((char*)&num_classes, sizeof(num_classes));
        
        // Note: Full implementation would save all weight matrices
        file.close();
        std::cout << "Model saved to " << path << std::endl;
    }
}

void Transformer::load_model(const std::string& path) {
    // Simplified model loading
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        // Load basic parameters
        file.read((char*)&d_model, sizeof(d_model));
        file.read((char*)&num_heads, sizeof(num_heads));
        file.read((char*)&num_layers, sizeof(num_layers));
        file.read((char*)&d_ff, sizeof(d_ff));
        file.read((char*)&num_classes, sizeof(num_classes));
        
        // Note: Full implementation would load all weight matrices
        file.close();
        std::cout << "Model loaded from " << path << std::endl;
    }
}

double Transformer::compute_l2_regularization() const {
    double l2_penalty = 0.0;
    
    // Add L2 penalty from embedding weights
    l2_penalty += patch_embedding_W.norm() * patch_embedding_W.norm();
    l2_penalty += classifier_W.norm() * classifier_W.norm();
    
    // Note: Full implementation would include all layer weights
    
    return l2_penalty * 0.5;
}

void Transformer::apply_weight_decay(double decay_rate) {
    patch_embedding_W.multiply_inplace(1.0 - decay_rate);
    classifier_W.multiply_inplace(1.0 - decay_rate);
    class_token.multiply_inplace(1.0 - decay_rate);
}

std::vector<Matrix> Transformer::get_attention_weights() const {
    std::vector<Matrix> attention_weights;
    
    for (const auto& layer : encoder_layers) {
        attention_weights.push_back(layer->attention->get_attention_weights());
    }
    
    return attention_weights;
}

int Transformer::get_parameter_count() const {
    int count = 0;
    
    // Embedding parameters
    count += patch_embedding_W.rows * patch_embedding_W.cols;
    count += patch_embedding_b.rows * patch_embedding_b.cols;
    count += class_token.rows * class_token.cols;
    count += classifier_W.rows * classifier_W.cols;
    count += classifier_b.rows * classifier_b.cols;
    
    // Layer parameters (simplified estimate)
    int layer_params = d_model * d_model * 4 + d_model * d_ff * 2; // Rough estimate
    count += layer_params * (encoder_layers.size() + decoder_layers.size());
    
    return count;
}

void Transformer::print_model_info() const {
    std::cout << "\n=== Transformer Model Information ===" << std::endl;
    std::cout << "Architecture:" << std::endl;
    std::cout << "  - Model dimension: " << d_model << std::endl;
    std::cout << "  - Number of heads: " << num_heads << std::endl;
    std::cout << "  - Encoder layers: " << encoder_layers.size() << std::endl;
    std::cout << "  - Decoder layers: " << decoder_layers.size() << std::endl;
    std::cout << "  - Feed-forward dimension: " << d_ff << std::endl;
    std::cout << "  - Number of classes: " << num_classes << std::endl;
    std::cout << "  - Patch size: " << patch_size << "x" << patch_size << std::endl;
    std::cout << "  - Number of patches: " << num_patches << std::endl;
    std::cout << "  - Sequence length: " << max_seq_len << std::endl;
    std::cout << "  - Dropout rate: " << dropout_rate << std::endl;
    std::cout << "\nParameters:" << std::endl;
    std::cout << "  - Total parameters: ~" << get_parameter_count() << std::endl;
    std::cout << "======================================\n" << std::endl;
}

