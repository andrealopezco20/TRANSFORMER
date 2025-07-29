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
