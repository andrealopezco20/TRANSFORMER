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