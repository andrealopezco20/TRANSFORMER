#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "matrix.h"
#include <memory>
#include <vector>
#include <unordered_map>

class AdamOptimizer;

// Cache para activaciones durante forward pass
struct ForwardCache {
    Matrix input;
    Matrix query, key, value;
    Matrix attention_scores;
    Matrix attention_weights;
    Matrix attended_values;
    Matrix ff_hidden;
    Matrix layer_norm_out1, layer_norm_out2;
    Matrix residual1, residual2;
};

// Clase para almacenar gradientes
struct Gradients {
    Matrix dW, db;
    
    Gradients() = default;
    Gradients(int rows, int cols) : dW(rows, cols), db(1, cols) {
        dW.zero();
        db.zero();
    }
    
    void zero() {
        dW.zero();
        db.zero();
    }
    
    void clip(double max_norm) {
        dW = dW.clip_gradients(max_norm);
        db = db.clip_gradients(max_norm);
    }
};

// Positional Encoding con gradientes
class PositionalEncoding {
public:
    int max_len, d_model;
    Matrix encoding;
    
    PositionalEncoding(int max_len, int d_model);
    Matrix encode(const Matrix& input) const;
    Matrix backward(const Matrix& grad_output) const;
};

// Multi-Head Attention con retropropagación completa
class MultiHeadAttention {
public:
    int d_model, num_heads, d_k;
    Matrix W_q, W_k, W_v, W_o;
    Matrix b_q, b_k, b_v, b_o;
    
    // Gradientes
    Gradients grad_q, grad_k, grad_v, grad_o;
    
    // Cache para backward pass
    mutable ForwardCache cache;
    
    MultiHeadAttention(int d_model, int num_heads);
    
    Matrix attention(const Matrix& Q, const Matrix& K, const Matrix& V, bool mask = false) const;
    Matrix forward(const Matrix& query, const Matrix& key, const Matrix& value, bool mask = false);
    
    // Backward pass
    std::tuple<Matrix, Matrix, Matrix> backward(const Matrix& grad_output);
    Matrix attention_backward(const Matrix& grad_output, const Matrix& Q, const Matrix& K, const Matrix& V) const;
    
    // Weight updates
    void update_weights(double learning_rate);
    void update_weights_adam(AdamOptimizer& optimizer, int step);
    
    // Utilities
    void zero_gradients();
    void clip_gradients(double max_norm);
    Matrix get_attention_weights() const { return cache.attention_weights; }
};

// Feed Forward Network con gradientes
class FeedForward {
public:
    int d_model, d_ff;
    Matrix W1, W2, b1, b2;
    
    // Gradientes
    Gradients grad_1, grad_2;
    
    // Cache
    mutable Matrix hidden_cache;
    mutable Matrix input_cache;
    
    FeedForward(int d_model, int d_ff);
    
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& grad_output);
    
    void update_weights(double learning_rate);
    void update_weights_adam(AdamOptimizer& optimizer, int step);
    void zero_gradients();
    void clip_gradients(double max_norm);
};

// Layer Normalization con gradientes
class LayerNorm {
public:
    int d_model;
    double eps;
    Matrix gamma, beta;
    
    // Gradientes
    Matrix grad_gamma, grad_beta;
    
    // Cache para backward
    mutable Matrix input_cache;
    mutable Matrix normalized_cache;
    mutable Matrix mean_cache;
    mutable Matrix std_cache;
    
    LayerNorm(int d_model, double eps = 1e-6);
    
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& grad_output);
    
    void update_weights(double learning_rate);
    void update_weights_adam(AdamOptimizer& optimizer, int step);
    void zero_gradients();
};

// Transformer Encoder Layer
class TransformerEncoderLayer {
public:
    double dropout_rate;
    std::unique_ptr<MultiHeadAttention> attention;
    std::unique_ptr<FeedForward> feed_forward;
    std::unique_ptr<LayerNorm> norm1;
    std::unique_ptr<LayerNorm> norm2;
    
    // Cache para backward
    mutable Matrix input_cache;
    mutable Matrix attention_out_cache;
    mutable Matrix norm1_out_cache;
    mutable Matrix ff_out_cache;
    
    TransformerEncoderLayer(int d_model, int num_heads, int d_ff, double dropout_rate = 0.1);
    
    Matrix forward(const Matrix& input, bool training = true);
    Matrix backward(const Matrix& grad_output);
    
    void update_weights(double learning_rate);
    void update_weights_adam(AdamOptimizer& optimizer, int step);
    void zero_gradients();
    void clip_gradients(double max_norm);
    void scale_gradients(double scale_factor);
};

// Transformer Decoder Layer  
class TransformerDecoderLayer {
public:
    double dropout_rate;
    std::unique_ptr<MultiHeadAttention> self_attention;
    std::unique_ptr<MultiHeadAttention> cross_attention;
    std::unique_ptr<FeedForward> feed_forward;
    std::unique_ptr<LayerNorm> norm1;
    std::unique_ptr<LayerNorm> norm2;
    std::unique_ptr<LayerNorm> norm3;
    
    // Cache
    mutable Matrix input_cache;
    mutable Matrix encoder_cache;
    mutable Matrix self_att_cache;
    mutable Matrix cross_att_cache;
    mutable Matrix norm1_cache, norm2_cache;
    mutable Matrix ff_cache;
    
    TransformerDecoderLayer(int d_model, int num_heads, int d_ff, double dropout_rate = 0.1);
    
    Matrix forward(const Matrix& input, const Matrix& encoder_output, bool training = true);
    std::pair<Matrix, Matrix> backward(const Matrix& grad_output);
    
    void update_weights(double learning_rate);
    void update_weights_adam(AdamOptimizer& optimizer, int step);
    void zero_gradients();
    void clip_gradients(double max_norm);
    void scale_gradients(double scale_factor);
};

// Adam Optimizer
class AdamOptimizer {
private:
    double beta1, beta2, eps;
    std::unordered_map<void*, Matrix> m_weights, v_weights;
    std::unordered_map<void*, Matrix> m_biases, v_biases;
    
public:
    AdamOptimizer(double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);
    
    void update(Matrix& weight, const Matrix& gradient, void* param_id, int step, double learning_rate);
    void clear();
};

// Learning Rate Scheduler
class LRScheduler {
private:
    double initial_lr;
    double decay_factor;
    int decay_steps;
    
public:
    LRScheduler(double initial_lr, double decay_factor = 0.95, int decay_steps = 1000);
    double get_lr(int step) const;
    double cosine_annealing(int step, int max_steps) const;
    double warmup_cosine(int step, int warmup_steps, int max_steps) const;
};

// Main Transformer Class
class Transformer {
private:
    int d_model, num_heads, num_layers, d_ff, num_classes, patch_size;
    int num_patches, max_seq_len;
    double dropout_rate;
    
    // Components
    std::unique_ptr<PositionalEncoding> pos_encoding;
    std::vector<std::unique_ptr<TransformerEncoderLayer>> encoder_layers;
    std::vector<std::unique_ptr<TransformerDecoderLayer>> decoder_layers;
    
    // Embeddings
    Matrix patch_embedding_W, patch_embedding_b;
    Matrix class_token;
    Matrix classifier_W, classifier_b;
    
    // Gradientes de embeddings
    Gradients patch_grad, classifier_grad;
    Matrix class_token_grad;
    
    // Optimizer y scheduler
    std::unique_ptr<AdamOptimizer> optimizer;
    std::unique_ptr<LRScheduler> scheduler;
    
    // Training state
    bool training_mode;
    int current_step;
    
    // Cache para backward pass
    mutable Matrix patches_cache;
    mutable Matrix embedded_cache;
    mutable Matrix tokens_cache;
    mutable Matrix encoded_cache;
    mutable Matrix decoded_cache;
    
public:
    Transformer(int d_model, int num_heads, int num_layers, int d_ff, 
                int num_classes, int patch_size = 4, double dropout_rate = 0.1);
    
    // Forward pass
    Matrix create_patches(const Matrix& image);
    Matrix encode(const Matrix& input);
    Matrix decode(const Matrix& encoded, const Matrix& target = Matrix());
    Matrix classify(const Matrix& encoded);
    Matrix forward(const Matrix& input);
    
    // Backward pass
    Matrix backward(const Matrix& grad_output, const std::vector<int>& labels);
    void compute_loss_gradients(const Matrix& predictions, const std::vector<int>& labels);
    
    // Training functions
    void train_step(const Matrix& input, const std::vector<int>& labels, double learning_rate);
    std::pair<double, double> train_batch(const std::vector<Matrix>& inputs, 
                                         const std::vector<int>& labels, 
                                         double learning_rate);
    
    // Loss and metrics
    double compute_loss(const Matrix& predictions, const std::vector<int>& labels);
    double compute_accuracy(const Matrix& predictions, const std::vector<int>& labels);
    double compute_loss_with_regularization(const Matrix& predictions, 
                                           const std::vector<int>& labels, 
                                           double l2_lambda = 0.01);
    
    // Weight management
    void update_weights(double learning_rate);
    void update_weights_adam(int step, double learning_rate);
    void zero_gradients();
    void clip_gradients(double max_norm = 1.0);
    
    // Training utilities
    void set_training(bool training) { training_mode = training; }
    void save_model(const std::string& path);
    void load_model(const std::string& path);
    
    // Regularization
    double compute_l2_regularization() const;
    void apply_weight_decay(double decay_rate);
    
    // Attention visualization
    std::vector<Matrix> get_attention_weights() const;
    
    // Model info
    int get_parameter_count() const;
    void print_model_info() const;
};

#endif