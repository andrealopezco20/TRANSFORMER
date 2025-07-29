#pragma once

// Interfaz de operaciones aceleradas (CPU optimizado o CUDA)
// No requiere CUDA para compilar

extern "C" {
    void cuda_matrix_scalar_mul(const float* A, float* C, float scalar, int size);
    void cuda_matmul(const float* A, const float* B, float* C, int m, int n, int p);
    void cuda_matrix_add(const float* A, const float* B, float* C, int size);
    void cuda_matrix_sub(const float* A, const float* B, float* C, int size);
    void cuda_matrix_transpose(const float* A, float* B, int rows, int cols);
    void cuda_matrix_softmax(const float* A, float* B, int rows, int cols);
    void cuda_matrix_relu(const float* A, float* B, int size);
    void cuda_matrix_gelu(const float* A, float* B, int size);
    void cuda_matrix_dropout(const float* A, float* B, int size, float dropout_rate, unsigned int seed);
    void cuda_matrix_sigmoid(const float* A, float* B, int size);
    void cuda_matrix_tanh(const float* A, float* B, int size);
    void cuda_matrix_batchnorm(const float* A, float* B, int size, float mean, float var, float gamma, float beta);
    void cuda_matrix_layernorm(const float* A, float* B, int rows, int cols, float gamma, float beta);
    void cuda_matrix_reduce_sum(const float* A, float* B, int size);
    void cuda_matrix_reduce_max(const float* A, float* B, int size);
    void cuda_matrix_exp(const float* A, float* B, int size);
}
