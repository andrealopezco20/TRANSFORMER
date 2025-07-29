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