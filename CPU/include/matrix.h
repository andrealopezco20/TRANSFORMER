#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>

class Matrix {
public:
    std::vector<std::vector<double>> data;
    int rows, cols;

    Matrix(int rows, int cols);
    Matrix(const std::vector<std::vector<double>>& data);
    Matrix();

    // Basic operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;
    
    // Element access
    std::vector<double>& operator[](int index);
    const std::vector<double>& operator[](int index) const;
    
    // Utility functions
    Matrix transpose() const;
    void randomize(double min = -1.0, double max = 1.0);
    void xavier_init();
    void zero();
    void print() const;
    
    // Activation functions
    Matrix relu() const;
    Matrix relu_derivative() const;
    Matrix sigmoid() const;
    Matrix sigmoid_derivative() const;
    Matrix tanh_activation() const;
    Matrix tanh_derivative() const;
    Matrix softmax() const;
    Matrix softmax_derivative() const;
    Matrix gelu() const;
    Matrix gelu_derivative() const;
    
    // Advanced operations
    Matrix layer_norm() const;
    Matrix layer_norm_derivative() const;
    Matrix dropout(double rate) const;
    Matrix hadamard(const Matrix& other) const; // Element-wise multiplication
    Matrix mean(int axis = 0) const;
    Matrix std_dev(int axis = 0) const;
    double sum() const;
    double mean_value() const;
    double norm() const;
    Matrix slice(int start_row, int end_row, int start_col, int end_col) const;
    void set_slice(int start_row, int start_col, const Matrix& other);
    Matrix reshape(int new_rows, int new_cols) const;
    
    // Gradient operations
    Matrix clip_gradients(double max_norm) const;
    void add_inplace(const Matrix& other);
    void subtract_inplace(const Matrix& other);
    void multiply_inplace(double scalar);
    
    // Statistical functions
    double variance() const;
    Matrix normalize() const;
    
    // Dimensions
    int get_rows() const { return rows; }
    int get_cols() const { return cols; }
};

// Utility functions
Matrix concatenate(const Matrix& a, const Matrix& b, int axis = 0);
Matrix zeros(int rows, int cols);
Matrix ones(int rows, int cols);
Matrix eye(int size);
double uniform_random(double min, double max);
double normal_random(double mean = 0.0, double std = 1.0);

#endif