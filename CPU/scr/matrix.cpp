#include "../include/matrix.h"
#include <stdexcept>
#include <iomanip>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
    data.resize(rows, std::vector<double>(cols, 0.0));
}

Matrix::Matrix(const std::vector<std::vector<double>>& data) : data(data) {
    rows = data.size();
    cols = rows > 0 ? data[0].size() : 0;
}

Matrix::Matrix() : rows(0), cols(0) {}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }
    return result;
}



Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] - other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            for (int k = 0; k < cols; k++) {
                result.data[i][j] += data[i][k] * other.data[k][j];
            }
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] * scalar;
        }
    }
    return result;
}

Matrix Matrix::operator/(double scalar) const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] / scalar;
        }
    }
    return result;
}

std::vector<double>& Matrix::operator[](int index) {
    return data[index];
}

const std::vector<double>& Matrix::operator[](int index) const {
    return data[index];
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[j][i] = data[i][j];
        }
    }
    return result;
}

void Matrix::randomize(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = dis(gen);
        }
    }
}

void Matrix::xavier_init() {
    double limit = sqrt(6.0 / (rows + cols));
    randomize(-limit, limit);
}

void Matrix::zero() {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = 0.0;
        }
    }
}

void Matrix::print() const {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(8) << std::setprecision(4) << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

Matrix Matrix::relu() const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = std::max(0.0, data[i][j]);
        }
    }
    return result;
}

Matrix Matrix::relu_derivative() const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] > 0 ? 1.0 : 0.0;
        }
    }
    return result;
}

Matrix Matrix::sigmoid() const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = 1.0 / (1.0 + exp(-data[i][j]));
        }
    }
    return result;
}

Matrix Matrix::sigmoid_derivative() const {
    Matrix s = sigmoid();
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = s.data[i][j] * (1.0 - s.data[i][j]);
        }
    }
    return result;
}

Matrix Matrix::tanh_activation() const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = tanh(data[i][j]);
        }
    }
    return result;
}

Matrix Matrix::tanh_derivative() const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double t = tanh(data[i][j]);
            result.data[i][j] = 1.0 - t * t;
        }
    }
    return result;
}

Matrix Matrix::softmax() const {
    Matrix result(rows, cols);
    
    for (int i = 0; i < rows; i++) {
        double max_val = data[i][0];
        for (int j = 1; j < cols; j++) {
            max_val = std::max(max_val, data[i][j]);
        }
        
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = exp(data[i][j] - max_val);
            sum += result.data[i][j];
        }
        
        for (int j = 0; j < cols; j++) {
            result.data[i][j] /= sum;
        }
    }
    
    return result;
}

Matrix Matrix::softmax_derivative() const {
    Matrix s = softmax();
    Matrix result(rows, cols);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < cols; k++) {
                if (j == k) {
                    result.data[i][j] += s.data[i][j] * (1.0 - s.data[i][k]);
                } else {
                    result.data[i][j] -= s.data[i][j] * s.data[i][k];
                }
            }
        }
    }
    return result;
}

Matrix Matrix::gelu() const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double x = data[i][j];
            result.data[i][j] = 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
        }
    }
    return result;
}

Matrix Matrix::gelu_derivative() const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double x = data[i][j];
            double tanh_term = tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x));
            double sech2_term = 1.0 - tanh_term * tanh_term;
            result.data[i][j] = 0.5 * (1.0 + tanh_term) + 
                               0.5 * x * sech2_term * sqrt(2.0 / M_PI) * (1.0 + 3.0 * 0.044715 * x * x);
        }
    }
    return result;
}

Matrix Matrix::layer_norm() const {
    Matrix result(rows, cols);
    const double eps = 1e-6;
    
    for (int i = 0; i < rows; i++) {
        double mean = 0.0;
        for (int j = 0; j < cols; j++) {
            mean += data[i][j];
        }
        mean /= cols;
        
        double variance = 0.0;
        for (int j = 0; j < cols; j++) {
            double diff = data[i][j] - mean;
            variance += diff * diff;
        }
        variance /= cols;
        
        double std_dev = sqrt(variance + eps);
        
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = (data[i][j] - mean) / std_dev;
        }
    }
    return result;
}