
#include "../include/matrix.h"
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifdef USE_CUDA
extern "C" void cuda_matmul(const float* A, const float* B, float* C, int M, int N, int K);
#include "../include/cuda_ops.h"
#endif
// Suma de matrices en GPU
Matrix Matrix::cudaAdd(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix dimensions must match for addition (GPU)");
#ifdef USE_CUDA
    Matrix result(rows, cols);
    std::vector<float> A(rows * cols), B(rows * cols), C(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            A[i * cols + j] = static_cast<float>(data[i][j]);
            B[i * cols + j] = static_cast<float>(other.data[i][j]);
        }
    cuda_matrix_add(A.data(), B.data(), C.data(), rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result.data[i][j] = static_cast<double>(C[i * cols + j]);
    return result;
#else
    throw std::runtime_error("CUDA no está habilitado. Compila con -DUSE_CUDA");
#endif
}
Matrix Matrix::cudaMultiply(float scalar) const {
    Matrix result(rows, cols);
    std::vector<float> A(rows * cols);
    std::vector<float> C(rows * cols, 0.0f);
    // Copia los datos de la matriz original a un buffer float
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double val = data[i][j];
            if (std::isnan(val) || std::isinf(val)) val = 0.0; // Evita valores no válidos
            A[i * cols + j] = static_cast<float>(val);
        }
    }
    cuda_matrix_scalar_mul(A.data(), C.data(), scalar, rows * cols);
    // Copia el resultado de vuelta a double
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.data[i][j] = static_cast<double>(C[i * cols + j]);
        }
    }
    return result;
}
// Resta de matrices en GPU
Matrix Matrix::cudaSub(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix dimensions must match for subtraction (GPU)");
#ifdef USE_CUDA
    Matrix result(rows, cols);
    std::vector<float> A(rows * cols), B(rows * cols), C(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            A[i * cols + j] = static_cast<float>(data[i][j]);
            B[i * cols + j] = static_cast<float>(other.data[i][j]);
        }
    cuda_matrix_sub(A.data(), B.data(), C.data(), rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result.data[i][j] = static_cast<double>(C[i * cols + j]);
    return result;
#else
    throw std::runtime_error("CUDA no está habilitado. Compila con -DUSE_CUDA");
#endif
}

// Transposición en GPU
Matrix Matrix::cudaTranspose() const {
#ifdef USE_CUDA
    Matrix result(cols, rows);
    std::vector<float> A(rows * cols), B(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A[i * cols + j] = static_cast<float>(data[i][j]);
    cuda_matrix_transpose(A.data(), B.data(), rows, cols);
    for (int i = 0; i < cols; ++i)
        for (int j = 0; j < rows; ++j)
            result.data[i][j] = static_cast<double>(B[i * rows + j]);
    return result;
#else
    throw std::runtime_error("CUDA no está habilitado. Compila con -DUSE_CUDA");
#endif
}

// Softmax en GPU
Matrix Matrix::cudaSoftmax() const {
#ifdef USE_CUDA
    Matrix result(rows, cols);
    std::vector<float> A(rows * cols), B(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A[i * cols + j] = static_cast<float>(data[i][j]);
    cuda_matrix_softmax(A.data(), B.data(), rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result.data[i][j] = static_cast<double>(B[i * cols + j]);
    return result;
#else
    throw std::runtime_error("CUDA no está habilitado. Compila con -DUSE_CUDA");
#endif
}

// ReLU en GPU
Matrix Matrix::cudaRelu() const {
#ifdef USE_CUDA
    Matrix result(rows, cols);
    std::vector<float> A(rows * cols), B(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A[i * cols + j] = static_cast<float>(data[i][j]);
    cuda_matrix_relu(A.data(), B.data(), rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result.data[i][j] = static_cast<double>(B[i * cols + j]);
    return result;
#else
    throw std::runtime_error("CUDA no está habilitado. Compila con -DUSE_CUDA");
#endif
}

// GELU en GPU - CRÍTICO para Transformers
Matrix Matrix::cudaGelu() const {
#ifdef USE_CUDA
    Matrix result(rows, cols);
    std::vector<float> A(rows * cols), B(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A[i * cols + j] = static_cast<float>(data[i][j]);
    cuda_matrix_gelu(A.data(), B.data(), rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result.data[i][j] = static_cast<double>(B[i * cols + j]);
    return result;
#else
    throw std::runtime_error("CUDA no está habilitado. Compila con -DUSE_CUDA");
#endif
}

// Dropout en GPU - CRÍTICO IMPLEMENTADO para regularización >85% accuracy
Matrix Matrix::cudaDropout(double dropout_rate, unsigned int seed) const {
#ifdef USE_CUDA
    Matrix result(rows, cols);
    std::vector<float> A(rows * cols), B(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A[i * cols + j] = static_cast<float>(data[i][j]);
    cuda_matrix_dropout(A.data(), B.data(), rows * cols, static_cast<float>(dropout_rate), seed);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result.data[i][j] = static_cast<double>(B[i * cols + j]);
    return result;
#else
    throw std::runtime_error("CUDA no está habilitado. Compila con -DUSE_CUDA");
#endif
}

// Sigmoid en GPU
Matrix Matrix::cudaSigmoid() const {
#ifdef USE_CUDA
    Matrix result(rows, cols);
    std::vector<float> A(rows * cols), B(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A[i * cols + j] = static_cast<float>(data[i][j]);
    cuda_matrix_sigmoid(A.data(), B.data(), rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result.data[i][j] = static_cast<double>(B[i * cols + j]);
    return result;
#else
    throw std::runtime_error("CUDA no está habilitado. Compila con -DUSE_CUDA");
#endif
}

// Tanh en GPU
Matrix Matrix::cudaTanh() const {
#ifdef USE_CUDA
    Matrix result(rows, cols);
    std::vector<float> A(rows * cols), B(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A[i * cols + j] = static_cast<float>(data[i][j]);
    cuda_matrix_tanh(A.data(), B.data(), rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result.data[i][j] = static_cast<double>(B[i * cols + j]);
    return result;
#else
    throw std::runtime_error("CUDA no está habilitado. Compila con -DUSE_CUDA");
#endif
}

// Reducción suma en GPU
double Matrix::cudaReduceSum() const {
#ifdef USE_CUDA
    std::vector<float> A(rows * cols);
    float result = 0.0f;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A[i * cols + j] = static_cast<float>(data[i][j]);
    cuda_matrix_reduce_sum(A.data(), &result, rows * cols);
    return static_cast<double>(result);
#else
    throw std::runtime_error("CUDA no está habilitado. Compila con -DUSE_CUDA");
#endif
}

// Reducción máximo en GPU
double Matrix::cudaReduceMax() const {
#ifdef USE_CUDA
    std::vector<float> A(rows * cols);
    float result = 0.0f;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A[i * cols + j] = static_cast<float>(data[i][j]);
    cuda_matrix_reduce_max(A.data(), &result, rows * cols);
    return static_cast<double>(result);
#else
    throw std::runtime_error("CUDA no está habilitado. Compila con -DUSE_CUDA");
#endif
}

// Exponenciación en GPU
Matrix Matrix::cudaExp() const {
#ifdef USE_CUDA
    Matrix result(rows, cols);
    std::vector<float> A(rows * cols), B(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A[i * cols + j] = static_cast<float>(data[i][j]);
    cuda_matrix_exp(A.data(), B.data(), rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result.data[i][j] = static_cast<double>(B[i * cols + j]);
    return result;
#else
    throw std::runtime_error("CUDA no está habilitado. Compila con -DUSE_CUDA");
#endif
}


// Multiplicación matricial en GPU usando CUDA
Matrix Matrix::cudaMultiply(const Matrix& other) const {
    // Si cualquiera de las dos matrices es 1x1, usar multiplicación escalar
    if (other.rows == 1 && other.cols == 1) {
        float scalar = static_cast<float>(other.data[0][0]);
        Matrix result(rows, cols);
        std::vector<float> A(rows * cols);
        std::vector<float> C(rows * cols, 0.0f);
        int idx = 0;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j, ++idx)
                A[idx] = static_cast<float>(data[i][j]);
        cuda_matrix_scalar_mul(A.data(), C.data(), scalar, rows * cols);
        idx = 0;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j, ++idx)
                result.data[i][j] = static_cast<double>(C[idx]);
        return result;
    }
    if (rows == 1 && cols == 1) {
        float scalar = static_cast<float>(data[0][0]);
        Matrix result(other.rows, other.cols);
        std::vector<float> B(other.rows * other.cols);
        std::vector<float> C(other.rows * other.cols, 0.0f);
        for (int i = 0; i < other.rows; ++i)
            for (int j = 0; j < other.cols; ++j)
                B[i * other.cols + j] = static_cast<float>(other.data[i][j]);
        cuda_matrix_scalar_mul(B.data(), C.data(), scalar, other.rows * other.cols);
        for (int i = 0; i < other.rows; ++i)
            for (int j = 0; j < other.cols; ++j)
                result.data[i][j] = static_cast<double>(C[i * other.cols + j]);
        return result;
    }
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication (GPU)");
    }
#ifdef USE_CUDA
    Matrix result(rows, other.cols);
    std::vector<float> A(rows * cols);
    std::vector<float> B(other.rows * other.cols);
    std::vector<float> C(rows * other.cols, 0.0f);
    // Copiar datos a float
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A[i * cols + j] = static_cast<float>(data[i][j]);
    for (int i = 0; i < other.rows; ++i)
        for (int j = 0; j < other.cols; ++j)
            B[i * other.cols + j] = static_cast<float>(other.data[i][j]);
    // Llamar kernel CUDA
    cuda_matmul(A.data(), B.data(), C.data(), rows, cols, other.cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < other.cols; ++j)
            result.data[i][j] = static_cast<double>(C[i * other.cols + j]);
    return result;
#else
    throw std::runtime_error("CUDA no está habilitado. Compila con -DUSE_CUDA");
#endif
}

//#ifndef M_PI
//#define M_PI 3.14159265358979323846
//#endif
#ifdef USE_CUDA
extern void cuda_matmul(const float* A, const float* B, float* C, int M, int N, int K);
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
    // Suma normal
    if (rows == other.rows && cols == other.cols) {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }
    // Broadcasting escalar
    if (other.rows == 1 && other.cols == 1) {
        double scalar = other.data[0][0];
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + scalar;
            }
        }
        return result;
    }
    // Broadcasting columna (sumar vector columna a cada columna)
    if (rows == other.rows && other.cols == 1) {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][0];
            }
        }
        return result;
    }
    // Broadcasting fila (sumar vector fila a cada fila)
    if (cols == other.cols && other.rows == 1) {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[0][j];
            }
        }
        return result;
    }
    throw std::invalid_argument("Matrix dimensions incompatible for addition");
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
    // Caso especial: multiplicación por matriz 1x1 (escalar)
    if (other.rows == 1 && other.cols == 1) {
        double scalar = other.data[0][0];
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }
    // Multiplicación estándar de matrices
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

Matrix Matrix::layer_norm_derivative() const {
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
        double inv_std = 1.0 / std_dev;
        
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = inv_std;
        }
    }
    return result;
}

Matrix Matrix::dropout(double rate) const {
    Matrix result(rows, cols);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    double keep_prob = 1.0 - rate;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (dis(gen) < keep_prob) {
                result.data[i][j] = data[i][j] / keep_prob;
            } else {
                result.data[i][j] = 0.0;
            }
        }
    }
    return result;
}

Matrix Matrix::hadamard(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for Hadamard product");
    }
    
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] * other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::mean(int axis) const {
    if (axis == 0) { // Mean along rows
        Matrix result(1, cols);
        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int i = 0; i < rows; i++) {
                sum += data[i][j];
            }
            result.data[0][j] = sum / rows;
        }
        return result;
    } else { // Mean along columns
        Matrix result(rows, 1);
        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                sum += data[i][j];
            }
            result.data[i][0] = sum / cols;
        }
        return result;
    }
}

Matrix Matrix::std_dev(int axis) const {
    Matrix mean_matrix = mean(axis);
    
    if (axis == 0) {
        Matrix result(1, cols);
        for (int j = 0; j < cols; j++) {
            double variance = 0.0;
            for (int i = 0; i < rows; i++) {
                double diff = data[i][j] - mean_matrix.data[0][j];
                variance += diff * diff;
            }
            result.data[0][j] = sqrt(variance / rows);
        }
        return result;
    } else {
        Matrix result(rows, 1);
        for (int i = 0; i < rows; i++) {
            double variance = 0.0;
            for (int j = 0; j < cols; j++) {
                double diff = data[i][j] - mean_matrix.data[i][0];
                variance += diff * diff;
            }
            result.data[i][0] = sqrt(variance / cols);
        }
        return result;
    }
}

double Matrix::sum() const {
    double total = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            total += data[i][j];
        }
    }
    return total;
}

double Matrix::mean_value() const {
    return sum() / (rows * cols);
}

double Matrix::norm() const {
    double sum_sq = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum_sq += data[i][j] * data[i][j];
        }
    }
    return sqrt(sum_sq);
}

Matrix Matrix::slice(int start_row, int end_row, int start_col, int end_col) const {
    Matrix result(end_row - start_row, end_col - start_col);
    for (int i = start_row; i < end_row; i++) {
        for (int j = start_col; j < end_col; j++) {
            result.data[i - start_row][j - start_col] = data[i][j];
        }
    }
    return result;
}

void Matrix::set_slice(int start_row, int start_col, const Matrix& other) {
    for (int i = 0; i < other.rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            if (start_row + i < rows && start_col + j < cols) {
                data[start_row + i][start_col + j] = other.data[i][j];
            }
        }
    }
}

Matrix Matrix::reshape(int new_rows, int new_cols) const {
    if (new_rows * new_cols != rows * cols) {
        throw std::invalid_argument("Total elements must remain the same");
    }
    
    Matrix result(new_rows, new_cols);
    int idx = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int new_i = idx / new_cols;
            int new_j = idx % new_cols;
            result.data[new_i][new_j] = data[i][j];
            idx++;
        }
    }
    return result;
}

Matrix Matrix::clip_gradients(double max_norm) const {
    double current_norm = norm();
    if (current_norm <= max_norm) {
        return *this;
    }
    
    double scale = max_norm / current_norm;
    return (*this) * scale;
}

void Matrix::add_inplace(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match");
    }
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] += other.data[i][j];
        }
    }
}

void Matrix::subtract_inplace(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match");
    }
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] -= other.data[i][j];
        }
    }
}

void Matrix::multiply_inplace(double scalar) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] *= scalar;
        }
    }
}

double Matrix::variance() const {
    double mean_val = mean_value();
    double variance = 0.0;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double diff = data[i][j] - mean_val;
            variance += diff * diff;
        }
    }
    
    return variance / (rows * cols);
}

Matrix Matrix::normalize() const {
    double mean_val = mean_value();
    double std_val = sqrt(variance());
    
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = (data[i][j] - mean_val) / std_val;
        }
    }
    return result;
}

// Utility functions
Matrix concatenate(const Matrix& a, const Matrix& b, int axis) {
    if (axis == 0) { // Concatenate along rows
        if (a.cols != b.cols) {
            throw std::invalid_argument("Column dimensions must match");
        }
        Matrix result(a.rows + b.rows, a.cols);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j];
            }
        }
        for (int i = 0; i < b.rows; i++) {
            for (int j = 0; j < b.cols; j++) {
                result.data[a.rows + i][j] = b.data[i][j];
            }
        }
        return result;
    } else { // Concatenate along columns
        if (a.rows != b.rows) {
            throw std::invalid_argument("Row dimensions must match");
        }
        Matrix result(a.rows, a.cols + b.cols);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j];
            }
            for (int j = 0; j < b.cols; j++) {
                result.data[i][a.cols + j] = b.data[i][j];
            }
        }
        return result;
    }
}

Matrix zeros(int rows, int cols) {
    Matrix result(rows, cols);
    result.zero();
    return result;
}

Matrix ones(int rows, int cols) {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = 1.0;
        }
    }
    return result;
}

Matrix eye(int size) {
    Matrix result(size, size);
    result.zero();
    for (int i = 0; i < size; i++) {
        result.data[i][i] = 1.0;
    }
    return result;
}

double uniform_random(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

double normal_random(double mean, double std) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<> dis(mean, std);
    return dis(gen);
}