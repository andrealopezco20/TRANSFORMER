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