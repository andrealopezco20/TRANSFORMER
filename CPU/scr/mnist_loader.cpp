#include "../include/mnist_loader.h"
#include <iostream>
#include <fstream>

int MNISTLoader::reverse_int(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

std::vector<Matrix> MNISTLoader::load_images(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);

    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverse_int(number_of_images);

    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverse_int(n_rows);

    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverse_int(n_cols);

    std::cout << "Loading " << number_of_images << " images of size " << n_rows << "x" << n_cols << std::endl;

    std::vector<Matrix> images;
    images.reserve(number_of_images);

    for (int i = 0; i < number_of_images; ++i) {
        Matrix image(n_rows, n_cols);
        for (int r = 0; r < n_rows; ++r) {
            for (int c = 0; c < n_cols; ++c) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                image.data[r][c] = static_cast<double>(temp);  // Don't normalize here
            }
        }
        images.push_back(image);
    }

    file.close();
    return images;
}

std::vector<int> MNISTLoader::load_labels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    int magic_number = 0;
    int number_of_labels = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);

    file.read((char*)&number_of_labels, sizeof(number_of_labels));
    number_of_labels = reverse_int(number_of_labels);

    std::cout << "Loading " << number_of_labels << " labels" << std::endl;

    std::vector<int> labels;
    labels.reserve(number_of_labels);

    for (int i = 0; i < number_of_labels; ++i) {
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(temp));
        labels.push_back(static_cast<int>(temp));
    }

    file.close();
    return labels;
}