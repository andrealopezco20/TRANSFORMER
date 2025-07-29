#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <string>
#include "matrix.h"

class MNISTLoader {
public:
    static std::vector<Matrix> load_images(const std::string& filename);
    static std::vector<int> load_labels(const std::string& filename);
    
private:
    static int reverse_int(int i);
};

#endif
//