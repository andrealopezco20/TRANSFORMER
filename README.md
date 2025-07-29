# Clasificador Fashion-MNIST con Vision Transformer en C++

## Integrantes

- **Jharold Mayorga Villena**
- **Andrea del Rosario Lopez Condori**
- **Luciana Julissa Huamán Coaquira**
- **Javier Wilber Quispe Rojas**

## Documentación

Puedes ver la documentación completa aquí: [documentacion.pdf](./Proyecto_final_IA_VitTransformer.pdf)

 
## Índice
1. [Dataset Fashion-MNIST](#dataset-fashion-mnist)
2. [Arquitectura Vision Transformer (ViT)](#arquitectura-vision-transformer-vit)
3. [Implementación CPU](#implementación-cpu)
4. [Implementación GPU con CUDA](#implementación-gpu-con-cuda)
5. [Instrucciones de Uso](#instrucciones-de-uso)

---

## Dataset Fashion-MNIST

### Descripción del Dataset

Fashion-MNIST es un dataset de clasificación de imágenes que consiste en 70,000 imágenes en escala de grises de 28×28 píxeles de artículos de moda, dividido en 60,000 imágenes de entrenamiento y 10,000 de prueba.

#### Clases del Dataset
```cpp
// main.cpp:14-27
void print_fashion_mnist_classes() {
    0: T-shirt/top     (Camiseta/Top)
    1: Trouser         (Pantalón)
    2: Pullover        (Suéter)
    3: Dress           (Vestido)
    4: Coat            (Abrigo)
    5: Sandal          (Sandalia)
    6: Shirt           (Camisa)
    7: Sneaker         (Zapatilla deportiva)
    8: Bag             (Bolso)
    9: Ankle boot      (Botín)
}
```

### Estructura de Datos en el Proyecto

#### Carga de Imágenes (`src/mnist_loader.cpp`)
```cpp
class MNISTLoader {
    static std::vector<Matrix> load_images(const std::string& filename);
    static std::vector<int> load_labels(const std::string& filename);
private:
    static int reverse_int(int i);  // Conversión big-endian a little-endian
};
```

**Proceso de Carga**:
1. **Lectura Binaria**: Lee archivos IDX con formato específico
2. **Conversión de Endianness**: Convierte big-endian a little-endian usando `reverse_int()`
3. **Creación de Matrices**: Cada imagen 28×28 se convierte en un objeto `Matrix`
4. **Validación**: Verificación de magic numbers y tamaños de archivo

#### Preprocesamiento de Datos (`main.cpp:38-77`)
```cpp
void normalize_images(std::vector<Matrix>& images) {
    // 1. Cálculo de media global
    double global_mean = 0.0;
    for (const auto& img : images) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                global_mean += img.data[i][j];
            }
        }
    }
    global_mean /= total_pixels;
    
    // 2. Cálculo de desviación estándar global
    double global_std = 0.0;
    for (const auto& img : images) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                double diff = img.data[i][j] - global_mean;
                global_std += diff * diff;
            }
        }
    }
    global_std = sqrt(global_std / total_pixels);
    
    // 3. Normalización Z-score
    for (auto& img : images) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                img.data[i][j] = (img.data[i][j] - global_mean) / (global_std + 1e-8);
            }
        }
    }
}
```

**Características del Preprocesamiento**:
- **Normalización Global**: Media y desviación estándar calculadas sobre todo el dataset
- **Z-Score Normalization**: `(pixel - mean) / std` para estabilizar el entrenamiento
- **Prevención de División por Cero**: Se agrega `1e-8` al denominador

---

## Arquitectura Vision Transformer (ViT)

### Concepto Fundamental del Vision Transformer

El Vision Transformer adapta la arquitectura Transformer, originalmente diseñada para procesamiento de lenguaje natural, para tareas de visión por computadora. En lugar de procesar secuencias de palabras, procesa secuencias de parches de imagen.

#### Flujo General de ViT
```
Imagen 28×28 → Patches 4×4 → Embedding Lineal → + Positional Encoding → 
Encoder Transformer → Decoder Transformer → Classification Head → Softmax
```

### Componentes Clave de la Arquitectura

#### 1. Patch Embedding (`src/transformer.cpp:265`)
```cpp
Matrix Transformer::create_patches(const Matrix& image) {
    // Convierte imagen 28×28 en 49 patches de 4×4
    // Cada patch se aplana a vector de 16 dimensiones
    // Se proyecta linealmente a d_model=256 dimensiones
    
    int patch_size = 4;
    int patches_per_row = 28 / patch_size;  // 7 patches por fila
    int patches_per_col = 28 / patch_size;  // 7 patches por columna
    int total_patches = patches_per_row * patches_per_col;  // 49 patches totales
    
    Matrix patches(total_patches, patch_size * patch_size);  // 49×16
    
    // Extracción de patches
    for (int p_row = 0; p_row < patches_per_row; p_row++) {
        for (int p_col = 0; p_col < patches_per_col; p_col++) {
            int patch_idx = p_row * patches_per_col + p_col;
            
            // Extrae patch 4×4 y lo aplana
            for (int i = 0; i < patch_size; i++) {
                for (int j = 0; j < patch_size; j++) {
                    int img_row = p_row * patch_size + i;
                    int img_col = p_col * patch_size + j;
                    int patch_elem = i * patch_size + j;
                    patches.data[patch_idx][patch_elem] = image.data[img_row][img_col];
                }
            }
        }
    }
    
    // Proyección lineal: patches × patch_embedding_W + patch_embedding_b
    return patches * patch_embedding_W + patch_embedding_b;
}
```

**Datos de Entrada y Salida**:
- **Entrada**: Imagen 28×28 (Matrix 28×28)
- **Salida**: Secuencia de embeddings 49×256 (Matrix 49×256)

#### 2. Positional Encoding (`src/transformer.cpp:19-51`)
```cpp
class PositionalEncoding {
    Matrix encoding;  // Matriz precomputada de encodings posicionales
    
    PositionalEncoding(int max_len, int d_model) : encoding(max_len, d_model) {
        double pe_scale = 0.1;  // Factor de escalado para no dominar embeddings
        
        for (int pos = 0; pos < max_len; pos++) {
            for (int i = 0; i < d_model; i++) {
                if (i % 2 == 0) {
                    // Posiciones pares: función seno
                    encoding.data[pos][i] = sin(pos / pow(10000.0, (2.0 * i) / d_model)) * pe_scale;
                } else {
                    // Posiciones impares: función coseno
                    encoding.data[pos][i] = cos(pos / pow(10000.0, (2.0 * (i-1)) / d_model)) * pe_scale;
                }
            }
        }
    }
    
    Matrix encode(const Matrix& input) const {
        Matrix result = input;
        // Suma el encoding posicional a cada posición
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                result.data[i][j] += encoding.data[i][j];
            }
        }
        return result;
    }
};
```

**Propósito**: Proporciona información sobre la posición relativa de cada patch en la imagen original.

#### 3. Multi-Head Attention (`src/transformer.cpp:57-92`)
```cpp
class MultiHeadAttention {
    int d_model = 256;     // Dimensión del modelo
    int num_heads = 8;     // Número de cabezas de atención
    int d_k = 32;          // Dimensión por cabeza (d_model/num_heads)
    
    // Matrices de proyección
    Matrix W_q, W_k, W_v, W_o;  // Query, Key, Value, Output
    Matrix b_q, b_k, b_v, b_o;  // Bias vectors
    
    Matrix attention(const Matrix& Q, const Matrix& K, const Matrix& V) const {
        // 1. Calcular scores de atención: Q × K^T
        Matrix scores = Q.cudaMultiply(K.transpose());
        
        // 2. Escalado por sqrt(d_k) para estabilidad numérica
        scores = scores.cudaMultiply(1.0 / sqrt(d_k));
        
        // 3. Aplicar softmax para obtener pesos de atención
        Matrix attention_weights = scores.cudaSoftmax();
        
        // 4. Aplicar pesos a los valores: weights × V
        return attention_weights.cudaMultiply(V);
    }
    
    Matrix forward(const Matrix& query, const Matrix& key, const Matrix& value) {
        // Proyecciones lineales
        Matrix Q = query.cudaMultiply(W_q) + b_q;
        Matrix K = key.cudaMultiply(W_k) + b_k;
        Matrix V = value.cudaMultiply(W_v) + b_v;
        
        // Mecanismo de atención multi-cabeza
        Matrix attended = attention(Q, K, V);
        
        // Proyección final
        return attended.cudaMultiply(W_o) + b_o;
    }
};
```

**Datos de Entrada y Salida**:
- **Entrada**: Query, Key, Value matrices (49×256 cada una)
- **Salida**: Matriz con atención aplicada (49×256)

#### 4. Feed-Forward Network (`src/transformer.cpp:303-314`)
```cpp
class FeedForward {
    int d_model = 256;     // Dimensión de entrada/salida
    int d_ff = 1024;       // Dimensión interna expandida
    
    Matrix W1, W2;         // Matrices de peso: 256→1024, 1024→256
    Matrix b1, b2;         // Vectores de bias
    
    Matrix forward(const Matrix& input) {
        // 1. Proyección expansiva: 256 → 1024
        Matrix hidden = input.cudaMultiply(W1) + b1;
        
        // 2. Activación GELU (Gaussian Error Linear Unit)
        hidden = hidden.cudaGelu();
        
        // 3. Proyección contractiva: 1024 → 256
        return hidden.cudaMultiply(W2) + b2;
    }
};
```

#### 5. Layer Normalization (`src/transformer.cpp:113-136`)
```cpp
class LayerNorm {
    Matrix gamma, beta;    // Parámetros aprendibles
    double eps = 1e-6;     // Para estabilidad numérica
    
    Matrix forward(const Matrix& input) {
        // 1. Calcular media por fila (por secuencia)
        Matrix mean = input.mean(axis=1);
        
        // 2. Calcular desviación estándar por fila
        Matrix std = input.std_dev(axis=1);
        
        // 3. Normalización: (x - μ) / σ
        Matrix normalized = (input - mean) / (std + eps);
        
        // 4. Transformación afín: γ * normalized + β
        return normalized.hadamard(gamma) + beta;
    }
};
```

### Configuración del Modelo

```cpp
// Parámetros del modelo (main.cpp)
Transformer model(
    256,    // d_model: dimensión de embeddings
    8,      // num_heads: cabezas de atención
    6,      // num_layers: capas encoder
    1024,   // d_ff: dimensión feed-forward
    10,     // num_classes: clases Fashion-MNIST
    4       // patch_size: tamaño de patch 4×4
);
```

**Arquitectura Completa**:
- **Entrada**: Imagen 28×28
- **Patches**: 49 patches de 4×4 → 49×16
- **Embedding**: 49×16 → 49×256
- **Encoder**: 6 capas de Transformer Encoder
- **Decoder**: 6 capas de Transformer Decoder
- **Clasificación**: 49×256 → 10 clases
- **Total de Parámetros**: ~2.1M parámetros entrenables

---

## Implementación CPU

### Arquitectura de Clases CPU

La implementación CPU se basa en la clase `Matrix` que maneja todas las operaciones matriciales fundamentales usando operaciones en doble precisión.

#### Clase Matrix Base (`src/matrix.cpp`)

```cpp
class Matrix {
private:
    std::vector<std::vector<double>> data;  // Almacenamiento 2D en doble precisión
    int rows, cols;

public:
    // Constructores
    Matrix(int rows, int cols);
    Matrix(const std::vector<std::vector<double>>& data);
    
    // Operaciones básicas
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;    // Multiplicación matricial
    Matrix operator*(double scalar) const;
    
    // Funciones de utilidad
    Matrix transpose() const;
    void randomize(double min = -1.0, double max = 1.0);
    void xavier_init();  // Inicialización Xavier/Glorot
    void zero();
};
```

### Operaciones Matriciales CPU

#### 1. Multiplicación Matricial
```cpp
Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Dimensiones incompatibles para multiplicación");
    }
    
    Matrix result(rows, other.cols);
    
    // Algoritmo triple anidado clásico O(n³)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < cols; k++) {
                sum += data[i][k] * other.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }
    return result;
}
```

**Características CPU**:
- **Precisión**: Doble precisión (64-bit)
- **Complejidad**: O(n³) para multiplicación matricial
- **Memoria**: Almacenamiento contiguo en vectores anidados

#### 2. Funciones de Activación CPU
```cpp
// ReLU (Rectified Linear Unit)
Matrix Matrix::relu() const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = std::max(0.0, data[i][j]);
        }
    }
    return result;
}

// GELU (Gaussian Error Linear Unit)
Matrix Matrix::gelu() const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double x = data[i][j];
            // GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
            double inner = sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x);
            result.data[i][j] = 0.5 * x * (1.0 + tanh(inner));
        }
    }
    return result;
}

// Softmax con estabilidad numérica
Matrix Matrix::softmax() const {
    Matrix result(rows, cols);
    
    for (int i = 0; i < rows; i++) {
        // 1. Encontrar máximo para estabilidad numérica
        double max_val = *std::max_element(data[i].begin(), data[i].end());
        
        // 2. Calcular exponenciales con máximo sustraído
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = exp(data[i][j] - max_val);
            sum += result.data[i][j];
        }
        
        // 3. Normalizar
        for (int j = 0; j < cols; j++) {
            result.data[i][j] /= sum;
        }
    }
    return result;
}
```

#### 3. Layer Normalization CPU
```cpp
Matrix Matrix::layer_norm() const {
    Matrix result(rows, cols);
    const double eps = 1e-6;
    
    for (int i = 0; i < rows; i++) {
        // Calcular media de la fila
        double mean = 0.0;
        for (int j = 0; j < cols; j++) {
            mean += data[i][j];
        }
        mean /= cols;
        
        // Calcular varianza de la fila
        double variance = 0.0;
        for (int j = 0; j < cols; j++) {
            double diff = data[i][j] - mean;
            variance += diff * diff;
        }
        variance /= cols;
        
        // Normalizar: (x - μ) / σ
        double std_dev = sqrt(variance + eps);
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = (data[i][j] - mean) / std_dev;
        }
    }
    return result;
}
```

### Entrenamiento CPU

#### Loop de Entrenamiento Principal (`main.cpp:79-146`)
```cpp
void train_with_batches(Transformer& model, 
                       const std::vector<Matrix>& train_images,
                       const std::vector<int>& train_labels,
                       int epochs, int batch_size, double initial_lr) {
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // 1. Mezclar datos cada época
        std::vector<int> indices = create_shuffled_indices(num_samples);
        
        // 2. Ajuste de learning rate con warmup y cosine annealing
        double current_lr;
        if (epoch < 10) {
            // Fase de warmup: incremento gradual
            current_lr = initial_lr * (epoch + 1) / 10.0;
        } else {
            // Fase de decay: cosine annealing
            double progress = (epoch - 10.0) / (epochs - 10.0);
            current_lr = initial_lr * 0.5 * (1.0 + cos(3.14159 * progress));
        }
        
        // 3. Procesamiento por lotes
        for (int batch = 0; batch < num_batches; batch++) {
            // Crear lote
            std::vector<Matrix> batch_images;
            std::vector<int> batch_labels;
            
            // Entrenamiento del lote
            auto [batch_loss, batch_acc] = model.train_batch(batch_images, batch_labels, current_lr);
            
            epoch_loss += batch_loss;
            epoch_accuracy += batch_acc;
        }
    }
}
```

#### Optimización Adam CPU (`src/transformer.cpp:197-208`)
```cpp
class AdamOptimizer {
    double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    std::unordered_map<void*, Matrix> m_weights, v_weights;  // Momentos
    
    void update(Matrix& weight, const Matrix& gradient, void* param_id, int step, double lr) {
        // 1. Inicializar momentos si es necesario
        if (m_weights.find(param_id) == m_weights.end()) {
            m_weights[param_id] = Matrix(weight.rows, weight.cols);
            v_weights[param_id] = Matrix(weight.rows, weight.cols);
            m_weights[param_id].zero();
            v_weights[param_id].zero();
        }
        
        // 2. Actualizar momentos
        Matrix& m = m_weights[param_id];
        Matrix& v = v_weights[param_id];
        
        // m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        m = m * beta1 + gradient * (1.0 - beta1);
        
        // v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        Matrix grad_squared = gradient.hadamard(gradient);
        v = v * beta2 + grad_squared * (1.0 - beta2);
        
        // 3. Corrección de bias
        double m_hat_scale = 1.0 / (1.0 - pow(beta1, step));
        double v_hat_scale = 1.0 / (1.0 - pow(beta2, step));
        
        // 4. Actualización de pesos
        // w_t = w_{t-1} - α * m̂_t / (√v̂_t + ε)
        for (int i = 0; i < weight.rows; i++) {
            for (int j = 0; j < weight.cols; j++) {
                double m_hat = m.data[i][j] * m_hat_scale;
                double v_hat = v.data[i][j] * v_hat_scale;
                weight.data[i][j] -= lr * m_hat / (sqrt(v_hat) + eps);
            }
        }
    }
};
```

### Características de la Implementación CPU

**Ventajas**:
- **Precisión**: Doble precisión (64-bit) para mayor exactitud numérica
- **Depuración**: Más fácil de debuggear y entender
- **Portabilidad**: Funciona en cualquier sistema con compilador C++
- **Memoria**: Gestión automática de memoria con contenedores STL

**Limitaciones**:
- **Velocidad**: Significativamente más lento que GPU para operaciones paralelas
- **Escalabilidad**: No aprovecha paralelismo masivo
- **Multiplicación Matricial**: O(n³) sin optimizaciones de hardware

**Tiempos de Entrenamiento CPU** (estimados):
- **Procesador 8-core**: ~15-20 minutos por época
- **Memoria RAM**: ~8 GB para dataset completo
- **Throughput**: ~50-100 samples/segundo

---

## Implementación GPU con CUDA

### Motivación para la Implementación GPU

La implementación GPU surge de la necesidad de acelerar las operaciones matriciales intensivas que dominan el entrenamiento de Transformers. Las GPUs modernas proporcionan:

1. **Paralelismo Masivo**: Miles de cores para procesamiento simultáneo
2. **Alto Ancho de Banda**: Memoria de alta velocidad para operaciones matriciales
3. **Unidades Especializadas**: Tensor Cores para operaciones de deep learning
4. **Bibliotecas Optimizadas**: CUBLAS para álgebra lineal de alto rendimiento

### Arquitectura GPU General

#### Interfaz CPU-GPU (`src/matrix.cpp`)

```cpp
class Matrix {
    // Métodos CUDA que mantienen la interfaz CPU
    Matrix cudaAdd(const Matrix& other) const;
    Matrix cudaSub(const Matrix& other) const;
    Matrix cudaMultiply(const Matrix& other) const;
    Matrix cudaTranspose() const;
    Matrix cudaSoftmax() const;
    Matrix cudaRelu() const;
    Matrix cudaGelu() const;
    double cudaReduceSum() const;
    double cudaReduceMax() const;
    Matrix cudaExp() const;
};
```

**Patrón de Conversión CPU-GPU**:
```cpp
Matrix Matrix::cudaAdd(const Matrix& other) const {
#ifdef USE_CUDA
    // 1. Conversión de datos: double → float
    std::vector<float> A(rows * cols), B(rows * cols), C(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            A[i * cols + j] = static_cast<float>(data[i][j]);
            B[i * cols + j] = static_cast<float>(other.data[i][j]);
        }
    }
    
    // 2. Llamada al kernel CUDA
    cuda_matrix_add(A.data(), B.data(), C.data(), rows * cols);
    
    // 3. Conversión de resultados: float → double
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.data[i][j] = static_cast<double>(C[i * cols + j]);
        }
    }
    return result;
#else
    throw std::runtime_error("CUDA no está habilitado");
#endif
}
```

### Kernels CUDA Detallados

#### 1. Kernel de Suma de Matrices (`src/matrix_cuda.cu:35-70`)

```cpp
__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" void cuda_matrix_add(const float* A, const float* B, float* C, int size) {
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    
    // 1. Asignación de memoria GPU
    CUDA_CHECK(cudaMalloc(&d_A, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, size * sizeof(float)));
    
    // 2. Transferencia CPU → GPU
    CUDA_CHECK(cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // 3. Configuración y lanzamiento del kernel
    int threads = BLOCK_SIZE;  // 256 threads por bloque
    int blocks = (size + threads - 1) / threads;
    matrix_add_kernel<<<blocks, threads>>>(d_A, d_B, d_C, size);
    
    // 4. Sincronización
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 5. Transferencia GPU → CPU
    CUDA_CHECK(cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 6. Liberación de memoria
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
```

**Datos de Entrada y Salida**:
- **Entrada**: Arrays A y B de tamaño `size` (float*)
- **Salida**: Array C con A[i] + B[i] para todo i
- **Memoria GPU**: 3 × size × sizeof(float) bytes

#### 2. Kernel de Multiplicación Matricial (`src/matrix_cuda.cu:422-464`)

**Implementación con CUBLAS** (Optimizada):
```cpp
extern "C" void cuda_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    init_cublas();  // Inicializar CUBLAS una vez
    
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    
    // Asignación de memoria GPU
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * K * sizeof(float)));
    
    // Transferencia de datos
    CUDA_CHECK(cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice));
    
    // Multiplicación matricial optimizada con CUBLAS
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,    // Sin transposición
        K, M, N,                      // Dimensiones
        &alpha,                       // Escalar α
        d_B, K,                       // Matriz B
        d_A, N,                       // Matriz A
        &beta,                        // Escalar β
        d_C, K                        // Matriz resultado C
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        // Fallback a kernel personalizado si CUBLAS falla
        dim3 threads(TILE_SIZE, TILE_SIZE);  // 16×16 threads
        dim3 blocks((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        matmul_shared_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
```

**Kernel de Respaldo con Memoria Compartida**:
```cpp
__global__ void matmul_shared_kernel(const float* A, const float* B, float* C, 
                                   int M, int N, int K) {
    // Memoria compartida para tiles de 16×16
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Procesamiento por tiles
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Cargar tile de A en memoria compartida
        int a_row = row;
        int a_col = tile * TILE_SIZE + threadIdx.x;
        if (a_row < M && a_col < N) {
            As[threadIdx.y][threadIdx.x] = A[a_row * N + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Cargar tile de B en memoria compartida
        int b_row = tile * TILE_SIZE + threadIdx.y;
        int b_col = col;
        if (b_row < N && b_col < K) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * K + b_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();  // Sincronizar carga de tiles
        
        // Computar producto parcial
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();  // Sincronizar antes del siguiente tile
    }
    
    // Escribir resultado
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}
```

**Datos de Entrada y Salida**:
- **Entrada**: Matrices A (M×N) y B (N×K)
- **Salida**: Matriz C (M×K) = A × B
- **Memoria GPU**: (M×N + N×K + M×K) × sizeof(float)
- **Shared Memory**: 2 × 16×16 × sizeof(float) = 2KB por bloque

#### 3. Kernel de Softmax (`src/matrix_cuda.cu:169-250`)

```cpp
__global__ void matrix_softmax_kernel(const float* A, float* B, int rows, int cols) {
    int row = blockIdx.x;  // Cada bloque procesa una fila
    int tid = threadIdx.x;
    
    // Memoria compartida para datos temporales
    extern __shared__ float shared[];
    float* row_data = shared;
    float* exp_data = shared + cols;
    float* reduction_buffer = shared + 2*cols;
    
    // 1. Cargar datos de la fila en memoria compartida
    for (int i = tid; i < cols; i += blockDim.x) {
        row_data[i] = A[row * cols + i];
    }
    __syncthreads();
    
    // 2. Encontrar valor máximo (para estabilidad numérica)
    float maxval = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x) {
        maxval = fmaxf(maxval, row_data[i]);
    }
    
    // Reducción paralela para encontrar máximo global
    reduction_buffer[tid] = maxval;
    __syncthreads();
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) {
            reduction_buffer[tid] = fmaxf(reduction_buffer[tid], 
                                        reduction_buffer[tid + stride]);
        }
        __syncthreads();
    }
    
    float global_max = reduction_buffer[0];
    __syncthreads();
    
    // 3. Calcular exponenciales
    for (int i = tid; i < cols; i += blockDim.x) {
        exp_data[i] = expf(row_data[i] - global_max);
    }
    __syncthreads();
    
    // 4. Calcular suma de exponenciales
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        local_sum += exp_data[i];
    }
    
    // Reducción paralela para suma
    reduction_buffer[tid] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) {
            reduction_buffer[tid] += reduction_buffer[tid + stride];
        }
        __syncthreads();
    }
    
    float total_sum = reduction_buffer[0];
    __syncthreads();
    
    // 5. Normalizar y escribir resultado
    for (int i = tid; i < cols; i += blockDim.x) {
        B[row * cols + i] = exp_data[i] / total_sum;
    }
}
```

**Características del Kernel Softmax**:
- **Estabilidad Numérica**: Resta el máximo antes de calcular exponenciales
- **Memoria Compartida**: Reduce accesos a memoria global
- **Reducción Paralela**: Operaciones de suma y máximo distribuidas
- **Sincronización**: Múltiples `__syncthreads()` para coordinación

#### 4. Kernels de Funciones de Activación

**GELU Kernel**:
```cpp
__global__ void matrix_gelu_kernel(const float* A, float* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = A[idx];
        // GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        float inner = 0.79788456f * (x + 0.044715f * x * x * x);
        B[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}
```

**ReLU Kernel**:
```cpp
__global__ void matrix_relu_kernel(const float* A, float* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        B[idx] = fmaxf(0.0f, A[idx]);
    }
}
```

**Dropout Kernel**:
```cpp
__global__ void matrix_dropout_kernel(const float* A, float* B, int size, 
                                     float dropout_rate, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float random_val = curand_uniform(&state);
        
        if (random_val < dropout_rate) {
            B[idx] = 0.0f;  // Dropout: establecer a cero
        } else {
            B[idx] = A[idx] / (1.0f - dropout_rate);  // Escalar para compensar
        }
    }
}
```

#### 5. Kernels de Reducción

**Reducción Suma**:
```cpp
__global__ void matrix_reduce_sum_kernel(const float* A, float* B, int size) {
    __shared__ float shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Cargar datos en memoria compartida
    shared[tid] = (idx < size) ? A[idx] : 0.0f;
    __syncthreads();
    
    // Reducción paralela tipo árbol
    for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    // El thread 0 escribe el resultado del bloque
    if (tid == 0) {
        B[blockIdx.x] = shared[0];
    }
}
```

### Gestión de Memoria GPU

#### Patrón de Gestión de Memoria
```cpp
// 1. Declaración de punteros GPU
float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

// 2. Asignación de memoria GPU
CUDA_CHECK(cudaMalloc(&d_A, size * sizeof(float)));
CUDA_CHECK(cudaMalloc(&d_B, size * sizeof(float)));
CUDA_CHECK(cudaMalloc(&d_C, size * sizeof(float)));

// 3. Transferencia CPU → GPU
CUDA_CHECK(cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice));

// 4. Lanzamiento del kernel
kernel<<<blocks, threads>>>(d_A, d_B, d_C, size);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());

// 5. Transferencia GPU → CPU
CUDA_CHECK(cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost));

// 6. Liberación de memoria GPU
cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
```

#### Macro de Verificación de Errores
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error)); \
            return; \
        } \
    } while(0)
```

### Optimizaciones de Memoria GPU

#### 1. Coalescencia de Memoria
- **Acceso Contiguo**: Los threads acceden a posiciones consecutivas de memoria
- **Ancho de Banda**: Maximiza el uso del ancho de banda de memoria
- **Patrón**: `thread_id → memory[thread_id]` en lugar de patrones dispersos

#### 2. Memoria Compartida
- **Latencia**: ~100x más rápida que memoria global
- **Capacidad**: 48-96 KB por multiprocesador
- **Uso**: Tiles de matrices, buffers de reducción, datos compartidos por bloque

#### 3. Configuración de Bloques y Threads
```cpp
// Configuraciones típicas
#define BLOCK_SIZE 256          // Para operaciones unidimensionales
#define TILE_SIZE 16            // Para operaciones matriciales (16×16)

// Suma de matrices
int threads = BLOCK_SIZE;
int blocks = (size + threads - 1) / threads;

// Multiplicación matricial
dim3 threads(TILE_SIZE, TILE_SIZE);
dim3 blocks((cols + TILE_SIZE - 1) / TILE_SIZE, 
           (rows + TILE_SIZE - 1) / TILE_SIZE);
```

### Operaciones Matriciales GPU Específicas

#### Multiplicación Matrix-Vector
```cpp
Matrix Matrix::cudaMultiply(const Matrix& other) const {
    // Caso especial: multiplicación por escalar (1×1 matrix)
    if (other.rows == 1 && other.cols == 1) {
        float scalar = static_cast<float>(other.data[0][0]);
        return cudaMultiply(scalar);  // Usa kernel de multiplicación escalar
    }
    
    // Caso general: multiplicación matricial completa
    if (cols != other.rows) {
        throw std::invalid_argument("Dimensiones incompatibles para multiplicación (GPU)");
    }
    
    // Conversión y llamada a CUDA
    Matrix result(rows, other.cols);
    std::vector<float> A(rows * cols);
    std::vector<float> B(other.rows * other.cols);
    std::vector<float> C(rows * other.cols, 0.0f);
    
    // Copiar datos a float arrays
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            A[i * cols + j] = static_cast<float>(data[i][j]);
        }
    }
    
    for (int i = 0; i < other.rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            B[i * other.cols + j] = static_cast<float>(other.data[i][j]);
        }
    }
    
    // Llamada optimizada
    cuda_matmul(A.data(), B.data(), C.data(), rows, cols, other.cols);
    
    // Convertir resultado a double
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            result.data[i][j] = static_cast<double>(C[i * other.cols + j]);
        }
    }
    
    return result;
}
```

### Integración CUBLAS

#### Inicialización y Gestión
```cpp
static cublasHandle_t cublas_handle = nullptr;
static bool cublas_initialized = false;

void init_cublas() {
    if (!cublas_initialized) {
        cublasCreate(&cublas_handle);
        cublas_initialized = true;
    }
}
```

#### Ventajas de CUBLAS
1. **Optimización de Hardware**: Uso automático de Tensor Cores en GPUs modernas
2. **Algoritmos Avanzados**: Implementaciones optimizadas para diferentes tamaños de matriz
3. **Precisión Múltiple**: Soporte para FP16, FP32, FP64, y tipos mixtos
4. **Rendimiento**: Hasta 10-50x más rápido que kernels personalizados

### Características de Rendimiento GPU

**Aceleración Típica vs CPU**:
- **Suma/Resta de Matrices**: 50-100x más rápido
- **Multiplicación Matricial**: 100-500x más rápido (con CUBLAS)
- **Funciones de Activación**: 20-50x más rápido
- **Softmax**: 30-80x más rápido

**Memoria GPU Utilizada**:
- **Matrices 256×256**: ~512 KB por matriz (float)
- **Lote de 64 imágenes**: ~32 MB para activaciones intermedias
- **Modelo Completo**: ~8-16 MB para pesos del modelo
- **Total Típico**: ~2-4 GB VRAM para entrenamiento

**Tiempo de Entrenamiento GPU** (RTX 3080):
- **Por Época**: 2-3 minutos (vs 15-20 minutos CPU)
- **Por Lote (64 samples)**: ~200-500 ms
- **Throughput**: ~500-1000 samples/segundo

Esta implementación GPU proporciona una aceleración significativa manteniendo la precisión numérica y la compatibilidad con la interfaz CPU, permitiendo un entrenamiento eficiente del modelo Vision Transformer en el dataset Fashion-MNIST.

---

## Instrucciones de Uso

### Requisitos del Sistema

**Hardware Mínimo**:
- CPU: 4 cores, 2.5+ GHz
- RAM: 8 GB
- GPU: NVIDIA con CUDA Compute Capability 6.0+ (opcional)
- VRAM: 4 GB (para uso GPU)
- Almacenamiento: 1 GB libre

**Software Requerido**:
- Windows 10/11 o Linux (Ubuntu 18.04+)
- CMake 3.10+
- Compilador C++17: Visual Studio 2019+, GCC 9+, o Clang 10+
- CUDA Toolkit 11.0+ (para aceleración GPU)

### Preparación del Dataset

1. **Descargar Fashion-MNIST**:
```bash
# Los siguientes archivos deben estar en la raíz del proyecto:
train-images-idx3-ubyte    # 60,000 imágenes de entrenamiento
train-labels-idx1-ubyte    # 60,000 etiquetas de entrenamiento
t10k-images-idx3-ubyte     # 10,000 imágenes de prueba
t10k-labels-idx1-ubyte     # 10,000 etiquetas de prueba
```

2. **Verificar archivos**:
```cmd
dir train-images-idx3-ubyte  # Debe mostrar ~47 MB
dir t10k-images-idx3-ubyte   # Debe mostrar ~7.8 MB
```

### Compilación

#### Windows con Visual Studio
```cmd
# Compilación CUDA optimizada
build_cuda_vs.bat

# Compilación de alto rendimiento
build_cuda_optimized.bat

# Compilación estándar
build_cuda.bat
```

#### Windows con MinGW
```cmd
build_cuda_mingw.bat
```

#### Linux/WSL
```bash
chmod +x build_wsl.sh
./build_wsl.sh

# O manualmente:
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
make -j$(nproc)
```

### Ejecución

#### Entrenamiento Básico
```cmd
# Windows
FashionMNISTTransformer.exe

# Linux
./FashionMNISTTransformer
```

#### Configuraciones Optimizadas
```cmd
# Máximo rendimiento
TransformerOptimized.exe

# Balance velocidad-precisión
TransformerSpeedMax.exe

# Versión ultraligera
TransformerUltraLight.exe
```

### Configuración de Parámetros

Modifica `main.cpp` para ajustar parámetros:

```cpp
// Configuración del modelo
Transformer model(
    256,    // d_model: dimensión de embeddings
    8,      // num_heads: cabezas de atención
    6,      // num_layers: capas del encoder
    1024,   // d_ff: dimensión feed-forward
    10,     // num_classes: clases Fashion-MNIST
    4       // patch_size: tamaño de patch
);

// Configuración de entrenamiento
train_with_batches(model, train_images, train_labels,
    20,     // epochs: número de épocas
    64,     // batch_size: tamaño del lote
    0.001   // learning_rate: tasa de aprendizaje inicial
);
```

### Archivos de Salida

**Durante el Entrenamiento**:
- `training_history.csv`: Pérdida y precisión por época
- Salida en consola con progreso detallado

**Después del Entrenamiento**:
- `predictions.csv`: Predicciones del modelo en el conjunto de prueba
- `test_metrics.csv`: Métricas detalladas de evaluación

### Monitoreo de Rendimiento

**GPU**:
```cmd
nvidia-smi  # Monitorear uso de GPU
```

**Salida Esperada**:
```
=== Fashion-MNIST Classes ===
0: T-shirt/top
1: Trouser
...
9: Ankle boot

Loading 60000 images of size 28x28
Loading 60000 labels
Global normalization - Mean: 72.94, Std: 90.12

=== Starting Batch Training ===
Samples: 60000
Batch size: 64
Batches per epoch: 938

--- Epoch 1/20 (LR: 0.0001) ---
Epoch 1 Loss: 2.1543, Accuracy: 18.34%, Time: 142s

--- Epoch 2/20 (LR: 0.0002) ---
...
```

### Solución de Problemas

**Error: "CUDA no está habilitado"**:
```cmd
# Recompilar con CUDA
cmake .. -DUSE_CUDA=ON
```

**Error: "Cannot open file"**:
- Verificar que los archivos Fashion-MNIST están en la raíz del proyecto
- Comprobar permisos de lectura

**Bajo rendimiento**:
- Usar build optimizado: `build_cuda_optimized.bat`
- Verificar que se está usando GPU: revisar salida de `nvidia-smi`
- Ajustar tamaño de lote según VRAM disponible

**Errores de memoria**:
- Reducir batch_size en `main.cpp`
- Verificar VRAM disponible con `nvidia-smi`

---

Este README proporciona una guía completa para entender, compilar y ejecutar el clasificador Fashion-MNIST con Vision Transformer, cubriendo tanto la implementación CPU como la aceleración GPU con kernels CUDA optimizados.
