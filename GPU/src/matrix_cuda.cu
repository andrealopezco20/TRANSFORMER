#include "../include/cuda_ops.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <float.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define TILE_SIZE 16
#define MAX_THREADS_PER_BLOCK 1024

// Variables globales para optimización
static cublasHandle_t cublas_handle = nullptr;
static bool cublas_initialized = false;

// Función para verificar errores CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            return; \
        } \
    } while(0)

// Inicializar CUBLAS una sola vez
void init_cublas() {
    if (!cublas_initialized) {
        cublasCreate(&cublas_handle);
        cublas_initialized = true;
    }
}

// Suma de matrices - Optimizada
__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" void cuda_matrix_add(const float* A, const float* B, float* C, int size) {
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    
    // Asignar memoria en GPU
    CUDA_CHECK(cudaMalloc(&d_A, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, size * sizeof(float)));
    
    // Copiar datos a GPU
    CUDA_CHECK(cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Configurar grid y threads
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    
    // Ejecutar kernel
    matrix_add_kernel<<<blocks, threads>>>(d_A, d_B, d_C, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copiar resultado de vuelta a CPU
    CUDA_CHECK(cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Liberar memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Resta de matrices - Optimizada
__global__ void matrix_sub_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] - B[idx];
    }
}

extern "C" void cuda_matrix_sub(const float* A, const float* B, float* C, int size) {
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_A, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, size * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice));
    
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    
    matrix_sub_kernel<<<blocks, threads>>>(d_A, d_B, d_C, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Transposición de matriz
__global__ void matrix_transpose_kernel(const float* A, float* B, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        B[x * rows + y] = A[y * cols + x];
    }
}
extern "C" void cuda_matrix_transpose(const float* A, float* B, int rows, int cols) {
    float *d_A, *d_B;
    cudaMalloc(&d_A, rows * cols * sizeof(float));
    cudaMalloc(&d_B, rows * cols * sizeof(float));
    cudaMemcpy(d_A, A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
    matrix_transpose_kernel<<<blocks, threads>>>(d_A, d_B, rows, cols);
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}

// Multiplicación escalar
__global__ void matrix_scalar_mul_kernel(const float* A, float* C, float scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * scalar;
    }
}
extern "C" void cuda_matrix_scalar_mul(const float* A, float* C, float scalar, int size) {
    float *d_A = nullptr, *d_C = nullptr;
    cudaError_t err;
    err = cudaMalloc(&d_A, size * sizeof(float));
    if (err != cudaSuccess) {
        return;
    }
    err = cudaMalloc(&d_C, size * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_A);
        return;
    }
    err = cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_A); cudaFree(d_C);
        return;
    }
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    matrix_scalar_mul_kernel<<<blocks, threads>>>(d_A, d_C, scalar, size);
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        // Silenciar error
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_A); cudaFree(d_C);
        return;
    }
    err = cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        // Silenciar error
    }
    cudaFree(d_A);
    cudaFree(d_C);
}

// Softmax por filas
__global__ void matrix_softmax_kernel(const float* A, float* B, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float shared[];
    float* row_data = shared;
    float* exp_data = shared + cols;
    float* reduction_buffer = shared + 2*cols;
    
    // Load row data into shared memory
    for (int i = tid; i < cols; i += blockDim.x) {
        row_data[i] = A[row * cols + i];
    }
    __syncthreads();
    
    // Find maximum value in the row (reduction)
    float maxval = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x) {
        maxval = fmaxf(maxval, row_data[i]);
    }
    
    // Store local max in reduction buffer
    reduction_buffer[tid] = maxval;
    __syncthreads();
    
    // Reduce to find global maximum
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) {
            reduction_buffer[tid] = fmaxf(reduction_buffer[tid], reduction_buffer[tid + stride]);
        }
        __syncthreads();
    }
    
    // Global maximum is now in reduction_buffer[0]
    float global_max = reduction_buffer[0];
    __syncthreads();
    
    // Compute exp(x - max) for numerical stability
    for (int i = tid; i < cols; i += blockDim.x) {
        exp_data[i] = expf(row_data[i] - global_max);
    }
    __syncthreads();
    
    // Compute sum of exponentials (reduction)
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        sum += exp_data[i];
    }
    
    // Store local sum in reduction buffer
    reduction_buffer[tid] = sum;
    __syncthreads();
    
    // Reduce to find global sum
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) {
            reduction_buffer[tid] += reduction_buffer[tid + stride];
        }
        __syncthreads();
    }
    
    // Global sum is now in reduction_buffer[0]
    float global_sum = reduction_buffer[0];
    __syncthreads();
    
    // Compute final softmax: exp(x - max) / sum
    for (int i = tid; i < cols; i += blockDim.x) {
        B[row * cols + i] = exp_data[i] / global_sum;
    }
}
extern "C" void cuda_matrix_softmax(const float* A, float* B, int rows, int cols) {
    float *d_A, *d_B;
    cudaMalloc(&d_A, rows * cols * sizeof(float));
    cudaMalloc(&d_B, rows * cols * sizeof(float));
    cudaMemcpy(d_A, A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    
    // Shared memory: row_data + exp_data + reduction_buffer
    int shared_size = (2 * cols + 32) * sizeof(float);
    matrix_softmax_kernel<<<rows, 32, shared_size>>>(d_A, d_B, rows, cols);
    
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}

// ReLU
__global__ void matrix_relu_kernel(const float* A, float* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) B[idx] = fmaxf(A[idx], 0.0f);
}
extern "C" void cuda_matrix_relu(const float* A, float* B, int size) {
    float *d_A, *d_B;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    matrix_relu_kernel<<<blocks, threads>>>(d_A, d_B, size);
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}

// Sigmoid
__global__ void matrix_sigmoid_kernel(const float* A, float* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) B[idx] = 1.0f / (1.0f + expf(-A[idx]));
}
extern "C" void cuda_matrix_sigmoid(const float* A, float* B, int size) {
    float *d_A, *d_B;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    matrix_sigmoid_kernel<<<blocks, threads>>>(d_A, d_B, size);
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}

// Tanh
__global__ void matrix_tanh_kernel(const float* A, float* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) B[idx] = tanhf(A[idx]);
}
extern "C" void cuda_matrix_tanh(const float* A, float* B, int size) {
    float *d_A, *d_B;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    matrix_tanh_kernel<<<blocks, threads>>>(d_A, d_B, size);
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}

// Reduce sum
__global__ void matrix_reduce_sum_kernel(const float* A, float* B, int size) {
    __shared__ float shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tid] = (idx < size) ? A[idx] : 0.0f;
    __syncthreads();
    for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0) B[blockIdx.x] = shared[0];
}
extern "C" void cuda_matrix_reduce_sum(const float* A, float* B, int size) {
    float *d_A, *d_B;
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, blocks * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    matrix_reduce_sum_kernel<<<blocks, BLOCK_SIZE>>>(d_A, d_B, size);
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}

// Reduce max
__global__ void matrix_reduce_max_kernel(const float* A, float* B, int size) {
    __shared__ float shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tid] = (idx < size) ? A[idx] : -FLT_MAX;
    __syncthreads();
    for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
        if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) B[blockIdx.x] = shared[0];
}
extern "C" void cuda_matrix_reduce_max(const float* A, float* B, int size) {
    float *d_A, *d_B;
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, blocks * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    matrix_reduce_max_kernel<<<blocks, BLOCK_SIZE>>>(d_A, d_B, size);
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}

// Exp
__global__ void matrix_exp_kernel(const float* A, float* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) B[idx] = expf(A[idx]);
}
extern "C" void cuda_matrix_exp(const float* A, float* B, int size) {
    float *d_A, *d_B;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    matrix_exp_kernel<<<blocks, threads>>>(d_A, d_B, size);
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}

// Kernel de respaldo para multiplicación de matrices
__global__ void matmul_shared_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Cargar tile de A
        int a_row = row;
        int a_col = tile * TILE_SIZE + threadIdx.x;
        if (a_row < M && a_col < N) {
            As[threadIdx.y][threadIdx.x] = A[a_row * N + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Cargar tile de B
        int b_row = tile * TILE_SIZE + threadIdx.y;
        int b_col = col;
        if (b_row < N && b_col < K) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * K + b_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Calcular producto parcial
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

// Multiplicación de matrices ULTRA OPTIMIZADA con CUBLAS
extern "C" void cuda_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    init_cublas();
    
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    
    // Asignar memoria GPU
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * K * sizeof(float)));
    
    // Copiar datos a GPU
    CUDA_CHECK(cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice));
    
    // Usar CUBLAS para máximo rendimiento (MUCHO más rápido que kernels custom)
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasSgemm(cublas_handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        K, M, N,
                                        &alpha,
                                        d_B, K,
                                        d_A, N,
                                        &beta,
                                        d_C, K);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS Error: %d\n", status);
        // Fallback a kernel custom si CUBLAS falla
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        matmul_shared_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copiar resultado a CPU
    CUDA_CHECK(cudaMemcpy(C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Liberar memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// LayerNorm CUDA REAL - CORREGIDO
__global__ void matrix_layernorm_kernel(const float* A, float* B, int rows, int cols, float gamma, float beta) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float eps = 1e-5f;
    
    extern __shared__ float shared[];
    float* row_data = shared;
    float* sum_buffer = shared + cols;
    
    // Cargar datos de la fila en memoria compartida
    for (int i = tid; i < cols; i += blockDim.x) {
        row_data[i] = A[row * cols + i];
    }
    __syncthreads();
    
    // Calcular media
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        local_sum += row_data[i];
    }
    sum_buffer[tid] = local_sum;
    __syncthreads();
    
    // Reducción para la suma
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) {
            sum_buffer[tid] += sum_buffer[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = sum_buffer[0] / cols;
    __syncthreads();
    
    // Calcular varianza
    float local_var = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float diff = row_data[i] - mean;
        local_var += diff * diff;
    }
    sum_buffer[tid] = local_var;
    __syncthreads();
    
    // Reducción para la varianza
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) {
            sum_buffer[tid] += sum_buffer[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = sum_buffer[0] / cols;
    float std_dev = sqrtf(variance + eps);
    __syncthreads();
    
    // Aplicar normalización + escala
    for (int i = tid; i < cols; i += blockDim.x) {
        float normalized = (row_data[i] - mean) / std_dev;
        B[row * cols + i] = gamma * normalized + beta;
    }
}

extern "C" void cuda_matrix_layernorm(const float* A, float* B, int rows, int cols, float gamma, float beta) {
    float *d_A, *d_B;
    int size = rows * cols;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configurar threads y memoria compartida para LayerNorm REAL
    int threads = min(cols, 256);
    int shared_size = (cols + threads) * sizeof(float);
    matrix_layernorm_kernel<<<rows, threads, shared_size>>>(d_A, d_B, rows, cols, gamma, beta);
    
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}

// GELU Activation - CRÍTICO para Transformers
__global__ void matrix_gelu_kernel(const float* A, float* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = A[idx];
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        float x_cubed = x * x * x;
        float tanh_arg = 0.7978845608f * (x + 0.044715f * x_cubed);  // sqrt(2/π) ≈ 0.7978845608
        B[idx] = 0.5f * x * (1.0f + tanhf(tanh_arg));
    }
}

extern "C" void cuda_matrix_gelu(const float* A, float* B, int size) {
    float *d_A, *d_B;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    matrix_gelu_kernel<<<blocks, threads>>>(d_A, d_B, size);
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}

// Dropout - CRÍTICO para regularización
__global__ void matrix_dropout_kernel(const float* A, float* B, int size, float dropout_rate, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Usar random simple (cuRAND requiere más setup)
        unsigned int local_seed = seed + idx;
        local_seed = local_seed * 1103515245 + 12345;  // LCG
        float rand_val = (float)(local_seed % 1000) / 1000.0f;
        
        if (rand_val < dropout_rate) {
            B[idx] = 0.0f;  // Drop out
        } else {
            B[idx] = A[idx] / (1.0f - dropout_rate);  // Scale up
        }
    }
}

extern "C" void cuda_matrix_dropout(const float* A, float* B, int size, float dropout_rate, unsigned int seed) {
    float *d_A, *d_B;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    matrix_dropout_kernel<<<blocks, threads>>>(d_A, d_B, size, dropout_rate, seed);
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_B, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
}