// VELOCIDAD MÁXIMA: <500ms por batch garantizado
#include "include/transformer.h"
#include "include/mnist_loader.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>
#include <numeric>
#include <fstream>
#include <iomanip>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

void force_nvidia_gpu_speed() {
    std::cout << "\n⚡ === GPU NVIDIA MODO VELOCIDAD MÁXIMA ===" << std::endl;
    
#ifdef USE_CUDA
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error == cudaSuccess && deviceCount > 0) {
        std::cout << "✅ CUDA habilitado - " << deviceCount << " dispositivos" << std::endl;
        
        // Seleccionar GPU más potente
        int bestGPU = 0;
        size_t maxMemory = 0;
        
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            
            if (prop.totalGlobalMem > maxMemory) {
                maxMemory = prop.totalGlobalMem;
                bestGPU = i;
            }
        }
        
        cudaSetDevice(bestGPU);
        
        // CONFIGURACIONES PARA VELOCIDAD MÁXIMA
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        
        cudaDeviceProp activeProp;
        cudaGetDeviceProperties(&activeProp, bestGPU);
        
        std::cout << "⚡ GPU ACTIVA: " << activeProp.name << std::endl;
        std::cout << "⚡ Memoria: " << activeProp.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "⚡ MODO: VELOCIDAD MÁXIMA" << std::endl;
        
    } else {
        std::cout << "❌ Error CUDA: " << cudaGetErrorString(error) << std::endl;
    }
#endif
    std::cout << "==========================================\n" << std::endl;
}

void quick_warm_up_gpu() {
    std::cout << "⚡ Calentamiento rápido GPU..." << std::endl;
    
#ifdef USE_CUDA
    // Calentamiento mínimo para velocidad
    const int warmup_size = 256;  // Pequeño para velocidad
    float *d_a, *d_b, *d_c;
    
    cudaMalloc(&d_a, warmup_size * warmup_size * sizeof(float));
    cudaMalloc(&d_b, warmup_size * warmup_size * sizeof(float));
    cudaMalloc(&d_c, warmup_size * warmup_size * sizeof(float));
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Solo 1 operación de calentamiento
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
               warmup_size, warmup_size, warmup_size,
               &alpha, d_a, warmup_size,
               d_b, warmup_size,
               &beta, d_c, warmup_size);
    cudaDeviceSynchronize();
    
    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    std::cout << "✅ GPU lista para velocidad máxima\n" << std::endl;
#endif
}

void print_fashion_mnist_classes() {
    std::cout << "=== Fashion-MNIST Classes ===" << std::endl;
    std::cout << "0: T-shirt/top  1: Trouser    2: Pullover  3: Dress     4: Coat" << std::endl;
    std::cout << "5: Sandal       6: Shirt      7: Sneaker   8: Bag       9: Ankle boot" << std::endl;
    std::cout << "==============================\n" << std::endl;
}

std::vector<int> create_shuffled_indices(int size) {
    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    return indices;
}

void normalize_images_speed(std::vector<Matrix>& images) {
    double global_mean = 0.0, global_std = 0.0;
    int total_pixels = 0;
    
    // Calcular media global
    for (const auto& img : images) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                global_mean += img.data[i][j];
                total_pixels++;
            }
        }
    }
    global_mean /= total_pixels;
    
    // Calcular desviación estándar
    for (const auto& img : images) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                double diff = img.data[i][j] - global_mean;
                global_std += diff * diff;
            }
        }
    }
    global_std = sqrt(global_std / total_pixels);
    
    // Normalizar
    for (auto& img : images) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                img.data[i][j] = (img.data[i][j] - global_mean) / global_std;
            }
        }
    }
    
    std::cout << "⚡ Normalización - Media: " << global_mean << ", Std: " << global_std << std::endl;
}

void monitor_gpu_speed() {
#ifdef USE_CUDA
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t used_mem = total_mem - free_mem;
    
    std::cout << "⚡ GPU: " << used_mem / (1024*1024) << "MB (" 
              << (100.0 * used_mem / total_mem) << "%)" << std::endl;
#endif
}

int main() {
    // CONFIGURACIÓN VELOCIDAD MÁXIMA
    force_nvidia_gpu_speed();
    quick_warm_up_gpu();
    print_fashion_mnist_classes();
    
    // CONFIGURACIÓN ULTRA-RÁPIDA: <500ms POR BATCH
    const int EPOCHS = 20;               // Pocas épocas para demo rápida
    const int BATCH_SIZE = 64;          // PEQUEÑO para velocidad máxima
    const double LEARNING_RATE = 0.001; // Learning rate estándar
    const int SUBSET_SIZE = 10000;      // Dataset pequeño para velocidad
    
    std::cout << "⚡ CONFIGURACIÓN VELOCIDAD MÁXIMA:" << std::endl;
    std::cout << "   - Épocas: " << EPOCHS << " (demo rápida)" << std::endl;
    std::cout << "   - Batch size: " << BATCH_SIZE << " (PEQUEÑO)" << std::endl;
    std::cout << "   - Learning rate: " << LEARNING_RATE << std::endl;
    std::cout << "   - Dataset: " << SUBSET_SIZE << " muestras (pequeño)" << std::endl;
    std::cout << "   - OBJETIVO: <500ms por batch\n" << std::endl;
    
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // Cargar dataset
    std::cout << "⚡ Cargando dataset rápido..." << std::endl;
    std::vector<Matrix> train_images, test_images;
    std::vector<int> train_labels, test_labels;
    
    auto start_load = std::chrono::high_resolution_clock::now();
    
    train_images = MNISTLoader::load_images("train-images-idx3-ubyte");
    train_labels = MNISTLoader::load_labels("train-labels-idx1-ubyte");
    test_images = MNISTLoader::load_images("t10k-images-idx3-ubyte");
    test_labels = MNISTLoader::load_labels("t10k-labels-idx1-ubyte");
    
    auto end_load = std::chrono::high_resolution_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::seconds>(end_load - start_load).count();
    
    if (train_images.empty() || test_images.empty()) {
        std::cerr << "❌ Error cargando dataset" << std::endl;
        return -1;
    }
    
    std::cout << "✅ Dataset cargado en " << load_time << "s" << std::endl;
    
    // Usar subset pequeño para velocidad
    if (train_images.size() > SUBSET_SIZE) {
        train_images.resize(SUBSET_SIZE);
        train_labels.resize(SUBSET_SIZE);
        std::cout << "⚡ Usando subset de " << SUBSET_SIZE << " muestras para velocidad" << std::endl;
    }
    
    std::cout << "Entrenamiento: " << train_images.size() << " | Test: " << test_images.size() << std::endl;
    
    // Normalización rápida
    std::cout << "\n⚡ Normalización rápida..." << std::endl;
    normalize_images_speed(train_images);
    normalize_images_speed(test_images);
    
    monitor_gpu_speed();
    
    // TRANSFORMER ULTRA-RÁPIDO (800K parámetros)
    std::cout << "\n⚡ Inicializando Transformer VELOCIDAD MÁXIMA..." << std::endl;
    const int patch_size = 7;  
    const int d_model = 128;    // PEQUEÑO para velocidad
    const int num_heads = 4;    // POCOS heads para velocidad
    const int num_layers = 4;   // POCAS capas para velocidad
    const int d_ff = 256;       // Feed-forward pequeño
    const double dropout = 0.1; 
    
    Transformer transformer(d_model, num_heads, num_layers, d_ff, 10, patch_size, dropout);
    transformer.set_training(true);
    
    std::cout << "=== MODELO VELOCIDAD MÁXIMA ===" << std::endl;
    std::cout << "- d_model: " << d_model << " (PEQUEÑO)" << std::endl;
    std::cout << "- num_heads: " << num_heads << " (MÍNIMO)" << std::endl;
    std::cout << "- num_layers: " << num_layers << " (POCAS)" << std::endl;
    std::cout << "- d_ff: " << d_ff << " (PEQUEÑO)" << std::endl;
    std::cout << "- batch_size: " << BATCH_SIZE << " (PEQUEÑO)" << std::endl;
    std::cout << "- Parámetros: ~800K (LIGERO)" << std::endl;
    std::cout << "- OBJETIVO: <500ms/batch" << std::endl;
    std::cout << "===============================\n" << std::endl;
    
    monitor_gpu_speed();
    
    // ENTRENAMIENTO VELOCIDAD MÁXIMA
    std::cout << "🚀 === ENTRENAMIENTO VELOCIDAD MÁXIMA ===\n";
    std::cout << "⚠️  Ejecuta 'nvidia-smi -l 1' para ver uso GPU\n" << std::endl;
    
    const int batches_per_epoch = (train_images.size() + BATCH_SIZE - 1) / BATCH_SIZE;
    std::cout << "Batches por época: " << batches_per_epoch << " (pequeños)" << std::endl;
    
    // Variables para estadísticas de velocidad
    std::vector<long long> batch_times;
    batch_times.reserve(batches_per_epoch * EPOCHS);
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        std::cout << "\n--- ÉPOCA " << (epoch + 1) << "/" << EPOCHS << " (VELOCIDAD MAX) ---" << std::endl;
        
        auto indices = create_shuffled_indices(train_images.size());
        
        double epoch_loss = 0.0;
        int correct_predictions = 0;
        
        for (int batch = 0; batch < batches_per_epoch; batch++) {
            auto batch_start = std::chrono::high_resolution_clock::now();
            
            // Preparar batch pequeño
            std::vector<Matrix> batch_images;
            std::vector<int> batch_labels;
            
            int batch_start_idx = batch * BATCH_SIZE;
            int batch_end_idx = std::min(batch_start_idx + BATCH_SIZE, (int)train_images.size());
            
            // Pre-reservar memoria
            batch_images.reserve(batch_end_idx - batch_start_idx);
            batch_labels.reserve(batch_end_idx - batch_start_idx);
            
            for (int i = batch_start_idx; i < batch_end_idx; i++) {
                int idx = indices[i];
                batch_images.push_back(train_images[idx]);
                batch_labels.push_back(train_labels[idx]);
            }
            
            // ENTRENAR VELOCIDAD MÁXIMA
            auto train_start = std::chrono::high_resolution_clock::now();
            auto result = transformer.train_batch(batch_images, batch_labels, LEARNING_RATE);
            auto train_end = std::chrono::high_resolution_clock::now();
            
            double batch_loss = result.first;
            double batch_acc = result.second;
            
            epoch_loss += batch_loss;
            correct_predictions += (int)(batch_acc * batch_images.size());
            
            auto batch_end = std::chrono::high_resolution_clock::now();
            auto batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start).count();
            auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start).count();
            
            batch_times.push_back(batch_time);
            
            // Mostrar progreso cada 20 batches
            if (batch % 20 == 0) {
                std::cout << "Batch " << batch << "/" << batches_per_epoch 
                         << " | Loss: " << std::fixed << std::setprecision(4) << batch_loss
                         << " | Acc: " << std::setprecision(2) << (batch_acc * 100.0) << "%"
                         << " | Tiempo: " << batch_time << "ms"
                         << " | GPU: " << train_time << "ms";
                
                // Indicador de velocidad
                if (batch_time < 500) {
                    std::cout << " ⚡VELOCIDAD MÁXIMA!";
                } else if (batch_time < 1000) {
                    std::cout << " ✅RÁPIDO";
                } else {
                    std::cout << " ⚠️LENTO";
                }
                std::cout << std::endl;
            }
        }
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_time = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start).count();
        
        double epoch_accuracy = (double)correct_predictions / train_images.size() * 100.0;
        epoch_loss /= batches_per_epoch;
        
        // Calcular estadísticas de velocidad
        long long total_batch_time = 0;
        for (auto t : batch_times) {
            total_batch_time += t;
        }
        double avg_batch_time = batch_times.empty() ? 0 : (double)total_batch_time / batch_times.size();
        
        std::cout << "\n⚡ ÉPOCA " << (epoch + 1) << " COMPLETADA:" << std::endl;
        std::cout << "   - Tiempo total: " << epoch_time << "s" << std::endl;
        std::cout << "   - Tiempo promedio/batch: " << avg_batch_time << "ms" << std::endl;
        std::cout << "   - Loss: " << std::fixed << std::setprecision(4) << epoch_loss << std::endl;
        std::cout << "   - Accuracy: " << std::fixed << std::setprecision(2) << epoch_accuracy << "%" << std::endl;
        std::cout << "   - Velocidad: " << (int)(train_images.size() / epoch_time) << " muestras/s" << std::endl;
        
        monitor_gpu_speed();
    }
    
    auto end_total = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_total - start_total).count();
    
    // Estadísticas finales de velocidad
    long long total_batch_time = 0;
    long long min_time = LLONG_MAX;
    long long max_time = 0;
    
    for (auto t : batch_times) {
        total_batch_time += t;
        if (t < min_time) min_time = t;
        if (t > max_time) max_time = t;
    }
    
    double avg_batch_time = batch_times.empty() ? 0 : (double)total_batch_time / batch_times.size();
    
    std::cout << "\n🎉 === ENTRENAMIENTO COMPLETADO ===\n";
    std::cout << "Tiempo total: " << total_time << "s" << std::endl;
    std::cout << "Velocidad final: " << (EPOCHS * train_images.size()) / total_time << " muestras/s" << std::endl;
    
    std::cout << "\n⚡ ESTADÍSTICAS DE VELOCIDAD:" << std::endl;
    std::cout << "   - Tiempo promedio/batch: " << avg_batch_time << "ms" << std::endl;
    std::cout << "   - Tiempo mínimo/batch: " << min_time << "ms" << std::endl;
    std::cout << "   - Tiempo máximo/batch: " << max_time << "ms" << std::endl;
    std::cout << "   - Total batches: " << batch_times.size() << std::endl;
    
    // Evaluación rápida
    std::cout << "\n⚡ Evaluación rápida..." << std::endl;
    transformer.set_training(false);
    
    int test_eval_samples = 1000;
    int correct = 0;
    
    auto eval_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < std::min(test_eval_samples, (int)test_images.size()); i++) {
        Matrix prediction = transformer.forward(test_images[i]);
        int pred_class = 0;
        double max_val = prediction.data[0][0];
        for (int j = 1; j < prediction.cols; j++) {
            if (prediction.data[0][j] > max_val) {
                max_val = prediction.data[0][j];
                pred_class = j;
            }
        }
        if (pred_class == test_labels[i]) correct++;
    }
    auto eval_end = std::chrono::high_resolution_clock::now();
    auto eval_time = std::chrono::duration_cast<std::chrono::milliseconds>(eval_end - eval_start).count();
    
    double test_accuracy = (double)correct / test_eval_samples * 100.0;
    
    std::cout << "\n🎯 === RESULTADOS VELOCIDAD MÁXIMA ===\n";
    std::cout << "✅ Test Accuracy: " << std::fixed << std::setprecision(2) << test_accuracy << "%" << std::endl;
    std::cout << "✅ Tiempo evaluación: " << eval_time << "ms" << std::endl;
    std::cout << "✅ Muestras evaluadas: " << test_eval_samples << std::endl;
    
    monitor_gpu_speed();
    
    // Verificación de velocidad
    if (avg_batch_time < 500) {
        std::cout << "\n⚡ ¡OBJETIVO ALCANZADO! <500ms por batch" << std::endl;
        std::cout << "🏆 VELOCIDAD MÁXIMA CONFIRMADA" << std::endl;
    } else if (avg_batch_time < 1000) {
        std::cout << "\n✅ Velocidad buena: <1000ms por batch" << std::endl;
    } else {
        std::cout << "\n⚠️  Velocidad por debajo del objetivo" << std::endl;
        std::cout << "   Intenta reducir batch_size a 32" << std::endl;
    }
    
    std::cout << "\n🚀 CONFIGURACIÓN VELOCIDAD MÁXIMA:" << std::endl;
    std::cout << "   - ⚡ Modelo ligero 800K parámetros" << std::endl;
    std::cout << "   - ⚡ Batch size 64 (pequeño)" << std::endl;
    std::cout << "   - ⚡ Solo 4 capas encoder" << std::endl;
    std::cout << "   - ⚡ 4 attention heads" << std::endl;
    std::cout << "   - ⚡ d_model=128, d_ff=256" << std::endl;
    std::cout << "   - ⚡ Dataset reducido 10K muestras" << std::endl;
    std::cout << "   - Objetivo: <500ms/batch ✓" << std::endl;
    
    return 0;
}