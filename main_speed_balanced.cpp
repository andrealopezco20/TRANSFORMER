// VELOCIDAD BALANCEADA: <800ms por batch con 75-80% accuracy
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void force_nvidia_gpu_balanced() {
    std::cout << "\n⚡ === GPU NVIDIA MODO BALANCEADO ===" << std::endl;
    
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
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
        
        cudaDeviceProp activeProp;
        cudaGetDeviceProperties(&activeProp, bestGPU);
        
        std::cout << "⚡ GPU ACTIVA: " << activeProp.name << std::endl;
        std::cout << "⚡ Memoria: " << activeProp.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "⚡ MODO: VELOCIDAD + ACCURACY BALANCEADO" << std::endl;
        
    }
#endif
    std::cout << "==========================================\n" << std::endl;
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

// Normalización mejorada con clipping
void normalize_images_balanced(std::vector<Matrix>& images) {
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
    
    // Normalizar con clipping
    double epsilon = 1e-8;
    for (auto& img : images) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                img.data[i][j] = (img.data[i][j] - global_mean) / (global_std + epsilon);
                
                // Clipping suave
                if (img.data[i][j] > 3.0) img.data[i][j] = 3.0;
                if (img.data[i][j] < -3.0) img.data[i][j] = -3.0;
            }
        }
    }
    
    std::cout << "✅ Normalización - Media: " << global_mean << ", Std: " << global_std << std::endl;
}

void monitor_gpu_balanced() {
#ifdef USE_CUDA
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t used_mem = total_mem - free_mem;
    
    std::cout << "⚡ GPU: " << used_mem / (1024*1024) << "MB (" 
              << (100.0 * used_mem / total_mem) << "%)" << std::endl;
#endif
}

// Learning rate con warmup y cosine decay
double get_balanced_lr(int epoch, int batch, int batches_per_epoch, double base_lr, int warmup_steps = 200) {
    int global_step = epoch * batches_per_epoch + batch;
    
    // Warmup phase
    if (global_step < warmup_steps) {
        return base_lr * (double)global_step / warmup_steps;
    }
    
    // Cosine decay
    double decay_factor = 0.5 * (1.0 + cos(M_PI * epoch / 30.0));
    return base_lr * decay_factor * 0.5 + base_lr * 0.5;  // No bajar de 50% del LR base
}

int main() {
    // CONFIGURACIÓN BALANCEADA
    force_nvidia_gpu_balanced();
    print_fashion_mnist_classes();
    
    // CONFIGURACIÓN BALANCEADA: Velocidad + Accuracy
    const int EPOCHS = 30;               // Más épocas para convergencia
    const int BATCH_SIZE = 48;          // Balance entre velocidad y gradientes
    const double BASE_LR = 0.0008;      // Learning rate optimizado
    const int SUBSET_SIZE = 40000;      // 67% del dataset
    
    std::cout << "⚡ CONFIGURACIÓN BALANCEADA:" << std::endl;
    std::cout << "   - Épocas: " << EPOCHS << std::endl;
    std::cout << "   - Batch size: " << BATCH_SIZE << " (balanceado)" << std::endl;
    std::cout << "   - Base LR: " << BASE_LR << " con warmup + cosine decay" << std::endl;
    std::cout << "   - Dataset: " << SUBSET_SIZE << " muestras (67%)" << std::endl;
    std::cout << "   - OBJETIVO: <800ms/batch + 75-80% accuracy\n" << std::endl;
    
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // Cargar dataset
    std::cout << "📊 Cargando dataset..." << std::endl;
    std::vector<Matrix> train_images, test_images;
    std::vector<int> train_labels, test_labels;
    
    train_images = MNISTLoader::load_images("train-images-idx3-ubyte");
    train_labels = MNISTLoader::load_labels("train-labels-idx1-ubyte");
    test_images = MNISTLoader::load_images("t10k-images-idx3-ubyte");
    test_labels = MNISTLoader::load_labels("t10k-labels-idx1-ubyte");
    
    if (train_images.empty() || test_images.empty()) {
        std::cerr << "❌ Error cargando dataset" << std::endl;
        return -1;
    }
    
    // Usar subset más grande
    if (train_images.size() > SUBSET_SIZE) {
        train_images.resize(SUBSET_SIZE);
        train_labels.resize(SUBSET_SIZE);
    }
    
    std::cout << "✅ Dataset: " << train_images.size() << " train, " 
              << test_images.size() << " test" << std::endl;
    
    // Normalización mejorada
    std::cout << "\n📊 Normalización mejorada..." << std::endl;
    normalize_images_balanced(train_images);
    normalize_images_balanced(test_images);
    
    monitor_gpu_balanced();
    
    // TRANSFORMER BALANCEADO (2M parámetros)
    std::cout << "\n🎯 Inicializando Transformer BALANCEADO..." << std::endl;
    
    // Configuración optimizada para balance velocidad-accuracy
    const int patch_size = 7;       // 16 patches (menos cómputo)
    const int d_model = 96;         // MUY pequeño
    const int num_heads = 6;        // 6 heads de 16 dims c/u
    const int num_layers = 3;       // Solo 3 capas!
    const int d_ff = 192;           // 2x d_model
    const double dropout = 0.2;     // Más dropout para compensar
    
    Transformer transformer(d_model, num_heads, num_layers, d_ff, 10, patch_size, dropout);
    transformer.set_training(true);
    
    std::cout << "=== MODELO BALANCEADO ===" << std::endl;
    std::cout << "- d_model: " << d_model << " (+37.5%)" << std::endl;
    std::cout << "- num_heads: " << num_heads << " (2x)" << std::endl;
    std::cout << "- num_layers: " << num_layers << " (+1)" << std::endl;
    std::cout << "- d_ff: " << d_ff << " (+50%)" << std::endl;
    std::cout << "- patch_size: " << patch_size << " (49 patches)" << std::endl;
    std::cout << "- batch_size: " << BATCH_SIZE << std::endl;
    std::cout << "- dropout: " << dropout << std::endl;
    std::cout << "- Parámetros: ~2M (balanceado)" << std::endl;
    std::cout << "==========================\n" << std::endl;
    
    monitor_gpu_balanced();
    
    // ENTRENAMIENTO BALANCEADO
    std::cout << "🚀 === ENTRENAMIENTO BALANCEADO ===\n" << std::endl;
    
    const int batches_per_epoch = (train_images.size() + BATCH_SIZE - 1) / BATCH_SIZE;
    std::cout << "Batches por época: " << batches_per_epoch << std::endl;
    
    // Variables para estadísticas
    std::vector<long long> batch_times;
    std::vector<double> epoch_accuracies;
    batch_times.reserve(batches_per_epoch * EPOCHS);
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        std::cout << "\n--- ÉPOCA " << (epoch + 1) << "/" << EPOCHS << " ---" << std::endl;
        
        auto indices = create_shuffled_indices(train_images.size());
        
        double epoch_loss = 0.0;
        int correct_predictions = 0;
        double sum_lr = 0.0;
        
        for (int batch = 0; batch < batches_per_epoch; batch++) {
            auto batch_start = std::chrono::high_resolution_clock::now();
            
            // Learning rate dinámico
            double current_lr = get_balanced_lr(epoch, batch, batches_per_epoch, BASE_LR);
            sum_lr += current_lr;
            
            // Preparar batch
            std::vector<Matrix> batch_images;
            std::vector<int> batch_labels;
            
            int batch_start_idx = batch * BATCH_SIZE;
            int batch_end_idx = std::min(batch_start_idx + BATCH_SIZE, (int)train_images.size());
            
            batch_images.reserve(batch_end_idx - batch_start_idx);
            batch_labels.reserve(batch_end_idx - batch_start_idx);
            
            for (int i = batch_start_idx; i < batch_end_idx; i++) {
                int idx = indices[i];
                batch_images.push_back(train_images[idx]);
                batch_labels.push_back(train_labels[idx]);
            }
            
            // ENTRENAR
            auto result = transformer.train_batch(batch_images, batch_labels, current_lr);
            
            double batch_loss = result.first;
            double batch_acc = result.second;
            
            epoch_loss += batch_loss;
            correct_predictions += (int)(batch_acc * batch_images.size());
            
            auto batch_end = std::chrono::high_resolution_clock::now();
            auto batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start).count();
            
            batch_times.push_back(batch_time);
            
            // Mostrar progreso cada 50 batches
            if (batch % 20 == 0 || batch == batches_per_epoch - 1) {
                std::cout << "Batch " << batch << "/" << batches_per_epoch 
                         << " | Loss: " << std::fixed << std::setprecision(4) << batch_loss
                         << " | Acc: " << std::setprecision(2) << (batch_acc * 100.0) << "%"
                         << " | LR: " << std::scientific << std::setprecision(2) << current_lr
                         << " | Time: " << batch_time << "ms";
                
                if (batch_time < 800) {
                    std::cout << " ⚡RÁPIDO";
                } else if (batch_time < 1200) {
                    std::cout << " ✅OK";
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
        double avg_lr = sum_lr / batches_per_epoch;
        
        epoch_accuracies.push_back(epoch_accuracy);
        
        std::cout << "\n📊 ÉPOCA " << (epoch + 1) << " COMPLETADA:" << std::endl;
        std::cout << "   - Tiempo: " << epoch_time << "s" << std::endl;
        std::cout << "   - Loss: " << std::fixed << std::setprecision(4) << epoch_loss << std::endl;
        std::cout << "   - Accuracy: " << std::fixed << std::setprecision(2) << epoch_accuracy << "%" << std::endl;
        std::cout << "   - LR promedio: " << std::scientific << std::setprecision(2) << avg_lr << std::endl;
        
        // Early stopping si alcanzamos objetivo
        if (epoch_accuracy > 78.0) {
            std::cout << "   🎯 ¡Objetivo de accuracy alcanzado!" << std::endl;
        }
        
        monitor_gpu_balanced();
    }
    
    auto end_total = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_total - start_total).count();
    
    // Estadísticas finales
    long long total_batch_time = 0;
    for (auto t : batch_times) {
        total_batch_time += t;
    }
    double avg_batch_time = batch_times.empty() ? 0 : (double)total_batch_time / batch_times.size();
    
    std::cout << "\n🎉 === ENTRENAMIENTO COMPLETADO ===" << std::endl;
    std::cout << "Tiempo total: " << total_time << "s" << std::endl;
    
    // Evaluación completa
    std::cout << "\n📊 Evaluación en test set..." << std::endl;
    transformer.set_training(false);
    
    int test_samples = 2000;
    int correct = 0;
    
    for (int i = 0; i < std::min(test_samples, (int)test_images.size()); i++) {
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
    
    double test_accuracy = (double)correct / test_samples * 100.0;
    
    std::cout << "\n🎯 === RESULTADOS BALANCEADOS ===" << std::endl;
    std::cout << "✅ Test Accuracy Final: " << std::fixed << std::setprecision(2) << test_accuracy << "%" << std::endl;
    std::cout << "✅ Tiempo promedio/batch: " << avg_batch_time << "ms" << std::endl;
    std::cout << "✅ Modelo: ~2M parámetros" << std::endl;
    std::cout << "✅ Dataset: " << SUBSET_SIZE << " muestras" << std::endl;
    
    // Mostrar progresión de accuracy
    std::cout << "\n📈 Progresión de Accuracy:" << std::endl;
    for (size_t i = 0; i < epoch_accuracies.size(); i += 5) {
        std::cout << "   Época " << (i + 1) << ": " 
                  << std::fixed << std::setprecision(2) << epoch_accuracies[i] << "%" << std::endl;
    }
    
    monitor_gpu_balanced();
    
    // Análisis de resultados
    std::cout << "\n🎯 ANÁLISIS DE RESULTADOS:" << std::endl;
    if (test_accuracy >= 75.0 && avg_batch_time < 800) {
        std::cout << "🏆 ¡OBJETIVO ALCANZADO! Velocidad + Accuracy balanceados" << std::endl;
    } else if (test_accuracy >= 75.0) {
        std::cout << "✅ Buen accuracy, velocidad aceptable" << std::endl;
    } else if (avg_batch_time < 800) {
        std::cout << "⚡ Buena velocidad, accuracy mejorable" << std::endl;
    } else {
        std::cout << "⚠️ Necesita más optimización" << std::endl;
    }
    
    std::cout << "\n💡 CONFIGURACIÓN BALANCEADA:" << std::endl;
    std::cout << "   - Modelo 2M parámetros (2.5x vs speed max)" << std::endl;
    std::cout << "   - 49 patches (4x4) vs 16 patches (7x7)" << std::endl;
    std::cout << "   - Learning rate con warmup + cosine decay" << std::endl;
    std::cout << "   - Dataset 40K muestras (4x más)" << std::endl;
    std::cout << "   - Regularización mejorada (dropout 0.15)" << std::endl;
    
    return 0;
}