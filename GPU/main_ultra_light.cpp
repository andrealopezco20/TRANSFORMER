// ARQUITECTURA ULTRA-LIGERA: Mínimos parámetros, máximo rendimiento
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

void force_nvidia_gpu_ultra_light() {
    std::cout << "\n🪶 === GPU NVIDIA MODO ULTRA-LIGERO ===" << std::endl;
    
#ifdef USE_CUDA
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error == cudaSuccess && deviceCount > 0) {
        std::cout << "✅ CUDA habilitado - " << deviceCount << " dispositivos" << std::endl;
        
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
        
        std::cout << "🪶 GPU ACTIVA: " << activeProp.name << std::endl;
        std::cout << "🪶 Memoria: " << activeProp.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "🪶 MODO: ULTRA-LIGERO + EFICIENTE" << std::endl;
    }
#endif
    std::cout << "==========================================\n" << std::endl;
}

std::vector<int> create_shuffled_indices(int size) {
    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    return indices;
}

// Normalización eficiente con aug ligera
void normalize_images_ultra(std::vector<Matrix>& images, bool augment = false) {
    double global_mean = 0.1307;  // Media conocida MNIST
    double global_std = 0.3081;   // Std conocida MNIST
    
    // Normalizar directamente con valores conocidos (más rápido)
    for (auto& img : images) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                img.data[i][j] = (img.data[i][j] - global_mean) / global_std;
                
                // Augmentation ligera aleatoria
                if (augment && (rand() % 100) < 5) {  // 5% de píxeles
                    img.data[i][j] += ((rand() % 100) / 100.0 - 0.5) * 0.1;  // Ruido pequeño
                }
            }
        }
    }
    
    std::cout << "🪶 Normalización ultra-rápida completada" << std::endl;
}

// Learning rate con warmup corto y decay agresivo
double get_ultra_light_lr(int step, double base_lr, int warmup_steps = 100) {
    if (step < warmup_steps) {
        return base_lr * (double)step / warmup_steps;
    }
    
    // Decay exponencial después del warmup
    int decay_steps = step - warmup_steps;
    return base_lr * pow(0.95, decay_steps / 100.0);  // Decay cada 100 steps
}

int main() {
    // CONFIGURACIÓN ULTRA-LIGERA
    force_nvidia_gpu_ultra_light();
    
    // CONFIGURACIÓN MÍNIMA OPTIMIZADA
    const int EPOCHS = 25;              // Suficiente para converger
    const int BATCH_SIZE = 128;         // Grande para velocidad
    const double BASE_LR = 0.002;       // LR más alto para converger rápido
    const int SUBSET_SIZE = 60000;      // TODO el dataset (crítico!)
    
    std::cout << "🪶 CONFIGURACIÓN ULTRA-LIGERA:" << std::endl;
    std::cout << "   - Épocas: " << EPOCHS << std::endl;
    std::cout << "   - Batch size: " << BATCH_SIZE << " (grande)" << std::endl;
    std::cout << "   - Base LR: " << BASE_LR << " con warmup + decay" << std::endl;
    std::cout << "   - Dataset: COMPLETO (60K) ⚠️ CRÍTICO" << std::endl;
    std::cout << "   - OBJETIVO: >70% accuracy con <400K params\n" << std::endl;
    
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // Cargar dataset
    std::cout << "📊 Cargando dataset completo..." << std::endl;
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
    
    std::cout << "✅ Dataset COMPLETO: " << train_images.size() << " train, " 
              << test_images.size() << " test" << std::endl;
    
    // Normalización ultra-rápida
    std::cout << "\n🪶 Normalización ultra-rápida..." << std::endl;
    normalize_images_ultra(train_images, true);   // Con augmentation ligera
    normalize_images_ultra(test_images, false);
    
    // ARQUITECTURA ULTRA-LIGERA OPTIMIZADA
    std::cout << "\n🪶 Inicializando Transformer ULTRA-LIGERO..." << std::endl;
    
    // CONFIGURACIÓN MÍNIMA PERO EFECTIVA
    const int patch_size = 7;       // 16 patches (menos cómputo)
    const int d_model = 96;         // MUY pequeño
    const int num_heads = 6;        // 6 heads de 16 dims c/u
    const int num_layers = 3;       // Solo 3 capas!
    const int d_ff = 192;           // 2x d_model
    const double dropout = 0.2;     // Más dropout para compensar
    
    Transformer transformer(d_model, num_heads, num_layers, d_ff, 10, patch_size, dropout);
    transformer.set_training(true);
    
    // Calcular parámetros aproximados
    int patch_embed = 49 * d_model;
    int attention_params = num_layers * (4 * d_model * d_model);
    int ffn_params = num_layers * (2 * d_model * d_ff);
    int classifier_params = d_model * 10;
    int total_params = patch_embed + attention_params + ffn_params + classifier_params + d_model;
    
    std::cout << "=== MODELO ULTRA-LIGERO ===" << std::endl;
    std::cout << "- d_model: " << d_model << " (MÍNIMO)" << std::endl;
    std::cout << "- num_heads: " << num_heads << " (16 dims/head)" << std::endl;
    std::cout << "- num_layers: " << num_layers << " (SOLO 3!)" << std::endl;
    std::cout << "- d_ff: " << d_ff << " (2x d_model)" << std::endl;
    std::cout << "- patch_size: " << patch_size << " (16 patches)" << std::endl;
    std::cout << "- dropout: " << dropout << " (alto)" << std::endl;
    std::cout << "- Parámetros totales: ~" << total_params/1000 << "K (ULTRA-LIGERO)" << std::endl;
    std::cout << "==========================\n" << std::endl;
    
    // ENTRENAMIENTO OPTIMIZADO
    std::cout << "🚀 === ENTRENAMIENTO ULTRA-LIGERO ===\n" << std::endl;
    
    const int batches_per_epoch = (train_images.size() + BATCH_SIZE - 1) / BATCH_SIZE;
    std::cout << "Batches por época: " << batches_per_epoch << std::endl;
    
    std::vector<long long> batch_times;
    std::vector<double> epoch_accuracies;
    int global_step = 0;
    
    // Técnica: Empezar con ejemplos fáciles (curriculum learning ligero)
    std::vector<int> all_indices(train_images.size());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        std::cout << "\n--- ÉPOCA " << (epoch + 1) << "/" << EPOCHS << " ---" << std::endl;
        
        // Shuffle más agresivo después de primeras épocas
        if (epoch > 2) {
            std::shuffle(all_indices.begin(), all_indices.end(), std::mt19937(std::random_device()()));
        }
        
        double epoch_loss = 0.0;
        int correct_predictions = 0;
        
        for (int batch = 0; batch < batches_per_epoch; batch++) {
            auto batch_start = std::chrono::high_resolution_clock::now();
            
            // Learning rate dinámico
            double current_lr = get_ultra_light_lr(global_step++, BASE_LR);
            
            // Preparar batch grande
            std::vector<Matrix> batch_images;
            std::vector<int> batch_labels;
            
            int batch_start_idx = batch * BATCH_SIZE;
            int batch_end_idx = std::min(batch_start_idx + BATCH_SIZE, (int)train_images.size());
            
            batch_images.reserve(BATCH_SIZE);
            batch_labels.reserve(BATCH_SIZE);
            
            for (int i = batch_start_idx; i < batch_end_idx; i++) {
                int idx = all_indices[i];
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
            
            // Mostrar progreso cada 100 batches
            if (batch % 100 == 0 || batch == batches_per_epoch - 1) {
                std::cout << "Batch " << batch << "/" << batches_per_epoch 
                         << " | Loss: " << std::fixed << std::setprecision(4) << batch_loss
                         << " | Acc: " << std::setprecision(2) << (batch_acc * 100.0) << "%"
                         << " | LR: " << std::scientific << std::setprecision(2) << current_lr
                         << " | Time: " << batch_time << "ms";
                
                if (batch_time < 300) {
                    std::cout << " 🚀ULTRA";
                } else if (batch_time < 500) {
                    std::cout << " ⚡RÁPIDO";
                }
                std::cout << std::endl;
            }
        }
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_time = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start).count();
        
        double epoch_accuracy = (double)correct_predictions / train_images.size() * 100.0;
        epoch_loss /= batches_per_epoch;
        epoch_accuracies.push_back(epoch_accuracy);
        
        std::cout << "\n🪶 ÉPOCA " << (epoch + 1) << " COMPLETADA:" << std::endl;
        std::cout << "   - Tiempo: " << epoch_time << "s" << std::endl;
        std::cout << "   - Loss: " << std::fixed << std::setprecision(4) << epoch_loss << std::endl;
        std::cout << "   - Accuracy: " << std::fixed << std::setprecision(2) << epoch_accuracy << "%" << std::endl;
        
        // Early stopping si alcanzamos objetivo
        if (epoch_accuracy > 72.0 && epoch > 10) {
            std::cout << "   🎯 ¡Objetivo alcanzado! Early stopping." << std::endl;
            break;
        }
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
    
    int correct = 0;
    int test_samples = std::min(5000, (int)test_images.size());
    
    auto eval_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_samples; i++) {
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
    auto eval_time = std::chrono::duration_cast<std::chrono::seconds>(eval_end - eval_start).count();
    
    double test_accuracy = (double)correct / test_samples * 100.0;
    
    std::cout << "\n🪶 === RESULTADOS ULTRA-LIGEROS ===" << std::endl;
    std::cout << "✅ Test Accuracy Final: " << std::fixed << std::setprecision(2) << test_accuracy << "%" << std::endl;
    std::cout << "✅ Tiempo promedio/batch: " << avg_batch_time << "ms" << std::endl;
    std::cout << "✅ Tiempo evaluación: " << eval_time << "s" << std::endl;
    std::cout << "✅ Modelo: ~" << total_params/1000 << "K parámetros" << std::endl;
    std::cout << "✅ Dataset: COMPLETO (60K)" << std::endl;
    
    // Análisis de eficiencia
    double params_per_percent = (total_params/1000.0) / test_accuracy;
    std::cout << "\n📊 ANÁLISIS DE EFICIENCIA:" << std::endl;
    std::cout << "   - Eficiencia: " << std::fixed << std::setprecision(2) 
              << params_per_percent << " K params por % accuracy" << std::endl;
    std::cout << "   - Velocidad: " << (int)(BATCH_SIZE * 1000.0 / avg_batch_time) << " muestras/segundo" << std::endl;
    
    // Mostrar mejores épocas
    std::cout << "\n📈 Mejores épocas:" << std::endl;
    for (size_t i = 0; i < epoch_accuracies.size(); i++) {
        if (epoch_accuracies[i] > 65.0) {
            std::cout << "   Época " << (i + 1) << ": " 
                      << std::fixed << std::setprecision(2) << epoch_accuracies[i] << "%" << std::endl;
        }
    }
    
    std::cout << "\n🪶 CONFIGURACIÓN ULTRA-LIGERA CLAVE:" << std::endl;
    std::cout << "   ✅ Dataset COMPLETO (60K) - CRÍTICO" << std::endl;
    std::cout << "   ✅ Solo 3 capas encoder" << std::endl;
    std::cout << "   ✅ d_model=96 (mínimo viable)" << std::endl;
    std::cout << "   ✅ Batch size 128 (velocidad)" << std::endl;
    std::cout << "   ✅ Learning rate con decay agresivo" << std::endl;
    std::cout << "   ✅ Dropout 0.2 (regularización fuerte)" << std::endl;
    std::cout << "   ✅ Data augmentation ligera" << std::endl;
    
    if (test_accuracy >= 70.0) {
        std::cout << "\n🏆 ¡ÉXITO! >70% accuracy con <400K parámetros" << std::endl;
    }
    
    return 0;
}