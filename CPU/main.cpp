#include "include/transformer.h"
#include "include/mnist_loader.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>
#include <numeric>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <cmath>

void print_fashion_mnist_classes() {
    std::cout << "\n=== Fashion-MNIST Classes ===" << std::endl;
    std::cout << "0: T-shirt/top" << std::endl;
    std::cout << "1: Trouser" << std::endl;
    std::cout << "2: Pullover" << std::endl;
    std::cout << "3: Dress" << std::endl;
    std::cout << "4: Coat" << std::endl;
    std::cout << "5: Sandal" << std::endl;
    std::cout << "6: Shirt" << std::endl;
    std::cout << "7: Sneaker" << std::endl;
    std::cout << "8: Bag" << std::endl;
    std::cout << "9: Ankle boot" << std::endl;
    std::cout << "========================\n" << std::endl;
}

std::vector<int> create_shuffled_indices(int size) {
    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    return indices;
}

void normalize_images(std::vector<Matrix>& images) {
    // Calculate global mean and std for better normalization
    double global_mean = 0.0;
    double global_std = 0.0;
    int total_pixels = 0;
    
    // Calculate mean
    for (const auto& img : images) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                global_mean += img.data[i][j];
                total_pixels++;
            }
        }
    }
    global_mean /= total_pixels;
    
    // Calculate std
    for (const auto& img : images) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                double diff = img.data[i][j] - global_mean;
                global_std += diff * diff;
            }
        }
    }
    global_std = sqrt(global_std / total_pixels);
    
    // Normalize with calculated mean and std
    for (auto& img : images) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                img.data[i][j] = (img.data[i][j] - global_mean) / (global_std + 1e-8);
            }
        }
    }
    
    std::cout << "Global normalization - Mean: " << global_mean 
              << ", Std: " << global_std << std::endl;
}

void train_with_batches(Transformer& model, 
                       const std::vector<Matrix>& train_images,
                       const std::vector<int>& train_labels,
                       int epochs, int batch_size, double initial_lr) {
    
    int num_samples = train_images.size();
    int num_batches = (num_samples + batch_size - 1) / batch_size;
    
    std::cout << "\n=== Starting Batch Training ===" << std::endl;
    std::cout << "Samples: " << num_samples << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Batches per epoch: " << num_batches << std::endl;
    
    // Open CSV for training history
    std::ofstream history_csv("training_history.csv");
    if (!history_csv.is_open()) {
        std::cerr << "Error opening training_history.csv" << std::endl;
        return;
    }
    history_csv << "epoch,loss,accuracy,learning_rate\n";
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        // Shuffle data each epoch
        std::vector<int> indices = create_shuffled_indices(num_samples);
        
        double epoch_loss = 0.0;
        double epoch_accuracy = 0.0;
        
        // Improved learning rate scheduling for better convergence
        double current_lr;
        if (epoch < 10) {
            // Warmup phase: gradually increase LR
            current_lr = initial_lr * (epoch + 1) / 10.0;
        } else {
            // Decay phase: cosine annealing
            double progress = (epoch - 10.0) / (epochs - 10.0);
            current_lr = initial_lr * 0.5 * (1.0 + cos(3.14159 * progress));
        }
        
        std::cout << "\n--- Epoch " << (epoch + 1) << "/" << epochs 
                  << " (LR: " << current_lr << ") ---" << std::endl;
        
        for (int batch = 0; batch < num_batches; batch++) {
            std::vector<Matrix> batch_images;
            std::vector<int> batch_labels;
            
            // Create batch
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, num_samples);
            
            for (int i = start_idx; i < end_idx; i++) {
                batch_images.push_back(train_images[indices[i]]);
                batch_labels.push_back(train_labels[indices[i]]);
            }
            
            // Train on batch
            std::pair<double, double> batch_result = model.train_batch(batch_images, batch_labels, current_lr);
            double batch_loss = batch_result.first;
            double batch_acc = batch_result.second;
            
            epoch_loss += batch_loss;
            epoch_accuracy += batch_acc;
            
            if (batch % 10 == 0) {
                std::cout << "Batch " << batch << "/" << num_batches 
                         << " | Loss: " << std::fixed << std::setprecision(5) << batch_loss 
                         << " | Acc: " << std::fixed << std::setprecision(3) << (batch_acc * 100.0) << "%" << std::endl;
            }
        }
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);
        
        double avg_loss = epoch_loss / num_batches;
        double avg_accuracy = (epoch_accuracy / num_batches) * 100.0;
        
        std::cout << "\nEpoch " << (epoch + 1) << " completed in " 
                  << epoch_duration.count() << " seconds" << std::endl;
        std::cout << "Average Loss: " << std::fixed << std::setprecision(5) << avg_loss << std::endl;
        std::cout << "Average Accuracy: " << std::fixed << std::setprecision(4) << avg_accuracy << "%" << std::endl;
        
        // Save to CSV
        history_csv << (epoch + 1) << "," << avg_loss << "," << avg_accuracy << "," << current_lr << "\n";
        
        // More aggressive early stopping for better results
        if (avg_accuracy > 85.0) {
            std::cout << "Early stopping: Excellent accuracy reached!" << std::endl;
            break;
        }
        
        // Stop if loss is not improving (simple check)
        if (epoch > 20 && avg_accuracy < 40.0) {
            std::cout << "Early stopping: Model not learning effectively" << std::endl;
            break;
        }
    }
    
    history_csv.close();
}

void evaluate_model(Transformer& model, 
                   const std::vector<Matrix>& test_images,
                   const std::vector<int>& test_labels,
                   int num_samples = 1000) {
    
    std::cout << "\n=== Evaluating Model ===" << std::endl;
    
    model.set_training(false); // Set to evaluation mode
    
    auto test_start = std::chrono::high_resolution_clock::now();
    
    int correct = 0;
    double total_loss = 0.0;
    
    num_samples = std::min(num_samples, static_cast<int>(test_images.size()));
    
    // Vectors to store predictions for CSV
    std::vector<int> true_labels_vec;
    std::vector<int> predicted_labels_vec;
    std::vector<double> confidences_vec;
    
    for (int i = 0; i < num_samples; i++) {
        if (i % 100 == 0) {
            std::cout << "Testing sample " << i << "/" << num_samples << "\r" << std::flush;
        }
        
        Matrix predictions = model.forward(test_images[i]);
        
        // Compute loss
        std::vector<int> single_label = {test_labels[i]};
        total_loss += model.compute_loss(predictions, single_label);
        
        // Get prediction
        int predicted_class = 0;
        double max_prob = predictions.data[0][0];
        for (int j = 1; j < 10; j++) {
            if (predictions.data[0][j] > max_prob) {
                max_prob = predictions.data[0][j];
                predicted_class = j;
            }
        }
        
        if (predicted_class == test_labels[i]) {
            correct++;
        }
        
        // Store for CSV
        true_labels_vec.push_back(test_labels[i]);
        predicted_labels_vec.push_back(predicted_class);
        confidences_vec.push_back(max_prob);
    }
    
    auto test_end = std::chrono::high_resolution_clock::now();
    auto test_duration = std::chrono::duration_cast<std::chrono::seconds>(test_end - test_start);
    
    double test_loss = total_loss / num_samples;
    double test_accuracy = (double)correct / num_samples * 100.0;
    
    std::cout << "\nTesting completed in " << test_duration.count() << " seconds" << std::endl;
    std::cout << "Test Loss: " << std::fixed << std::setprecision(5) << test_loss << std::endl;
    std::cout << "Test Accuracy: " << static_cast<int>(test_accuracy) << "%" << std::endl;
    
    // Save test results to CSV
    std::ofstream predictions_csv("predictions.csv");
    if (!predictions_csv.is_open()) {
        std::cerr << "Error opening predictions.csv" << std::endl;
    } else {
        predictions_csv << "true_label,predicted_label,confidence\n";
        for (int i = 0; i < num_samples; i++) {
            predictions_csv << true_labels_vec[i] << "," << predicted_labels_vec[i] << "," << confidences_vec[i] << "\n";
        }
        predictions_csv.close();
    }
    
    // Append test metrics to a separate CSV or console only
    std::ofstream test_metrics_csv("test_metrics.csv");
    if (!test_metrics_csv.is_open()) {
        std::cerr << "Error opening test_metrics.csv" << std::endl;
    } else {
        test_metrics_csv << "test_loss,test_accuracy\n";
        test_metrics_csv << test_loss << "," << test_accuracy << "\n";
        test_metrics_csv.close();
    }
    
    model.set_training(true); // Back to training mode
}

void show_predictions(Transformer& model,
                     const std::vector<Matrix>& images,
                     const std::vector<int>& labels,
                     int num_samples = 5) {
    
    std::vector<std::string> class_names = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };
    
    std::cout << "\n=== Sample Predictions ===" << std::endl;
    
    model.set_training(false);
    
    for (int i = 0; i < num_samples && i < images.size(); i++) {
        Matrix predictions = model.forward(images[i]);
        
        int predicted_class = 0;
        double max_prob = predictions.data[0][0];
        for (int j = 1; j < 10; j++) {
            if (predictions.data[0][j] > max_prob) {
                max_prob = predictions.data[0][j];
                predicted_class = j;
            }
        }
        
        std::cout << "Sample " << (i + 1) << ":" << std::endl;
        std::cout << "  True: " << class_names[labels[i]] << " (" << labels[i] << ")" << std::endl;
        std::cout << "  Predicted: " << class_names[predicted_class] << " (" << predicted_class << ")" << std::endl;
        std::cout << "  Confidence: " << std::fixed << std::setprecision(4) << (max_prob * 100.0) << "%" << std::endl;
        std::cout << "  Correct: " << (predicted_class == labels[i] ? "Yes" : "No") << std::endl;
        std::cout << std::endl;
    }
    
    model.set_training(true);
}

int main() {
    std::cout << "=== Fashion-MNIST Transformer Classifier (Optimized) ===" << std::endl;
    print_fashion_mnist_classes();
    
    // File paths
    std::string train_images_path = "train-images-idx3-ubyte";
    std::string train_labels_path = "train-labels-idx1-ubyte";
    std::string test_images_path = "t10k-images-idx3-ubyte";
    std::string test_labels_path = "t10k-labels-idx1-ubyte";
    
    
    try {
        std::cout << "Loading Fashion-MNIST dataset..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        MNISTLoader loader;
        std::vector<Matrix> train_images = loader.load_images(train_images_path);
        std::vector<int> train_labels = loader.load_labels(train_labels_path);
        std::vector<Matrix> test_images = loader.load_images(test_images_path);
        std::vector<int> test_labels = loader.load_labels(test_labels_path);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        std::cout << "Data loading completed in " << duration.count() << " seconds." << std::endl;
        
        // Use more data for better training (if computationally feasible)
        int train_samples = std::min(60000, static_cast<int>(train_images.size())); // Todo el dataset
        int test_samples = std::min(10000, static_cast<int>(test_images.size()));   // Todo el test set
        
        // Resize vectors to use subset
        train_images.resize(train_samples);
        train_labels.resize(train_samples);
        test_images.resize(test_samples);
        test_labels.resize(test_samples);
        
        std::cout << "\nDataset Summary:" << std::endl;
        std::cout << "Training samples: " << train_samples << std::endl;
        std::cout << "Test samples: " << test_samples << std::endl;
        
        // Normalize data
        std::cout << "\nNormalizing images..." << std::endl;
        normalize_images(train_images);
        normalize_images(test_images);
        
        // Create more powerful model for better accuracy
        std::cout << "\nInitializing Transformer model..." << std::endl;
        Transformer model(
            128,  // d_model (duplicado para más capacidad)
            8,    // num_heads (más cabezas de atención)
            4,    // num_layers (más capas para aprender mejor)
            512,  // d_ff (red feedforward más grande)
            10,   // num_classes
            4,    // patch_size
            0.1   // dropout_rate
        );
        
        model.print_model_info();
        
        // Optimized training parameters for higher accuracy
        int epochs = 20;        // Más épocas para convergencia completa
        int batch_size = 64;    // Batch más pequeño para mejor gradientes
        double learning_rate = 0.001;  // LR un poco más alto para aprender más rápido
        
        std::cout << "\nTraining Configuration:" << std::endl;
        std::cout << "- Epochs: " << epochs << std::endl;
        std::cout << "- Batch size: " << batch_size << std::endl;
        std::cout << "- Learning rate: " << learning_rate << std::endl;
        
        // Train the model
        train_with_batches(model, train_images, train_labels, epochs, batch_size, learning_rate);
        
        // Evaluate on test set
        evaluate_model(model, test_images, test_labels, test_samples);
        
        // Show sample predictions
        show_predictions(model, test_images, test_labels, 10);
        
        std::cout << "\n=== Training and Testing Completed ===" << std::endl;
        
        // Additional info for Python parser
        std::cout << "\n=== Files Generated ===" << std::endl;
        std::cout << "📊 training_history.csv - " << epochs << " epochs of training data" << std::endl;
        std::cout << "🎯 predictions.csv - " << test_samples << " test predictions" << std::endl;
        std::cout << "📈 test_metrics.csv - Final test results" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}