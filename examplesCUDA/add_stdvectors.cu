/**
 * @file add_two_stdvectors.cu
 * @author remzerrr (remi.helleboid@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-09-24
 *
 * @copyright Copyright (c) 2022
 *
 * In this example, we will add two std::vectors on the GPU.
 *
 */

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

void add_serial(const std::vector<double>& x, std::vector<double>& y, double alpha, double beta) {
    for (size_t i = 0; i < x.size(); i++) {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

// Kernel function to compute the linear combination of two vectors.
__global__ void add(int n, double* x, double* y, double alpha, double beta) {
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = alpha * x[i] + beta * y[i];
}

int main(void) {
    const unsigned int SIZE_DOUBLE = sizeof(double);
    int                nb_values   = 1 << 26;  // 1M elements
    std::cout << "Number values = " << nb_values << std::endl;

    // Create input vectors on host and initialize them.
    std::vector<double> x_device(nb_values);
    std::vector<double> y_device(nb_values);
    for (int i = 0; i < nb_values; i++) {
        x_device[i] = 1.0;
        y_device[i] = 2.0;
    }

    // Start the timer
    auto start_gpu = std::chrono::high_resolution_clock::now();

    // Allocate memory on device to receive the input vectors.
    double* x_host;
    double* y_host;
    cudaMalloc(&x_host, nb_values * SIZE_DOUBLE);
    cudaMalloc(&y_host, nb_values * SIZE_DOUBLE);

    // Copy input vectors from host to device.
    cudaMemcpy(x_host, x_device.data(), nb_values * SIZE_DOUBLE, cudaMemcpyHostToDevice);
    cudaMemcpy(y_host, y_device.data(), nb_values * SIZE_DOUBLE, cudaMemcpyHostToDevice);

    // Determine the number of threads per block and the number of blocks.
    int blockSize = 256;
    int numBlocks = (nb_values + blockSize - 1) / blockSize;

    // Launch the kernel.
    const double alpha = M_PI;
    const double beta  = M_E;
    add<<<numBlocks, blockSize>>>(nb_values, x_host, y_host, alpha, beta);

    // Copy output vector from device to host.
    cudaMemcpy(y_device.data(), y_host, nb_values * SIZE_DOUBLE, cudaMemcpyDeviceToHost);

    // Free memory on device.
    cudaFree(x_host);
    cudaFree(y_host);

    // Stop the timer
    auto stop_gpu     = std::chrono::high_resolution_clock::now();
    auto duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_gpu - start_gpu);

    // Verify the result.
    double max_error = 0.0;
    for (int i = 0; i < nb_values; i++) {
        max_error = std::max(max_error, std::abs(y_device[i] - (alpha * 1 + beta * 2)));
    }
    std::cout << "Max error GPU = " << max_error << std::endl;

    // Start the timer
    for (int i = 0; i < nb_values; i++) {
        x_device[i] = 1.0;
        y_device[i] = 2.0;
    }
    auto start_cpu = std::chrono::high_resolution_clock::now();
    add_serial(x_device, y_device, alpha, beta);
    auto stop_cpu     = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu);

    // Verify the result.
    max_error = 0.0;
    for (int i = 0; i < nb_values; i++) {
        max_error = std::max(max_error, std::abs(y_device[i] - (alpha * 1 + beta * 2)));
    }
    std::cout << "Max error CPU = " << max_error << std::endl;

    std::cout << "GPU time = " << duration_gpu.count() << " us" << std::endl;
    std::cout << "CPU time = " << duration_cpu.count() << " us" << std::endl;

    return 0;
}