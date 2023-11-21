#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// Function for CPU matrix multiplication
void matrix_mul_CPU(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            double sum = 0.0;
            for (int k = 0; k < colsA; k++)
                sum += A[i * colsA + k] * B[k * colsB + j];
            C[i * colsB + j] = sum;
        }
    }
}

// Kernel for GPU matrix multiplication
__global__ void matrix_mul_GPU(const double* A, const double* B, double* C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    if (row < rowsA && col < colsB) {
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

int main() {
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    int m = 1000;
    int n = 1000;
    int p = 1000;

    std::vector<double> a(m * p);
    std::vector<double> b(p * n);
    std::vector<double> c(m * n, 0.0);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            a[i * p + j] = distribution(generator);
        }
    }

    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < n; ++j) {
            b[i * n + j] = distribution(generator);
        }
    }

    std::vector<double> hc(m * n, 0.0);

    // CPU matrix multiplication
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrix_mul_CPU(a, b, c, m, n, p);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;

    double* da, * db, * dc;
    cudaMalloc(&da, m * p * sizeof(double));
    cudaMalloc(&db, p * n * sizeof(double));
    cudaMalloc(&dc, m * n * sizeof(double));

    cudaMemcpy(da, a.data(), m * p * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), p * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, hc.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block_dim(32, 32);
    dim3 grid_dim((n + block_dim.x - 1) / block_dim.x, (m + block_dim.y - 1) / block_dim.y);

    cudaEvent_t begin, stop;
    cudaEventCreate(&begin);
    cudaEventCreate(&stop);

    cudaEventRecord(begin, 0);
    matrix_mul_GPU << <grid_dim, block_dim >> > (da, db, dc, m, n, p);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, begin, stop);

    cudaMemcpy(hc.data(), dc, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Verify results
    bool is_equal = true;
    for (int i = 0; i < m * n; ++i) {
        if (c[i] != hc[i]) {
            is_equal = false;
            break;
        }
    }

    std::cout << "CPU time (s): " << cpu_duration.count() << std::endl;
    std::cout << "GPU time (s): " << gpu_time / 1000.0 << std::endl;

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    cudaEventDestroy(begin);
    cudaEventDestroy(stop);

    return 0;
}
