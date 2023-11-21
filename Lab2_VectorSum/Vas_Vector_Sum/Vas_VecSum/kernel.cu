#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <vector>



using namespace std;

// Функция для сложения элементов вектора на CPU
float CPUArraySum(const float* arr, int size) {
    float sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += arr[i];
    }
    return sum;
}


// Функция для сложения элементов вектора на GPU
__global__ void GPUVectorSum(float* d_vec, float* d_result, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float local_sum = 0.0;
    while (tid < size) {
        local_sum += d_vec[tid];
        tid += blockDim.x * gridDim.x;
    }
    atomicAdd(d_result, local_sum);
}

// Функция для заполнения вектора псевдослучайными значениями
void FillVectorWithRandomValues(vector<float>& vec) {
    srand(static_cast<unsigned int>(time(nullptr)));
    for (float& element : vec) {
        element = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    setlocale(LC_ALL, "Russian");

    const int vectorSize = 600000000





        ;  // Размер вектора (измените по необходимости)

    // Создание и заполнение векторов на CPU
    vector<float> hostVector(vectorSize);
    vector<float> resultOnCPU(1, 0.0);
    FillVectorWithRandomValues(hostVector);

    // Копирование данных с CPU на GPU
    float* d_vec;
    float* d_result;
    cudaMalloc((void**)&d_vec, vectorSize * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));
    cudaMemcpy(d_vec, hostVector.data(), vectorSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, resultOnCPU.data(), sizeof(float), cudaMemcpyHostToDevice);

    // Измерение времени выполнения на CPU
    auto startCPU = chrono::high_resolution_clock::now();
    float resultOnCPUSum = CPUArraySum(hostVector.data(), hostVector.size());
    auto endCPU = chrono::high_resolution_clock::now();
    chrono::duration<double> cpuDuration = endCPU - startCPU;


    // Вычисление на GPU и измерение времени выполнения
    int threadsPerBlock = 256;
    int numBlocks = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;

    auto startGPU = chrono::high_resolution_clock::now();
    GPUVectorSum << <numBlocks, threadsPerBlock >> > (d_vec, d_result, vectorSize);
    auto endGPU = chrono::high_resolution_clock::now();
    chrono::duration<double> gpuDuration = endGPU - startGPU;

    // Копирование результата с GPU на CPU
    cudaMemcpy(resultOnCPU.data(), d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Вывод результатов и времени выполнения
    cout << "Сумма на CPU: " << resultOnCPUSum << endl;
    cout << "Сумма на GPU: " << resultOnCPU[0] << endl;
    cout << "Время выполнения на CPU: " << cpuDuration.count() << " секунд" << endl;
    cout << "Время выполнения на GPU: " << gpuDuration.count() << " секунд" << endl;

    cudaFree(d_vec);
    cudaFree(d_result);

    return 0;
}
