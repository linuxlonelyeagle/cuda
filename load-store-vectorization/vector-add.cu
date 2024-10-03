#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include<assert.h>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define FLOAT4(ptr) ((float4*)ptr)

#define checkCudaErrors(func)               \
{                                   \
    cudaError_t e = (func);         \
    if(e != cudaSuccess)                                        \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));        \
}

__global__ void vector_add(float* x, float* y, float* z, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    z[idx] = x[idx] + y[idx];
  }
}

__global__ void vector_add_vectorial(float* x, float* y, float *z, int N) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x);
  if (idx < N) {
    float4 xTmp = FLOAT4(x)[idx];
    float4 yTmp = FLOAT4(y)[idx];
    float4 zTmp;
    zTmp.x = xTmp.x + yTmp.x;
    zTmp.y = xTmp.y + yTmp.y;
    zTmp.z = xTmp.z + yTmp.z;
    zTmp.w = xTmp.w + xTmp.w;
    FLOAT4(z)[idx] = zTmp;
  }
}

int main(){
    size_t size = 1 << 27;
    size_t blockSize = 512;
    size_t bytes = sizeof(float) * size;

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    for( int i = 0; i < size; i++ ){
        h_A[i] = 1.0;
        h_B[i] = 1.0;
    }
    float* d_A;
    float* d_B;
    float* d_C;
    checkCudaErrors(cudaMalloc(&d_A, bytes));
    checkCudaErrors(cudaMalloc(&d_B, bytes));
    checkCudaErrors(cudaMalloc(&d_C, bytes));
    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes, cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
    float msec;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start));    
    vector_add<<<(size + blockSize -1 ) / blockSize, blockSize>>>(d_A, d_B, d_C, size);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
    printf("vector-add takes %.3f msec\n", msec);

    checkCudaErrors(cudaEventRecord(start));
    vector_add_vectorial<<<(size + blockSize - 1) / (blockSize), blockSize / 4 >>>(d_A, d_B, d_C, size);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
    printf("vector-add-vectorial takes %.3f msec\n", msec);
    checkCudaErrors(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}