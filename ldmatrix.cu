#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

// https://www.jianshu.com/p/c3c2f7d30eda
// https://stackoverflow.com/questions/76992939/confusion-about-cvta-generic-to-shared
__global__ void helloFromGPU(void) {
  __shared__ uint32_t aTile[4 * 8 * 4];
  int tidx = threadIdx.x + blockDim.x * threadIdx.y;
  if (tidx == 0) {
    for (int i = 0; i < 4 * 8 * 4; ++i) {
      aTile[i] = i;
    }
  }
  __syncthreads();
  int aTile_index = tidx % 16 * 8 + tidx / 16 * 4;
  uint32_t a[4];
  uint32_t smem = __cvta_generic_to_shared(aTile + aTile_index);
  asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [ %4 ];\n"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
      : "r"(smem));

  if (tidx == 1) {
    printf("%d \n", (a[0]));
    printf("%d \n", (a[1]));
    printf("%d \n", (a[2]));
    printf("%d \n", (a[3]));
  }
}

__global__ void singleMatrix() {
  __shared__ uint32_t shareData[8 * 4];
  shareData[threadIdx.x] = threadIdx.x;
  __syncthreads();
  uint32_t a;
  if (threadIdx.x < 8) {
    uint32_t smem = __cvta_generic_to_shared(shareData + threadIdx.x * 4);
    asm("ldmatrix.sync.aligned.m8n8.x1.shared.b16 { %0 }, [ %1 ];\n"
        : "=r"(a)
        : "r"(smem));
  }
  printf("%u\n", a);
}

int main(int argc, char *argv[]) {
  uint3 block = {32, 1, 1};
  uint3 grid = {1, 1, 1};
  helloFromGPU<<<grid, block>>>();
  singleMatrix<<<grid, block>>>();
  cudaDeviceReset();
  return 0;
}
