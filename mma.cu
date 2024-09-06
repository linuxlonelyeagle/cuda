#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>

// from https://github.com/Bruce-Lee-LY/cuda_hgemm
using namespace std;

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_SIZE 32

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"     \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr)                                              \
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"    \
               : "=r"(R0), "=r"(R1)                                            \
               : "r"(addr))

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)            \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, "  \
               "{%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"                       \
               : "=r"(RD0), "=r"(RD1)                                          \
               : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1),   \
                 "r"(RC0), "r"(RC1))

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void mmaNaiveKernel(const half *__restrict__ A,
                               const half *__restrict__ B, half *__restrict__ C,
                               size_t M, size_t N, size_t K) {
  const size_t K_tiles = div_ceil(K, MMA_K);

  const size_t warp_row = blockIdx.y * MMA_M;
  const size_t warp_col = blockIdx.x * MMA_N;

  if (warp_row >= M || warp_col >= N) {
    return;
  }

  __shared__ half A_smem[MMA_M][MMA_K];
  __shared__ half B_smem[MMA_N][MMA_K];
  __shared__ half C_smem[MMA_M][MMA_N];

  const size_t lane_id = threadIdx.x % WARP_SIZE;

  uint32_t RC[2] = {0, 0};

#pragma unroll
  for (size_t i = 0; i < K_tiles; ++i) {
    // int4 struct {x, y , z, x}
    *((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
        *((int4 *)(&A[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2);

    if (lane_id < MMA_N * 2) {
      *((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) = *(
          (int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2);
    }

    __syncthreads();

    uint32_t RA[4];
    uint32_t RB[2];

    uint32_t A_smem_lane_addr =
        __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
    LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

    uint32_t B_smem_lane_addr =
        __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
    LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

    HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0],
              RC[1]);

    __syncthreads();
  }

  *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
  *((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];

  __syncthreads();

  if (lane_id < MMA_M) {
    *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) =
        *((int4 *)(&C_smem[lane_id][0]));
  }
}

void mmaNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
  dim3 block(WARP_SIZE);
  dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));
  mmaNaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
}

int main() {
  int m = 32, n = 32, k = 32;
  half a[m][k];
  half b[k][n];
  half c[m][n];

  half *A, *B, *C;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      a[i][j] = half(1.0);
    }
  }
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      b[i][j] = half(1.0);
    }
  }
  cudaMalloc(&A, m * n * sizeof(half));
  cudaMalloc(&B, k * n * sizeof(half));
  cudaMalloc(&C, m * n * sizeof(half));
  cudaMemcpy(A, a, m * k * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(B, b, k * n * sizeof(half), cudaMemcpyHostToDevice);
  mmaNaive(A, B, C, m, n, k);
  cudaMemcpy(c, C, m * n * sizeof(half), cudaMemcpyDeviceToHost);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      cout << __half2float(c[i][j]) << " ";
    }
    cout << "\n";
  }
  return 0;
}
