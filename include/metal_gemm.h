#ifndef METAL_GEMM_H
#define METAL_GEMM_H

void gemm_metal(float* A, float* B, float* C, int M, int N, int K);

#endif // METAL_GEMM_H