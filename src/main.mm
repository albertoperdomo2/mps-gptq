#import <iostream>
#include "metal_add.h"
#include "metal_gemm.h"
#include "metal_hessian.h"
#include "metal_quantize_layer.h"

void test_quantize_layer_metal() {
    std::cout << "[ TEST ]: metal_quantize_layer" << std::endl;

    const int M = 3;

    // Original weights
    float W[M] = {-0.6f, 0.4f, 1.0f};

    // Simple diagonal inverse Hessian (identity approximation)
    float H_inv[M * M] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };

    // Output arrays
    float W_q[M] = {0.0f};           // quantized weights
    float error_out[M] = {0.0f};     // optional: stores the quantization error

    // Run Metal quantization kernel
    quantize_layer_metal(W, H_inv, W_q, error_out, M);

    // Print results
    std::cout << "input weights:\n";
    for (int i = 0; i < M; ++i) {
        printf("%f ", W[i]);
    }
    printf("\nquantized weights:\n");
    for (int i = 0; i < M; ++i) {
        printf("%f ", W_q[i]);
    }
    printf("\nerror output:\n");
    printf("%f ", error_out[0]);
    printf("\n");

    std::cout << "[ TEST ]: metal_quantize_layer done!" << std::endl;
}

void test_hessian_approximation() {
    std::cout << "[ TEST ]: metal_hessian_approximation" << std::endl;
    int M = 3, N = 2;

    float X[6] = {1.0f, 2.0f, 3.0f, 
                4.0f, 5.0f, 6.0f};
    
    float H[9] = {0.0f};

    // Initial inverse Hessian H_inv (Identity Matrix)
    float H_inv[9] = {1.0f, 0.0f, 0.0f, 
                      0.0f, 1.0f, 0.0f, 
                      0.0f, 0.0f, 1.0f};

    // float H[9] = {0.0f};
    float u[3] = {0.1f, 0.2f, 0.3f};

    hessian_approximation_metal(X, H, u, M, N);

    printf("result:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            printf("%f ", H[i * M + j]);
        }
        printf("\n");
    }

    std::cout << "[ TEST ]: metal_hessian_approximation done!" << std::endl;
}

void test_tiled_gemm() {
    std::cout << "[ TEST ]: metal_gemm" << std::endl;
    int M = 4, N = 4, K = 4;

    float A[16] = {1.0f, 2.0f, 3.0f, 4.0f, 
                   5.0f, 6.0f, 7.0f, 8.0f, 
                   9.0f, 10.0f, 11.0f, 12.0f, 
                   13.0f, 14.0f, 15.0f, 16.0f};

    float B[16] = {16.0f, 15.0f, 14.0f, 13.0f, 
                   12.0f, 11.0f, 10.0f, 9.0f, 
                   8.0f, 7.0f, 6.0f, 5.0f, 
                   4.0f, 3.0f, 2.0f, 1.0f};

    float C[16] = {0.0f};

    /*
    C:
    {
    80, 70, 60, 50
    240, 214, 188, 162,
    400, 358, 316, 274,
    570, 502, 444, 386
    }
    */

    gemm_metal(A, B, C, M, N, K);

    printf("result:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", C[i * N + j]);
        }
        printf("\n");
    }

    std::cout << "[ TEST ]: metal_gemm done!" << std::endl;
}

void test_gemm() {
    std::cout << "[ TEST ]: metal_gemm" << std::endl;
    int M = 2, N = 2, K = 2;
    float A[4] = {1, 2, 3, 4};
    float B[4] = {5, 6, 7, 8};
    float C[4] = {0, 0, 0, 0};

    gemm_metal(A, B, C, M, N, K);

    printf("result:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", C[i * N + j]);
        }
        printf("\n");
    }
    std::cout << "[ TEST ]: metal_gemm done!" << std::endl;
}

void test_add() {
    std::cout << "[ TEST ]: metal_add" << std::endl;
    const int size = 5;
    float inA[size] = {1, 2, 3, 4, 5};
    float inB[size] = {10, 20, 30, 40, 50};
    float out[size];
    
    add_arrays_metal(inA, inB, out, size);

    std::cout << "result:\n";
    for (int i = 0; i < size; i++) {
        std::cout << out[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "[ TEST ]: metal_add done!" << std::endl;
}

int main() {
    // test_add();
    // test_tiled_gemm();
    // test_hessian_approximation();
    test_quantize_layer_metal();
    return 0;
}
