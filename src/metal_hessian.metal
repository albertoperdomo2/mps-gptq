#include <metal_stdlib>
#include <metal_compute>
using namespace metal;

/*
GPTQ approximates the importance of each weight using the second-order Taylor expansion of the model's loss function. 
The Hessian H (the matrix of second derivatives of the loss with respect to the weights) would ideally be used to estimate 
the impact of quantization. However, computing the full Hessian is infeasible for large models.

Therefore, the Hessian is approximated as H = diag(X @ X^T)

But computing this matrix per each layer update is computationally expensive. It uses the Gaussian elimination to efficiently 
update the inverse Hessian.
*/

kernel void hessian_approximation(
    device const half* X,
    device half* H,
    device const uint* M_ptr,
    device const uint* N_ptr,
    uint2 id [[thread_position_in_grid]]
) {
    uint M = M_ptr[0];
    uint N = N_ptr[0];

    if (id.x < M) {
        half sum = 0.0h;
        for (uint i = 0; i < N; i++) {
            half value = X[id.x * N + i]; // Metal uses row-major order
            sum += value * value;
        }
        H[id.x * M + id.x] = sum;
    }
}