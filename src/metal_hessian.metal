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

kernel void hessian_sherman_morrison(
    device const half* X,
    device half* H_inv,
    device const half* u,  // update vector (M x 1)
    device const uint* M_ptr,
    device const uint* N_ptr,
    uint2 id [[thread_position_in_grid]]
) {
    uint M = M_ptr[0];
    uint N = N_ptr[0];

    if (id.x < M && id.y < M) {
        half sum = 0.0h;
        for (uint i = 0; i < N; i++) {
            half value = X[id.x * N + i]; // Metal uses row-major order
            sum += value * value;
        }
        H_inv[id.x * M + id.x] = 1.0h / sum;
    }

    // wait for all threads to complete before updating inverse
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sherman-Morrison update
    /*
    H⁻¹_new = H⁻¹ - (H⁻¹ u u^T H⁻¹) / (1 + u^T H⁻¹ u)
    */
    if (id.x < M && id.y < M) {
        half dot_product = 0.0h;

        for (uint k = 0; k < M; k++) {
            dot_product += u[k] * H_inv[k * M + id.y];
        }

        half factor = dot_product / (1.0h + dot_product);

        H_inv[id.x * M + id.y] -= factor * u[id.x] * dot_product;
    }
}

/*
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
*/