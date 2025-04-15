#include <metal_stdlib>
#include <metal_compute>
using namespace metal;

/*
The main goal of the GPTQ algorithm is quantize the weights of a column of the weights
matrix W (which are the weights of a layer), producing an approximated version q, while
minimizing the quantization error under the inverse Hessian.

Here, things can get pretty complex but for now, I'm going to just quantize the vector
using symmetric quantization.
*/

kernel void quantize_layer(
    device const half* W,
    device half* H_inv,
    device half* W_q,
    device atomic_int* error_out,
    device const uint* M_ptr,
    uint2 id [[thread_position_in_grid]]
) {
    uint M = M_ptr[0];

    // only thread 0 computes global values like scale and error
    threadgroup half shared_scale;
    threadgroup half shared_error;
    half local_error = 0.0h;

    if (id.x == 0) {
        // compute max(abs(W)) to get the scale factor for the symmetric quantization
        half max_val = 0.0h; // TODO: make this the smallest possible half value
        for (uint i = 0; i < M; i++) {
            half val = fabs(W[i]);
            if (val > max_val) max_val = val;
        }

        // 4-bit quantization: symmetric range [-7, 7] (or [-8, 7] depending on implementation)
        half scale = max_val / 7.0h;

        shared_scale = scale;
        shared_error = 0.0h;
    }

    // synch all threads before computation
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // per-element quantization (parallel across M threads)
    // if (id.x < M && id.y < N) {
    if (id.x < M) {
        half scale = shared_scale;
        half val = W[id.x];
        short q = short(round(val / scale)); // quantize to int
        q = clamp(q, short(-7), short(7)); // clamp to 4-bit range
        half q_val = half(q) * scale; // dequantized value
        W_q[id.x] = q_val;

        // compute contribution to quantization error
        half diff_i = val - q_val;
        // half local_error = 0.0h;

        for (uint j = 0; j < M; j++) {
            half diff_j = W[j] - W_q[j];
            local_error += diff_i * H_inv[id.x * M + j] * diff_j;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (id.x == 0) {
        // atomic_fetch_add_explicit((device atomic_int*)&shared_error, int(local_error * 10000.0h), memory_order_relaxed);
        atomic_fetch_add_explicit(error_out, int(local_error * 10000.0h), memory_order_relaxed);
    }

    // synch all threads before computation
    // threadgroup_barrier(mem_flags::mem_threadgroup);

    // store total error (only once)
    // if (id == 0 && error_out != nullptr) {
        // error_out[0] = half(shared_error) / 10000.0h;
    // }
}