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
    device const half* H_inv,
    device half* W_q,
    device atomic_int* error_out,
    device const uint* M_ptr,
    uint id [[thread_position_in_grid]]
) {
    uint M = M_ptr[0];

    threadgroup half shared_scale;
    half local_error = 0.0h;

    if (id == 0) {
        // compute max(abs(W)) to get scale
        half max_val = 0.0h;
        for (uint i = 0; i < M; i++) {
            half val = fabs(W[i]);
            if (val > max_val) max_val = val;
        }
        shared_scale = max_val / 7.0h;
        atomic_store_explicit(error_out, 0, memory_order_relaxed);  // explicitly zero the error buffer
    }

    // synch all threads before computation
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (id < M) {
        half scale = shared_scale;
        half val = W[id];
        short q = short(round(val / scale));
        q = clamp(q, short(-7), short(7));
        half q_val = half(q) * scale;
        W_q[id] = q_val;

        half diff_i = val - q_val;
        for (uint j = 0; j < M; j++) {
            half diff_j = W[j] - W_q[j];
            local_error += diff_i * H_inv[id * M + j] * diff_j;
        }
        atomic_fetch_add_explicit(error_out, int(local_error * 10000.0h), memory_order_relaxed);
    }
}