#include <metal_stdlib>
#include <metal_compute>
using namespace metal;

#define TILE_SIZE 32

kernel void gemm(
    device const float* A,
    device const float* B,
    device float* C,
    device const uint* K_ptr,
    device const uint* N_ptr,
    device const uint* M_ptr,
    uint2 threadPosition [[thread_position_in_grid]],
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 positionInThreadgroup [[thread_position_in_threadgroup]]
) {
    const uint K = K_ptr[0];
    const uint N = N_ptr[0];
    const uint M = M_ptr[0];
    
    // calculate the actual row and column this thread is responsible for
    const uint row = threadgroupPosition.x * TILE_SIZE + positionInThreadgroup.x;
    const uint col = threadgroupPosition.y * TILE_SIZE + positionInThreadgroup.y;
    
    // exit if outside matrix bounds
    if (row >= M || col >= N) {
        return;
    }
    
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    for (uint tileIdx = 0; tileIdx < (K + TILE_SIZE - 1) / TILE_SIZE; tileIdx++) {
        const uint aCol = tileIdx * TILE_SIZE + positionInThreadgroup.y;
        const uint bRow = tileIdx * TILE_SIZE + positionInThreadgroup.x;
        
        // load A tile with bounds checking
        if (row < M && aCol < K) {
            tileA[positionInThreadgroup.x][positionInThreadgroup.y] = A[row * K + aCol];
        } else {
            tileA[positionInThreadgroup.x][positionInThreadgroup.y] = 0.0f; // not sure if this is the desired behavior
        }
        
        // load B tile with bounds checking
        if (bRow < K && col < N) {
            tileB[positionInThreadgroup.x][positionInThreadgroup.y] = B[bRow * N + col];
        } else {
            tileB[positionInThreadgroup.x][positionInThreadgroup.y] = 0.0f; // not sure if this is the desired behavior
        }
        
        // synch all threads before computation
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint k = 0; k < TILE_SIZE; k++) {
            if (tileIdx * TILE_SIZE + k < K) {
                sum += tileA[positionInThreadgroup.x][k] * tileB[k][positionInThreadgroup.y];
            }
        }
        
        // synch again before loading the next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    C[row * N + col] = sum;
}