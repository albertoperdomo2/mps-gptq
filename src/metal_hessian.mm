#import <Metal/Metal.h>
#include "metal_hessian.h"
#import "metal_utils.h"

void hessian_approximation_metal(float* X, float* H_inv, float* u, int M, int N) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    if (!device) {
        printf("Metal is not supported on this device.\n");
        return;
    }

    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    NSError *error = nil;
    NSString *shaderSource = [NSString stringWithContentsOfFile:@"src/metal_hessian.metal" encoding:NSUTF8StringEncoding error:&error];

    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];

    if (!library) {
        printf("Failed to load metal library: %s\n", error.localizedDescription.UTF8String);
        return;
    }

    id<MTLFunction> function = [library newFunctionWithName:@"hessian_sherman_morrison"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];

    __fp16* hX = new __fp16[M * N];
    __fp16* hU = new __fp16[M * 1];

    // convert to half
    convertToHalf(X, hX, M * N);
    convertToHalf(u, hU, M * 1);

    id<MTLBuffer> bufferX = [device newBufferWithBytes:hX length:M * N * sizeof(__fp16) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferH = [device newBufferWithLength:M * M * sizeof(__fp16) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferU = [device newBufferWithBytes:hU length:M * sizeof(__fp16) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferN = [device newBufferWithBytes:&N length:sizeof(uint) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferM = [device newBufferWithBytes:&M length:sizeof(uint) options:MTLResourceStorageModeShared];

    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:bufferX offset:0 atIndex:0];
    [encoder setBuffer:bufferH offset:0 atIndex:1];
    [encoder setBuffer:bufferU offset:0 atIndex:2];
    [encoder setBuffer:bufferM offset:0 atIndex:3];
    [encoder setBuffer:bufferN offset:0 atIndex:4];

    MTLSize gridSize = MTLSizeMake(M, M, 1);
    MTLSize threadGroupSize = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // convert back
    __fp16* hH_inv = (__fp16*)bufferH.contents;
    convertToFloat(hH_inv, H_inv, M * M);

    delete[] hX;
    delete[] hU;

    // memcpy(H, bufferH.contents, M * M * sizeof(float));
}
