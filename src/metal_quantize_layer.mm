#import <Metal/Metal.h>
#include "metal_hessian.h"
#import "metal_utils.h"

void quantize_layer_metal(float* W, float* H_inv, float* W_q, float* error_out, int M) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    if (!device) {
        printf("Metal is not supported on this device.\n");
        return;
    }

    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    NSError *error = nil;
    NSString *shaderSource = [NSString stringWithContentsOfFile:@"src/metal_quantize_layer.metal" encoding:NSUTF8StringEncoding error:&error];

    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];

    if (!library) {
        printf("Failed to load metal library: %s\n", error.localizedDescription.UTF8String);
        return;
    }

    id<MTLFunction> function = [library newFunctionWithName:@"quantize_layer"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];

    __fp16* hW = new __fp16[M * 1];
    __fp16* hHinv = new __fp16[M * M];
    // __fp16* hWq = new __fp16[M * 1];
    // __fp16* hErrorOut = new __fp16[1];

    // convert to half
    convertToHalf(W, hW, M * 1);
    convertToHalf(H_inv, hHinv, M * M);
    // convertToHalf(W_q, hWq, M * 1);
    // convertToHalf(error_out, hErrorOut, 1);

    id<MTLBuffer> bufferW = [device newBufferWithBytes:hW length:M * sizeof(__fp16) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferHinv = [device newBufferWithBytes:hHinv length:M * M * sizeof(__fp16) options:MTLResourceStorageModeShared];
    // id<MTLBuffer> bufferWq = [device newBufferWithLength:W_q * sizeof(__fp16) options:MTLResourceStorageModeShared];
    // id<MTLBuffer> bufferErrorOut = [device newBufferWithLength:error_out * sizeof(__fp16) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferWq = [device newBufferWithLength:M * sizeof(__fp16) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferErrorOut = [device newBufferWithLength: sizeof(__fp16) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferM = [device newBufferWithBytes:&M length:sizeof(uint) options:MTLResourceStorageModeShared];

    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:bufferW offset:0 atIndex:0];
    [encoder setBuffer:bufferHinv offset:0 atIndex:1];
    [encoder setBuffer:bufferWq offset:0 atIndex:2];
    [encoder setBuffer:bufferErrorOut offset:0 atIndex:3];
    [encoder setBuffer:bufferM offset:0 atIndex:4];

    MTLSize gridSize = MTLSizeMake(M, M, 1);
    MTLSize threadGroupSize = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // convert back
    __fp16* hWq = (__fp16*)bufferWq.contents;
    __fp16* hErrorOut = (__fp16*)bufferErrorOut.contents;

    convertToFloat(hWq, W_q, M * 1);
    convertToFloat(hErrorOut, error_out, 1);

    delete[] hW;
    delete[] hHinv;
}
