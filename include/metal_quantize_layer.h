#ifndef METAL_QUANTIZE_LAYER_H
#define METAL_QUANTIZE_LAYER_H

void quantize_layer_metal(float* W, float* H_inv, float* W_q, float* error_out, int M);

#endif // METAL_QUANTIZE_LAYER_H