#ifndef CMSIS_NN_CNN_ENGINE_H
#define CMSIS_NN_CNN_ENGINE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initializes CMSIS-NN ping-pong buffers & context
void cmsis_nn_cnn_init(void);

// Runs the 5-layer 2D CNN with shared 98.5 KB ping-pong buffers directly on Cortex-M33
// Input: [52, 313] INT8 Spectrogram (scale=0.093125, zp=20)
// Output: [13, 40, 32] INT8 Feature Map (scale=0.155061, zp=5)
int cmsis_nn_cnn_run(const int8_t *input_spec_int8, int8_t *ping_pong_A, int8_t *ping_pong_B, int8_t *out_features_int8);

// Output Feature Map Quantization Parameters
#define CMSIS_CNN_OUTPUT_SCALE      0.035800964f
#define CMSIS_CNN_OUTPUT_ZERO_POINT -17

#ifdef __cplusplus
}
#endif

#endif // CMSIS_NN_CNN_ENGINE_H
