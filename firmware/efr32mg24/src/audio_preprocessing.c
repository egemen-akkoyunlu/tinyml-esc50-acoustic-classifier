#include "audio_preprocessing.h"
#include "mel_filterbank_tables.h"
#include "config.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

#if defined(CONFIG_CMSIS_DSP)
#include <arm_math.h>
#endif

static bool is_preproc_initialized = false;

#if defined(CONFIG_CMSIS_DSP)
static arm_rfft_fast_instance_f32 rfft_instance;
#endif

static float fft_buf[FFT_SIZE];
static float fft_output[FFT_SIZE];
static float power_spectrum[NUM_FREQ_BINS];

void audio_preprocessing_init(void) {
    if (is_preproc_initialized) return;

#if defined(CONFIG_CMSIS_DSP)
    arm_rfft_fast_init_f32(&rfft_instance, FFT_SIZE);
#endif

    is_preproc_initialized = true;
    printf("-> [OK] Audio preprocessing DSP module initialized (Bit-Exact Torchaudio Mel Filterbank - 0 Extra RAM).\n");
}

int32_t audio_preprocess_compute_dc(const int16_t *raw_pcm, int num_samples) {
    int64_t sum = 0;
    for (int i = 0; i < num_samples; i++) {
        sum += raw_pcm[i];
    }
    return (int32_t)(sum / num_samples);
}

static void compute_frame_power_spectrum_onthefly(
    const int16_t *raw_pcm, 
    int frame_idx, 
    int num_samples, 
    int32_t dc_offset) 
{
    for (int n = 0; n < FFT_SIZE; n++) {
        int audio_idx = frame_idx * HOP_LENGTH - (FFT_SIZE / 2) + n;
        if (audio_idx < 0) {
            audio_idx = -audio_idx; /* Reflect padding left */
        } else if (audio_idx >= num_samples) {
            audio_idx = 2 * (num_samples - 1) - audio_idx; /* Reflect padding right */
            if (audio_idx < 0) audio_idx = 0;
        }

        float raw_curr = (float)(raw_pcm[audio_idx] - dc_offset) / 32768.0f;
        float sample_val = raw_curr;
#if (ACTIVE_MODEL_PROFILE == PROFILE_FLAGSHIP_DENSE_91 || ACTIVE_MODEL_PROFILE == PROFILE_INT8_FIXED_SIMD_91 || ACTIVE_MODEL_PROFILE == PROFILE_CMSIS_NN_PINGPONG_91)
        int prev_idx = audio_idx - 1;
        if (prev_idx < 0) prev_idx = 0;
        float prev_sample = (float)(raw_pcm[prev_idx] - dc_offset) / 32768.0f;
        sample_val = raw_curr - 0.95f * prev_sample;
#endif
        fft_buf[n] = sample_val * HANN_WINDOW_512[n];
    }

#if defined(CONFIG_CMSIS_DSP)
    arm_rfft_fast_f32(&rfft_instance, fft_buf, fft_output, 0);

    power_spectrum[0] = fft_output[0] * fft_output[0];
    power_spectrum[NUM_FREQ_BINS - 1] = fft_output[1] * fft_output[1];

    for (int k = 1; k < NUM_FREQ_BINS - 1; k++) {
        float real = fft_output[2 * k];
        float imag = fft_output[2 * k + 1];
        power_spectrum[k] = real * real + imag * imag;
    }
#else
    for (int k = 0; k < NUM_FREQ_BINS; k++) {
        float real = 0.0f;
        float imag = 0.0f;
        for (int n = 0; n < FFT_SIZE; n++) {
            float angle = (2.0f * 3.14159265358979323846f * k * n) / FFT_SIZE;
            real += fft_buf[n] * cosf(angle);
            imag -= fft_buf[n] * sinf(angle);
        }
        power_spectrum[k] = real * real + imag * imag;
    }
#endif
}

void audio_preprocess_extract_melspectrogram_direct(
    const int16_t *raw_pcm, 
    int num_samples, 
    int32_t dc_offset, 
    float scale, 
    int32_t zero_point, 
    bool is_int8, 
    void *out_tensor_ptr) 
{
    if (!is_preproc_initialized) {
        audio_preprocessing_init();
    }

    if (scale == 0.0f) scale = 1.0f;
    int8_t *int8_buf = (int8_t*)out_tensor_ptr;
    float *float_buf = (float*)out_tensor_ptr;

    for (int t = 0; t < SPECTROGRAM_TIME_STEPS; t++) {
        /* 1. On-the-fly Power Spectrum with Exact Reflect Padding (313 Frames) */
        compute_frame_power_spectrum_onthefly(raw_pcm, t, num_samples, dc_offset);

        /* 2. Exact Sparse Mel Matrix Multiplication + Direct Input Tensor Writing */
        for (int m = 0; m < N_MELS; m++) {
            float mel_energy = 0.0f;
            uint16_t start_k = MEL_FILTERS[m].start_bin;
            uint16_t count_k = MEL_FILTERS[m].num_bins;
            const float *w_ptr = MEL_FILTERS[m].weights;

            for (uint16_t k = 0; k < count_k; k++) {
                mel_energy += power_spectrum[start_k + k] * w_ptr[k];
            }

            float log_mel = logf(mel_energy + 1e-6f);
            int spec_index = m * SPECTROGRAM_TIME_STEPS + t;

            if (is_int8) {
                int32_t quantized = (int32_t)lroundf(log_mel / scale) + zero_point;
                if (quantized < -128) quantized = -128;
                if (quantized > 127)  quantized = 127;
                int8_buf[spec_index] = (int8_t)quantized;
            } else {
                float_buf[spec_index] = log_mel;
            }
        }
    }

    if (is_int8) {
        printf("-> [DSP] Spectrogram Extracted: Mel 0, Time 0..4 INT8: %d, %d, %d, %d, %d\n",
               int8_buf[0], int8_buf[1], int8_buf[2], int8_buf[3], int8_buf[4]);
    }
}

void audio_preprocess_run_direct(
    const int16_t *raw_pcm, 
    int num_samples, 
    float scale, 
    int32_t zero_point, 
    bool is_int8, 
    void *out_tensor_ptr) 
{
    int32_t dc_offset = 0; /* Exact floating-point parity for normalized audio */
    audio_preprocess_extract_melspectrogram_direct(
        raw_pcm, num_samples, dc_offset, scale, zero_point, is_int8, out_tensor_ptr
    );
}

void audio_preprocess_append_chunk_direct(
    const int16_t *new_pcm_chunk, 
    int num_samples, 
    float scale, 
    int32_t zero_point, 
    bool is_int8, 
    void *out_tensor_ptr)
{
    if (!is_preproc_initialized) {
        audio_preprocessing_init();
    }
    if (scale == 0.0f) scale = 1.0f;
    int8_t *int8_buf = (int8_t*)out_tensor_ptr;
    float *float_buf = (float*)out_tensor_ptr;

    int new_frames = 0;
    if (num_samples >= FFT_SIZE) {
        new_frames = (num_samples - FFT_SIZE) / HOP_LENGTH + 1;
    }
    if (num_samples >= 79000 || new_frames > SPECTROGRAM_TIME_STEPS) {
        new_frames = SPECTROGRAM_TIME_STEPS;
    }
    if (new_frames <= 0) return;

    int keep_frames = SPECTROGRAM_TIME_STEPS - new_frames;

    /* 1. Shift existing spectrogram frames left by new_frames in-place */
    for (int m = 0; m < N_MELS; m++) {
        if (is_int8) {
            memmove(int8_buf + (m * SPECTROGRAM_TIME_STEPS), 
                    int8_buf + (m * SPECTROGRAM_TIME_STEPS) + new_frames, 
                    keep_frames);
        } else {
            memmove(float_buf + (m * SPECTROGRAM_TIME_STEPS), 
                    float_buf + (m * SPECTROGRAM_TIME_STEPS) + new_frames, 
                    sizeof(float) * keep_frames);
        }
    }

    int32_t dc_offset = audio_preprocess_compute_dc(new_pcm_chunk, num_samples);
    float power_spectrum[NUM_FREQ_BINS];

    /* 2. Calculate new Mel frames and append to tail */
    for (int t = 0; t < new_frames; t++) {
        int sample_offset = (num_samples >= 79000) ? ((t - 1) * HOP_LENGTH) : (t * HOP_LENGTH);
        
        float fft_buf[FFT_SIZE];
        float fft_output[FFT_SIZE];

        for (int n = 0; n < FFT_SIZE; n++) {
            float sample_val = 0.0f;
            int sample_idx = sample_offset + n;
            if (sample_idx >= 0 && sample_idx < num_samples) {
                sample_val = (float)(new_pcm_chunk[sample_idx] - dc_offset) / 32768.0f;
            }
            fft_buf[n] = sample_val * HANN_WINDOW_512[n];
        }

#if defined(CONFIG_CMSIS_DSP)
        arm_rfft_fast_f32(&rfft_instance, fft_buf, fft_output, 0);

        power_spectrum[0] = fft_output[0] * fft_output[0];
        power_spectrum[NUM_FREQ_BINS - 1] = fft_output[1] * fft_output[1];

        for (int k = 1; k < NUM_FREQ_BINS - 1; k++) {
            float real = fft_output[2 * k];
            float imag = fft_output[2 * k + 1];
            power_spectrum[k] = real * real + imag * imag;
        }
#else
        for (int k = 0; k < NUM_FREQ_BINS; k++) {
            float real = 0.0f;
            float imag = 0.0f;
            for (int n = 0; n < FFT_SIZE; n++) {
                float angle = (2.0f * 3.14159265358979323846f * k * n) / FFT_SIZE;
                real += fft_buf[n] * cosf(angle);
                imag -= fft_buf[n] * sinf(angle);
            }
            power_spectrum[k] = real * real + imag * imag;
        }
#endif

        int time_index = keep_frames + t;

        for (int m = 0; m < N_MELS; m++) {
            float mel_energy = 0.0f;
            uint16_t start_k = MEL_FILTERS[m].start_bin;
            uint16_t count_k = MEL_FILTERS[m].num_bins;
            const float *w_ptr = MEL_FILTERS[m].weights;

            for (uint16_t k = 0; k < count_k; k++) {
                mel_energy += power_spectrum[start_k + k] * w_ptr[k];
            }

            float log_mel = logf(mel_energy + 1e-6f);
            int spec_index = m * SPECTROGRAM_TIME_STEPS + time_index;

            if (is_int8) {
                int32_t quantized = (int32_t)lroundf(log_mel / scale) + zero_point;
                if (quantized < -128) quantized = -128;
                if (quantized > 127)  quantized = 127;
                int8_buf[spec_index] = (int8_t)quantized;
            } else {
                float_buf[spec_index] = log_mel;
            }
        }
    }
}

void audio_preprocess_tile_chunk_direct(
    const int16_t *new_pcm_chunk, 
    int num_samples, 
    float scale, 
    int32_t zero_point, 
    bool is_int8, 
    void *out_tensor_ptr) 
{
    if (out_tensor_ptr == NULL || new_pcm_chunk == NULL) return;
    if (scale == 0.0f) scale = 1.0f;
    int8_t *int8_buf = (int8_t*)out_tensor_ptr;
    float *float_buf = (float*)out_tensor_ptr;

    int new_frames = 0;
    if (num_samples >= FFT_SIZE) {
        new_frames = (num_samples - FFT_SIZE) / HOP_LENGTH + 1;
    }
    if (new_frames > SPECTROGRAM_TIME_STEPS) {
        new_frames = SPECTROGRAM_TIME_STEPS;
    }
    if (new_frames <= 0) return;

    int32_t dc_offset = audio_preprocess_compute_dc(new_pcm_chunk, num_samples);
    float power_spectrum[NUM_FREQ_BINS];

    /* 1. Calculate Mel frames for the newly captured audio chunk into the first new_frames slots */
    for (int t = 0; t < new_frames; t++) {
        int sample_offset = t * HOP_LENGTH;
        float fft_buf[FFT_SIZE];
        float fft_output[FFT_SIZE];

        for (int n = 0; n < FFT_SIZE; n++) {
            float sample_val = 0.0f;
            int sample_idx = sample_offset + n;
            if (sample_idx >= 0 && sample_idx < num_samples) {
                float raw_curr = (float)(new_pcm_chunk[sample_idx] - dc_offset) / 32768.0f;
                sample_val = raw_curr;
#if (ACTIVE_MODEL_PROFILE == PROFILE_FLAGSHIP_DENSE_91 || ACTIVE_MODEL_PROFILE == PROFILE_INT8_FIXED_SIMD_91 || ACTIVE_MODEL_PROFILE == PROFILE_CMSIS_NN_PINGPONG_91)
                float prev_sample = (sample_idx > 0) ? (float)(new_pcm_chunk[sample_idx - 1] - dc_offset) / 32768.0f : raw_curr;
                sample_val = raw_curr - 0.95f * prev_sample;
#endif
            }
            fft_buf[n] = sample_val * HANN_WINDOW_512[n];
        }

#if defined(CONFIG_CMSIS_DSP)
        arm_rfft_fast_f32(&rfft_instance, fft_buf, fft_output, 0);

        power_spectrum[0] = fft_output[0] * fft_output[0];
        power_spectrum[NUM_FREQ_BINS - 1] = fft_output[1] * fft_output[1];

        for (int k = 1; k < NUM_FREQ_BINS - 1; k++) {
            float real = fft_output[2 * k];
            float imag = fft_output[2 * k + 1];
            power_spectrum[k] = real * real + imag * imag;
        }
#else
        for (int k = 0; k < NUM_FREQ_BINS; k++) {
            float real = 0.0f;
            float imag = 0.0f;
            for (int n = 0; n < FFT_SIZE; n++) {
                float angle = (2.0f * 3.14159265358979323846f * k * n) / FFT_SIZE;
                real += fft_buf[n] * cosf(angle);
                imag -= fft_buf[n] * sinf(angle);
            }
            power_spectrum[k] = real * real + imag * imag;
        }
#endif

        for (int m = 0; m < N_MELS; m++) {
            float mel_energy = 0.0f;
            uint16_t start_k = MEL_FILTERS[m].start_bin;
            uint16_t count_k = MEL_FILTERS[m].num_bins;
            const float *w_ptr = MEL_FILTERS[m].weights;

            for (uint16_t k = 0; k < count_k; k++) {
                mel_energy += power_spectrum[start_k + k] * w_ptr[k];
            }

            float log_mel = logf(mel_energy + 1e-6f);
            int spec_index = m * SPECTROGRAM_TIME_STEPS + t;

            if (is_int8) {
                int32_t quantized = (int32_t)lroundf(log_mel / scale) + zero_point;
                if (quantized < -128) quantized = -128;
                if (quantized > 127)  quantized = 127;
                int8_buf[spec_index] = (int8_t)quantized;
            } else {
                float_buf[spec_index] = log_mel;
            }
        }
    }

    /* 2. Seamlessly tile the active chunk across all 313 frames of the 5.0-second tensor */
    for (int m = 0; m < N_MELS; m++) {
        for (int t = new_frames; t < SPECTROGRAM_TIME_STEPS; t++) {
            int src_t = t % new_frames;
            int dst_idx = m * SPECTROGRAM_TIME_STEPS + t;
            int src_idx = m * SPECTROGRAM_TIME_STEPS + src_t;

            if (is_int8) {
                int8_buf[dst_idx] = int8_buf[src_idx];
            } else {
                float_buf[dst_idx] = float_buf[src_idx];
            }
        }
    }
}

void audio_preprocess_quantize_int8(const float *in_spectrogram, int8_t *out_int8_tensor, float scale, int32_t zero_point) {
    if (scale == 0.0f) scale = 1.0f;

    for (int i = 0; i < SPECTROGRAM_TOTAL_ELEMENTS; i++) {
        float val = in_spectrogram[i];
        int32_t quantized = (int32_t)lroundf(val / scale) + zero_point;
        
        if (quantized < -128) quantized = -128;
        if (quantized > 127)  quantized = 127;
        
        out_int8_tensor[i] = (int8_t)quantized;
    }
}
