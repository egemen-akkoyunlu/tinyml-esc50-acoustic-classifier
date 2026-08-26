/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 *
 * Zephyr compatibility stub for <psa/crypto.h>.
 * Since models loaded in Zephyr (rodata flash) are unencrypted,
 * PSA crypto calls are cleanly stubbed out.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { int dummy; } psa_key_attributes_t;
typedef uint32_t psa_key_id_t;
typedef struct { int dummy; } psa_cipher_operation_t;
typedef int psa_status_t;

#define PSA_KEY_ATTRIBUTES_INIT {0}
#define PSA_KEY_ID_NULL 0
#define PSA_CIPHER_OPERATION_INIT {0}
#define PSA_KEY_USAGE_ENCRYPT 1
#define PSA_ALG_CTR 1
#define PSA_KEY_TYPE_AES 1

static inline void psa_set_key_usage_flags(psa_key_attributes_t *attr, uint32_t flags) { (void)attr; (void)flags; }
static inline void psa_set_key_algorithm(psa_key_attributes_t *attr, uint32_t alg) { (void)attr; (void)alg; }
static inline void psa_set_key_type(psa_key_attributes_t *attr, uint32_t type) { (void)attr; (void)type; }
static inline void psa_set_key_bits(psa_key_attributes_t *attr, size_t bits) { (void)attr; (void)bits; }
static inline psa_status_t psa_import_key(const psa_key_attributes_t *attr, const uint8_t *data, size_t data_length, psa_key_id_t *key_id) { (void)attr; (void)data; (void)data_length; *key_id = 1; return 0; }
static inline void psa_reset_key_attributes(psa_key_attributes_t *attr) { (void)attr; }
static inline psa_status_t psa_cipher_encrypt_setup(psa_cipher_operation_t *operation, psa_key_id_t key, uint32_t alg) { (void)operation; (void)key; (void)alg; return 0; }
static inline psa_status_t psa_cipher_set_iv(psa_cipher_operation_t *operation, const uint8_t *iv, size_t iv_length) { (void)operation; (void)iv; (void)iv_length; return 0; }
static inline psa_status_t psa_cipher_update(psa_cipher_operation_t *operation, const uint8_t *input, size_t input_length, uint8_t *output, size_t output_size, size_t *output_length) {
    (void)operation; (void)output_size;
    if (input && output && input != output) {
        memcpy(output, input, input_length);
    }
    if (output_length) {
        *output_length = input_length;
    }
    return 0;
}
static inline psa_status_t psa_cipher_finish(psa_cipher_operation_t *operation, uint8_t *output, size_t output_size, size_t *output_length) { (void)operation; (void)output; (void)output_size; if (output_length) *output_length = 0; return 0; }
static inline psa_status_t psa_destroy_key(psa_key_id_t key) { (void)key; return 0; }

#ifdef __cplusplus
}
#endif
