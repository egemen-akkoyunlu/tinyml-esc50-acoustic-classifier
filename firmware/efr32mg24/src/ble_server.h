#ifndef BLE_SERVER_H
#define BLE_SERVER_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void ble_server_init(void);
bool ble_is_connected_and_ready(void);
void ble_send_audio_data(int16_t *data, int total_samples);

#ifdef __cplusplus
}
#endif

#endif /* BLE_SERVER_H */