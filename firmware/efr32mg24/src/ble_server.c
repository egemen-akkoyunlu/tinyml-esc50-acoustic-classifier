#include "ble_server.h"
#include <zephyr/kernel.h>
#include <stdio.h>

#if defined(CONFIG_BT) && (CONFIG_BT == 1)
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/bluetooth/gatt.h>

#define CUSTOM_SERVICE_UUID \
    BT_UUID_DECLARE_128(BT_UUID_128_ENCODE(0x12345678, 0x1234, 0x5678, 0x1234, 0x56789abcdef0))

#define CUSTOM_CHAR_UUID \
    BT_UUID_DECLARE_128(BT_UUID_128_ENCODE(0x12345678, 0x1234, 0x5678, 0x1234, 0x56789abcdef1))

static bool is_connected = false;
static bool notify_enabled = false;
static struct bt_conn *active_conn = NULL;

static void ccc_cfg_changed(const struct bt_gatt_attr *attr, uint16_t value) {
    notify_enabled = (value == BT_GATT_CCC_NOTIFY);
    printf("[BLE] Karsi cihazdan Bildirim (Notify) %s!\n", notify_enabled ? "ACILDI" : "KAPATILDI");
}

BT_GATT_SERVICE_DEFINE(my_custom_svc,
    BT_GATT_PRIMARY_SERVICE(CUSTOM_SERVICE_UUID),
    BT_GATT_CHARACTERISTIC(CUSTOM_CHAR_UUID, BT_GATT_CHRC_NOTIFY, BT_GATT_PERM_NONE, NULL, NULL, NULL),
    BT_GATT_CCC(ccc_cfg_changed, BT_GATT_PERM_READ | BT_GATT_PERM_WRITE)
);

static const struct bt_data ad[] = {
    BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
};

static const struct bt_data sd[] = {
    BT_DATA(BT_DATA_NAME_COMPLETE, CONFIG_BT_DEVICE_NAME, sizeof(CONFIG_BT_DEVICE_NAME) - 1),
};

static void connected(struct bt_conn *conn, uint8_t err) {
    if (!err) {
        is_connected = true;
        active_conn = bt_conn_ref(conn);
        printf("\n[BLE] >>> ALICI (CENTRAL) CIHAZ BAGLANDI! <<<\n");
    }
}

static void disconnected(struct bt_conn *conn, uint8_t reason) {
    is_connected = false;
    notify_enabled = false;
    if (active_conn) {
        bt_conn_unref(active_conn);
        active_conn = NULL;
    }
    printf("\n[BLE] --- BAGLANTI KOPTU (Sebep: %u) ---\n", reason);
}

BT_CONN_CB_DEFINE(conn_callbacks) = {
    .connected = connected,
    .disconnected = disconnected,
};

#define MY_ADV_PARAM BT_LE_ADV_PARAM(BT_LE_ADV_OPT_CONN, \
                                     BT_GAP_ADV_FAST_INT_MIN_2, \
                                     BT_GAP_ADV_FAST_INT_MAX_2, \
                                     NULL)

void ble_server_init(void) {
    int err = bt_enable(NULL);
    if (err) {
        printf("[BLE HATA] Baslatilamadi (Kod: %d)\n", err);
        return;
    }
    bt_le_adv_start(MY_ADV_PARAM, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
    printf("[BLE] Yayin basladi. Alici cihaz bekleniyor...\n");
}

bool ble_is_connected_and_ready(void) {
    return (is_connected && notify_enabled);
}

void ble_send_audio_data(int16_t *data, int total_samples) {
    int samples_sent = 0;
    int enomem_count = 0;

    /* MTU'ya gore dinamik paket boyutu hesapla */
    /* ATT Notify header = 3 byte, geri kalani PCM verisi */
    uint16_t mtu = 23; /* varsayilan */
    if (active_conn) {
        mtu = bt_gatt_get_mtu(active_conn);
    }
    int max_payload = mtu - 3;  /* ATT Notify header cikar */
    int chunk_size = max_payload / (int)sizeof(int16_t);
    if (chunk_size < 1) chunk_size = 1;

    printf("\n[BLE] MTU=%u -> Paket basi %d ornek (%d byte)\n", mtu, chunk_size, chunk_size * 2);
    printf("[BLE] %d adet PCM ornegi havadan firlatiliyor...\n", total_samples);

    while (samples_sent < total_samples && ble_is_connected_and_ready()) {
        int to_send = total_samples - samples_sent;
        if (to_send > chunk_size) {
            to_send = chunk_size;
        }

        const struct bt_gatt_attr *attr = bt_gatt_attr_next(&my_custom_svc.attrs[1]);

        int ret = bt_gatt_notify(active_conn, attr, &data[samples_sent], to_send * sizeof(int16_t));

        if (ret == 0) {
            samples_sent += to_send;
            enomem_count = 0;
            /* Buyuk MTU ile daha az paket -> daha kisa uyku yeterli */
            k_sleep(K_MSEC(2));
            
        } else if (ret == -ENOMEM || ret == -ENOBUFS || ret == -EAGAIN) {
            enomem_count++;
            if (enomem_count % 50 == 0) {
                printf("[UYARI] Hafiza sisti! %d. ornekte radyonun bosalmasi bekleniyor...\n", samples_sent);
            }
            k_sleep(K_MSEC(10));
            
        } else {
            printf("[BLE KRIZ] Gonderim REDDEDILDI! (Kod: %d)\n", ret);
            k_sleep(K_MSEC(50));
        }
    printf("\n[BLE] ISLEM TAMAMLANDI! %d ornek gonderildi.\n", samples_sent);
}
#else
void ble_server_init(void) {}
bool ble_is_connected_and_ready(void) { return true; }
void ble_send_audio_data(int16_t *data, int total_samples) {
    (void)data;
    (void)total_samples;
}
#endif