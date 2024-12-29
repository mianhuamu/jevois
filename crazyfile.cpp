#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"
#include "cpx.h"

#include "FreeRTOS.h"
#include "task.h"

#define DEBUG_MODULE "APP"
#include "debug.h"

// Define a structure to hold the parsed coordinates
typedef struct {
    float x;
    float y;
} FloatCoordinates;

// Callback function that is called when a CPX packet arrives
static void cpxPacketCallback(const CPXPacket_t* cpxRx);

void appMain() {
    DEBUG_PRINT("Hello! I am the stm_gap8_cpx app\n");

    // Register the callback for CPX packets.
    // Packets sent to destination=CPX_T_STM32 and function=CPX_F_APP will be handled here
    cpxRegisterAppMessageHandler(cpxPacketCallback);

    while(1) {
        vTaskDelay(M2T(3000)); // Delay for 3000 milliseconds
        DEBUG_PRINT("waiting for data \n");
    }
}

static void cpxPacketCallback(const CPXPacket_t* cpxRx) {
    // Print the raw received packet data for debugging
    DEBUG_PRINT("Received packet: ");
    for(int i = 0; i < cpxRx->length; i++) {
        DEBUG_PRINT("%02X ", cpxRx->data[i]);
    }
    DEBUG_PRINT("\n");

    // Validate the packet length (minimum 5 bytes: 1 start + 1 length + 3 data)
    if (cpxRx->length < 5) {
        DEBUG_PRINT("Received packet too short: length=%d\n", cpxRx->length);
        return;
    }

    // Validate the start byte (should be 0xFF)
    if (cpxRx->data[0] != 0xFF) {
        DEBUG_PRINT("Invalid start byte: 0x%02X\n", cpxRx->data[0]);
        return;
    }

    // Parse the length byte
    uint8_t data_length = cpxRx->data[1];
    if (data_length != 3) {
        DEBUG_PRINT("Unexpected data length: %d\n", data_length);
        return;
    }

    // Verify that the actual packet length matches the expected length
    if (cpxRx->length != (2 + data_length + 1)) { // Start byte + length byte + data + checksum
        DEBUG_PRINT("Data length mismatch: expected %d, got %d\n", 2 + data_length + 1, cpxRx->length);
        return;
    }

    // Calculate the checksum by XOR-ing all bytes except the last one (checksum byte)
    uint8_t calculated_checksum = 0;
    for(int i = 0; i < (cpxRx->length - 1); i++) { // Exclude the last byte
        calculated_checksum ^= cpxRx->data[i];
    }

    // Retrieve the received checksum
    uint8_t received_checksum = cpxRx->data[cpxRx->length - 1];
    if (calculated_checksum != received_checksum) {
        DEBUG_PRINT("Checksum mismatch: calculated=0x%02X, received=0x%02X\n", calculated_checksum, received_checksum);
        return;
    }

    // Parse 'x' as a signed short (int16_t) in big-endian format
    int16_t x = (cpxRx->data[2] << 8) | cpxRx->data[3];
    float divergence = ((float)x) / 100.0f;

    // Parse 'y' as a signed byte (int8_t)
    int8_t y_byte = (int8_t)cpxRx->data[4];
    float obstacle = (float)y_byte;

    DEBUG_PRINT("Divergence: %.2f\n", (double)divergence);
    DEBUG_PRINT("Obstacle parameter: %.2f\n", (double)obstacle);

    if (obstacle == 1.0f) {
        DEBUG_PRINT("Drone is landing normally.\n");
    } else {
        DEBUG_PRINT("No obstacle detected.\n");
    }
}
