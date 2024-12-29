#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"
#include "cpx.h"

#include "FreeRTOS.h"
#include "task.h"

#define DEBUG_MODULE "APP"
#include "debug.h"

typedef struct {
    float x;
    float y;
} FloatCoordinates;

// Callback that is called when a CPX packet arrives
static void cpxPacketCallback(const CPXPacket_t* cpxRx);

void appMain() {
  DEBUG_PRINT("Hello! I am the stm_gap8_cpx app\n");

  // Register a callback for CPX packets.
  // Packets sent to destination=CPX_T_STM32 and function=CPX_F_APP will arrive here
  cpxRegisterAppMessageHandler(cpxPacketCallback);

  while(1) {
    vTaskDelay(M2T(3000));
    DEBUG_PRINT("waiting for data \n");
  }
}

static void cpxPacketCallback(const CPXPacket_t* cpxRx) {
    // 将 data[0] 解释为有符号的 8 位整数
    int8_t raw_x = (int8_t)cpxRx->data[0];
    float divergence = ((float)raw_x) / 100.0f;

    // data[1] 仍然作为无符号字节处理
    uint8_t raw_y = cpxRx->data[1];
    float obstacle = (float)raw_y;

    DEBUG_PRINT("Divergence: %.2f\n", (double)divergence);
    DEBUG_PRINT("Obstacle parameter: %.2f\n", (double)obstacle);

    if (obstacle == 1.0f) {
        DEBUG_PRINT("Drone is landing normally.\n");
    }
}
