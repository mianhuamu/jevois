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

    float divergence = cpxRx->data[0] / 100.0f;
    float obstacle = cpxRx->data[1];

    DEBUG_PRINT("Divergence: %.2f\n", divergence);
    DEBUG_PRINT("Obstacle parameter: %.2f\n", obstacle);

    if (obstacle == 1.0f) {
        DEBUG_PRINT("Drone is landing normally.\n");
    }
}
