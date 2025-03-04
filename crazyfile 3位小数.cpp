#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "app.h"
#include "cpx.h"
#include "FreeRTOS.h"
#include "task.h"
#include "debug.h"
#include "log.h"

static logVarId_t idX;
static float PositionX = 0.0f;

static logVarId_t idY;
static float PositionY = 0.0f;

static void cpxPacketCallback(const CPXPacket_t* cpxRx);

void appMain(void)
{
    DEBUG_PRINT("Hello! I am the stm_gap8_cpx app\n");

    idX = logGetVarId("stateEstimate", "x");
    idY = logGetVarId("stateEstimate", "y");

    cpxRegisterAppMessageHandler(cpxPacketCallback);

    while(1)
    {
        vTaskDelay(M2T(3000));
        DEBUG_PRINT("waiting for data \n");
    }
}

static void cpxPacketCallback(const CPXPacket_t* cpxRx)
{
    // Ensure the packet has at least 3 bytes: 2 for divergence and 1 for obstacle flag
    if (cpxRx->length < 3) {
        DEBUG_PRINT("Received packet with insufficient length: %d\n", cpxRx->length);
        return;
    }

    // Combine the first two bytes into a signed 16-bit integer (little endian)
    int16_t raw_x = (int16_t)(((uint16_t)cpxRx->data[0]) | ((uint16_t)cpxRx->data[1] << 8));
    // Scale to get the divergence with three decimal places
    float divergence = ((float)raw_x) / 1000.0f;

    // Read the third byte as the obstacle flag
    uint8_t raw_y = cpxRx->data[2];
    float obstacle = (float)raw_y;

    // Apply upper and lower limits to divergence
    if (divergence > 0.100f)
    {
        divergence = 0.100f;
        DEBUG_PRINT("Adjusted Divergence (upper limit): %.3f\n", (double)divergence);
    }
    else if (divergence < -0.200f)
    {
        divergence = -0.200f;
        DEBUG_PRINT("Adjusted Divergence (lower limit): %.3f\n", (double)divergence);
    }

    // Calculate the velocity parameter
    float k = 4.0f;
    float D_star = -0.05f;
    float v = k * (divergence - D_star);

    // Print the received and calculated values with three decimal places
    DEBUG_PRINT("Divergence: %.3f\n", (double)divergence);
    DEBUG_PRINT("Obstacle parameter: %.3f\n", (double)obstacle);
    DEBUG_PRINT("v: %.3f\n", (double)v);

    // If an obstacle is detected, perform additional actions
    if(obstacle == 1.0f)
    {
        DEBUG_PRINT("Drone is landing normally.\n");
        PositionX = logGetFloat(idX);
        DEBUG_PRINT("PositionX is now: %.3f deg\n", (double)PositionX);
        PositionY = logGetFloat(idY);
        DEBUG_PRINT("PositionY is now: %.3f deg\n", (double)PositionY);
    }
}
