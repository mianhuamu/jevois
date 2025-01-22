/**
 * ,---------,       ____  _ __
 * |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 * | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * | / ,--´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie control firmware
 *
 * Copyright (C) 2019 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * hello_world.c - App layer application of a simple hello world debug print every
 *   2 seconds.
 */
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
    int8_t  raw_x  = (int8_t)cpxRx->data[0];
    float   divergence = ((float)raw_x) / 100.0f;

    uint8_t raw_y  = cpxRx->data[1];
    float   obstacle = (float)raw_y;

    if (divergence > 0.2f)
    {
        divergence = 0.2f;
        DEBUG_PRINT("Adjusted Divergence (upper limit): %.2f\n", (double)divergence);
    }
    else if (divergence < -0.3f)
    {
        divergence = -0.3f;
        DEBUG_PRINT("Adjusted Divergence (lower limit): %.2f\n", (double)divergence);
    }

    float k = 1.0f;
    float D_star = -0.1f;
    float v = k * (divergence - D_star);

    DEBUG_PRINT("Divergence: %.2f\n", (double)divergence);
    DEBUG_PRINT("Obstacle parameter: %.2f\n", (double)obstacle);
    DEBUG_PRINT("v: %.2f\n", (double)v);

    if(obstacle == 1.0f)
    {
        DEBUG_PRINT("Drone is landing normally.\n");
        PositionX = logGetFloat(idX);
        DEBUG_PRINT("PositionX is now: %f deg\n", (double)PositionX);
        PositionY = logGetFloat(idY);
        DEBUG_PRINT("PositionY is now: %f deg\n", (double)PositionY);
    }
}
