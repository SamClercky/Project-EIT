/**
  Section: Included Files
 */

#include <xc.h>
#include "mcc_generated_files/mcc.h"
#include "PI.h"

/**
  Section: PI Module APIs
 */

static uint8_t sensorHeight = 0;
static uint8_t setpoint = 200; //255 is top
static int error = 0;
static float integral = 0;
static float ki = 0.005;
static float kp = 1;
static int dutycycle;

uint8_t PI_GetSensorHeight(void){
    return sensorHeight;
}

void PI_SetSetpoint(uint8_t value){
    setpoint = value;
}

void PI_SetKp(float value){
    kp = value;
}

void PI_SetKi(float value){
    ki = value;
}

void PI(void) {
	//printf("Hello from PI");
    
    PWM5_LoadDutyValue((uint8_t) setpoint);
}

/**
 End of File
 */
