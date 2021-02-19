/**
  Section: Included Files
 */

#include <xc.h>
#include "mcc_generated_files/mcc.h"
#include "PI.h"

/**
  Section: PI Module APIs
 */

uint8_t clamp(int);

static uint8_t sensorHeight = 0;
static uint8_t setpoint = 200; //255 is top
static int error = 0;
static float integral = 0;
static float ki = 0.566;
static float kp = 1.350;
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

// 0.01f = 10ms
// period = 2.860s
// KP_init = 3
void PI_Initialize() {
	float ku = 3;
	float T0 = 2.860;
	
	kp = 0.45 * ku;
	float T1 = T0/1.2;
	ki = kp/T1;
}

void PI(void) {
    //setpoint = (uint8_t) (ADC_GetConversion(Potentiometer) >> 8); //resultaat van ADC
    sensorHeight = (uint8_t) (ADC_GetConversion(Hoogtesensor) >> 8); //resultaat van ADC

	int error = ( (int)setpoint - (int)sensorHeight );
	//printf("%d\n\r", error);
	int P = kp * error;
	int I = ki * ((float)integral + 0.01f* (float)error); 
	
	// TODO: Find problem
	dutycycle = P + I;
   
    PWM5_LoadDutyValue(clamp(dutycycle)); // output pwm signaal voor hoogte
}

uint8_t clamp(int data) {
	data = data > 255? 255 : data;
	data = data < 0? 0 : data;
	return data;
}

/**
 End of File
 */
