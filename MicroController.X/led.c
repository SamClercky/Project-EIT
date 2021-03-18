#include <xc.h>
#include "mcc_generated_files/mcc.h"
#include "led.h"

void LED_SetStatus(uint8_t flag) {
	LED1_LAT = (flag >> 0) & 1;
	LED2_LAT = (flag >> 1) & 1;
	LED3_LAT = (flag >> 2) & 1;
	LED4_LAT = (flag >> 3) & 1;
}
