#include <xc.h>
#include "mcc_generated_files/mcc.h"
#include "led.h"

void LED_SetStatus(uint8_t flag) {
	LED1_LAT = flag & 0b0001;
	LED2_LAT = flag & 0b0010;
	LED3_LAT = flag & 0b0100;
	LED4_LAT = flag & 0b1000;
}
