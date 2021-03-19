#include <xc.h>
#include "mcc_generated_files/mcc.h"
#include "led.h"

#define NUMBER_OF_LEDS 60
#define COLOR 0xFFFFFF

//static uint64_t colorData = 2305843009213693951;
static uint8_t setpoint = 0;

void LEDSTRIP_Draw_Colors(uint24_t bgr) {
	SPI1_ExchangeByte(0x20); // intensity
	SPI1_ExchangeByte((bgr >> 16) & 255); // blue
	SPI1_ExchangeByte((bgr >> 8) & 255); // green
	SPI1_ExchangeByte(bgr & 255); // red
}

void LEDSTRIP_Draw_Frames() {
	// start frame
	// Alles achter elkaar voor perf
	SPI1_ExchangeByte(0x00);
	SPI1_ExchangeByte(0x00);
	SPI1_ExchangeByte(0x00);
	SPI1_ExchangeByte(0x00);

	// data
	for (char led = 0; led < NUMBER_OF_LEDS; led++) {
        //LEDSTRIP_Draw_Colors(COLOR);
		//if ((colorData >> led) % 2) {
		//	LEDSTRIP_Draw_Colors(COLOR); // color
		//} else {
		//	LEDSTRIP_Draw_Colors(0xFF0000); // no color
		//}

		if (led < setpoint) {
			LEDSTRIP_Draw_Colors(COLOR);
		} else {
			LEDSTRIP_Draw_Colors(0xFF0000);
		}
	}

	// end frame
	SPI1_ExchangeByte(0xFF);
	SPI1_ExchangeByte(0xFF);
	SPI1_ExchangeByte(0xFF);
	SPI1_ExchangeByte(0xFF);
}

// flag: bit 8: on/off -> rest is 0..60
void LEDSTRIP_Set_Locations(uint8_t flag) {
//	if (flag >> 7) {
//		colorData |= 1 << (flag - 255);
//	} else {
//		colorData &= ~(1<<(flag - 255));
//	}
	setpoint = flag;
}
