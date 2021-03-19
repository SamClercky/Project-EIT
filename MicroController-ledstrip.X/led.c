#include <xc.h>
#include "mcc_generated_files/mcc.h"
#include "led.h"

#define NUMBER_OF_LEDS 60
#define COLOR 0x00F00F

static uint64_t colorData = 25;

void LEDSTRIP_Draw_Colors(uint24_t bgr) {
	SPI1_ExchangeByte(0xFF * (bgr != 0)); // intensity
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
		if ((colorData >> led) & 1) {
			LEDSTRIP_Draw_Colors(COLOR); // no color
		} else {
			LEDSTRIP_Draw_Colors(0x000000); // color
		}
	}

	// end frame
	SPI1_ExchangeByte(0xFF);
	SPI1_ExchangeByte(0xFF);
	SPI1_ExchangeByte(0xFF);
	SPI1_ExchangeByte(0xFF);

	printf("Frame written");
}

// flag: bit 8: on/off -> rest is 0..60
void LEDSTRIP_Set_Locations(char flag) {
	if (flag >> 7) {
		colorData |= 1 << flag;
	} else {
		colorData &= ~(1<<flag);
	}
}
