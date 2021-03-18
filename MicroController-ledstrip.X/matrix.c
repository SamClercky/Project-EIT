#include <xc.h>
#include "mcc_generated_files/mcc.h"
#include "matrix.h"

uint8_t Matrix_ScanButtons() {
	// set all to LOW
	mbtn11_SetLow();
	mbtn12_SetLow();

	// ensure pullup
	mbtn21_SetPullup();
	mbtn22_SetPullup();

	uint8_t matrixState = 0;

	// Read first row
	mbtn11_SetHigh();

	matrixState |= mbtn21_GetValue() * 0b0001;
	matrixState |= mbtn22_GetValue() * 0b0010;

	mbtn11_SetLow();

	// Read first row
	mbtn12_SetHigh();

	matrixState |= mbtn21_GetValue() * 0b0100;
	matrixState |= mbtn22_GetValue() * 0b1000;

	mbtn12_SetLow();

	return matrixState;
}
