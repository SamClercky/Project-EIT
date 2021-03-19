/* 
 * File:   ledstrip.h
 * Author: samclercky
 *
 * Created on March 12, 2021, 2:05 PM
 */

#ifndef LEDSTRIP_H
#define	LEDSTRIP_H

#include <xc.h>
#include "mcc_generated_files/mcc.h"

#ifdef	__cplusplus
extern "C" {
#endif

void LEDSTRIP_Draw_Frames();

void LEDSTRIP_Set_Locations(uint8_t flag);

#ifdef	__cplusplus
}
#endif

#endif	/* LEDSTRIP_H */

