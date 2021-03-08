/* Demo project pingpongtoren + hoogtesensor 
 * 
 * standaard waarden: setpoint = 100
 *                    kp = 1
 *                    ki = 0.005
 * 
 * pinout:  RC2 = receiver input
 *          RC7 = transmitter output
 *          RB6 = pulse lengte output
 *          RB4 = pwm output
 *          RC1 = motor output
 */

#include "mcc_generated_files/mcc.h"
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <stdint.h>
#include "PI.h"
#include "UART.h"

/*
                         Main application
 */
void main(void) {
    // initialize the device
    SYSTEM_Initialize();

    // When using interrupts, you need to set the Global and Peripheral Interrupt Enable bits
    // Use the following macros to:

    // Enable the Global Interrupts
    INTERRUPT_GlobalInterruptEnable();

    // Enable the Peripheral Interrupts
    INTERRUPT_PeripheralInterruptEnable();

    // Disable the Global Interrupts
    //INTERRUPT_GlobalInterruptDisable();

    // Disable the Peripheral Interrupts
    //INTERRUPT_PeripheralInterruptDisable();

    // Initialiseer de hoogtemeting
    ADC_SelectChannel(Hoogtesensor);
    ADC_StartConversion();
    TMR2_Initialize();
    TMR2_StartTimer();
        
    while (1) {
                
        // loop moet op een vaste frequentie lopen voor de integrator
        if (TMR0_HasOverflowOccured()) {
            TMR0_Initialize();

            PI();
            UARTCommunication();
            // NIET ONDER 60 GAAN
            //PWM5_LoadDutyValue((uint8_t) 120);
        }
    }
}

/**
 End of File
 */
