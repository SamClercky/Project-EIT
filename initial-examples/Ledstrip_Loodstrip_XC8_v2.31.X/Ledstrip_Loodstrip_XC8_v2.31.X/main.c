/**
  Generated Main Source File

  Company:
    Microchip Technology Inc.

  File Name:
    main.c

  Summary:
    This is the main file generated using PIC10 / PIC12 / PIC16 / PIC18 MCUs

  Description:
    This header file provides implementations for driver APIs for all modules selected in the GUI.
    Generation Information :
        Product Revision  :  PIC10 / PIC12 / PIC16 / PIC18 MCUs - 1.81.7
        Device            :  PIC16F18345
        Driver Version    :  2.00
*/

/*
    (c) 2018 Microchip Technology Inc. and its subsidiaries. 
    
    Subject to your compliance with these terms, you may use Microchip software and any 
    derivatives exclusively with Microchip products. It is your responsibility to comply with third party 
    license terms applicable to your use of third party software (including open source software) that 
    may accompany Microchip software.
    
    THIS SOFTWARE IS SUPPLIED BY MICROCHIP "AS IS". NO WARRANTIES, WHETHER 
    EXPRESS, IMPLIED OR STATUTORY, APPLY TO THIS SOFTWARE, INCLUDING ANY 
    IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, AND FITNESS 
    FOR A PARTICULAR PURPOSE.
    
    IN NO EVENT WILL MICROCHIP BE LIABLE FOR ANY INDIRECT, SPECIAL, PUNITIVE, 
    INCIDENTAL OR CONSEQUENTIAL LOSS, DAMAGE, COST OR EXPENSE OF ANY KIND 
    WHATSOEVER RELATED TO THE SOFTWARE, HOWEVER CAUSED, EVEN IF MICROCHIP 
    HAS BEEN ADVISED OF THE POSSIBILITY OR THE DAMAGES ARE FORESEEABLE. TO 
    THE FULLEST EXTENT ALLOWED BY LAW, MICROCHIP'S TOTAL LIABILITY ON ALL 
    CLAIMS IN ANY WAY RELATED TO THIS SOFTWARE WILL NOT EXCEED THE AMOUNT 
    OF FEES, IF ANY, THAT YOU HAVE PAID DIRECTLY TO MICROCHIP FOR THIS 
    SOFTWARE.
*/

#include "mcc_generated_files/mcc.h"

/*
                         Main application
 */
uint8_t blue = 0x00, green = 0x00, red = 0xFF;

enum states {
    GREEN_UP, RED_DOWN, BLUE_UP, GREEN_DOWN, RED_UP, BLUE_DOWN
};
enum states change_color = GREEN_UP;
uint8_t step = 1;

enum count {
    UP, DOWN
};
enum count direction = UP;
char led_run = 0;
const int NumberOfLEDs = 60;

void Send_LED_Frame(uint8_t intensity, uint8_t blue, uint8_t green, uint8_t red) {
    SPI1_ExchangeByte(0xE0 | intensity);
    SPI1_ExchangeByte(blue);
    SPI1_ExchangeByte(green);
    SPI1_ExchangeByte(red);
}

void Send_LED_StartFrame() {
    for (int i = 0; i < 4; i++) {
        SPI1_ExchangeByte(0x00);
    }
}

void Send_LED_EndFrame() {
    for (int i = 0; i < 4; i++) {
        SPI1_ExchangeByte(0xFF);
    }
}
static uint8_t buffer[16] = "data";



void main(void) {
    // initialize the device
    SYSTEM_Initialize();
    spi_master_open(ledStrip);

    // When using interrupts, you need to set the Global and Peripheral Interrupt Enable bits
    // Use the following macros to:

    // Enable the Global Interrupts
    //INTERRUPT_GlobalInterruptEnable();

    // Enable the Peripheral Interrupts
    //INTERRUPT_PeripheralInterruptEnable();

    // Disable the Global Interrupts
    //INTERRUPT_GlobalInterruptDisable();

    // Disable the Peripheral Interrupts
    //INTERRUPT_PeripheralInterruptDisable();
    
    printf("\n\r");//print new line
    printf("ledstrip gestart met baudrate 9600\n\r");



    Send_LED_StartFrame();
    while (1) {
        switch (direction) {
            case UP: if (led_run < NumberOfLEDs - 1) {
                    led_run++;
                } else {
                    direction = DOWN;
                }
                break;
            case DOWN: if (led_run > 0) {
                    led_run--;
                } else {
                    direction = UP;
                }
                break;
        }

        switch (change_color) {
            case GREEN_UP: if (green < 0xFF) {
                    green += step;
                } else {
                    change_color = RED_DOWN;
                }
                break;
            case RED_DOWN: if (red > 0x00) {
                    red -= step;
                } else {
                    change_color = BLUE_UP;
                }
                break;
            case BLUE_UP: if (blue < 0xFF) {
                    blue += step;
                } else {
                    change_color = GREEN_DOWN;
                }
                break;
            case GREEN_DOWN: if (green > 0x00) {
                    green -= step;
                } else {
                    change_color = RED_UP;
                }
                break;
            case RED_UP: if (red < 0xFF) {
                    red += step;
                } else {
                    change_color = BLUE_DOWN;
                }
                break;
            case BLUE_DOWN: if (blue > 0x00) {
                    blue -= step;
                } else {
                    change_color = GREEN_UP;
                }
                break;
        }

        //start frame
        Send_LED_StartFrame();
        for (char led = 0; led < NumberOfLEDs; led++) {
            if (led == led_run) {
                Send_LED_Frame(0x1F, blue, green, red);
            } else {
                Send_LED_Frame(0x00, 0x00, 0x00, 0x00);
            }
        }
        //stop frame
        Send_LED_EndFrame();
        __delay_ms(10);
    };
}
/**
 End of File
 */