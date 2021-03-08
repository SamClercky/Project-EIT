/**
  Section: Included Files
 */

#include <xc.h>
#include "mcc_generated_files/mcc.h"
#include "PI.h"
#include <stdlib.h> //for atoi and atof functions
#include <ctype.h> //for toupper command

/**
  Section: UART Module APIs
 */

static uint8_t sensorHeight;
static uint8_t setpoint; //240 = top, 0 = bottom
static float ki;
static float kp;

static char command;
static int index;
static char data[8]; //"S20" of "p1.25"
static char value[7];
static int printCycle = 0;

#define cmd_str_max_len 20
#define CR (char)13
static int cmd_str_len;
static char cmd_str[cmd_str_max_len];

static enum {
	STATE_INIT = 0,
	STATE_START_READ_CMD,
	STATE_READ_CMD,
	STATE_EXEC_CMD,
	STATE_TERMINATE
} state = STATE_INIT;

void Display_Menu() {
	printf("\n=== Ping Pong Toren ===\n");
	printf("\tS + arg to change setpoint\n");
	printf("\tB to get button status in u4 bitflag read as char");
	printf("\tL + u4 bitflag; to set led status in u4 read as char");
	printf("\tH for help\n");
	printf("END\n");
}

bool Read_Command(char* cmd_str, int* cmd_str_len) {
	while (EUSART_DataReady) {
		char ch = EUSART_Read();
		if (ch == CR) {
			cmd_str[*cmd_str_len] = '\0';
			return true;
		} else {
			cmd_str[(*cmd_str_len)++] = ch;
			if (*cmd_str_len == cmd_str_max_len) {
				cmd_str[cmd_str_max_len - 1] = '\0'; // overflow
				return true;
			}
		}
	}
	return false;
}

/**
 * @return true als de applicatie nog niet afgesloten mag worden
 */
bool Execute_Command(char* data) {
	char command = data[0];

	switch(toupper(command)) { // normalize command
		case 'S': //Setpoint                            
			setpoint = (uint8_t) atoi(data + 1); //atoi = ASCII to integer
			PI_SetSetpoint(setpoint);
			printf("%d\n", PI_GetSensorHeight()); // send sensorheight back
			break;
		case 'H':
			Display_Menu();
			break;
		default:
			printf("\nInvalid command %c\n", command);
			Display_Menu();
	}
	return true;
}

void UARTCommunication(void) {
	switch (state) {
		case STATE_INIT:
			Display_Menu();
			state = STATE_START_READ_CMD;
			break;
		case STATE_START_READ_CMD:
			// reset cmd
			cmd_str_len = -1;
			cmd_str[0] = '\0';

			// TODO: Bekijk of volgende printf nodig is
			// printf(">");
			state = STATE_READ_CMD;
			break;
		case STATE_READ_CMD:
			if (Read_Command(cmd_str, &cmd_str_len)) {
				state = STATE_EXEC_CMD;
			}
			break;
		case STATE_EXEC_CMD:
			if (Execute_Command(cmd_str)) {
				state = STATE_START_READ_CMD;
			} else {
				state = STATE_TERMINATE;
			}
			break;
		default: // normaal nooit uitgevoerd
			state = STATE_TERMINATE;
			printf("ERROR: Invalid state");
			break;
	}
}

/**
 End of File
 */
