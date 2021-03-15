"""
This module manipulates and translates all serial communication with our microchip
"""

import serial
import serial.tools.list_ports
import random

from typing import Tuple

class PcSerial:
    _port: serial.Serial
    _clickedBtns = 0
    _ledState = 0

    def __init__(self, port: str = None):
        # Setup
        open_ports = list(serial.tools.list_ports.comports())
        selected_port = None

        if port != None:
            selected_port = port
        elif len(open_ports) == 0:
            print("Microchip not found: placebo is being served")
            selected_port = None
        elif len(open_ports) == 1:
            selected_port = open_ports[0].device
        else:
            # More then 1 port available, let the user deside
            print("Multiple ports found, please select one:")
            for idx, port in enumerate(open_ports):
                print(f"\t{idx}: {port.device}")
            port_id = int(input("> "))

            selected_port = open_ports[port_id].device

            print("Thx for helping me out ;)")

        if selected_port == None:
            self._port = None
        else:
            self._port = serial.Serial(selected_port, baudrate=9600)
            print(f"Connected to port: {selected_port}")

    def poll(self) -> bool:
        if self._port == None: # Placebo data
            return random.random() > 0.9
        else:
            self._resetPort()
            
            self._writeToPort(b"B")
            clickedBtns = int(self._port.readline().rstrip())
            
            needUpdate = (self._clickedBtns != 0 or
                    self._clickedBtns != clickedBtns)
            self._clickedBtns = clickedBtns

            return needUpdate

    def get_btn_state(self) -> Tuple[bool,bool,bool,bool]:
        return (
            self._clickedBtns & 0b0001 != 0,
            self._clickedBtns & 0b0010 != 0,
            self._clickedBtns & 0b0100 != 0,
            self._clickedBtns & 0b1000 != 0
            )

    def set_led_state(self, ledIndex: int, status: bool):
        """ledIndex: [0,1,2,3]"""
        if self._port == None:
            return

        self._resetPort()

        ledIndex = _clamp(ledIndex, 0, 3)
        
        # Update global status
        if status:
            self._ledState |= 1<<ledIndex
        else:
            self._ledState &= ~(1<<ledIndex)

        # Schrijf weg
        self._writeToPort(f"L{self._ledState}".encode("ascii"))
        self._port.readline() # Read echo
        print(f"Writing to device: L{self._ledState}")
    
    def set_height(self, height: int):
        if self._port == None:
            return
        
        height = _clamp(height)

        self._resetPort()

        self._writeToPort(f"S{height}".encode("ascii"))
        self._port.readline() # Read echo

        print(f"Writing to device: S{height}")
        
    def _resetPort(self):
        if self._port == None:
            return
        self._port.reset_input_buffer();
        self._port.reset_output_buffer();
            
    def _writeToPort(self, data):
        if self._port == None:
            return

        data = b' ' + data + b'\r'
        print(f"Writing to device: {data}")
        self._port.write(data)
        self._port.flush()

    def __del__(self): # Always clean after yourself :)
        if self._port != None:
            self._port.close()

def _clamp(n: int, min_n = 0, max_n = 255):
    return max(min_n, min(max_n, n))
