#!/bin/sh

sudo socat pty,raw,echo=0,link=/dev/ttyS20 pty,raw,echo=0,link=/dev/ttyS21 &
sleep 0.5
sudo chown samclercky:samclercky /dev/ttyS20
sudo chown samclercky:samclercky /dev/ttyS21
