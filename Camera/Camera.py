# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:22:00 2021

@author: Ramses
"""
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

print("ik ben iets aan het doen")

while True:
    _, frame = cap.read()

    cv2.imshow("Ik ben ook maar een persoon",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


