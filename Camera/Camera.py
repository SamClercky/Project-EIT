# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:22:00 2021

@author: Ramses
"""
import numpy as np
import cv2

#defenitie van geel we moeten enkel geel uit de feed halen omdat we met een tennisbal werken
yellow=[[0,255,255],'yellow']

cap = cv2.VideoCapture(0)

print("ik ben iets aan het doen")

while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frameP = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)



    cv2.imshow("Ik ben ook maar een persoon",frame)
    cv2.imshow("Ik ben ook maar een persoon (gray)", hsv_frame)
    #cv2.imshow("Ik ben ook maar een persoon (gray)",gray)
    #cv2.imshow("Ik ben ook maar een persoon (binary)",frameP)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


