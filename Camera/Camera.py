# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:22:00 2021

@author: Ramses
"""
import numpy as np
import cv2
import time
import math

#defenitie van geel we moeten enkel geel uit de feed halen omdat we met een tennisbal werken
yellow=[[0,255,255],'yellow']
blue = [[255, 0, 0],'blue']
positions = []
amountOfPoints = len(positions)
velocity = []
t0 = time.monotonic()


cap = cv2.VideoCapture(0)

print("ik ben iets aan het doen")

kernel = np.ones((5, 5), np.uint8)



def draw_contours(Mask,colour):
    text=str(colour[1] + "balls")
    # print(text)
    #why use .copy()?
    #RETR_EXTERNAL segt welke contours worden bijgehouden in dit geval alle child contours worden weggelaten
    #CHAIN_APPROX_SIMPLE zegt hoeveel punten bewaard worden voor elke contour in dit geval enkel de uiterste punten
    (cnts, _) = cv2.findContours(Mask.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # look for any contours
    if len(cnts) > 0:
        # Sort the contours using area and find the largest one
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        positions.append((int(x), int(y), time.monotonic() - t0))

        # Draw the circle around the contour
        cv2.circle(frame, (int(x-radius/2), int(y)), int(radius/2), colour[0], 2)
        cv2.circle(frame, (int(x+radius/2), int(y)), int(radius/2), colour[0], 2)
        cv2.rectangle(frame, (int(x-radius/4), int(y-radius*2)), (int(x+radius/4), int(y)), colour[0], 2)
        # Get the moments to calculate the center of the contour
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        centroid = str(center)

        #v2.putText(frame, centroid, center, cv2.FONT_HERSHEY_SIMPLEX,
        #            0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # write a text to frame
        #cv2.putText(frame, str(text), (int(x + 50), int(y + 50)), cv2.FONT_HERSHEY_SIMPLEX,
        #            0.9, colour[0], 2, cv2.LINE_AA)
def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def get_velocity():
    global amountOfPoints
    #print("amount of points = " + str(amountOfPoints) + ", len positions = " + str(len(positions)))
    for i in range(amountOfPoints-1, len(positions)-1):
        print(i)
        if (i > -1) & (positions[i+1][2] - positions[i][2] > 10^(-5)):
            dt = positions[i + 1][2] - positions[i][2]
            pos = (positions[i-1][0], positions[i-1][1])

            dx = positions[i+1][0] - positions[i][0]
            dy = positions[i+1][1] - positions[i][1]
            xv = (pos, dx / dt, dy / dt)

            velocity.append(xv)

            amountOfPoints = len(positions)
    i = 0
    gem = (0,0)
    for x in velocity:
        gem = (gem[0] + x[1] , gem[1]+ x[2])
        i = i+1
        cv2.circle(frame,x[0],5,(0,255,255),2)

    if len(velocity) > 0:
        cv2.putText(frame, "vx = " + str(round_up(velocity[-1][1])), (100, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "vy = " + str(round_up(velocity[-1][2])), (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2, cv2.LINE_AA)

        gem = (gem[0] / i, gem[1] / i)
        cv2.putText(frame, "gem. vx = " + str(round_up(gem[0])), (100, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "gem. vy = " + str(round_up(gem[1] )), (100, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)
    #print(velocity)



while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow("Ik ben ook maar een persoon (gray)", hsv_frame)

    # Blue color
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue_mask = cv2.erode(blue_mask, kernel, iterations=2)
    cv2.imshow("blue mask_1", blue_mask)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow("blue mask_2", blue_mask)
    blue_mask = cv2.dilate(blue_mask, kernel, iterations=1)
    cv2.imshow("blue mask_3",blue_mask)
    draw_contours(blue_mask, blue)
    # blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

    get_velocity()


    cv2.imshow("Ik ben ook maar een persoon",frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        positions = []
        amountOfPoints = len(positions)
        velocity = []
        t0 = time.monotonic()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


