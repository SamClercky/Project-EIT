


import pyrealsense2 as rs
import numpy as np
import cv2
import time
import math
import matplotlib.pyplot as plt


class STATE:

    def __init__(self, start_state):

        self.states = "getting_target", "getting_data", "analyzing_data"
        self.current_state = self.states[self.states.index(start_state)]

    def cycle(self):

        if self.states.index(self.current_state) == len(self.states) - 1:
            self.current_state = self.states[0]
        else:
            self.current_state = self.states[self.states.index(self.current_state) + 1]


state = STATE("getting_target")

green = [[0, 255, 0], "green"]
target_points = []
positions = []
amountOfPoints = len(positions)
velocity = []
t0 = time.monotonic()

pipeline = rs.pipeline()

config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth map uint16
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

kernel = np.ones((5, 5), np.uint8)


def get_target(Mask, colour):
    global target_points

    (cnts, _) = cv2.findContours(Mask.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)

        target_points.append([int(x), int(y)])
        cv2.putText(color_image, "Point found", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw the circle around the contour
        cv2.circle(color_image, (int(x), int(y)), int(radius), colour[0], 2)

        # Get the moments to calculate the center of the contour
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        diepte = (depth_image[center[1]][center[0]])

        centroid = str(center)

        cv2.putText(color_image, centroid, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(color_image, str(diepte), (100, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # code voor target
    point0 = [50, 50, 50]
    if len(target_points) == 3:
        state.cycle()

def draw_contours(Mask, colour):
    text = str(colour[1] + "balls")
    # print(text)
    # why use .copy()?
    # RETR_EXTERNAL segt welke contours worden bijgehouden in dit geval alle child contours worden weggelaten
    # CHAIN_APPROX_SIMPLE zegt hoeveel punten bewaard worden voor elke contour in dit geval enkel de uiterste punten
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
        cv2.circle(color_image, (int(x), int(y)), int(radius), colour[0], 2)

        # Get the moments to calculate the center of the contour
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        diepte = (depth_image[center[1]][center[0]])

        centroid = str(center)

        cv2.putText(color_image, centroid, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(color_image, str(diepte), (100, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # write a text to frame
        # cv2.putText(color_image, str(text), (int(x + 50), int(y + 50)), cv2.FONT_HERSHEY_SIMPLEX,
        #            0.9, colour[0], 2, cv2.LINE_AA)

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def vect_AB(p0, p1):

    x0, y0, z0 = p0
    x1, y1, z1 = p1

    return [x1 - x0, y1 - y0, z1 - z0]

def cross_product(u, v):

    ux, uy, uz = u
    vx, vy, vz = v

    uXv = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]

    return uXv

def quadratic_constants(points):
    p0, p1, p2 = points
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    g = y1 - y0
    k = x1 * x1 - x0 * x0
    h = x1 - x0

    a = ((y2 - y0) * h - x2 * g + x0 * g) / (x2 * x2 * h - x2 * k - x0 * x0 * h + x0 * k)
    b = (g - a * k) / h
    c = y0 - a * x0 * x0 - b * x0

    return a, b, c

def get_velocity():
    global amountOfPoints
    # print("amount of points = " + str(amountOfPoints) + ", len positions = " + str(len(positions)))
    for i in range(amountOfPoints - 1, len(positions) - 1):
        print(i)
        if (i > -1) & (positions[i + 1][2] - positions[i][2] > 10 ^ (-5)):
            dt = positions[i + 1][2] - positions[i][2]
            pos = (positions[i - 1][0], positions[i - 1][1])

            dx = positions[i + 1][0] - positions[i][0]
            dy = positions[i + 1][1] - positions[i][1]
            xv = (pos, dx / dt, dy / dt)

            velocity.append(xv)

            amountOfPoints = len(positions)
    i = 0
    gem = (0, 0)
    for x in velocity:
        gem = (gem[0] + x[1], gem[1] + x[2])
        i = i + 1
        # tekenen van gele cirkels op traject
        # cv2.circle(color_image,x[0],5,(0,255,255),2)

    if len(velocity) > 0:
        cv2.putText(color_image, "vx = " + str(round_up(velocity[-1][1])), (100, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(color_image, "vy = " + str(round_up(velocity[-1][2])), (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)

        gem = (gem[0] / i, gem[1] / i)
        cv2.putText(color_image, "gem. vx = " + str(round_up(gem[0])), (100, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(color_image, "gem. vy = " + str(round_up(gem[1])), (100, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)
    # print(velocity)

def planes_quadratic_intersect(target_points, trajectory_points):

    plt3d = plt.figure().gca(projection='3d')
    plt.xlabel("X axis")
    plt.ylabel("Y axis")

    p0, p1, p2 = target_points
    q0, q1, q2 = trajectory_points
    x_coo = []
    y_coo = []
    z_coo = []
    for i in target_points:
        x_coo.append(i[0])
        y_coo.append(i[1])
        z_coo.append(i[2])
    for i in trajectory_points:
        x_coo.append(i[0])
        y_coo.append(i[1])
        z_coo.append(i[2])

    plt3d.scatter3D(x_coo, y_coo, z_coo, color="purple")

    point0 = np.array(p0)
    normal0 = np.array(cross_product(vect_AB(p0, p1), vect_AB(p0, p2)))

    d0 = -point0.dot(normal0)

    xx0, yy0 = np.meshgrid(range(max(x_coo)), range(max(y_coo)))

    zz0 = (-normal0[0] * xx0 - normal0[1] * yy0 - d0) * 1. / normal0[2]

    plt3d.plot_surface(xx0, yy0, zz0, color="gray", alpha=0.15)



    # y = ax^2 + bx + c
    a, b, c = quadratic_constants(trajectory_points)

    point1 = np.array(q0)
    normal1 = np.array(cross_product(vect_AB(q0, q1), vect_AB(q0, q2)))

    d1 = -point1.dot(normal1)

    #plotting quadratic function
    z = []
    x = []
    y = []
    for j in range(min(x_coo)-10, max(x_coo)+10 , 10):
        x.append(j)
        y.append(a * j * j + b * j + c)
        z.append((-normal1[0] * j - normal1[1] * (a * j * j + b * j + c) - d1) * 1. / normal1[2])

    plt3d.scatter3D(x, y, z, color="red", alpha= 0.3)

    plt.show()

def procces_data():
    # selecting 3 points for calculation
    i = len(positions)
    if i % 2 == 0:
        trajectory_points = [positions[i - 1], positions[i], positions[i + 1]]
    else:
        i = (i - 1) / 2
        trajectory_points = [positions[i - 1], positions[i], positions[i + 1]]

    # plotting the target plane the plane of the trow and the trajectory of the trow
    planes_quadratic_intersect(target_points, trajectory_points)

try:
    while True:
        while state is "getting_target" or state is "getting_data":

            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth_image = np.asanyarray(depth_frame.get_data())

            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            hsv_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            cv2.imshow("Ik ben ook maar een persoon (gray)", hsv_frame)

            # green color
            low_green = np.array([40, 52, 72])
            high_green = np.array([102, 255, 255])
            green_mask = cv2.inRange(hsv_frame, low_green, high_green)
            green_mask = cv2.erode(green_mask, kernel, iterations=2)
            # cv2.imshow("blue mask_1", green_mask)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
            # cv2.imshow("blue mask_2", green_mask)
            green_mask = cv2.dilate(green_mask, kernel, iterations=1)
            # cv2.imshow("blue mask_3", green_mask)

            # blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

            #get_velocity()

            if state is "getting_target":
                get_target(green_mask, green)
            elif state is "getting_data":
                draw_contours(green_mask, green)

            cv2.imshow("Depth Image", depth_colormap)
            cv2.imshow("Color Image", color_image)

            if cv2.waitKey(1) & 0xFF == ord(' '):
                state.cycle()
            cv2.waitKey(1)

        if state is "analyzing_data":
            procces_data()

finally:
    pipeline.stop()