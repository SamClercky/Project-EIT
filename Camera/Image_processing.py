
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import math
import matplotlib.pyplot as plt


class STATE:

    def __init__(self, start_state):

        self.states = "getting_target", "waiting", "getting_data", "analyzing_data"
        self.current_state = self.states[self.states.index(start_state)]

    def cycle(self):

        if self.states.index(self.current_state) == len(self.states) - 1:
            self.current_state = self.states[0]
        else:
            self.current_state = self.states[self.states.index(self.current_state) + 1]


state = STATE("getting_target")

green = [[0, 255, 0], "green"]
blue = [[255, 0, 0],'blue']

positions = []
trpoints = []
amountOfPoints = len(positions)
velocity = []
t0 = time.monotonic()

target_point1 = []
target_point2 = []
target_point3 = []
target_points = [target_point1, target_point2, target_point3]

trajectory_points = np.array()

intersection = []

pipeline = rs.pipeline()

config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth map uint16
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pc = rs.pointcloud()                                                    # Point cloud object (Depth map --> 3D Points)

profile = pipeline.start(config)

depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

kernel = np.ones((5, 5), np.uint8)

plt3d = plt.figure().gca(projection='3d')



def get_target(Mask, colour):
    global target_points

    (cnts, _) = cv2.findContours(Mask.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # look for any contours
    if len(cnts) > 0:
        cntss = sorted(cnts, key=cv2.contourArea, reverse=True)
        for i in range(0, len(cntss)):
            if i < 3:
                cnt = cntss[i]
                # Get the radius of the enclosing circle around the found contour
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                # Draw the circle around the contour
                cv2.circle(color_image, (int(x - radius / 2), int(y)), int(radius / 2), colour[0], 2)
                # Get the moments to calculate the center of the contour
                M = cv2.moments(cnt)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                diepte = (depth_image[center[1]][center[0]])


                point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], diepte)
                print(point)

                target_points[i].append(point)

                centroid = str(point)

                cv2.putText(color_image, centroid, center, cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # write a text to frame
                # cv2.putText(frame, str(text), (int(x + 50), int(y + 50)), cv2.FONT_HERSHEY_SIMPLEX,
                #            0.9, colour[0], 2, cv2.LINE_AA)
            else:
                break

def finalyzing_target():
    global target_points
    n = min(len(target_point1), len(target_point2), len(target_point3))
    #print(n)
    gem1 = [0, 0, 0]
    gem2 = [0, 0, 0]
    gem3 = [0, 0, 0]
    gem = [gem1, gem2, gem3]
    for i in range(0, n):
        #print(i)
        for j in range(0, 3):
            #print(i , j)
            gem[j][0] = gem[j][0] + target_points[j][i][0]
            gem[j][1] = gem[j][1] + target_points[j][i][1]
            gem[j][2] = gem[j][2] + target_points[j][i][2]
    for i in range(0, len(gem)):
        gem[i][0] = gem[i][0] / n
        coordinaten_transformatie(gem[0])
        gem[i][1] = gem[i][1] / n
        coordinaten_transformatie(gem[1])
        gem[i][2] = gem[i][2] / n
        coordinaten_transformatie(gem[2])
    target_points = gem
    print(target_points)

def coordinaten_transformatie(point):
    x = point[0] + 240
    y = point[1] + 320
    z = int(math.sqrt(point[2]*point[2] - (point[0]*point[0] + point[1] * point[1])))


    return x, y, z

def get_bal(Mask, colour):
    text = str(colour[1])
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

        # Draw the circle around the contour
        cv2.circle(color_image, (int(x), int(y)), int(radius), colour[0], 2)

        # Get the moments to calculate the center of the contour
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            diepte = (depth_image[center[1]][center[0]])


            point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], diepte)
            print(point)
            positions.append(point)
            trpoints.append(point)

            centroid = str(point)

            cv2.putText(color_image, centroid, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)
            #cv2.putText(color_image, str(diepte), (100, 150), cv2.FONT_HERSHEY_SIMPLEX,
             #           0.5, (255, 255, 255), 2, cv2.LINE_AA)

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

def afstand_punt_vlak(normal, d, point):
    # afstand = | ap + bq + cr + d | / √(a2 + b2 + c2).
    teller = 0
    noemer = 0
    for i in range(0,len(normal)):
        teller = teller + normal[i] * point[i]
        noemer = noemer + point[i] * point[i]
    teller = teller + d

    return abs(teller)/math.sqrt(noemer)

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

def Newton(normal0, d0, normal1, d1, qc):
    print("Newton")
    global X
    X = np.array([[1000, 1000, 1000]])
    X = X.transpose()
    # print(X)
    # print(X[0][0])
    # print(X[1][0])
    # print(X[2][0])

    # X[0][0] = x
    # X[0][1] = y
    # X[0][2] = z
    # f1 = normal0[0]*x + normal0[1]*y + normal0[2] * z - d0
    # dxf1 = normal0[0]
    # dyf1 = normal0[1]
    # dzf1 = normal0[2]
    # f2 = normal1[0]*x + normal1[1]*y + normal1[2] * z - d1
    # dxf2 = normal1[0]
    # dyf2 = normal1[1]
    # dzf2 = normal1[2]
    # f3 = qc[0] * x * x + qc[1] * x + qc[2] - y
    # dxf3 = 2*qc[0] * x + qc[1]
    # dyf3 = -1
    # dzf3 = 0

    iterations_store = []
    error_store = []
    plt3d.scatter3D(X[0][0], X[1][0], X[2][0], color="blue", marker='o', )

    i = 0
    n = 30
    while i < n:
        # print(i)
        # print(X)
        if i > 1:
            plt3d.scatter3D(X[0][0], X[1][0], X[2][0], color="blue", marker='.', alpha=(1 / n) * i)
        i = i + 1
        iterations_store.append(i)
        F = []
        F = np.array([[normal0[0] * X[0][0] + normal0[1] * X[1][0] + normal0[2] * X[2][0] - d0,
                       normal1[0] * X[0][0] + normal1[1] * X[1][0] + normal1[2] * X[2][0] - d1,
                       qc[0] * X[0][0] * X[0][0] + qc[1] * X[0][0] + qc[2] - X[1][0]]])
        F = F.transpose()
        error_store.append(np.linalg.norm(F))
        J = np.array([[normal0[0], normal0[1], normal0[2]],
                      [normal1[0], normal1[1], normal1[2]],
                      [2 * qc[0] * X[0][0] + qc[1], -1, 0]])
        X = X - np.linalg.inv(J).dot(F)

    print(X)
    plt.figure()
    plt.legend("newton rapsody for calculating intersection")
    plt.xlabel("itarations")
    plt.ylabel("error")
    plt.plot(iterations_store, error_store)
    plt3d.scatter3D(X[0][0], X[1][0], X[2][0], color="blue", marker='x', )
    return X.transpose

    #return x,y,z

def planes_quadratic_intersect(target_points, trajectory_points):


    print("berekeningen")

    plt3d = plt.figure().gca(projection='3d')
    plt.xlabel("X axis")
    plt.ylabel("Y axis")

    p0, p1, p2 = target_points
    q0, q1, q2 = trajectory_points
    x_coo = []
    y_coo = []
    z_coo = []
    for i in target_points:
        x_coo.append(int(i[0]))
        y_coo.append(int(i[1]))
        z_coo.append(int(i[2]))
    for i in trajectory_points:
        x_coo.append(int(i[0]))
        y_coo.append(int(i[1]))
        z_coo.append(int(i[2]))

    plt3d.set_xlim3d(min(x_coo), max(x_coo))
    plt3d.set_ylim3d(min(y_coo), max(y_coo))
    plt3d.set_zlim3d(min(z_coo), max(z_coo))

    plt3d.plot_trisurf([x_coo[0], x_coo[1], x_coo[2]], [y_coo[0], y_coo[1], y_coo[2]], [z_coo[0], z_coo[1], z_coo[2]],
                       alpha=0.3)
    plt3d.plot_trisurf([x_coo[3], x_coo[4], x_coo[5]], [y_coo[3], y_coo[4], y_coo[5]], [z_coo[3], z_coo[4], z_coo[5]],
                       color="red", alpha=0.3)

    plt3d.scatter3D(x_coo, y_coo, z_coo, color="purple")

    e = []
    e0 = []
    for i in range(0, 100, 1):
        e.append(i)
        e0.append(0)
    plt3d.plot(e, e0, e0, color="black")
    plt3d.plot(e0, e, e0, color="black")
    plt3d.plot(e0, e0, e, color="black")

    #target equation and plane
    point0 = np.array(p0)
    normal0 = np.array(cross_product(vect_AB(p0, p1), vect_AB(p0, p2)))
    d0 = -point0.dot(normal0)
    #xx0, yy0 = np.meshgrid(range(1000), range(1000))
    #zz0 = (-normal0[0] * xx0 - normal0[1] * yy0 - d0) * 1. / normal0[2]
    #plt3d.plot_surface(xx0, yy0, zz0, color="gray", alpha=0.15)

    # y = ax^2 + bx + c
    qc = quadratic_constants(trajectory_points)
    a, b, c = qc

    #trajection equation and plane
    point1 = np.array(q0)
    normal1 = np.array(cross_product(vect_AB(q0, q1), vect_AB(q0, q2)))
    d1 = -point1.dot(normal1)
    #xx1, yy1 = np.meshgrid(range(1000), range(1000))
    #zz1 = (-normal1[0] * xx1 - normal1[1] * yy1 - d1) * 1. / normal1[2]
    #lt3d.plot_surface(xx1, yy1, zz1, color="red", alpha=0.15)

    #plotting quadratic function
    z = []
    x = []
    y = []
    for j in range(-100, 100, 10):
        x.append(j)
        y.append(a * j * j + b * j + c)
        z.append((-normal1[0] * (a * j * j + b * j + c) - normal1[1] * j - d1) * 1. / normal1[2])
        if (afstand_punt_vlak(normal0, d0, (x[-1], y[-1], z[-1])) < 1000):
            print(afstand_punt_vlak(normal0, d0, (x[-1], y[-1], z[-1])))
            print(x[-1], y[-1], z[-1])
            plt3d.scatter3D(x[-1], y[-1], z[-1], color="blue", marker='x', )
    plt3d.plot(x, y, z, color="red", alpha= 0.3)

    global intersection
    intersection = Newton(normal0, d0, normal1, d1, qc)

    plt.show(block = False)
    plt.pause(5)

def procces_data():
    global trajectory_points
    trajectory_points = []
    print("positions :")
    print(positions)
    # selecting 3 points for calculation
    i = len(positions)
    print("len = " + str(i))
    if i % 2 == 0:
        i = int(i/2)
        print("i = " + str(i))
        trajectory_pointst = [positions[i - 1], positions[i], positions[i + 1]]
    else:
        i = int((i - 1) / 2)
        print("i = " + str(i))
        trajectory_pointst = [positions[i - 1], positions[i], positions[i + 1]]

    trajectory_points = trajectory_pointst

    # plotting the target plane the plane of the trow and the trajectory of the trow
    print("trajectory points : ")
    print(trajectory_points)
    planes_quadratic_intersect(target_points, trajectory_points)

def get_distace_to_intersect():
    d = []
    for i in range(0, 3):
        d.append(math.sqrt((target_points[i][0] - intersection[0])*(target_points[i][0] - intersection[0]) + (target_points[i][1] - intersection[1])*(target_points[i][1] - intersection[1]) + (target_points[i][2] - intersection[2])*(target_points[i][2] - intersection[2])))
    return d

def draw_positions():
    #dit werkt niet meer omdat er alleen 3d punten opgeslagen worden
    for i in range(0, len(trpoints)):
        cv2.circle(color_image, (int(trpoints[i][0]), int(trpoints[i][1])), 5, (0,255,255), 2)


try:
    while True:
        while state.current_state is "getting_target" or state.current_state is "getting_data":

            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth_image = np.asanyarray(depth_frame.get_data())

            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            hsv_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            #plt.imshow(hsv_frame)
            #plt.show()

            #get_velocity()

            if state.current_state is "getting_target":
                # Blue color
                #low_blue = np.array([90, 250, 70])
                #high_blue = np.array([130, 255, 130])
                #blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)


                lower_red = np.array([0, 120, 70])
                upper_red = np.array([10, 255, 255])
                mask1 = cv2.inRange(hsv_frame, lower_red, upper_red)
                # Range for upper range
                lower_red = np.array([170, 120, 70])
                upper_red = np.array([180, 255, 255])
                mask2 = cv2.inRange(hsv_frame, lower_red, upper_red)
                # Generating the final mask to detect red color
                blue_mask = mask1 + mask2

                blue_mask = cv2.erode(blue_mask, kernel, iterations=1)
                blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
                blue_mask = cv2.dilate(blue_mask, kernel, iterations=2)
                cv2.imshow("blue mask", blue_mask)
                # blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
                get_target(blue_mask, blue)

            elif state.current_state is "getting_data":
                draw_positions()
                # green color
                #low_green = np.array([90, 70, 15])
                #high_green = np.array([110, 120, 40])

                lower_red = np.array([0, 120, 70])
                upper_red = np.array([10, 255, 255])
                mask1 = cv2.inRange(hsv_frame, lower_red, upper_red)
                # Range for upper range
                lower_red = np.array([170, 120, 70])
                upper_red = np.array([180, 255, 255])
                mask2 = cv2.inRange(hsv_frame, lower_red, upper_red)
                # Generating the final mask to detect red color
                green_mask = mask1 + mask2

                #green_mask = cv2.inRange(hsv_frame, low_green, high_green)
                #green_mask = cv2.erode(green_mask, kernel, iterations=2)
                # cv2.imshow("blue mask_1", green_mask)
                #green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
                # cv2.imshow("blue mask_2", green_mask)
                #green_mask = cv2.dilate(green_mask, kernel, iterations=1)
                # cv2.imshow("blue mask_3", green_mask)
                #gousian blur for tracking
                get_bal(green_mask, green)

            #cv2.imshow("Ik ben ook maar een persoon (gray)", hsv_frame)
            #cv2.imshow("Depth Image", depth_colormap)
            cv2.imshow("Color Image", color_image)


            if cv2.waitKey(1) & 0xFF == ord(' '):
                if state.current_state is "getting_target":
                    print("finalyzing_target")
                    finalyzing_target()
                    state.cycle()
                    print(state.current_state)
                elif len(positions) > 2 and state.current_state is "getting_data":
                    state.cycle()
                    print(state.current_state)
                    print("yes")

                    break
                else:
                    print("not enough points trow again")
                    print(positions)
                    positions = []
        if state.current_state is "analyzing_data":
            print(state.current_state)
            procces_data()
            state.cycle()
            state.cycle()
            print(state.current_state)
            positions = []
            trpoints = []
        elif state.current_state is "waiting":
            if cv2.waitKey(1) & 0xFF == ord(' '):
                state.cycle()



finally:
    pipeline.stop()
