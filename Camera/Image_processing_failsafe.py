import pyrealsense2 as rs
import numpy as np
import cv2
import time
import math
import matplotlib.pyplot as plt
from sty import fg
from pcserial.pcserial import *


class STATE:

    def __init__(self, start_state):

        self.states = "getting_target", "getting_data"
        self.current_state = self.states[self.states.index(start_state)]

    def cycle(self):

        if self.states.index(self.current_state) == len(self.states) - 1:
            self.current_state = self.states[0]
        else:
            self.current_state = self.states[self.states.index(self.current_state) + 1]


class CameraControl():


    def __init__(self):
        # voor als ik op kot werk -> "kot"
        self.place = ""
        self.calibreren = False
        self.speltype = 1
        self.state = STATE("getting_target")
        self.positions = []
        self.positions_on_color = []
        self.amountOfPoints = len(self.positions)
        self.target_point1 = []
        self.target_point2 = []
        self.target_point3 = []
        self.target_points = [self.target_point1, self.target_point2, self.target_point3]
        self.trajectory_points = np.array([])
        self.intersection = []
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth map uint16
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.profile = self.pipeline.start(self.config)

        self.depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        self.depth_intrinsics = self.depth_profile.get_intrinsics()

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.kernel = np.ones((5, 5), np.uint8)

        self.plt3d = 0

        self.color_image = 0
        self.depth_image = 0

    pcs = PcSerial()

    green = [[0, 255, 0], "green"]
    blue = [[255, 0, 0],'blue']

    pc = rs.pointcloud()                                                    # Point cloud object (Depth map --> 3D Points)

    def draw_positions_on_color_image(self):
        for i in self.positions_on_color:
            cv2.circle(self.color_image, i[0], int(i[1]/150), (255, 153, 255))

    def draw_instructions(self):
        cv2.putText(self.color_image, "instructies:"
                    , (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if(self.state.current_state is "getting_target"):
            cv2.putText(self.color_image, "-press ' ' for finalising the target"
                        , (120, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(self.color_image, "-press ' ' for getting score"
                        , (120, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(self.color_image, "-press 'p' for plotting positions (positions aren't thrown away)"
                    , (120, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(self.color_image, "-press 'f' pay respect and reset found points"
                    , (120, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def distance_between_points(self, point1, point2):
        sum = 0
        for i in range(0,3):
            sum = sum + (point1[i]-point2[i])*(point1[i]-point2[i])
        return math.sqrt(sum)

    def get_target(self, Mask, colour):

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
                    cv2.circle(self.color_image, (int(x), int(y)), int(radius), colour[0], 2)
                    # Get the moments to calculate the center of the contour
                    M = cv2.moments(cnt)
                    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

                    x, y = center
                    mask = self.depth_image[y - 3:y + 3, x - 3:x + 3]
                    depht = np.mean(mask)

                    point = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], depht)
                    #print(point)
                    if abs(self.distance_between_points(point, (0, 0, 0))) > 30:
                        self.target_points[i].append(point)
                        self.positions.append(point)
                        self.positions_on_color.append(((x,y), depht))

                    if any([np.isnan(value) for value in point]) is False:
                        centroid = f"{int(point[0])},{int(point[1])},{int(point[2])}"
                    else:
                        centroid = "nan"

                    cv2.putText(self.color_image, centroid, center, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2, cv2.LINE_AA)

                else:
                    break

    def finalyzing_target(self, ):
        n = min(len(self.target_point1), len(self.target_point2), len(self.target_point3))
        #print(n)
        gem1 = [0, 0, 0]
        gem2 = [0, 0, 0]
        gem3 = [0, 0, 0]
        gem = [gem1, gem2, gem3]
        for i in range(0, n):
            for j in range(0, 3):
                #print(i , j)
                gem[j][0] = gem[j][0] + self.target_points[j][i][0]
                gem[j][1] = gem[j][1] + self.target_points[j][i][1]
                gem[j][2] = gem[j][2] + self.target_points[j][i][2]

        for i in range(0, len(gem)):
            gem[i][0] = gem[i][0] / n
            gem[i][1] = gem[i][1] / n
            gem[i][2] = gem[i][2] / n

        self.target_points = gem

    def get_bal(self, Mask, colour):
        # why use .copy()?
        # RETR_EXTERNAL segt welke contours worden bijgehouden in dit geval alle child contours worden weggelaten
        # CHAIN_APPROX_SIMPLE zegt hoeveel punten bewaard worden voor elke contour in dit geval enkel de uiterste punten
        (cnts, _) = cv2.findContours(Mask.copy(),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) > 0:

            cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

            ((x, y), radius) = cv2.minEnclosingCircle(cnt)

            cv2.circle(self.color_image, (int(x), int(y)), int(radius), colour[0], 2)

            # Get the moments to calculate the center of the contour
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                #print(str(center))
                x,y = center

                mask = self.depth_image[y -3:y + 3, x - 3:x + 3]
                #cv2.imshow("patch", mask)
                depht = np.mean(mask)

                point = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], depht)
                #print(point , depht)
                cutoff = 200
                if len(self.positions) >= 1 and self.distance_between_points(point, self.positions[-1]) > cutoff  and abs(self.distance_between_points(point, (0, 0, 0))) > 30 :
                    self.positions.append(point)
                    self.positions_on_color.append(((x,y), depht))
                elif len(self.positions) == 0:
                    self.positions.append(point)
                    self.positions_on_color.append(((x,y), depht))

                if any([np.isnan(value) for value in point]) is False and abs(self.distance_between_points(point, (0, 0, 0))) > 30:
                    centroid = f"{int(point[0])},{int(point[1])},{int(point[2])}"
                else:
                    centroid = "nan"

                #cv2.putText(self.color_image, centroid, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                #            0.5, (255, 255, 255), 2, cv2.LINE_AA)

    def round_up(self, n, decimals=0):
        multiplier = 10 ** decimals
        return math.ceil(n * multiplier) / multiplier

    def vect_AB(self, p0, p1):

        x0, y0, z0 = p0
        x1, y1, z1 = p1

        return [x1 - x0, y1 - y0, z1 - z0]

    def cross_product(self, u, v):

        ux, uy, uz = u
        vx, vy, vz = v

        uXv = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]

        return uXv

    def plot_poitions(self):
        self.plt3d = plt.figure().gca(projection='3d')

        plt.xlabel("X axis")
        plt.ylabel("Y axis")

        e = []
        e0 = []
        for i in range(0, 100, 1):
            e.append(i)
            e0.append(0)
        self.plt3d.plot(e, e0, e0, color="black")
        self.plt3d.plot(e0, e, e0, color="black")
        self.plt3d.plot(e0, e0, e, color="black")

        x, y, z = [0], [0], [0]
        for i in range(0, len(self.positions)):
            x.append(self.positions[i][0])
            y.append(self.positions[i][1])
            z.append(self.positions[i][2])
        self.plt3d.set_xlim3d(min(x)-1, max(x)+1)
        self.plt3d.set_ylim3d(min(y)-1, max(y)+1)
        self.plt3d.set_zlim3d(min(z)-1, max(z)+1)

        print("plot")
        print(self.positions)
        self.plt3d.scatter3D(x, y, z, color="green", alpha=1)
        self.draw_positions_on_color_image()

        cv2.destroyAllWindows()
        self.draw_positions_on_color_image()
        cv2.imshow("positions", self.color_image)
        plt.show(block=True)

    def quadratic_constants(self, points):
        x, y, z = [], [], []
        for i in range(0, len(self.positions)):
            x.append(self.positions[i][0])
            y.append(self.positions[i][1])
            z.append(self.positions[i][2])
        print(self.positions)
        self.plt3d.scatter3D(x, y, z, color = "green", alpha= 0.2)
        #print(positions)
        y = np.array(y)
        z = np.array(z)
        q = np.polyfit(y, z, 2)

        #3 punten
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

        return a, b, c, q[0], q[1], q[2]

    def straight_constants(self):
        x, y, z = [], [], []
        for i in range(0, len(self.positions)):
            x.append(self.positions[i][0])
            y.append(self.positions[i][1])
            z.append(self.positions[i][2])
        #print(self.positions)
        self.plt3d.scatter3D(x, y, z, color="green", alpha=0.2)
        # print(positions)
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        q = np.polyfit(y, z, 1)
        a, b = q
        p = np.polyfit(y, x, 1)
        c, d = p
        return a, b, c, d

    def afstand_punt_vlak(self, normal, d, point):
        # afstand = | ap + bq + cr + d | / √(a2 + b2 + c2).
        teller = 0
        noemer = 0
        for i in range(0,len(normal)):
            teller = teller + normal[i] * point[i]
            noemer = noemer + point[i] * point[i]
        teller = teller + d

        return abs(teller)/math.sqrt(noemer)

    def Newton(self, normal0, d0, normal1, d1, qc, collor):
        X = np.array([[0, 0, 0]])
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
        #kwadratisch
        # f2 = normal1[0]*x + normal1[1]*y + normal1[2] * z - d1
        # dxf2 = normal1[0]
        # dyf2 = normal1[1]
        # dzf2 = normal1[2]
        # f3 = qc[0] * x * x + qc[1] * x + qc[2] - y
        # dxf3 = 2*qc[0] * x + qc[1]
        # dyf3 = -1
        # dzf3 = 0
        #rechte
        # f2 = qc[2]* y +qc[3] - x
        # dxf3 = -1
        # dyf3 = qc[2]
        # dzf3 = 0
        # f3 = qc[0] * y + qc[1] - z
        # dxf3 = 0
        # dyf3 = qc[0]
        # dzf3 = -1

        iterations_store = []
        error_store = []
        self.plt3d.scatter3D(X[0][0], X[1][0], X[2][0], color=collor, marker='o', )
        i = 0
        n = 30

        if self.speltype == 2:
            while i < n:
                if i > 1:
                    self.plt3d.scatter3D(X[0][0], X[1][0], X[2][0], color=collor, marker='.', alpha= 0.3)
                i = i + 1
                iterations_store.append(i)
                F = []
                F = np.array([[normal0[0] * X[0][0] + normal0[1] * X[1][0] + normal0[2] * X[2][0] + d0,
                               normal1[0] * X[0][0] + normal1[1] * X[1][0] + normal1[2] * X[2][0] + d1,
                               qc[0] * X[0][0] * X[0][0] + qc[1] * X[0][0] + qc[2] - X[1][0]]])
                F = F.transpose()
                error_store.append(np.linalg.norm(F))
                J = np.array([[normal0[0], normal0[1], normal0[2]],
                              [normal1[0], normal1[1], normal1[2]],
                              [2 * qc[0] * X[0][0] + qc[1], -1, 0]])
                X = X - np.linalg.inv(J).dot(F)
        elif self.speltype == 1:
            while i < n:
                if i > 1:
                    self.plt3d.scatter3D(X[0][0], X[1][0], X[2][0], color=collor, marker='.', alpha=0.3)
                i = i + 1
                iterations_store.append(i)
                F = []
                F = np.array([[normal0[0] * X[0][0] + normal0[1] * X[1][0] + normal0[2] * X[2][0] + d0,
                               qc[2] * X[1][0] + qc[3] - X[0][0],
                               qc[0] * X[1][0] + qc[1] - X[2][0]]])
                F = F.transpose()
                error_store.append(np.linalg.norm(F))
                J = np.array([[normal0[0], normal0[1], normal0[2]],
                              [-1, qc[2], 0],
                              [0, qc[0], -1]])
                X = X - np.linalg.inv(J).dot(F)

        #print("Newton :")
        self.X = X
        #print(X)
        #plt.figure()
        #plt.legend("newton rapsody for calculating intersection")
        #plt.xlabel("itarations")
        #plt.ylabel("error")
        #plt.plot(iterations_store, error_store)
        return X

        #return x,y,z

    def planes_quadratic_intersect(self, target_points, trajectory_points):

        self.plt3d = plt.figure().gca(projection='3d')

        plt.xlabel("X axis")
        plt.ylabel("Y axis")

        e = []
        e0 = []
        for i in range(0, 100, 1):
            e.append(i)
            e0.append(0)
        self.plt3d.plot(e, e0, e0, color="black")
        self.plt3d.plot(e0, e, e0, color="black")
        self.plt3d.plot(e0, e0, e, color="black")

        p0, p1, p2 = target_points
        q0, q1, q2 = trajectory_points
        x_coo = [0]
        y_coo = [0]
        z_coo = [0]
        for i in target_points:
            x_coo.append(int(i[0]))
            y_coo.append(int(i[1]))
            z_coo.append(int(i[2]))
        for i in trajectory_points:
            x_coo.append(int(i[0]))
            y_coo.append(int(i[1]))
            z_coo.append(int(i[2]))

        self.plt3d.set_xlim3d(min(x_coo), max(x_coo))
        self.plt3d.set_ylim3d(min(y_coo), max(y_coo))
        self.plt3d.set_zlim3d(min(z_coo), max(z_coo))

        self.plt3d.plot_trisurf([x_coo[3], x_coo[1], x_coo[2]], [y_coo[3], y_coo[1], y_coo[2]], [z_coo[3], z_coo[1], z_coo[2]],
                           alpha=0.3)

        self.plt3d.scatter3D(x_coo, y_coo, z_coo, color="purple")

        #target equation and plane
        point0 = np.array(p0)
        normal0 = np.array(self.cross_product(self.vect_AB(p0, p1), self.vect_AB(p0, p2)))
        d0 = -point0.dot(normal0)
        #xx0, yy0 = np.meshgrid(range(1000), range(1000))
        #zz0 = (-normal0[0] * xx0 - normal0[1] * yy0 - d0) * 1. / normal0[2]
        #plt3d.plot_surface(xx0, yy0, zz0, color="gray", alpha=0.15)

        #trajection equation and plane
        point1 = np.array(q0)
        normal1 = np.array(self.cross_product(self.vect_AB(q0, q1), self.vect_AB(q0, q2)))
        d1 = -point1.dot(normal1)
        #xx1, yy1 = np.meshgrid(range(1000), range(1000))
        #zz1 = (-normal1[0] * xx1 - normal1[1] * yy1 - d1) * 1. / normal1[2]
        #lt3d.plot_surface(xx1, yy1, zz1, color="red", alpha=0.15)

        if self.speltype == 2:
            #y = ax^2 + bx + c
            qc = self.quadratic_constants(trajectory_points)
            a, b, c, g, h, l = qc

            #plotting quadratic function
            x, y, z, xp, yp, zp = [], [], [], [], [], []
            for j in range(min(x_coo), max(x_coo), 10):
                x.append(j)
                xp.append((-normal1[1] * j - normal1[2] * (g * j * j + h * j + l) - d1) * 1. / normal1[0])
                y.append(a * j * j + b * j + c)
                yp.append(j)
                z.append((-normal1[0] * j - normal1[1] * (a * j * j + b * j + c) - d1) * 1. / normal1[2])
                zp.append(g*j*j+ h*j+ l)
                #barbaarse manier om nulpunt te vinden
                #if (afstand_punt_vlak(normal0, d0, (x[-1], y[-1], z[-1])) < 1000):
                    #print(afstand_punt_vlak(normal0, d0, (x[-1], y[-1], z[-1])))
                    #print(x[-1], y[-1], z[-1])
                    #plt3d.scatter3D(x[-1], y[-1], z[-1], color="blue", marker='x', )
            self.plt3d.plot(xp, yp, zp, color="green", alpha= 0.3)

            # 3punten :
            self.intersection = self.Newton(normal0, d0, normal1, d1, qc, "red")
            # alle punten :
            self.Newton(normal0, d0, normal1, d1, (qc[3], qc[4], qc[5]), "green")

        elif self.speltype ==1:
            a, b , c, d = self.straight_constants()
            x, y, z = [], [], []
            for j in range(min(y_coo), max(y_coo), 10):
                x.append(c*j + d)
                y.append(j)
                z.append(a*j + b)
            self.intersection = self.Newton(normal0, d0, normal1, d1, (a, b, c, d), "red")

        self.plt3d.plot(x, y, z, color="red", alpha=0.3)
        self.plt3d.scatter3D(self.intersection[0][0], self.intersection[1][0], self.intersection[2][0], color="red", marker='x', )




        #cv2.destroyAllWindows()
        self.draw_positions_on_color_image()
        plt.show(block = True)

    def procces_data(self):
        self.trajectory_points = []
        #print("positions :")
        #print(positions)
        # selecting 3 points for calculation
        i = len(self.positions)
        #print("len = " + str(i))
        if i % 2 == 0:
            i = int(i/2)
            #print("i = " + str(i))
            self.trajectory_points = [self.positions[i - 1], self.positions[i], self.positions[i + 1]]
        else:
            i = int((i - 1) / 2)
            #print("i = " + str(i))
            self.trajectory_points = [self.positions[i - 1], self.positions[i], self.positions[i + 1]]

        # plotting the target plane the plane of the trow and the trajectory of the trow
        self.planes_quadratic_intersect(self.target_points, self.trajectory_points)

    def get_distace_to_intersect(self):
        #print(self.target_points)
        #print(self.intersection)
        d = []
        for i in range(0, 3):
            xmxie2 = (self.target_points[i][0] - self.intersection[0][0])*(self.target_points[i][0] - self.intersection[0][0])
            ymyie2 = (self.target_points[i][1] - self.intersection[1][0])*(self.target_points[i][1] - self.intersection[1][0])
            zmzie2 = (self.target_points[i][2] - self.intersection[2][0])*(self.target_points[i][2] - self.intersection[2][0])
            dx = math.sqrt(xmxie2+ ymyie2+ zmzie2)
            d.append(dx)
        return d

    def run_code(self, wanted_state):
        try:
            print(fg.li_cyan + "___________________________________________________________Camera___________________________________________________________")
            self.state.current_state = wanted_state

            while self.state.current_state is "getting_target" :

                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                self.depth_image = np.asanyarray(depth_frame.get_data())
                self.color_image = np.asanyarray(color_frame.get_data())

                hsv_frame = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2HSV)

                if self.calibreren:
                    plt.imshow(hsv_frame)
                    plt.show()

                # Blue color
                if self.place is "kot":
                    low_orange = np.array([1, 1, 1])
                    high_orange = np.array([170, 100, 30])
                else:
                    low_orange = np.array([1, 200, 150])
                    high_orange = np.array([15, 255, 200])
                orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)

                #blue_mask = cv2.erode(blue_mask, self.kernel, iterations=1)
                #blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, self.kernel)
                orange_mask = cv2.dilate(orange_mask, self.kernel, iterations=10)
                cv2.imshow("orange mask", orange_mask)

                self.get_target(orange_mask, self.blue)

                self.draw_instructions()
                cv2.imshow("Color Image", self.color_image)

                #functie van andreas implementeren
                #self.pcs.poll()
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    if min(len(self.target_point1), len(self.target_point2), len(self.target_point3)) > 50:
                        print("finalyzing_target")
                        self.finalyzing_target()
                        print("target points :")
                        print(self.target_points)
                        self.positions = []
                        self.positions_on_color = []
                        cv2.destroyAllWindows()
                        break
                    else:
                        print("niet genoeg punten, rustig jong het komt goed nog even wachten # = " + str(min(len(self.target_point1), len(self.target_point2), len(self.target_point3))))
                        #print("target point1 : " + str(self.target_point1))
                        #print("target point2 : " + str(self.target_point2))
                        #print("target point3 : " + str(self.target_point3))
                        #print("total : " + str(self.target_points))
                if cv2.waitKey(1) & 0xFF == ord('p'):
                    if len(self.positions) != []:
                        print("plotting positions")
                        self.plot_poitions()
                    else:
                        print("not enough points to plot trow again")
                if cv2.waitKey(1) & 0xFF == ord('f'):
                    self.target_point1 = []
                    self.target_point2 = []
                    self.target_point3 = []
                    self.target_points = [self.target_point1, self.target_point2, self.target_point3]
                    self.positions = []
                    self.positions_on_color = []
                    print("reset")

            while self.state.current_state is "getting_data":
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                self.depth_image = np.asanyarray(depth_frame.get_data())
                self.color_image = np.asanyarray(color_frame.get_data())

                hsv_frame = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2HSV)

                if self.calibreren:
                    plt.imshow(hsv_frame)
                    plt.show()

                if self.place is "kot":
                    low_orange = np.array([1, 1, 1])
                    high_orange = np.array([170, 100, 30])
                else:
                    low_orange = np.array([1, 200, 150])
                    high_orange = np.array([15, 255, 200])

                orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)
                #orange_mask = cv2.erode(orange_mask, self.kernel, iterations=1)
                #orange_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, self.kernel)
                orange_mask = cv2.dilate(orange_mask, self.kernel, iterations=10)
                # posible gousian blur for tracking

                self.get_bal(orange_mask, self.green)

                self.draw_positions_on_color_image()


                cv2.imshow("green_mask", orange_mask)
                self.draw_instructions()
                cv2.imshow("Color Image", self.color_image)

                #functie van andreas implementeren
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    if len(self.positions) > 2 and any(i != [] for i in self.target_points):
                        self.procces_data()
                        self.positions = []
                        self.positions_on_color = []
                        cv2.destroyAllWindows()
                        return self.get_distace_to_intersect()
                    elif any(i == [] for i in self.target_points):
                        print("er zijn geen target points gevonden")
                    else:
                        print("not enough points to procces trow again")
                        print(self.positions)
                        self.positions = []
                        self.positions_on_color = []
                if cv2.waitKey(1) & 0xFF == ord('p'):
                    if len(self.positions) != []:
                        print("plotting positions")
                        self.plot_poitions()
                    else:
                        print("not enough points to plot trow again")
                if cv2.waitKey(1) & 0xFF == ord('f'):
                    self.positions = []
                    self.positions_on_color = []
                    print("reset")
        finally:
            print("____________________________________________________________________________________________________________________________" + fg.rs)

    def stop_pipline(self):
        self.pipeline.stop()

#cam = CameraControl()
#cam.run_code("getting_target")
#while True:
#    print(cam.run_code("getting_data"))
#print("done")
#cam.stop_pipline()
