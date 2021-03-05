from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# Creating dataset


# punten kwadratische funcite


# bepalen van roos vlak

# sources:
# https://stackoverflow.com/questions/53698635/how-to-define-a-plane-with-3-points-and-plot-it-in-3d


target_points = [[0, -20, 100], [0, 20, -150], [-60, 150, 50]]
trajectory_points = [[150, 222, 300], [-10, 40, 35], [50, -600, 26]]

plt3d = plt.figure().gca(projection='3d')
plt.xlabel("X axis")
plt.ylabel("Y axis")

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


def Newton(normal0, d0, normal1, d1, qc):
    print("Newton")
    x = 1
    y = 1
    z = 1
    f1 = normal0[0]*x + normal0[2]*y + normal0[2] * z - d0
    dxf1 = normal0[0]
    dyf1 = normal0[1]
    dzf1 = normal0[2]
    f2 = normal1[0]*x + normal1[2]*y + normal1[2] * z - d1
    dxf1 = normal1[0]
    dyf1 = normal1[1]
    dzf1 = normal1[2]
    f3 = qc[0] * x * x + qc[1] * x + qc[2] - y
    dxf3 = 2*qc[0] * x + qc[1]
    dyf3 = -1
    dzf3 = 0

    #return x,y,z


def planes_quadratic_intersect(target_points, trajectory_points):

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
    qc = quadratic_constants(trajectory_points)
    a, b, c = qc

    point1 = np.array(q0)
    normal1 = np.array(cross_product(vect_AB(q0, q1), vect_AB(q0, q2)))

    d1 = -point1.dot(normal1)

    Newton(normal0, d0, normal1, d1, qc)

    #plotting quadratic function
    z = []
    x = []
    y = []
    for j in range(min(x_coo)-100, max(x_coo)+100 , 1):
        x.append(j)
        y.append(a * j * j + b * j + c)
        z.append((-normal1[0] * j - normal1[1] * (a * j * j + b * j + c) - d1) * 1. / normal1[2])
        if (afstand_punt_vlak(normal0, d0, (x[-1], y[-1], z[-1])) < 1000):
            print(afstand_punt_vlak(normal0, d0, (x[-1], y[-1], z[-1])))
            print(x[-1], y[-1], z[-1])
            plt3d.scatter3D(x[-1], y[-1], z[-1], color="blue", marker='x', )

    plt3d.scatter3D(x, y, z, color="red", alpha= 0.03)
    plt.show()


planes_quadratic_intersect(target_points, trajectory_points)

plt.show()
