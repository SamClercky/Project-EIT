from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from sty import fg

print(fg.li_cyan + "tsdlqfkjqodfji" + fg.rs)
# Creating dataset


# punten kwadratische funcite


# bepalen van roos vlak

# sources:
# https://stackoverflow.com/questions/53698635/how-to-define-a-plane-with-3-points-and-plot-it-in-3d


target_points = [[0, -20, 100], [0, 20, 500], [-60, 150, 50]]
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
    x, y = [], []
    for i in range(0, len(points)):
        x.append(points[i][0])
        y.append(points[i][1])

    x = np.array(x)
    y = np.array(y)
    z = np.polyfit(x, y, 2)
    print("polyfit : ")
    print(z)
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

    print(a, b, c)
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
    global X
    X = np.array([[-1000, 1000, 1000]])
    X = X.transpose()
    #print(X)
    #print(X[0][0])
    #print(X[1][0])
    #print(X[2][0])

    #X[0][0] = x
    #X[0][1] = y
    #X[0][2] = z
    #f1 = normal0[0]*x + normal0[1]*y + normal0[2] * z - d0
    #dxf1 = normal0[0]
    #dyf1 = normal0[1]
    #dzf1 = normal0[2]
    #f2 = normal1[0]*x + normal1[1]*y + normal1[2] * z - d1
    #dxf2 = normal1[0]
    #dyf2 = normal1[1]
    #dzf2 = normal1[2]
    #f3 = qc[0] * x * x + qc[1] * x + qc[2] - y
    #dxf3 = 2*qc[0] * x + qc[1]
    #dyf3 = -1
    #dzf3 = 0

    iterations_store = []
    error_store = []
    plt3d.scatter3D(X[0][0], X[1][0], X[2][0], color="blue", marker='o', )

    i = 0
    n = 30
    while i < n:
        #print(i)
        #print(X)
        if i > 1:
            plt3d.scatter3D(X[0][0], X[1][0], X[2][0], color="blue", marker='.', alpha= (1/n)*i)
        i = i+1
        iterations_store.append(i)
        F = []
        F = np.array([[normal0[0]*X[0][0] + normal0[1]*X[1][0] + normal0[2] * X[2][0] + d0,
                       normal1[0]*X[0][0] + normal1[1]*X[1][0] + normal1[2] * X[2][0] + d1,
                       qc[0] * X[0][0] * X[0][0] + qc[1] * X[0][0] + qc[2] - X[1][0]]])
        F = F.transpose()
        error_store.append(np.linalg.norm(F))
        J = np.array([[normal0[0], normal0[1], normal0[2]],
                      [normal1[0], normal1[1], normal1[2]],
                      [2*qc[0] * X[0][0] + qc[1], -1, 0]])
        X = X - np.linalg.inv(J).dot(F)

    print(X)
    plt.figure()
    plt.legend("newton rapsody for calculating intersection")
    plt.xlabel("itarations")
    plt.ylabel("error")
    plt.plot(iterations_store, error_store)
    plt3d.scatter3D(X[0][0], X[1][0], X[2][0], color="blue", marker='x', )


def planes_quadratic_intersect(target_points, trajectory_points):
    #plotting axis
    e = []
    e0 = []
    for i in range(0,100,1):
        e.append(i)
        e0.append(0)
    plt3d.plot(e, e0, e0, color = "black")
    plt3d.plot(e0, e, e0, color = "black")
    plt3d.plot(e0, e0, e, color = "black")


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

    plt3d.set_xlim3d(min(x_coo), max(x_coo))
    plt3d.set_ylim3d(min(y_coo), max(y_coo))
    plt3d.set_zlim3d(min(z_coo), max(z_coo))

    plt3d.plot_trisurf([x_coo[0], x_coo[1], x_coo[2]], [y_coo[0], y_coo[1], y_coo[2]], [z_coo[0], z_coo[1], z_coo[2]], alpha= 0.3)
    plt3d.plot_trisurf([x_coo[3], x_coo[4], x_coo[5]], [y_coo[3], y_coo[4], y_coo[5]], [z_coo[3], z_coo[4], z_coo[5]], color = "red",  alpha= 0.3)

    plt3d.scatter3D(x_coo, y_coo, z_coo, color="purple")

    #target equation and plane
    point0 = np.array(p0)
    normal0 = np.array(cross_product(vect_AB(p0, p1), vect_AB(p0, p2)))
    d0 = -point0.dot(normal0)
    #xx0, yy0 = np.meshgrid(range(max(x_coo)+1000), range(max(y_coo)+1000))
    #zz0 = (-normal0[0] * xx0 - normal0[1] * yy0 - d0) * 1. / normal0[2]
    #plt3d.plot_surface(xx0, yy0, zz0, color="gray", alpha=0.15)

    # y = ax^2 + bx + c
    qc = quadratic_constants(trajectory_points)
    a, b, c = qc

    #trajectory equation and plane
    point1 = np.array(q0)
    normal1 = np.array(cross_product(vect_AB(q0, q1), vect_AB(q0, q2)))
    d1 = -point1.dot(normal1)
    #xx1, yy1 = np.meshgrid(range(max(x_coo)+1000), range(max(y_coo)+1000))
    #zz1 = (-normal1[0] * xx1 - normal1[1] * yy1 - d1) * 1. / normal1[2]
    #plt3d.plot_surface(xx1, yy1, zz1, color="red", alpha=0.15)

    Newton(normal0, d0, normal1, d1, qc, )

    #plotting quadratic function
    z = []
    x = []
    y = []
    print(min(x_coo)-10, max(x_coo)+10)
    for j in range(min(x_coo)-10, max(x_coo)+10 , 1):
        x.append(j)
        y.append(a * j * j + b * j + c)
        z.append((-normal1[0] * j - normal1[1] * (a * j * j + b * j + c) - d1) * 1. / normal1[2])
        #if (afstand_punt_vlak(normal0, d0, (x[-1], y[-1], z[-1])) < 5000):
            #print(afstand_punt_vlak(normal0, d0, (x[-1], y[-1], z[-1])))
            #print(x[-1], y[-1], z[-1])
            #plt3d.scatter3D(x[-1], y[-1], z[-1], color="green", marker='x', )


    plt3d.plot(x, y, z, color="red")


planes_quadratic_intersect(target_points, trajectory_points)

plt.show()
