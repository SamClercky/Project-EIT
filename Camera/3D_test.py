from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Creating dataset


# punten kwadratische funcite


# bepalen van roos vlak

# sources:
# https://stackoverflow.com/questions/53698635/how-to-define-a-plane-with-3-points-and-plot-it-in-3d

points = points0, points1 = [[[5, 6, 10], [13, 20, 22], [100, 150, 50]], [[20, 100, 40], [60, 18, 35], [50, 33, 26]]]

plt3d = plt.figure().gca(projection='3d')
plt.xlabel("X axis")
plt.ylabel("Y axis")


for i in range(0, 2):
    print(i)
    p0, p1, p2 = points[i]
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    u = ux, uy, uz = [x1 - x0, y1 - y0, z1 - z0]
    v = vx, vy, vz = [x2 - x0, y2 - y0, z2 - z0]

    uXv = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]

    point = np.array(p0)
    normal = np.array(uXv)

    d = -point.dot(normal)

    xx, yy = np.meshgrid(range(200), range(200))

    zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    plt3d.plot_surface(xx, yy, zz, color="blue", alpha=0.15)


    if i == 1:

        z = []
        x = []
        y = []

        g = y1-y0
        k = x1*x1-x0*x0
        h = x1-x0

        a = ((y2-y0)*h - x2*g + x0*g )/(x2*x2*h - x2*k - x0*x0*h + x0*k)
        b =(g - a*k)/h
        c = y0 - a*x0*x0 - b*x0

        for j in range(0, 200):
            x.append(j)
            y.append(a * j * j + b * j + c)
            z.append((-normal[0] * j - normal[1] * (a * j * j + b * j + c) - d) * 1. / normal[2])
        plt3d.scatter3D(x, y, z, color="green", alpha= 0.3)
        plt3d.scatter3D((x0, x1, x2), (y0, y1, y2), (z0, z1, z2), color="blue")

plt.show()
