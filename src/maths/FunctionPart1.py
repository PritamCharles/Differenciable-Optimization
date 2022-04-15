import numpy as np
import itertools
from matplotlib import pyplot as plt


class Function:
    # Définition de la fonction de Rosenbrock
    def g_function(self, x1, x2):
        return x1 ** 2 + x2 ** 2 * (x2 ** 2 + 2)

    # Question q
    def curve(self):
        X = np.linspace(-5, 5, 100)
        Y = np.linspace(-10, 10, 200)
        Xv, Yv = np.meshgrid(X, Y)

        values_g = []
        for i, j in itertools.zip_longest(X, Y, fillvalue=""):
            try:
                value = self.g_function(i, j)
                values_g.append(value)
            except TypeError:
                pass

        plt.figure(figsize=(15, 9))
        plt.plot(X, values_g, color="blue")
        plt.title("Function")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.show()

    # Question p et r
    def levels_card(self, xmin, xmax, ymin, ymax, step):
        subdivisions_x = int((xmax - xmin) / step)
        subdivisions_y = int((ymax - ymin) / step)

        X = np.linspace(xmin, xmax, subdivisions_x)
        Y = np.linspace(ymin, ymax, subdivisions_y)
        Xv, Yv = np.meshgrid(X, Y)
        Zv = self.g_function(Xv, Yv)

        plt.figure(figsize=(15, 9))
        h = plt.contour(Xv, Yv, Zv, [i for i in range(1, 6001, 50)])
        # plt.clabel(h, inline=1, fontsize=10, fmt='%3.2f')
        plt.title("Carte de niveaux de la fonction de Rosenbrock")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_3D(self, xmin, xmax, ymin, ymax, step):
        subdivisions_x = int((xmax - xmin) / step)
        subdivisions_y = int((ymax - ymin) / step)

        X = np.linspace(xmin, xmax, subdivisions_x)
        Y = np.linspace(ymin, ymax, subdivisions_y)
        Xv, Yv = np.meshgrid(X, Y)
        Zv = self.g_function(Xv, Yv)

        fig = plt.figure(figsize=(15, 9))
        ax = fig.gca(projection='3d')
        proj_3D = ax.plot_surface(Xv, Yv, Zv, cmap=plt.cm.coolwarm, linewidth=2, antialiased=False)
        ax.zaxis.set_major_locator(plt.LinearLocator(5))
        ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))
        fig.colorbar(proj_3D, shrink=0.5, aspect=5)
        plt.show()

    # Question z
    def evalFR(self, x):
        return x[0] ** 2 + x[1] ** 2 * (x[1] ** 2 + 2)

    def gradFR(self, x):
        return np.array([2 * x[0], -20 * (4 * x[1] ** 3 + 4 * x[1])])

    def HessianFR(self, x):
        return np.array([[2, 0], [0, 12 * x[1] ** 2 + 4]])
