import numpy as np
import itertools
from matplotlib import pyplot as plt


# Définition de la fonction de Rosenbrock
def Rosenbrock_function(x1, x2):
    return (x1 - 1) ** 2 + 10 * (x1 ** 2 - x2) ** 2


# Question q
X = np.linspace(-5, 5, 100)
Y = np.linspace(-10, 10, 200)
Xv, Yv = np.meshgrid(X, Y)

values_Rosenbrock = []
for i, j in itertools.zip_longest(X, Y, fillvalue=""):
    try:
        value = Rosenbrock_function(i, j)
        values_Rosenbrock.append(value)
    except TypeError:
        pass

plt.figure(figsize=(15, 9))
plt.plot(X, values_Rosenbrock, color="blue", label="Rosenbrock function")
plt.title("Rosenbrock function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()


#####
# Question p et r
# graph1 - Fenêtre de tracé large
X = np.linspace(-5, 5, 100)
Y = np.linspace(-30, 40, 700)
Xv, Yv = np.meshgrid(X, Y)
Zv = Rosenbrock_function(Xv, Yv)

plt.figure(figsize=(15, 9))
h = plt.contour(Xv, Yv, Zv, [i for i in range(1, 6001, 50)])
# plt.clabel(h, inline=1, fontsize=10, fmt='%3.2f')
plt.title("Carte de niveaux de la fonction de Rosenbrock")
plt.legend()
plt.grid()
plt.show()

# graph2 - Fenêtre de tracé plus proche de l'origine
X = np.linspace(-5, 5, 100)
Y = np.linspace(-10, 10, 200)
Xv, Yv = np.meshgrid(X, Y)
Zv = Rosenbrock_function(Xv, Yv)

plt.figure(figsize=(15, 9))
h = plt.contour(Xv, Yv, Zv, [i for i in range(1, 6001, 50)])
# plt.clabel(h, inline=1, fontsize=10, fmt='%3.2f')
plt.title("Carte de niveaux de la fonction de Rosenbrock")
plt.legend()
plt.grid()
plt.show()


#####
# Question s
# Le point [1, 1].T est un minimum global.


#####
# Question z
def evalFR(x):
    return (x[0] - 1) ** 2 + 10 * (x[0] ** 2 - x[1]) ** 2


def gradFR(x):
    return np.array([2 * (x[0] - 1) + 40 * ((x[0] ** 2) - x[1]) * x[0], -20 * (x[0] ** 2 - x[1])])


def HessianFR(x):
    return np.array([[2 + 120 * x[0] - 40 * x[1], -40 * x[0]], [-40 * x[0], 20 * x[1]]])


# Vérifications avec le vecteur x = [1, 1]
x = [1, 1]
print("Fonction de Rosenbrock :", evalFR(x))
print("Gradient de la fonction de Rosenbrock :", gradFR(x))
print("Hessienne de la fonction de Rosenbrock :\n", HessianFR(x))
