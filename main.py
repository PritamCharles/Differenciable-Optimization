from src.maths.FunctionPart1 import Function
from src.maths.RosenbrockFunction import RosebrockFunction
from src.maths.GradientAlgorithms import Gradients
from src.maths.PreconditioningAlgorithms import Preconditioning
from src.maths.NewtonAlgorithms import Newton
import numpy as np

### Fonction de Rosenbrock ###
rosenbrock = RosebrockFunction()
rosenbrock.curve()
rosenbrock.levels_card(xmin=-5, xmax=5, ymin=-10, ymax=10, step=0.1)
rosenbrock.levels_card(xmin=-5, xmax=5, ymin=-30, ymax=40, step=0.1)
rosenbrock.plot_3D(xmin=-1, xmax=1, ymin=-1, ymax=1, step=0.1)

# Vérifications avec le vecteur x = [1, 1]
x = [1, 1]
print("Fonction de Rosenbrock :", rosenbrock.evalFR(x))
print("Gradient de la fonction de Rosenbrock :", rosenbrock.gradFR(x))
print("Hessienne de la fonction de Rosenbrock :\n", rosenbrock.HessianFR(x))

# Question s : Le point [1, 1].T est un minimum global.

###

### Fonction de la partie 1 ###
fct = Function()
fct.curve()
fct.levels_card(xmin=-5, xmax=5, ymin=-10, ymax=10, step=0.1)
fct.plot_3D(xmin=-1, xmax=1, ymin=-1, ymax=1, step=0.1)

# Vérifications avec le vecteur x = [0, 0] (point critique)
x = [0, 0]
print("Fonction g(x1, x2) :", fct.evalFR(x))
print("Gradient de la fonction g(x1, x2) :", fct.gradFR(x))
print("Hessienne de la fonction g(x1, x2) :\n", fct.HessianFR(x))

###

### Méthodes du gradient ###
### Pas fixe ###
x0 = np.array([1, 2], dtype=float)
test = Gradients(step=10 ** -3, epsilonR=10 ** -8, epsilon=10 ** (-4), nmax=10 ** 5, maxIt=10 ** 4)
print(test.fixed_step_gradient_descent(x0, rosenbrock.gradFR))  # [1, 1] environ

### Pas optimal ###
x0 = np.array([1, 2], dtype=float)
rho_0 = 10 ** (-3)
test = Gradients(step=10 ** -3, epsilonR=10 ** -8, epsilon=10 ** (-4), nmax=10 ** 5, maxIt=10 ** 4)
print(test.optimal_step_gradient_descent(x0, rho_0, rosenbrock.gradFR, rosenbrock.HessianFR))  # [1, 1]

###

### Méthode du gradient à pas optimal préconditionné ###
x0 = np.array([1, 0.5], dtype=float)
test = Preconditioning(epsilonR=10 ** -8, epsilon=10 ** (-4), maxIt=10 ** 4, nmax=10 ** 5)
print(test.optimal_step_gradient_preconditionned(x0, rosenbrock.gradFR, rosenbrock.HessianFR))  # [1, 1] environ

###

### Méthodes de Newton et Quasi-Newton BFGS ###
# Quasi Newton
x0 = np.array([1, 0.5], dtype=float)
test = Newton(epsilonR=10 ** -8, epsilon=10 ** (-2), maxIt=10 ** 4, nmax=100)
print(test.quasi_Newton_BFGS(x0, rosenbrock.gradFR, rosenbrock.HessianFR))

# Newton
x0 = np.array([1, 0.5], dtype=float)
test = Newton(epsilonR=10 ** -8, epsilon=10 ** (-2), maxIt=10 ** 4, nmax=10)
print(test.Newton(x0, rosenbrock.gradFR, rosenbrock.HessianFR))
