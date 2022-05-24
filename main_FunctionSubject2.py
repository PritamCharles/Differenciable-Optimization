from src.maths.GradientAlgorithms import Gradients
from src.maths.PreconditioningAlgorithms import Preconditioning
from src.maths.NewtonAlgorithms import Newton
from src.maths.Function_Subject2 import GFunction
import numpy as np

### PROJET ###
### Fonction g(x1, x2)
func = GFunction()
func.levels_card(xmin=-5, xmax=5, ymin=-10, ymax=10, step=0.1)
#func.plot_3D(xmin=-5, xmax=5, ymin=-5, ymax=5, step=0.1)

### Méthodes du gradient ###
### Pas fixe ###
x0 = np.array([4, -2], dtype=float)
test = Gradients(step=10 ** -3, epsilonR=10 ** -8, epsilon=10 ** (-4), nmax=10 ** 5, maxIt=10 ** 4)
print(test.fixed_step_gradient_descent(x0, func.gradG))

### Pas optimal ###
x0 = np.array([4, -2], dtype=float)
rho_0 = 10 ** (-3)
test = Gradients(step=10 ** -3, epsilonR=10 ** -8, epsilon=10 ** (-4), nmax=10 ** 5, maxIt=10 ** 4)
print(test.optimal_step_gradient_descent(x0, rho_0, func.gradG, func.HessianG))

###

### Méthode du gradient à pas optimal préconditionné ###
x0 = np.array([-1, -5], dtype=float)
test = Preconditioning(epsilonR=10 ** -8, epsilon=10 ** (-4), maxIt=10 ** 4, nmax=10 ** 5)
print(test.optimal_step_gradient_preconditionned(x0, func.gradG, func.HessianG))

###

### Méthodes de Newton et Quasi-Newton BFGS ###
# Quasi Newton
x0 = np.array([1, 0.5], dtype=float)
test = Newton(epsilonR=10 ** -8, epsilon=10 ** (-2), maxIt=10 ** 4, nmax=100)
print(test.quasi_Newton_BFGS(x0, func.gradG, func.HessianG))

# Newton
x0 = np.array([1, 0.5], dtype=float)
test = Newton(epsilonR=10 ** -8, epsilon=10 ** (-2), maxIt=10 ** 4, nmax=10)
print(test.Newton(x0, func.gradG, func.HessianG))