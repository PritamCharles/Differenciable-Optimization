from src.maths.MathematicsUtils import MathematicsUtils
import numpy as np


class Gradients:
    def __init__(self, step, epsilonR, epsilon, nmax, maxIt):
        self.step_fixed = step
        self.epsilonR = epsilonR
        self.epsilon = epsilon
        self.nmax = nmax
        self.maxIt = maxIt
        self.maths_utils = MathematicsUtils(self.epsilonR, self.epsilon, self.maxIt, self.nmax)

    def gradient(self, x, grad):
        return grad(x)

    def fixed_step_gradient(self, x, grad):
        return self.step_fixed * (- grad(x))

    def optimal_step_gradient(self, x, xi, di, rho_j, grad, hess):
        return self.maths_utils.step_research(xi, di, rho_j, grad, hess) * (- grad(x))

    def fixed_step_gradient_descent(self, x0, grad):
        iteration = 1
        xi = x0
        di = grad(xi)
        while (np.linalg.norm(di) > self.epsilon) and (iteration < self.nmax):
            xi += self.fixed_step_gradient(xi, grad)
            di = grad(xi)
            iteration += 1
        print("Iterations :", iteration)

        return xi

    def optimal_step_gradient_descent(self, x0, rho_0, grad, hess):
        iteration = 1
        xi = x0
        while (np.linalg.norm(grad(xi)) > self.epsilon) and (iteration < self.nmax):
            di = - grad(xi)
            rho_opt = self.maths_utils.step_research(xi, di, rho_0, grad, hess)
            xi += rho_opt * di
            iteration += 1
        print("Iterations :", iteration)

        return xi
