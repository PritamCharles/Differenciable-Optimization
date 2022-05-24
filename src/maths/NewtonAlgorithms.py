from src.maths.MathematicsUtils import MathematicsUtils
import numpy as np


class Newton:
    def __init__(self, epsilonR, epsilon, maxIt, nmax):
        self.epsilonR = epsilonR
        self.epsilon = epsilon
        self.maxIt = maxIt
        self.nmax = nmax
        self.rho_0 = 10 ** (-3)
        self.maths_utils = MathematicsUtils(self.epsilonR, self.epsilon, self.maxIt, self.nmax)

    def quasi_Newton_BFGS(self, x0, grad, hess):
        iteration = 1
        xi = x0
        Di = hess(x0)

        while (np.linalg.norm(grad(xi)) > self.epsilon) and (iteration < self.nmax):
            di = np.linalg.solve(Di, - grad(xi))
            xi_old = xi
            xi = xi + self.maths_utils.step_research(xi, di, self.rho_0, grad, hess) * di
            yi = grad(xi) - grad(xi_old)
            si = self.maths_utils.step_research(xi, di, self.rho_0, grad, hess) * di

            mult_yi_yiT = np.dot(yi, np.transpose(yi))
            mult_yiT_si = np.dot(np.transpose(yi), si)
            mult_Di_si = np.dot(Di, si)
            mult_siT_Di = np.dot(np.transpose(si), Di)
            Di += ((mult_yi_yiT) / (mult_yiT_si)) - ((np.dot(mult_Di_si, mult_siT_Di)) / (np.dot(mult_siT_Di, si)))

            iteration += 1

        return xi

    def Newton(self, x0, grad, hess):
        iteration = 1
        xi = x0

        while (np.linalg.norm(grad(xi)) > self.epsilon) and (iteration < self.nmax):
            Di = hess(xi)
            di = np.linalg.solve(Di, - grad(xi))
            xi = xi + self.maths_utils.step_research(xi, di, self.rho_0, grad, hess) * di
            iteration += 1

            print(xi)

        return xi
