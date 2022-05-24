from src.maths.MathematicsUtils import MathematicsUtils
import numpy as np


class Preconditioning:
    def __init__(self, epsilonR, epsilon, maxIt, nmax):
        self.epsilonR = epsilonR
        self.epsilon = epsilon
        self.maxIt = maxIt
        self.nmax = nmax
        self.rho_0 = 10 ** (-3)
        self.maths_utils = MathematicsUtils(self.epsilonR, self.epsilon, self.maxIt, self.nmax)

    def optimal_step_gradient_preconditionned(self, x0, grad, hess):
        iteration = 1
        xi = x0

        while (np.linalg.norm(grad(xi)) > self.epsilon) and (iteration < self.nmax):
            Di = self.maths_utils.get_jacobian_matrix(xi, hess)
            di = np.linalg.solve(Di, - grad(xi))
            xi += self.maths_utils.step_research(xi, di, self.rho_0, grad, hess) * di
            iteration += 1

        return xi
