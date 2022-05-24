import numpy as np


class MathematicsUtils:
    def __init__(self, epsilonR, epsilon, maxIt, nmax):
        self.epsilonR = epsilonR
        self.epsilon = epsilon
        self.maxIt = maxIt
        self.nmax = nmax

    def get_jacobian_matrix(self, X, hess):
        N = len(X)
        D = np.eye(N)

        for i in range(N):
            D[i, i] = hess(X)[i, i]

        return D

    def step_research(self, xi, di, rho_j, grad, hess):
        iteration = 1
        rho_jmoins1 = 10 * rho_j

        while (np.linalg.norm(rho_j - rho_jmoins1) > self.epsilonR) and (iteration < self.maxIt):
            phi_p = np.dot(di.T, grad(xi + rho_j * di))
            phi_pp = np.dot(np.dot(di.T, hess(xi + rho_j * di)), di)
            rho_jmoins1 = rho_j
            rho_j -= (phi_p / phi_pp)
            iteration += 1

        rho_opt = rho_j

        return rho_opt
