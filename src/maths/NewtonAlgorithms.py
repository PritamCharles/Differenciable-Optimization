import numpy as np
import RosenbrockFunction as rf


class Newton:
    def __init__(self, epsilonR, epsilon, maxIt, nmax):
        self.epsilonR = epsilonR
        self.epsilon = epsilon
        self.maxIt = maxIt
        self.nmax = nmax
        self.rho_0 = 10 ** (-3)
        self.rosenbrock = rf.RosebrockFunction()

    def calculate_gradient(self, X):
        N = len(X)
        A = 2 * np.eye(N)

        for i in range(0, N - 1):
            A[i, i + 1] = -1
            A[i + 1, i] = -1

        b = np.ones(N)

        return A @ X - b

    def calculate_hessian(self, X):
        N = len(X)
        A = 2 * np.eye(N)

        for i in range(0, N - 1):
            A[i, i + 1] = -1
            A[i + 1, i] = -1

        return A

    def get_jacobian_matrix(self, X):
        N = len(X)
        D = np.eye(N)

        for i in range(N):
            D[i, i] = self.rosenbrock.HessianFR(X)[i, i]

        return D

    def get_optimal_step(self, xi, di, rho_j):
        iteration = 1
        rho_jmoins1 = 10 * rho_j

        while (np.linalg.norm(rho_j - rho_jmoins1) > self.epsilonR) and (iteration < self.maxIt):
            phi_p = np.dot(di.T, self.rosenbrock.gradFR(xi + rho_j * di))
            phi_pp = np.dot(np.dot(di.T, self.rosenbrock.HessianFR(xi + rho_j * di)), di)
            rho_jmoins1 = rho_j
            rho_j -= (phi_p / phi_pp)
            iteration += 1

        rho_opt = rho_j

        return rho_opt

    def quasi_Newton_BFGS(self, x0):
        iteration = 1
        xi = x0
        N = len(xi)
        Di = self.rosenbrock.HessianFR(x0)

        while (np.linalg.norm(self.rosenbrock.gradFR(xi)) > self.epsilon) and (iteration < self.nmax):
            di = np.linalg.solve(Di, - self.rosenbrock.gradFR(xi))
            xi_ancien = xi
            xi = xi + self.get_optimal_step(xi, di, self.rho_0) * di
            yi = self.rosenbrock.gradFR(xi) - self.rosenbrock.gradFR(xi_ancien)
            si = self.get_optimal_step(xi, di, self.rho_0) * di

            mult_yi_yiT = np.dot(yi, np.transpose(yi))
            mult_yiT_si = np.dot(np.transpose(yi), si)
            mult_Di_si = np.dot(Di, si)
            mult_siT_Di = np.dot(np.transpose(si), Di)
            Dipp = Di + ((mult_yi_yiT) / (mult_yiT_si)) - ((np.dot(mult_Di_si, mult_siT_Di)) / (np.dot(mult_siT_Di, si)))
            Di = Dipp

            iteration += 1
            print(iteration)

        return xi


x0 = np.array([1, 0.5], dtype=float)
test = Newton(epsilonR=10 ** -8, epsilon=10 ** (-4), maxIt=10 ** 4, nmax=1000)
print(test.quasi_Newton_BFGS(x0))  # [1, 1] environ