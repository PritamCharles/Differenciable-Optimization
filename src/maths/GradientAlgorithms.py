import numpy as np
import RosenbrockFunction as rf


class Gradients:
    def __init__(self, step, epsilonR, epsilon, nmax, maxIt):
        self.step_fixed = step
        self.epsilonR = epsilonR
        self.epsilon = epsilon
        self.nmax = nmax
        self.maxIt = maxIt

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

    def gradient(self, x):
        return self.calculate_gradient(x)

    def fixed_step_gradient(self, x):
        return self.step_fixed * (- self.calculate_gradient(x))

    def optimal_step_gradient(self, x, xi, di, rho_j):
        return self.get_optimal_step(xi, di, rho_j) * (- self.calculate_gradient(x))

    def fixed_step_gradient_descent(self, x0):
        iteration = 1
        xi = x0
        di = self.calculate_gradient(xi)
        while (np.linalg.norm(di) > self.epsilon) and (iteration < self.nmax):
            xi += self.fixed_step_gradient(xi)
            di = self.calculate_gradient(xi)
            iteration += 1

        return xi

    def optimal_step_gradient_descent(self, x0, rho_0):
        iteration = 1
        xi = x0
        while (np.linalg.norm(self.calculate_gradient(xi)) > self.epsilon) and (iteration < self.nmax):
            di = - self.calculate_gradient(xi)
            rho_opt = self.get_optimal_step(xi, di, rho_0)
            xi += rho_opt * di
            iteration += 1

        return xi


x0 = np.array([1, 2], dtype=float)
test = Gradients(step=10 ** -3, epsilonR=10 ** -8, epsilon=10 ** (-4), nmax = 10 ** 5, maxIt= 10 ** 4)
print(test.fixed_step_gradient_descent(x0))  # [1, 1] environ

x0 = np.array([1, 2], dtype=float)
rho_0 = 10 ** (-3)
test = Gradients(step=10 ** -3, epsilonR=10 ** -8, epsilon=10 ** (-4), nmax = 10 ** 5, maxIt=10 ** 4)
print(test.optimal_step_gradient_descent(x0, rho_0))  # [1, 1]

