import numpy as np

class Gradients:
    def __init__(self, step, epsilon, Nitermax):
        self.step_fixed = step
        self.epsilon = epsilon
        self.Nitermax = Nitermax

    def model(self, x):
        return np.gradient(x)

    def fixed_step_gradient(self, x):
        return self.step_fixed * (- self.model(x))

    def fixed_step_gradient_descent(self, x):
        iteration = 0
        xi = x
        di = self.model(xi)
        while (np.linalg.norm(di) > self.epsilon) and (iteration < self.Nitermax):
            xi += self.fixed_step_gradient(xi)
            di = self.model(xi)
            iteration += 1

        return xi


x0 = np.array([4, 1], dtype=float)
test = Gradients(step=10 ** -3, epsilon=10 ** -8, Nitermax=10 ** 4)
print(test.fixed_step_gradient_descent(x0))
