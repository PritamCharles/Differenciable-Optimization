import numpy as np
from MathematicsUtils import MathematicsUtils
from sympy import symbols, cos, diff

class RandomFunction():
    def __init__(self, epsilonR, epsilon, maxIt, nmax):
        self.epsilonR = epsilonR
        self.epsilon = epsilon
        self.maxIt = maxIt
        self.nmax = nmax
        self.rho_0 = 10 ** (-3)
        self.maths_utils = MathematicsUtils(self.epsilonR, self.epsilon, self.maxIt, self.nmax)

    def function_definition(self, x1, x2):
        return (x1 - 1) ** 2 + 2 * x2 ** 2

    def get_hessian