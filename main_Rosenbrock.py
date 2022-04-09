import src.maths.RosenbrockFunction as rf

rosenbrock = rf.RosebrockFunction()
rosenbrock.curve()
rosenbrock.levels_card(xmin=-5, xmax=5, ymin=-10, ymax=10, step=0.1)
rosenbrock.levels_card(xmin=-5, xmax=5, ymin=-30, ymax=40, step=0.1)

# Vérifications avec le vecteur x = [1, 1]
x = [1, 1]
print("Fonction de Rosenbrock :", rosenbrock.evalFR(x))
print("Gradient de la fonction de Rosenbrock :", rosenbrock.gradFR(x))
print("Hessienne de la fonction de Rosenbrock :\n", rosenbrock.HessianFR(x))


#####
# Question s
# Le point [1, 1].T est un minimum global.
