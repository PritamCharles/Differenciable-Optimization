import src.maths.FunctionPart1 as func

fct = func.Function()
fct.curve()
fct.levels_card(xmin=-5, xmax=5, ymin=-10, ymax=10, step=0.1)
fct.plot_3D(xmin=-1, xmax=1, ymin=-1, ymax=1, step=0.1)

# Vérifications avec le vecteur x = [0, 0] (point critique)
x = [0, 0]
print("Fonction g(x1, x2) :", fct.evalFR(x))
print("Gradient de la fonction g(x1, x2) :", fct.gradFR(x))
print("Hessienne de la fonction g(x1, x2) :\n", fct.HessianFR(x))
