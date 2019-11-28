import scipy.integrate
import numdifftools as nd
import numpy as np
import matplotlib.pyplot as plt


def result(a):
    # ode for current protocol AS FUNCTION OF PARAMS a
    b = -0.1
    def f(t, x):
        return a*x + b*x**2

    # solve ode
    t = np.linspace(0, 100, 100)
    result = scipy.integrate.solve_ivp(f, (0,100), [10], t_eval=t).y[0]

    return result

df = nd.Derivative(result, n=1)
print(df(1))
exit()
