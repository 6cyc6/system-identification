import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def deriv(ydy, t, u):
    y, dy = ydy[0], ydy[1]
    ddy = np.sin(y) + u
    return [dy, ddy]

class sinEnv(object):
    def __init__(self, y0dy0=[0, 1]):
        ydy = y0dy0

y0 = [0, 1]
t = np.linspace(0, 1, 100)
ydys = odeint(deriv, y0, t, args=(0,))
ys = ydys[:, 0]
plt.plot(t, ys)
plt.show()