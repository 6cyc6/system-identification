import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def g(v):
    Fc = 1.0
    Fs = 1.5
    vs = 0.001
    sigma0 = 100000
    return (Fc + (Fs - Fc) * np.exp(-(v/vs)**2)) / sigma0

def deriv(z, t, v):
    dzdt = v - np.abs(v) / g(v) * z
    return dzdt

def ode_step(z0, v):
    zs = odeint(deriv, z0, t, args=(v,))
    z_ = zs[1]
    dzdt = deriv(z_, 0, v)
    F_ = sigma0 * z_ + sigma1 * dzdt
    return z_, F_

def euler_step(z0, v):
    dzdt = deriv(z0, 0, v)
    dt = 1 / freq
    z_ = z0 + dt * dzdt
    F_ = sigma0 * z_ + sigma1 * dzdt
    return z_, F_

def vFun():
    vs = []
    for t_ in t:
        if t_ < np.pi or t_ > 3 * np.pi:
            v = 0.0001 * np.sin(100*t_)
        else:
            v = np.sin(w*t_)
        vs.append(v)
    return np.array(vs)

if __name__ == '__main__':
    freq = 100
    sigma0 = 10 ** 5
    sigma1 = np.sqrt(sigma0)
    sigma2 = 0.4
    w = 2

    t = np.linspace(0, 4*np.pi, 6*freq)
    v = vFun()
    z0 = 0.0
    z = z0
    Fs = []
    for v_ in v:
        z, F = ode_step(z, v_)
        Fs.append(F)
    veq0 = np.zeros_like(Fs)
    plt.plot(t, Fs, t, v, t, veq0)
    plt.legend(["Force", "Velocity", "Zero Velocity"])
    plt.show()
