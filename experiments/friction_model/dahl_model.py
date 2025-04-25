import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

def deriv(F, t, v):
    return sigma0 * v * (1-F/Fc*np.sign(v))

def Ode_step(F0, v):
    _Fs = odeint(deriv, F0, t, args=(v,))
    _F = _Fs[1]
    return _F[0]

def euler_step(F0, v):
    dFdt = deriv(F0, 0, v)
    step_ = (1 / freq) * dFdt
    _F = F0 + step_
    return _F

def euler_step_clip(F0, v):
    dFdt = deriv(F0, 0, v)
    step_ = (1 / freq) * dFdt
    _F = F0 + step_
    _F = np.clip(_F, -1.0, 1.0)
    return _F

def RK45_step(F0, v):
    t_span = (0., 0.001)
    F0 = np.array(F0).reshape(1, )
    d_ = solve_ivp(deriv, t_span, F0, method='RK45', args=(v, ), first_step=0.001)
    _Fs = d_.y
    _F = _Fs[0][1]
    return _F

def genVQ():
    vs = []
    qs = []
    for t_ in t:
        if t_ < np.pi or t_ > 3 * np.pi:
            scale = 0.0001
            w_ = 100
            v = scale * np.sin(w_*t_)
            q = scale * -1/w_ * np.cos(w_*t_) - 0.5
        else:
            v = np.sin(w*t_)
            q = -1/w*np.cos(w*t_)
        vs.append(v)
        qs.append(q)
    return np.array(vs), np.array(qs)

if __name__ == '__main__':
    F0 = 0.0
    Fc = 1.0
    sigma0 = 100000

    freq = 1000
    dt = 1 / freq
    i = 1
    w = 2
    t = np.linspace(0, 4*np.pi, 6*freq)
    v, q = genVQ()

    Fs_Dq = []
    vFDs = []
    qprev = q[0]
    F_Dq = F0
    for i in range(len(q)):
        Dq = q[i] - qprev
        vT = v[i]
        vFD = Dq / dt
        F_Dq = euler_step_clip(F_Dq, vFD)
        Fs_Dq.append(F_Dq)
        qprev = q[i]

    veq0 = np.zeros_like(Fs_Dq)
    plt.plot(t, v, t, veq0, t, Fs_Dq)
    plt.legend(["Velocity", "Zero Velocity", "Friction Dq"])
    plt.show()
