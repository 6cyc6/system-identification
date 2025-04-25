import numpy as np
import matplotlib.pyplot as plt

def F_STIC(v):
    F = []
    for v_tmp in v:
        if abs(v_tmp) > v0 and abs(v_tmp) < v_brk:
            F.append(np.sqrt(2 * np.e) * (F_brk - F_c) * np.exp(-(v_tmp / v_st) ** 2) * (v_tmp / v_st) + F_c * np.tanh(
            v_tmp / v_coul))
        elif abs(v_tmp) < v0:
            F.append(0)
        else:
            if v_tmp > 0:
                F.append(F_c)
            else:
                F.append(-F_c)
    return F

def F_COULOMB(v):
    F_ = []
    for v_tmp in v:
        if v_tmp < -v_brk:
            F_.append(-F_c)
        elif v_tmp > v_brk:
            F_.append(F_c)
        else:
            F_.append(0)
    return np.array(F_)

def vFun():
    vs = []
    for t_ in t:
        # if t_ < np.pi or t_ > 3 * np.pi:
        #     v = 0 #0.0001 * np.sin(100*t_)
        # else:
        #     v = np.sin(w*t_)
        v = np.sin(w * t_)
        vs.append(v)
    return np.array(vs)



F_brk = 3
F_c = 2
v_brk = 0.1
v_st = v_brk * np.sqrt(2)
v_coul = v_brk/10
v0 = 0.0006

w = 1
freq = 100
t = np.linspace(-np.pi, np.pi, 6 * freq)
v_ = vFun()
F_stic = F_STIC(v_)
F_coulomb = F_COULOMB(v_)
F_residual = F_stic - F_coulomb
veq0 = np.zeros_like(F_stic)
plt.rcParams['axes.axisbelow'] = True
plt.plot(t, F_stic, t, v_, t, F_coulomb, t, F_residual)
plt.legend(["F static", "Velocity", "F Coulomb", "F residual"])
plt.show()



