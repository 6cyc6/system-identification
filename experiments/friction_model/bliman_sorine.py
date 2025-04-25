import numpy as np
import matplotlib.pyplot as plt

def dxs_ds(xs, v, a, b):
    vs = np.sign(v)
    dxs_ds = a * xs + b * vs
    return dxs_ds

def intode(s, num_samples):
    xs1 = xs2 = 0
    h = s / num_samples
    v = 0.00001
    F = []
    xs1s = []
    xs2s = []
    for i in range(num_samples):
        F.append(xs1+xs2)
        xs1s.append(xs1)
        xs2s.append(xs2)
        dxs1_ds = dxs_ds(xs1, v, a1, b1)
        dxs2_ds = dxs_ds(xs2, v, a2, b2)
        dxs1 = h * dxs1_ds
        dxs2 = h * dxs2_ds
        xs1 += dxs1
        xs2 += dxs2
    s = np.linspace(0, s, num_samples)
    return F, xs1s, xs2s, s

a1, b1 = -1/0.01, 30/(0.01)
a2, b2 = -1/0.1, -10/(0.1)
F, xs1s, xs2s, t = intode(1000, int(1e6))
plt.plot(t, F, t, xs1s, t, xs2s)
plt.legend(["F", "fast", "slow"])
plt.title("Bliman Sorine Friction Model")
plt.show()


