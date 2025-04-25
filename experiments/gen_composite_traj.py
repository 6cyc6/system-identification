import numpy as np
import matplotlib.pyplot as plt

def dydx(a):
    return a

def tanh(y):
    return (np.exp(y)-np.exp(-y)) / (np.exp(y)+np.exp(-y))

def dtanh(y):
    return 1-tanh(y)**2

def ddtanh(y):
    return -2*tanh(y)*dtanh(y)

def q(x, q_f, q_0, omega):
    return -np.cos(omega * x) * (q_f - q_0) / 2 + (q_f + q_0) / 2

def dq(x, q_f, q_0, omega):
    return omega * np.sin(omega * x) * (q_f - q_0) / 2

def ddq(x, q_f, q_0, omega):
    return (omega) ** 2 * np.cos(omega * x) * (q_f - q_0) / 2

def gen_tanh_traj_jointwise(t_input, i):
    qs = []
    dqs = []
    ddqs = []
    init_position = np.array([0., 0., 0., -0., 0., 0., 0.]) * np.pi / 6.
    goal_position = np.array([2., 2., 2., -2., 2., 2, 2.]) * np.pi / 6.
    t_final = 1.0
    omega = np.pi / 2 / t_final
    a = 1.6

    q_f = goal_position[i]
    q_0 = init_position[i]
    q_i = q(t_input, q_f, q_0, omega)
    dq_i = dq(t_input, q_f, q_0, omega)
    ddq_i = ddq(t_input, q_f, q_0, omega)

    y = a*t_input
    tanh_i = tanh(y)
    dtanh_di = dtanh(y) * dydx(a)
    ddtanh_di = ddtanh(y) * (dydx(a) ** 2)

    tanh_q_i = tanh_i * q_i + q_0 - q_0 * tanh_i
    dtanh_q_i = dtanh_di * q_i + tanh_i * dq_i - q_0 * dtanh_di
    ddtanh_q_i = ddtanh_di * q_i + 2 * dtanh_di * dq_i + tanh_i * ddq_i - q_0 * ddtanh_di

    qs.append(tanh_q_i.reshape(-1, 1))
    dqs.append(dtanh_q_i.reshape(-1, 1))
    ddqs.append(ddtanh_q_i.reshape(-1, 1))

    qs = np.hstack(qs)
    dqs = np.hstack(dqs)
    ddqs = np.hstack(ddqs)
    return qs, dqs, ddqs

def gen_tanh_traj(t_input):
    qs = []
    dqs = []
    ddqs = []
    init_position = np.array([0., 0., 0., -0., 0., 0., 0.]) * np.pi / 6.
    goal_position = np.array([2., 2., 2., -2., 2., 2, 2.]) * np.pi / 6.
    t_final = 1.0
    omega = np.pi / 2 / t_final
    a = 1.6

    for i in range(7):
        q_f = goal_position[i]
        q_0 = init_position[i]
        q_i = q(t_input, q_f, q_0, omega)
        dq_i = dq(t_input, q_f, q_0, omega)
        ddq_i = ddq(t_input, q_f, q_0, omega)

        y = a*t_input
        tanh_i = tanh(y)
        dtanh_di = dtanh(y) * dydx(a)
        ddtanh_di = ddtanh(y) * (dydx(a) ** 2)

        tanh_q_i = tanh_i * q_i + q_0 - q_0 * tanh_i
        dtanh_q_i = dtanh_di * q_i + tanh_i * dq_i - q_0 * dtanh_di
        ddtanh_q_i = ddtanh_di * q_i + 2 * dtanh_di * dq_i + tanh_i * ddq_i - q_0 * ddtanh_di

        qs.append(tanh_q_i.reshape(-1, 1))
        dqs.append(dtanh_q_i.reshape(-1, 1))
        ddqs.append(ddtanh_q_i.reshape(-1, 1))

    qs = np.hstack(qs)
    dqs = np.hstack(dqs)
    ddqs = np.hstack(ddqs)
    return qs, dqs, ddqs

if __name__ == '__main__':
    # parameters for sigmoid and
    a = 1.6
    joint_id = [0]
    t_final = 1.0
    init_position = np.array([0., 0., 0., -0., 0., 0., 0.]) * np.pi / 6.
    goal_position = np.array([2., 2., 2., -2., 2., 2, 2.]) * np.pi / 6.
    period = 2
    ts = np.linspace(0, 4 * period * t_final, int(4 * period * t_final * 100) + 1)
    q_0_joints = init_position
    q_f_joints = goal_position
    omega = np.pi / 2 / t_final

    qs = []
    dqs = []
    ddqs = []

    for i in range(7):
        q_f = q_f_joints[i]
        q_0 = q_0_joints[i]
        q_i = q(ts, q_f, q_0, omega)
        dq_i = dq(ts, q_f, q_0, omega)
        ddq_i = ddq(ts, q_f, q_0, omega)

        y = a*ts
        tanh_i = tanh(y)
        dtanh_di = dtanh(y) * dydx(a)
        ddtanh_di = ddtanh(y) * (dydx(a) ** 2)

        tanh_q_i = tanh_i * q_i + q_0 - q_0 * tanh_i
        dtanh_q_i = dtanh_di * q_i + tanh_i * dq_i - q_0 * dtanh_di
        ddtanh_q_i = ddtanh_di * q_i + 2 * dtanh_di * dq_i + tanh_i * ddq_i - q_0 * ddtanh_di

        qs.append(tanh_q_i.reshape(-1, 1))
        dqs.append(dtanh_q_i.reshape(-1, 1))
        ddqs.append(ddtanh_q_i.reshape(-1, 1))

        plt.plot(ts, q_i, ts, dq_i, ts, ddq_i, ts, tanh_q_i, ts, dtanh_q_i, ts, ddtanh_q_i)
        plt.legend(["q", "dq", "ddq", "tanh*q", "dtanh*dq", "ddtanh*ddq"])
        plt.show()
    qs = np.hstack(qs)
    dqs = np.hstack(dqs)
    ddqs = np.hstack(ddqs)
