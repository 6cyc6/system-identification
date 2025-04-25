import numpy as np
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from system_identification.utils import *

def quadratic_coeff(theta_0, theta_f, tf=1):
    a0 = theta_0
    a1 = 0
    a2 = (theta_f - theta_0) / (tf ** 2)
    return a0, a1, a2

def quadratic_interpolate(theta_0, theta_f, tf, freq=1000):
    a0, a1, a2 = quadratic_coeff(theta_0, theta_f, tf)
    time = np.linspace(0, tf, int(tf*freq))
    theta_t = a0 + a1 * time + a2 * time ** 2
    dtheta_t = a1 + 2 * a2 * time
    ddtheta_t = 2 * a2 * np.ones_like(time)
    return time, theta_t, dtheta_t, ddtheta_t

def cubic_coeff(theta_0, theta_f, tf=1):
    a0 = theta_0
    a1 = 0
    a2 = 3 * (theta_f - theta_0) / (tf ** 2)
    a3 = -2 * (theta_f - theta_0) / (tf ** 3)
    return a0, a1, a2, a3

def cubic_interpolate(theta_0, theta_f, tf, freq=1000):
    a0, a1, a2, a3 = cubic_coeff(theta_0, theta_f, tf)
    time = np.linspace(0, tf, int(tf*freq))
    theta_t = a0 + a1 * time + a2 * time ** 2 + a3 * time**3
    dtheta_t = a1 + 2 * a2 * time + 3 * a3 * time ** 2
    ddtheta_t = 2 * a2 + 6 * a3 * time
    # ddtheta_t = np.zeros_like(time)
    return time, theta_t, dtheta_t, ddtheta_t

def quintic_coeff(theta_0, theta_f, tf=1):
    # assume dtheta_0 == dtheta_f == 0, ddtheta_0 == ddtheta_f == 0
    a0 = theta_0
    a1 = 0
    a2 = 0
    a3 = (20 * theta_f - 20 * theta_0) / (2 * (tf ** 3))
    a4 = (30 * theta_0 - 30 * theta_f) / (2 * (tf ** 4))
    a5 = (12 * theta_f - 12 * theta_0) / (2 * (tf ** 5))
    return a0, a1, a2, a3, a4, a5

def quintic_interpolate(theta_0, theta_f, tf, freq=1000):
    a0, a1, a2, a3, a4, a5 = quintic_coeff(theta_0, theta_f, tf)
    time = np.linspace(0, tf, int(tf*freq))
    theta_t = a0 + a1 * time + a2 * time ** 2 + a3 * time**3 + a4 * time**4 + a5 * time ** 5
    dtheta_t = a1 + 2 * a2 * time + 3 * a3 * time ** 2 + 4 * a4 * time ** 3 + 5 * a5 * time ** 4
    ddtheta_t = 2 * a2 + 6 * a3 * time + 12 * a4 * time ** 2 + 20 * a5 * time ** 3
    return time, theta_t, dtheta_t, ddtheta_t

def generate_traj(thetas_0, thetas_tf, tf):
    q, dq, ddq = [], [], []
    for theta_0, theta_f in zip(thetas_0, thetas_tf):
        time, theta_t, dtheta_t, ddtheta_t = cubic_interpolate(theta_0, theta_f, tf)
        q.append(theta_t)
        dq.append(dtheta_t)
        ddq.append(ddtheta_t)
    q = np.array(q).T
    dq = np.array(dq).T
    ddq = np.array(ddq).T
    traj_data = {}
    traj_data["t"] = time
    traj_data["q"] = q
    traj_data["qd"] = dq
    traj_data["qdd"] = ddq
    return traj_data

def publish_ros_iiwas(traj_data):
    rospy.init_node("publish_trajectory", anonymous=True)
    rospy.sleep(2.0)
    controller = "bspline_adrc"
    # controller = 'adrc'
    # controller = 'joint_position'
    # controller = 'joint_torque'
    # controller = 'joint_feedforward'
    use_front = True
    print("read_data")
    t, q, dq, ddq = traj_data["t"], traj_data["q"], traj_data["qd"], traj_data["qdd"]

    robot = "/iiwa_front/" if use_front else "/iiwa_back/"
    joint_prefix = "F" if use_front else "B"
    topic = robot + controller + "_joint_trajectory_controller/command"
    print(topic)
    cmdPub = rospy.Publisher(topic, JointTrajectory, queue_size=1)
    rospy.sleep(2.0)

    traj_msg = JointTrajectory()
    for i in range(1, 1 + 7):
        joint_name = joint_prefix + f"_joint_{i}"
        traj_msg.joint_names.append(joint_name)

    traj_point_goal = JointTrajectoryPoint()
    traj_point_goal.positions = q[0]
    traj_point_goal.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    traj_point_goal.time_from_start = rospy.Time(5.0)
    traj_msg.points.append(traj_point_goal)
    traj_msg.header.stamp = rospy.Time.now()
    cmdPub.publish(traj_msg)
    rospy.sleep(6.0)

    traj_msg.points.clear()
    for t_i, q_i, dq_i, ddq_i in zip(t, q, dq, ddq):
        traj_point = JointTrajectoryPoint()
        traj_point.positions = q_i
        traj_point.velocities = dq_i
        traj_point.accelerations = ddq_i
        traj_point.time_from_start = rospy.Time(
            t_i+0.5
        )  # the previous code has weird t_i+0.5
        traj_msg.points.append(traj_point)

    traj_msg.header.stamp = rospy.Time.now()
    cmdPub.publish(traj_msg)

if __name__ == "__main__":
    thetas_0 = [0., 0., 0., 0., 0., 0. ,0.]
    thetas_f = [0., 0., 0., 0., 0., 0., 1.5]
    traj_data = generate_traj(thetas_0, thetas_f, tf=1.0)
    t, q, dq, ddq = traj_data["t"], traj_data["q"], traj_data["qd"], traj_data["qdd"]
    njoints = q.shape[1]
    fig, axs = plt.subplots(3, 1, figsize=(19.20, 10.03))
    print(axs)
    axs[0].plot(t, q)
    axs[1].plot(t, dq)
    axs[2].plot(t, ddq)
    legends = [f"Joint {i}" for i in range(1, 8)]
    plt.xlabel("Time: s")
    axs[0].set_ylabel("Joint Position")
    axs[1].set_ylabel("Joint Velocity")
    axs[2].set_ylabel("Joint Acceleration")
    axs[0].legend(legends)
    plt.show()

    # for t_i, q_i, dq_i, ddq_i in zip(t, q, dq, ddq):
    #     print(t_i, q_i)
    #     print(dq_i)
    #     print(ddq_i)
    #     print("=========")

    # publish_ros_iiwas(traj_data)