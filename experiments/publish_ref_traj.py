from system_identification.my_utils import *
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import subprocess
import os
import signal

def vis_trajs(file_name):
    traj_data = np.load(f"{file_name}", allow_pickle=True)
    traj_data = traj_data.item()
    t, q, dq, ddq = traj_data["t"], traj_data["q"], traj_data["dq"], traj_data["ddq"]
    # ts = [t[::10]]
    # qs = [q[::10,:]]
    # dqs = [dq[::10,:]]
    # ddqs = [ddq[::10,:]]
    ts = [t]
    qs = [q]
    dqs = [dq]
    ddqs = [ddq]
    vis_compare_seqs(ts, qs, ["q"], ["time: s"])
    vis_compare_seqs(ts, dqs, ["dq"], ["time: s"])
    vis_compare_seqs(ts, ddqs, ["ddq"], ["time: s"])
    plt.show()

def publish(traj_name, record=False, index=0):
    file_name = f"{traj_dir}/{traj_name}.npy"
    traj_data = np.load(f"{file_name}", allow_pickle=True)
    traj_data = traj_data.item()
    t, q, dq, ddq = traj_data["t"], traj_data["q"], traj_data["dq"], traj_data["ddq"]
    print(f"========== read_data finished ==========")

    rospy.init_node("excitation_trajectory", anonymous=True)
    rospy.sleep(2.0)
    type_name = "bspline_adrc_joint"
    # type_name = 'adrc'
    # type_name = 'joint_position'
    # type_name = 'joint_torque'
    # type_name = 'joint_feedforward'
    use_front = True
    topic_name = "/iiwa_front/" if use_front else "/iiwa_back/"
    joint_prefix = "F" if use_front else "B"
    ctrl_topic_name = topic_name + type_name + "_trajectory_controller_puze"
    topic = f"{ctrl_topic_name}/command"
    print(f"topic name: {topic}")
    cmdPub = rospy.Publisher(topic, JointTrajectory, queue_size=1)

    rospy.sleep(2.0)
    if record:
        p = subprocess.Popen(f"rosbag record {ctrl_topic_name}/state {ctrl_topic_name}/observer_state -O {traj_name}_{index}.bag",
                             stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid) # refer to https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
                                                                                       # we want to kill the child process without killing the parene

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
    print(f"============ reset starting position finished ==========")

    traj_msg.points.clear()
    for t_i, q_i, dq_i, ddq_i in zip(t, q, dq, ddq):
        traj_point = JointTrajectoryPoint()
        traj_point.positions = q_i
        traj_point.velocities = dq_i
        traj_point.accelerations = ddq_i
        traj_point.time_from_start = rospy.Time(t_i)# the previous code has weird t_i+0.5
        traj_msg.points.append(traj_point)

    while not rospy.is_shutdown():
        traj_msg.header.stamp = rospy.Time.now()
        cmdPub.publish(traj_msg)
        rospy.sleep(traj_msg.points[-1].time_from_start.to_sec())
        break

    # kill the subprocess to stop data recording
    if record:
        rospy.sleep(16.0)
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IIWAS KUKA")
    parser.add_argument("--record", type=bool, default=True)
    args = parser.parse_args()

    # ===================================================================================
    # Publishing trajectories to the robot and collect real data
    # ===================================================================================
    # TODO: add the file name interface
    """
    Notice!!!
        traj_20 to traj_24 are tested correctly, don't test traj_1-traj_3, the assigned velocities
        are computed wrongly and then cause weird interpolation in the ROS interface.
    """
    traj_dir = "./npys"
    traj_name = "traj_20"
    # vis_trajs(file_name)
    for i in range(50):
        publish(traj_name, args.record, i)