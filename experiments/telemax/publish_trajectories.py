import numpy as np
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import subprocess
import os
import signal

def publish_ros_telemax(index, record_bag=0):
    rospy.init_node("excitation_trajectory", anonymous=True)
    rospy.sleep(2.0)

    print("read_data")
    file_name = f"data/telemax_{index}"
    traj_data = np.load(f"{file_name}.npy", allow_pickle=True)
    print(f"Load file {file_name}.npy")
    traj_data = traj_data.item()
    t, q, dq, ddq = traj_data["t"], traj_data["q"], traj_data["qd"], traj_data["qdd"]
    print(f"Load successfully, Shape of joint positions: {q.shape}")

    topic = "/telemax_control/manipulator_arm_traj_controller/command"
    cmdPub = rospy.Publisher(topic, JointTrajectory, queue_size=1)
    rospy.sleep(2.0)
    if record_bag:
        print("Recording bag")
        p = subprocess.Popen(
            [
                "rosbag",
                "record",
                "/telemax_control/manipulator_arm_traj_controller/state",
                "/telemax_control/joint_states",
                "/telemax_control/manipulator_arm_traj_controller/command"
            ]
        )

    traj_msg = JointTrajectory()
    for i in range(6):
        joint_name = f"arm_joint_{i}"
        traj_msg.joint_names.append(joint_name)

    # =====================================================
    # Initialize the joint positions
    # =====================================================
    traj_point_goal = JointTrajectoryPoint()
    traj_point_goal.positions = q[0]
    traj_point_goal.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    traj_point_goal.time_from_start = rospy.Time(5.0)
    traj_msg.points.append(traj_point_goal)
    traj_msg.header.stamp = rospy.Time.now()
    cmdPub.publish(traj_msg)
    rospy.sleep(6.0)

    # =====================================================
    # Publish one trajectory
    # =====================================================
    traj_msg.points.clear()
    for t_i, q_i, dq_i, ddq_i in zip(t, q, dq, ddq):
        traj_point = JointTrajectoryPoint()
        traj_point.positions = q_i
        traj_point.velocities = dq_i
        traj_point.accelerations = ddq_i
        traj_point.time_from_start = rospy.Time(
            t_i + 0.5
        )  # here 0.5 to avoid the mismatch for current time in
        # python library and current ros
        traj_msg.points.append(traj_point)

    traj_msg.header.stamp = rospy.Time.now()
    cmdPub.publish(traj_msg)

    # kill the subprocess to stop data recording
    if record_bag:
        rospy.sleep(1.0)
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        print("Terminate recording bag")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulator")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--record_bag", type=int, default=1)
    args = parser.parse_args()
    publish_ros_telemax(args.index, args.record_bag)
