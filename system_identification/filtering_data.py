import numpy as np
import rosbag
from scipy.signal import filtfilt, butter
from system_identification.utils import *

def read_bag(bag, lower_bd, vis_cutting=False, end_index=10000, mode="all"):
    # =================================================================
    # Logging data for model learning: positions and efforts, the data
    # is storing in a episodic-based manner: [traj1, traj2, traj3 ....]
    # =================================================================
    tau_measurement = []
    collect = False
    t_start = -1
    time_controlstate = []

    # =================================================================
    # Logging data for visualization: desired position, vel, acc
    # =================================================================
    desired_position = []
    desired_vel = []
    desired_acc = []
    actual_position = []
    actual_velocity = []
    tau_cmd = []
    actual_acc = []

    # error buffer
    pos_errs = []
    vel_errs = []

    for topic, msg, t_bag in bag.read_messages():
        if t_start < 0:
            t_start = msg.header.stamp.to_sec()
        if topic in [
            "/iiwa_front/joint_position_trajectory_controller/state",
            "/iiwa_front/joint_feedforward_trajectory_controller/state",
            "/iiwa_front/adrc_trajectory_controller/state",
            "/iiwa_front/bspline_adrc_joint_trajectory_controller/state",
            "/iiwa_front/bspline_adrc_joint_trajectory_controller_puze/state",
        ]:
            des_pos_ = msg.desired.positions
            des_vel_ = msg.desired.velocities
            des_acc_ = msg.desired.accelerations
            actual_pos_ = msg.actual.positions
            tau_cmd_ = msg.desired.effort
            tau_measurement_ = msg.actual.effort
            pos_err_ = msg.error.positions
            vel_err_ = msg.error.velocities
            actual_vel_ = msg.actual.velocities
            actual_acc_ = msg.actual.accelerations
            t_cur = msg.header.stamp.to_sec() - t_start

            if not collect:
                if mode == "all":
                    if np.all(np.abs(msg.desired.velocities[:]) > lower_bd[:]):
                        collect = True
                    else:
                        continue
                elif mode == "any":
                    if np.any(np.abs(msg.desired.velocities[:]) > lower_bd[:]):
                        collect = True
                    else:
                        continue
                else:
                    raise ValueError("mode must be 'all' or 'any'")
            else:
                desired_position.append(des_pos_)
                desired_vel.append(des_vel_)
                desired_acc.append(des_acc_)
                actual_position.append(actual_pos_)
                time_controlstate.append([t_cur])
                tau_cmd.append(tau_cmd_)
                pos_errs.append(pos_err_)
                vel_errs.append(vel_err_)
                actual_velocity.append(actual_vel_)
                tau_measurement.append(tau_measurement_)
                actual_acc.append(actual_acc_)
        if len(time_controlstate) > end_index:
            break

    tau_measurement = np.array(tau_measurement)
    desired_position = np.array(desired_position)
    desired_vel = np.array(desired_vel)
    desired_acc = np.array(desired_acc)
    actual_position = np.array(actual_position)
    actual_velocity = np.array(actual_velocity)
    actual_acc = np.array(actual_acc)
    time_controlstate = np.array(time_controlstate)
    tau_cmd = np.array(tau_cmd)
    pos_errs = np.array(pos_errs)
    vel_errs = np.array(vel_errs)

    # =================================================================
    # Comparing the cutted data and the original data
    # =================================================================
    if vis_cutting:
        vis_compare_seqs(
            [time_controlstate],
            [desired_position],
            ["desired position"],
            ["time (s)"],
        )
        
        vis_compare_seqs(
            [time_controlstate],
            [desired_vel],
            ["desired velocity"],
            ["time (s)"],
        )
        
        vis_compare_seqs(
            [time_controlstate],
            [actual_position],
            ["actual position"],
            ["time (s)"],
        )

    time_even = np.arange(len(time_controlstate)).reshape(-1, 1)
    # =================================================================
    # Regenerate time indexes for filtering, here frequency is 1000
    # =================================================================
    return (
        time_even,
        time_controlstate,
        desired_position,
        desired_vel,
        desired_acc,
        tau_measurement,
        tau_cmd,
        actual_position,
        actual_velocity,
        actual_acc,
        pos_errs,
        vel_errs,
    )

def cal_d(fs, dts):
    dfs = np.zeros_like(fs)
    dfs[1:-1] = (fs[2:, :] - fs[:-2, :]) / (dts[2:] - dts[:-2])
    dfs[0] = dfs[1]
    dfs[-1] = dfs[-2]
    return dfs

def deriv_poly_filter(time, measurement, freq=1000.0, window=50, poly_order=3, vis_filtering=False):
    finite_diff_derive = cal_d(measurement, time)
    filtered = savitzy_filter(measurement, freq, window, poly_order, 0)
    dot_filtered = savitzy_filter(measurement, freq, window, poly_order, 1)
    dot_filtered[0] = np.zeros(measurement.shape[1])
    ddot_filtered = savitzy_filter(measurement, freq, window, poly_order, 2)
    if vis_filtering:
        vis_compare_seqs(
            [time, time],
            [measurement, filtered],
            legends=["Before filtered", "After filtered"],
            labels=["Time (s)"],
        )
        vis_compare_seqs(
            [time, time],
            [finite_diff_derive, dot_filtered],
            legends=["Before filtered", "After filtered"],
            labels=["Time (s)"],
        )
    return filtered, dot_filtered, ddot_filtered

# ===============================================================================
# Preprocessing: read the bag file and filter the data, save the data as npz file
# ===============================================================================
def preprocess_bag(bag_path, lower_bd, freq=1000, window=50, poly_order=3, mode="all", vis_cutting=False, end_index=10000, vis_filtering=False, npz_root_path=""):
    bag = rosbag.Bag(bag_path)
    (
        time_even,
        time_controlstate,
        desired_position,
        desired_vel,
        desired_acc,
        tau_measurement,
        tau_cmd,
        actual_position,
        actual_velocity,
        actual_acc,
        pos_errs,
        vel_errs,
    ) = read_bag(bag, lower_bd, mode=mode, vis_cutting=vis_cutting, end_index=end_index)
    filtered_pos, filtered_vel, filtered_acc = deriv_poly_filter(
        time_controlstate, actual_position, freq, window, poly_order, vis_filtering=vis_filtering
    )
    file_name = bag_path.split("/")[-1].split(".")[0]
    file_name = npz_root_path + file_name + ".npz"
    print(file_name)
    np.savez(
        file_name,
        time_even=time_even,
        time_controlstate=time_controlstate,
        desired_position=desired_position,
        desired_vel=desired_vel,
        desired_acc=desired_acc,
        tau_measurement=tau_measurement,
        tau_cmd=tau_cmd,
        actual_position=actual_position,
        actual_velocity=actual_velocity,
        actual_acc=actual_acc,
        pos_errs=pos_errs,
        vel_errs=vel_errs,
        filtered_pos=filtered_pos,
        filtered_vel=filtered_vel,
        filtered_acc=filtered_acc,
    )

# ===============================================================================
# Preprocessing: read npz files
# ===============================================================================
def preprocess(bag_paths, lower_bd, freq=1000, window=50, poly_order=3, mode="all", read_bags=False, vis_cutting=False, end_index=10000, vis_filtering=False, npz_root_path=""):
    # assume one bag for one trajectory
    if read_bags:
        for bag_path in bag_paths:
            preprocess_bag(bag_path, lower_bd, freq, window, poly_order, mode, vis_cutting, end_index, vis_filtering, npz_root_path)
    traj_data = []
    for bag_path in bag_paths:
        file_name = bag_path.split("/")[-1].split(".")[0]
        file_name = npz_root_path + file_name + ".npz"
        data = np.load(file_name)
        traj_data.append(data)
    traj_data_file_name = npz_root_path + f"traj_data_{len(bag_paths)}traj.npz"
    np.savez(traj_data_file_name, traj_data=traj_data)

if __name__ == "__main__":
    bag_paths = [
        "traj22_0.bag",
        "traj22_1.bag",
        "traj23_0.bag",
        "traj23_1.bag",
        "traj24_0.bag",
        "traj24_1.bag",
    ]
    bag_root_path = "/home/junninghuang/Desktop/Codes/system-identification/experiments/bags/"
    for i in range(len(bag_paths)):
        bag_paths[i] = bag_root_path + bag_paths[i]
    njoints = 7
    lower_bd = np.array([1e-5] * njoints)
    vis_cutting = True
    vis_filtering = True
    npz_root_path = "/home/junninghuang/Desktop/Codes/system-identification/experiments/npzs/"
    preprocess(bag_paths, lower_bd, read_bags=True, npz_root_path=npz_root_path, vis_cutting=vis_cutting, vis_filtering=vis_filtering)