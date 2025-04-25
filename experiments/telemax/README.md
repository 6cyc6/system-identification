## Trajectories for SysID
The folder named data contains trajectories for identifying the model of the telemax robot. The trajectories are stored as npy files, and each npy file contains a 60s trajectory with frequency 100 Hz, overall 6000 data points for one file. Currently the folder has four npy files, namely, four trajectories.

## Requirement
The script named publish_trajectories.py is used for publishing the trajectories, the only requirements are: numpy, rospy
## Run Script
There are two steps to run the script

1. Launch the simulator
    ```bash
    roslaunch gazebo_ros empty_world.launch
    roslaunch drz_telemax_gazebo_launch spawn_default.launch
    ```
2. Run the script
    ```bash
    python publish_trajectories.py --index 0
    ```
    here index represents the index number of the npy file, valid choices of index are 0, 1, 2, 3. 
