import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg, PhysxCfg


@configclass
class PushTEnv6Cfg(DirectRLEnvCfg):
    """PushT v6 config — 7 DOF joint space action (delta joint positions).

    Obs (37): joint pos norm (7), joint vel (7), EE xyz (3), EE quat (4), fingertip xyz (3),
              obj pose xyz+quat (7), goal xy (2), goal quat (4).
    Actions (7): delta joint positions for fr3_joint1..7, scaled by action_scale.
    """
    # env
    episode_length_s: float = 15.0
    decimation: int = 4                    # 120 Hz physics / 30 Hz policy
    action_space: int = 7                  # delta joint positions (7 DOF)
    observation_space: int = 37            # 7 joint pos (norm), 7 joint vel, 3 ee xyz, 4 ee quat, 3 fingertip xyz, 7 obj pose (xyz+quat), 2 goal xy, 4 goal quat
    state_space: int = 0

    # EE body name in the fr3_pusher USD
    ee_body_name: str = "fr3_link8"
    # Max joint displacement per policy step (radians)
    action_scale: float = 0.02
    # Workspace bounds for object sampling in robot root frame (x, y)
    workspace_x: tuple = (0.30, 0.70)
    workspace_y: tuple = (-0.30, 0.30)

    # Goal position in env-local frame (x, y)
    goal_pos: tuple = (0.03, 0.0)
    # Goal yaw (radians); T-shape target orientation around Z-axis
    goal_yaw: float = 0.0

    # Object initial pose randomisation
    obj_init_r_range: tuple = (0.05, 0.12)
    obj_init_x_range: tuple = (0.0, 0.12)
    obj_init_y_range: tuple = (-0.12, 0.12)
    obj_init_yaw_range: tuple = (-3.14159, 3.14159)

    # Success thresholds
    success_pos_threshold: float = 0.003    # metres
    success_orient_threshold: float = 0.08  # radians

    # Rewards (same formula as PushTEnv5):
    #   r_robot  = 1[d_robot_obj^{t-1} > obj_null_dist] * (d_robot_obj^{t-1} - d_robot_obj^t) * weight
    #   r_pos    = weight * (exp(-2*pos_err) - 1)
    #   r_orient = weight * (exp(-2*|yaw_err|) - 1)
    #   r_smooth = -weight * ||a_t - a_{t-1}||^2
    #   r_success = reward_success * success

    reward_robot_dist_weight: float = 100.0
    box_null_dist: float = 0.12  # 0.1
    # Tilt penalty weight (applied when EE z-axis deviates from pointing down)
    reward_tilt_weight: float = 1.0
    # Dot-product threshold below which tilt penalty is applied (cos(30°) ≈ 0.866)
    tilt_threshold: float = 0.866
    reward_pos_weight: float = 1.5
    reward_orient_weight: float = 0.5
    reward_smooth_weight: float = 0.0001
    reward_success: float = 100.0

    static_penalty: float = -0.6           # penalty when T-block barely moves
    static_pos_threshold: float = 0.001    # metres; position movement below this triggers penalty
    static_rot_threshold: float = 0.01     # radians; yaw change below this triggers penalty

    # Flip penalty: applied when T-shape CoM z > this threshold (object is lifted/flipped)
    flip_z_threshold: float = 0.121        # metres (local frame); resting flat ≈ 0.021
    reward_flip_penalty: float = -0.5      # penalty per step when object is airborne/flipped
    
    reward_table_penalty: float = -0.5     # penalty per step when fingertip is too close to the table

    # sim
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            gpu_collision_stack_size=2**30,  # 256 MB; default 64 MB overflows with many envs
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8, env_spacing=3.0, replicate_physics=True)

    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 2.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
    )

    robot = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/galois/assets/assets/franka/fr3_pusher_v8.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.5, 0.0, 0.1),
            joint_pos={
                "fr3_joint1":  0.6127,
                "fr3_joint2":  0.1663,
                "fr3_joint3":  0.1232,
                "fr3_joint4": -2.6854,
                "fr3_joint5": -0.0707,
                "fr3_joint6":  2.8499,
                "fr3_joint7":  1.5874,
            },
        ),
        actuators={
            "fr3_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=400.0,
                damping=40.0,
            ),
            "fr3_forearm": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=400.0,
                damping=40.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # T-shape object
    obj = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Tshape",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/galois/assets/assets/obj/Tshape/t_shape.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.8),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.121)),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
