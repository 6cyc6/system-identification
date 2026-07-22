# Repository Overview — `system-identification`

A Python toolkit for **dynamic parameter identification of robot manipulators** (inertial parameters, friction, rotor inertia). It covers the full system-identification pipeline:

1. **Excitation trajectory generation** — optimize periodic Fourier-series joint trajectories that minimize the condition number of the dynamics regressor (via SciPy, CasADi/IPOPT, or Drake's `MathematicalProgram`), subject to joint limits, self-collision, and workspace obstacle (camera box) constraints.
2. **Trajectory execution & data collection** — play trajectories on simulators (MuJoCo, Drake, Gazebo) or real robots via ROS, and record data (rosbags / `.npy`).
3. **Data preprocessing** — filtering, numerical differentiation of measured joint states.
4. **Parameter estimation** — fit inertial + friction + rotor parameters from torque data using LMI-constrained QP (CVXPY) or log-Cholesky over-parameterized optimization (JAX), with physical-consistency constraints.

Supported robots (URDFs in `robot_description/`): KUKA iiwa14, Franka FR3, Telemax, Barrett WAM, Unitree A1, cart-pole, Furuta pendulum, 3-DOF planar arm.

---

## Top-level layout

| Path | Purpose |
|---|---|
| `system_identification/` | The installable Python package (core library) |
| `script/` | Current entry-point scripts (trajectory generation / simulation / checking) |
| `experiments/` | Older experiment scripts, ROS publishing, friction-model demos, recorded data |
| `robot_description/` | URDF / xacro / mesh files for all supported robots |
| `fr3/` | FR3-specific plotting utility for recorded real-robot data |
| `test/` | Minimal MuJoCo C-API smoke test (CMake) |
| `docs/` | This documentation |
| `setup.py`, `environment.yml`, `pixi.toml` | Packaging & environments (conda env `sysid`; newer pixi workspace) |

---

## Core package: `system_identification/`

### Excitation trajectory generation

| File | What it does |
|---|---|
| [excitation_generator.py](../system_identification/excitation_generator.py) | Original Fourier-series trajectory generator. Parameterizes `q, dq, ddq` as a finite Fourier series with constraints `q(0)=dq(0)=ddq(0)=0`, samples random/truncated-Gaussian coefficients, and rejects trajectories violating joint position/velocity limits or (Pinocchio-based) collision checks (`obtain_valid_traj_param`, `is_traj_valid`). |
| [excitation_generator_new.py](../system_identification/excitation_generator_new.py) | Updated version of the above (used by `script/`). Adds `camera_box_clearance()`: samples points along the robot body (Pinocchio FK) and computes clearance to fixed camera-box obstacles around the workspace. |
| [excitation_optimization.py](../system_identification/excitation_optimization.py) | Objective/constraint functions for SciPy-based optimization: `params2cond` (regressor condition number), `params2condFriction` (incl. symmetric/asymmetric friction regressor columns), `params2coverage` (joint-space coverage), and the position/velocity `constraints()` builder for SLSQP / trust-constr. |
| [fourier_utils.py](../system_identification/fourier_utils.py) | Small helpers to flatten/unflatten Fourier parameter arrays `(2, order, njoints)` ↔ flat vectors and convert flat params directly to trajectories. |
| [ipopt_solver.py](../system_identification/ipopt_solver.py) | CasADi/IPOPT excitation solver (`IpoptExcitationSolver`). Builds the regressor, selects identifiable columns via pivoted QR (`select_identifiable_columns`), defines condition-number-style eigenvalue objectives, Fourier position/velocity constraints, camera-clearance constraints (finite-difference CasADi callbacks), and per-iteration condition-number logging. |

### Drake integration: `system_identification/drake/`

| File | What it does |
|---|---|
| [camera_collision.py](../system_identification/drake/camera_collision.py) | `DrakeCameraCollisionChecker`: builds a Drake `MultibodyPlant`+`SceneGraph` from the robot URDF, adds the four camera boxes (specs in `CAMERA_BOX_SPECS_MM`, scaled by `CAMERA_BOX_MARGIN_SCALE`) as collision geometry, and exposes minimum-distance constraints for trajectory feasibility checks. |
| [ipopt_solver.py](../system_identification/drake/ipopt_solver.py) | `IpoptExcitationDrakeSolver`: same CasADi/IPOPT formulation as the core solver, but with Drake-based collision / link-Y-bound / path-constraint callbacks instead of the Pinocchio point-sampling clearance check. |
| [mathematical_program_solver.py](../system_identification/drake/mathematical_program_solver.py) | `DrakeMathematicalProgramExcitationSolver`: formulates excitation optimization natively in Drake's `MathematicalProgram` (AutoDiff gradients, `IpoptSolver` backend) with `fourier_constraint_bounds/values` helpers. |

### Dynamics models

| File | What it does |
|---|---|
| [inertia_model.py](../system_identification/inertia_model.py) | `InertiaModel`: rigid-body dynamics regressor built with Pinocchio (based on Atkeson/An/Hollerbach and the Springer Handbook formulation). Loads a URDF from `robot_description/`, computes the standard 10-parameter-per-link regressor, predicts torques, and converts between parameterizations. `InertiaModelGeometryInvariant` is a variant with geometry-invariant parameterization. |
| [friction_model.py](../system_identification/friction_model.py) | Family of joint-friction regressor models sharing the `FrictionModel` base: Coulomb, viscous, offset/bias, symmetric & asymmetric MATLAB-style Stribeck (`MatlabFModel`, `AsymMatlabFModel`), combined Coulomb+viscous (sym/asym), and `StribeckModel`. Each provides `regressor()` + `predict()`. |
| [rotor_model.py](../system_identification/rotor_model.py) | Rotor-inertia residual models (`BaseRotor`, `NegRotor`, `RotorAndGear`): regressor features `ddq * gear_ratio²` for reflected rotor inertia. |

### Parameter-estimation solvers

| File | What it does |
|---|---|
| [abstract_solver.py](../system_identification/abstract_solver.py) | `AbstractSolver` base class: composes an `InertiaModel` with a configurable list of residual models (friction, rotor) from a models-config dict; defines the `generate_regressor / objective / fit` interface and logging helpers. |
| [qp_solver.py](../system_identification/qp_solver.py) | CVXPY-based estimation. `BaseQP`: least-squares torque regression. `LmiQP`: adds LMI (linear matrix inequality) physical-consistency constraints (positive-definite pseudo-inertia matrices) — convex SDP identification. |
| [overparam_solver.py](../system_identification/overparam_solver.py) | Nonlinear estimation with JAX autodiff + SciPy `minimize`. `LogCholesky`: log-Cholesky over-parameterization guaranteeing physically consistent inertial parameters; variants `LogCholeskyOnlyInertia` and `LogCholeskyOnlyMCoM` restrict which parameters are free. |

### Data & utilities

| File | What it does |
|---|---|
| [filtering_data.py](../system_identification/filtering_data.py) | Rosbag data pipeline: `read_bag()` extracts measured/desired joint states and torques per trajectory episode; Butterworth and Savitzky-Golay filtering; `cal_d`/`deriv_poly_filter` numerical differentiation; `preprocess()` converts bags → filtered `.npz` datasets. |
| [utils.py](../system_identification/utils.py) | Grab-bag of shared helpers: `WandbLogger`, `find_path`, YAML/pickle I/O, `yml2urdf` (write identified parameters back into a URDF), `Dataset` container, `retrieve_robot_config` (per-robot joint limits / initial pose / sampling config), `QR_dim_reduction` & `SVD_dim_reduction` (base-parameter extraction + condition number), `feature2regressor`, plotting (`vis_compare_seqs`, `draw_spectrum`), filtering helpers. |
| [my_utils/path_utils.py](../system_identification/my_utils/path_utils.py) | `safe_mkdir` helper. |
| `crba_rotor_py.cpython-38-*.so` | Prebuilt (Python 3.8) extension exposing `crba_rotor` — composite-rigid-body inertia including rotor terms, used by `experiments/crba.py`. |

---

## Entry-point scripts: `script/`

These are the current-generation scripts (the FR3-focused workflow). All save trajectories as CSV (`t, q_i, dq_i, ddq_i`) under `script/saves/`.

| File | What it does |
|---|---|
| [generate_excitation.py](../script/generate_excitation.py) | Excitation generation with **SciPy** (`SLSQP` / `trust-constr`): minimize regressor condition number (optionally with friction columns or coverage objective) under joint-limit + camera-clearance constraints. |
| [generate_excitation_ipopt.py](../script/generate_excitation_ipopt.py) | Same problem solved with **CasADi/IPOPT** (`IpoptExcitationSolver`), Pinocchio-based camera clearance. |
| [generate_excitation_ipopt_drake.py](../script/generate_excitation_ipopt_drake.py) | CasADi/IPOPT with **Drake collision checking** (`IpoptExcitationDrakeSolver`); includes trajectory validation and comparison plots. |
| [generate_excitation_ipopt_drake_solver.py](../script/generate_excitation_ipopt_drake_solver.py) | Excitation generation fully inside **Drake's `MathematicalProgram`** (AutoDiff + IPOPT), with multi-seed parallel solves (`ProcessPoolExecutor`), constraint-violation reporting, and per-run save folders. |
| [generate_traj.py](../script/generate_traj.py) | Slim variant of SciPy-based generation (random valid Fourier trajectory → optimize → save CSV). |
| [check_traj.py](../script/check_traj.py) | Validate a saved trajectory (CSV/`.npy`): joint limits, camera-box clearance, regressor condition number / singular values (with and without friction columns). |
| [run_mujoco.py](../script/run_mujoco.py) | Play a reference trajectory in **MuJoCo** with feedforward (inverse dynamics) + PD control; records `t, q, qd, qdd, tau` to `.npy` (→ `experiments/traj_data/`). Supports FR3 and iiwa. |
| [run_drake.py](../script/run_drake.py) | Visualize/replay a trajectory in **Drake + Meshcat**, with the camera boxes added to the scene for visual collision inspection. |
| `saves/` | Generated excitation trajectory CSVs (`cond_fr3_*.csv`, `cond_ipopt_*` etc.). |

---

## `experiments/` (older / robot-deployment scripts)

| File | What it does |
|---|---|
| [generate_excitation.py](../experiments/generate_excitation.py) | Older iiwa-focused excitation generation (SciPy; uses the original `excitation_generator`). |
| [offline_torque_reg.py](../experiments/offline_torque_reg.py) | **The actual identification step**: loads preprocessed data, builds a solver (`LmiQP` / `LogCholesky` … via config), fits inertial+friction parameters, and visualizes measured-vs-predicted torque. |
| [crba.py](../experiments/crba.py) | Hand-written Composite Rigid Body Algorithm (Featherstone ch. 6.2) including rotor inertia; cross-checks against Pinocchio and the `crba_rotor` C extension. |
| [gen_composite_traj.py](../experiments/gen_composite_traj.py) | Composite point-to-point trajectories from cosine/tanh segments (smooth rest-to-rest motions). |
| [publish_traj.py](../experiments/publish_traj.py) | ROS publisher for joint trajectories (quadratic/cubic interpolation to start pose, then trajectory streaming). |
| [publish_ref_traj.py](../experiments/publish_ref_traj.py) | ROS publisher for saved `.npy` reference trajectories; can trigger rosbag recording. |
| [telemax/](../experiments/telemax/) | Trajectories + `publish_trajectories.py` for the Telemax robot in Gazebo (see its own [README](../experiments/telemax/README.md)). |
| [friction_model/](../experiments/friction_model/) | Standalone friction-model study scripts: LuGre, Dahl, Bliman-Sorine, MATLAB Stribeck contact model, Gaussian peak, scalar ODE toy system — each simulates/plots one friction model. |
| `traj_data/` | Recorded MuJoCo runs and older iiwa excitation results (`.npy`). |
| [data_description](../experiments/data_description) | Notes on the recorded rosbag/PKL datasets (iiwa periodic + striking motions). |

---

## Other directories

- **`robot_description/`** — robot models: `fr3_description/` (FR3 variants: gripper, soft gripper, pusher), `iiwas_description/` (CAD, edited, and *identified* URDF variants — i.e., URDFs regenerated from identification results), `telemax_description/`, `wam_description/`, `a1_description/`, plus cart-pole / Furuta pendulum / 3-link toy models for testing.
- **`fr3/plot_traj.py`** — plots real FR3 recordings (`.npy` files per signal) against reference trajectories, with numerical velocity differentiation.
- **`test/`** — `t1.cpp` + CMake: minimal MuJoCo C API load-and-step check (`hello.xml`).

---

## Typical workflow

```bash
# 1. Generate an optimized excitation trajectory (FR3, IPOPT + Drake collision)
python script/generate_excitation_ipopt_drake.py

# 2. Sanity-check it (limits, camera clearance, condition number)
python script/check_traj.py --path script/saves/<traj>.csv

# 3. Execute in simulation and record data
python script/run_mujoco.py --robot franka   # or visualize: python script/run_drake.py

# 4. (Real robot) publish via ROS and record bags
python experiments/publish_ref_traj.py

# 5. Preprocess recorded data (filtering + differentiation)
#    system_identification/filtering_data.py :: preprocess()

# 6. Identify parameters
python experiments/offline_torque_reg.py
```

## Installation

See the top-level [README](../README.md): conda env `sysid` from `environment.yml`, then `pip install -e .`. A newer `pixi.toml` workspace (Python 3.10-3.11, numpy/scipy/cvxpy/casadi stack) is also provided. Key dependencies: **Pinocchio** (dynamics/regressor), **CasADi + IPOPT** and **Drake** (trajectory optimization), **CVXPY** (LMI-QP), **JAX** (log-Cholesky solver), **MuJoCo**, **ROS** (rospy/rosbag, for real-robot data).
