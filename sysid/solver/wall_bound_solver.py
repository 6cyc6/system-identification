"""Drake excitation solver with FR3 workspace and self-collision constraints."""

import numpy as np

from ..model.workspace_constraints import robot_link_workspace_margins
from ..utils.fourier_utils import flat_params_to_traj
from .mathematical_program_solver import DrakeMathematicalProgramExcitationSolver


class WallBoundDrakeMathematicalProgramExcitationSolver(
    DrakeMathematicalProgramExcitationSolver
):
    """Add FR3 workspace walls and self-collision pairs to the shared solver."""

    def __init__(
        self,
        *args,
        link_x_lower=-0.2,
        link_z_lower=0.1,
        use_self_collision_constraints=True,
        self_collision_body_pairs=(),
        self_collision_clearance=0.0,
        **kwargs,
    ):
        self.link_x_lower = float(link_x_lower)
        self.link_z_lower = float(link_z_lower)
        self.use_self_collision_constraints = bool(use_self_collision_constraints)
        self.self_collision_body_pairs = tuple(self_collision_body_pairs)
        self.self_collision_clearance = float(self_collision_clearance)
        super().__init__(*args, **kwargs)

    def _path_constraint_values(self, flat_params):
        """Evaluate path margins at the collision-check sampling rate."""
        flat_params = np.asarray(flat_params, dtype=float).reshape(-1)
        _, q, _, _ = flat_params_to_traj(
            flat_params,
            self.fourier_config,
            self.robot_config,
        )
        q_sampled = q[:: self.camera_collision_stride]

        # Keep this order synchronized with _path_constraint_bounds().
        values = [
            self.collision_checker.minimum_distance_constraint_values(q_sampled)
        ]
        if self.use_self_collision_constraints:
            self_collision_margins = (
                self.collision_checker.robot_self_collision_pair_margins(
                    q_sampled,
                    body_pairs=self.self_collision_body_pairs,
                )
            )
            values.append(self_collision_margins.reshape(-1))
        if self.use_link_y_bounds:
            workspace_margins = robot_link_workspace_margins(
                self.collision_checker,
                q_sampled,
                x_lower=self.link_x_lower,
                y_lower=self.link_y_lower,
                y_upper=self.link_y_upper,
                z_lower=self.link_z_lower,
            )
            values.append(workspace_margins.reshape(-1))
        return np.concatenate(values)

    def _path_constraint_bounds(self):
        """Build bounds matching the flattened path-constraint vector."""
        n_samples = len(
            range(
                0,
                int(self.fourier_config["duration"] * 100) + 1,
                self.camera_collision_stride,
            )
        )
        n_self_collision_outputs = (
            len(self.self_collision_body_pairs)
            if self.use_self_collision_constraints
            else 0
        )
        n_workspace_outputs = 4 if self.use_link_y_bounds else 0
        n_outputs = n_samples * (
            1 + n_self_collision_outputs + n_workspace_outputs
        )

        lower = np.zeros(n_outputs, dtype=float)
        upper = np.full(n_outputs, np.inf, dtype=float)
        upper[:n_samples] = 1.0
        if n_self_collision_outputs:
            start = n_samples
            stop = start + n_samples * n_self_collision_outputs
            lower[start:stop] = self.self_collision_clearance
        return lower, upper
