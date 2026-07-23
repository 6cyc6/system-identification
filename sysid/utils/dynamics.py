"""Rigid-body dynamics model composition helpers."""


def dynamics_prediction(models, q, dq, ddq, des_pos=None, des_vel=None, des_acc=None):
    """Predict joint torques using an inertia model and optional friction model."""
    if len(models) == 1:
        return models[0].predict(q, dq, ddq)

    if len(models) == 2:
        inertia_model, friction_model = models
        tau_inertia = inertia_model.predict(q, dq, ddq)
        friction_velocity = dq if des_vel is None else des_vel
        tau_friction = friction_model.predict(q, friction_velocity, ddq)
        return tau_inertia + tau_friction

    raise NotImplementedError(
        "Expected one inertia model and at most one friction model"
    )


def load_model_params(params, inertia_model, friction_model=None):
    """Synchronize a flattened parameter vector into dynamics models."""
    total_inertia_params = inertia_model.njoints * 10
    inertia_model.sync_param(params[:total_inertia_params])
    if friction_model is not None:
        friction_model.sync_param(params[total_inertia_params:])
