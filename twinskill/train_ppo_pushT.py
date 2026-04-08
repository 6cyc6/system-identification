"""Train PPO on PushT-MJX using Brax.

Usage:
    python twinskill/train_ppo_pushT.py [--num_envs 1024] [--num_timesteps 50_000_000]

This script mirrors the pattern from mujoco_playground/learning/train_jax_ppo.py
but is self-contained for the PushT task.
"""

from __future__ import annotations

import datetime
import functools
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
)

import jax
import jax.numpy as jp
import mediapy as media
import mujoco
from absl import app, flags, logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from ml_collections import config_dict
from mujoco_playground._src import wrapper

from twinskill.envs.pushT_env_mjx import PushTEnvMjx, default_config

logging.set_verbosity(logging.WARNING)

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
_NUM_ENVS       = flags.DEFINE_integer("num_envs",       1024, "Number of parallel envs")
_NUM_EVAL_ENVS  = flags.DEFINE_integer("num_eval_envs",   128, "Number of eval envs")
_NUM_TIMESTEPS  = flags.DEFINE_integer("num_timesteps", 50_000_000, "Total env steps")
_NUM_EVALS      = flags.DEFINE_integer("num_evals",         10, "Eval checkpoints")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length",   450, "Steps per episode")
_UNROLL_LENGTH  = flags.DEFINE_integer("unroll_length",     20, "PPO rollout length")
_NUM_MINIBATCHES= flags.DEFINE_integer("num_minibatches",   16, "Minibatches per update")
_UPDATES_PER_BATCH = flags.DEFINE_integer("num_updates_per_batch", 8, "Gradient steps")
_LEARNING_RATE  = flags.DEFINE_float("learning_rate",    3e-4, "Adam LR")
_DISCOUNTING    = flags.DEFINE_float("discounting",       0.97, "Discount factor")
_ENTROPY_COST   = flags.DEFINE_float("entropy_cost",      1e-2, "Entropy bonus coeff")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling",     1.0, "Global reward scale")
_BATCH_SIZE     = flags.DEFINE_integer("batch_size",       256, "Minibatch size")
_CLIPPING_EPS   = flags.DEFINE_float("clipping_epsilon",   0.3, "PPO clip range")
_MAX_GRAD_NORM  = flags.DEFINE_float("max_grad_norm",      1.0, "Gradient clip")
_SEED           = flags.DEFINE_integer("seed",               1, "RNG seed")
_LOGDIR         = flags.DEFINE_string("logdir",          None,  "Log directory (default: logs/)")
_NUM_VIDEOS     = flags.DEFINE_integer("num_videos",         1, "Rollout videos to save")
_USE_WANDB      = flags.DEFINE_boolean("use_wandb",      False, "Log to Weights & Biases")
_PLAY_ONLY      = flags.DEFINE_boolean("play_only",      False, "Skip training, run inference only")


# ---------------------------------------------------------------------------
# PPO config
# ---------------------------------------------------------------------------

def make_ppo_params() -> config_dict.ConfigDict:
    return config_dict.create(
        num_timesteps=_NUM_TIMESTEPS.value,
        num_evals=_NUM_EVALS.value,
        episode_length=_EPISODE_LENGTH.value,
        normalize_observations=True,
        action_repeat=1,
        reward_scaling=_REWARD_SCALING.value,
        unroll_length=_UNROLL_LENGTH.value,
        num_minibatches=_NUM_MINIBATCHES.value,
        num_updates_per_batch=_UPDATES_PER_BATCH.value,
        discounting=_DISCOUNTING.value,
        learning_rate=_LEARNING_RATE.value,
        entropy_cost=_ENTROPY_COST.value,
        num_envs=_NUM_ENVS.value,
        num_eval_envs=_NUM_EVAL_ENVS.value,
        batch_size=_BATCH_SIZE.value,
        max_grad_norm=_MAX_GRAD_NORM.value,
        clipping_epsilon=_CLIPPING_EPS.value,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(256, 256, 256),
            value_hidden_layer_sizes=(256, 256, 256),
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv):
    del argv

    ppo_params = make_ppo_params()

    if _PLAY_ONLY.value:
        ppo_params.num_timesteps = 0

    # Log directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name  = f"PushT-MJX-{timestamp}"
    logdir    = Path(_LOGDIR.value or "logs").resolve() / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Experiment: {exp_name}")
    print(f"Log dir:    {logdir}")

    # Weights & Biases
    if _USE_WANDB.value and not _PLAY_ONLY.value:
        import wandb  # type: ignore
        wandb.init(project="pushT-mjx", name=exp_name)

    # Environments
    env_cfg  = default_config()
    env      = PushTEnvMjx(env_cfg)
    eval_env = PushTEnvMjx(env_cfg)

    print(f"Observation size: {env.observation_size}")
    print(f"Action size:      {env.action_size}")
    print(f"n_substeps:       {env.n_substeps}")
    print(f"PPO params:\n{ppo_params}")

    # Network factory
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory,
    )

    training_params = {k: v for k, v in ppo_params.items() if k != "network_factory"}

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        seed=_SEED.value,
        save_checkpoint_path=ckpt_path,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=ppo_params.num_eval_envs,
    )

    times = [time.monotonic()]

    def progress(num_steps: int, metrics: dict):
        times.append(time.monotonic())
        rew = metrics.get("eval/episode_reward", float("nan"))
        print(f"  step={num_steps:>10d}  eval_reward={rew:.3f}")
        if _USE_WANDB.value and not _PLAY_ONLY.value:
            import wandb  # type: ignore
            wandb.log(metrics, step=num_steps)

    print("Starting training …")
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress,
        eval_env=eval_env,
    )
    print("Training complete.")

    if len(times) > 1:
        print(f"JIT compile time: {times[1] - times[0]:.1f}s")
        print(f"Training time:    {times[-1] - times[1]:.1f}s")

    # ---------------------------------------------------------------
    # Inference rollout + video
    # ---------------------------------------------------------------
    print("Running inference rollout …")
    inference_fn     = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)

    infer_env = PushTEnvMjx(env_cfg)
    wrapped_infer_env = wrapper.wrap_for_brax_training(
        infer_env,
        episode_length=ppo_params.episode_length,
        action_repeat=ppo_params.get("action_repeat", 1),
    )

    n = _NUM_VIDEOS.value
    rng = jax.random.split(jax.random.PRNGKey(_SEED.value + 99), n)
    reset_states = jax.jit(wrapped_infer_env.reset)(rng)

    # Thin State sentinel for scanning
    empty_data = jax.tree.map(lambda x: None, reset_states.data)
    empty_traj = reset_states.replace(data=empty_data)
    empty_traj = jax.tree.map(lambda _: None, empty_traj)

    def rollout_step(carry, _):
        state, rng = carry
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, n)
        act  = jax.vmap(jit_inference_fn)(state.obs, keys)[0]
        state = wrapped_infer_env.step(state, act)
        traj_data = jax.tree.map(
            lambda full: full, state
        )
        return (state, rng), state

    @jax.jit
    def do_rollout(init_state, rng):
        _, traj = jax.lax.scan(
            rollout_step, (init_state, rng), None,
            length=ppo_params.episode_length
        )
        return traj

    traj = do_rollout(reset_states, jax.random.PRNGKey(_SEED.value + 100))
    # traj shape: (episode_length, n, ...) → per-video: (n, episode_length, ...)
    traj = jax.tree.map(lambda x: jp.moveaxis(x, 0, 1), traj)

    render_every = 2
    fps = 1.0 / infer_env.dt / render_every
    scene_option = mujoco.MjvOption()

    for i in range(n):
        traj_i = jax.tree.map(lambda x: x[i], traj)
        frames_list = [
            jax.tree.map(lambda x: x[j], traj_i)
            for j in range(ppo_params.episode_length)
        ]
        frames = infer_env.render(
            frames_list[::render_every],
            height=480, width=640,
            scene_option=scene_option,
        )
        video_path = logdir / f"rollout_{i}.mp4"
        media.write_video(str(video_path), frames, fps=fps)
        print(f"Saved video: {video_path}")

    print("Done.")


def run():
    app.run(main)


if __name__ == "__main__":
    run()
