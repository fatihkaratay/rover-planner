"""
PPO training script for the RoverEnv.

Usage:
    python -m backend.rl.train                        # default 500k steps
    python -m backend.rl.train --timesteps 1000000
    python -m backend.rl.train --timesteps 100000 --run-name quick-test
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

from backend.env.rover_env import RoverEnv

MODELS_DIR = Path("backend/rl/models")
LOGS_DIR   = Path("backend/rl/logs")


def _make_env_fn(seed: int):
    """Return a no-arg factory for a seeded, monitored RoverEnv."""
    def _init():
        return Monitor(RoverEnv(seed=seed))
    return _init


def train(timesteps: int = 500_000, run_name: str = "ppo_rover") -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Vectorised envs — PPO benefits from parallel rollouts
    n_envs = 4
    vec_env = DummyVecEnv([_make_env_fn(i) for i in range(n_envs)])

    # Separate eval env (single, fixed seed)
    eval_env = Monitor(RoverEnv(seed=999))

    # ── PPO hyperparameters ────────────────────────────────────────────────
    # Tuned for a dense 125-dim obs, discrete 9-action space, sparse-ish reward.
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,           # rollout length per env before update
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # entropy bonus keeps exploration alive
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(LOGS_DIR),
        seed=42,
        policy_kwargs=dict(
            net_arch=[256, 256],   # two hidden layers
        ),
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=str(MODELS_DIR / run_name),
        name_prefix="ckpt",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / run_name / "best"),
        log_path=str(LOGS_DIR / run_name),
        eval_freq=max(25_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    print(f"Training PPO for {timesteps:,} timesteps — run: {run_name}")
    print(f"Obs dim: {vec_env.observation_space.shape}  Actions: {vec_env.action_space.n}")

    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name=run_name,
        progress_bar=True,
    )

    final_path = MODELS_DIR / run_name / "final_model"
    model.save(str(final_path))
    print(f"\nSaved final model → {final_path}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--run-name",  type=str, default="ppo_rover")
    args = parser.parse_args()
    train(args.timesteps, args.run_name)
