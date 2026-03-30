"""
Evaluation script — compares trained PPO against baselines.

Metrics (per POMDP.md Section 10 / PLAN.md paper evaluation):
  - Success rate       : % of episodes reaching the goal
  - Mean path length   : cells travelled (efficiency)
  - Mean energy used   : energy consumed per episode
  - Mean slip events   : risk proxy
  - Mean episode reward: overall performance signal

Baselines:
  - Random policy      : lower bound
  - A* (oracle)        : upper bound — optimal path on known cost map

Usage:
    python -m backend.rl.evaluate                              # latest best model
    python -m backend.rl.evaluate --model backend/rl/models/ppo_rover_v1/best/best_model
    python -m backend.rl.evaluate --episodes 50
"""

import argparse
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from stable_baselines3 import PPO

from backend.env.rover_env import RoverEnv
from backend.terrain.cost_map import build_cost_map
from backend.terrain.dem_loader import load_dem
from backend.planner.astar import plan, NoPathError


@dataclass
class EpisodeStats:
    success: bool
    steps: int
    energy_used: float
    slip_events: int
    total_reward: float
    dist_final: float


@dataclass
class EvalResult:
    name: str
    episodes: list[EpisodeStats] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return np.mean([e.success for e in self.episodes]) * 100

    @property
    def mean_steps(self) -> float:
        return np.mean([e.steps for e in self.episodes])

    @property
    def mean_energy(self) -> float:
        return np.mean([e.energy_used for e in self.episodes])

    @property
    def mean_slips(self) -> float:
        return np.mean([e.slip_events for e in self.episodes])

    @property
    def mean_reward(self) -> float:
        return np.mean([e.total_reward for e in self.episodes])

    def print_summary(self) -> None:
        print(f"\n{'─'*50}")
        print(f"  {self.name}")
        print(f"{'─'*50}")
        print(f"  Success rate   : {self.success_rate:.1f}%")
        print(f"  Mean reward    : {self.mean_reward:.1f}")
        print(f"  Mean steps     : {self.mean_steps:.0f}")
        print(f"  Mean energy    : {self.mean_energy:.1f}")
        print(f"  Mean slips     : {self.mean_slips:.2f}")
        print(f"{'─'*50}")


# ── Episode runners ────────────────────────────────────────────────────────

def run_episode_ppo(env: RoverEnv, model: PPO) -> EpisodeStats:
    obs, info = env.reset()
    total_reward = 0.0
    slip_events = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        if info["slipped"]:
            slip_events += 1
        if terminated or truncated:
            break

    return EpisodeStats(
        success=terminated and info["dist_to_goal"] <= 1.5,
        steps=info["steps"],
        energy_used=env.energy_max - info["energy"],
        slip_events=slip_events,
        total_reward=total_reward,
        dist_final=info["dist_to_goal"],
    )


def run_episode_random(env: RoverEnv) -> EpisodeStats:
    obs, info = env.reset()
    total_reward = 0.0
    slip_events = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info["slipped"]:
            slip_events += 1
        if terminated or truncated:
            break

    return EpisodeStats(
        success=terminated and info["dist_to_goal"] <= 1.5,
        steps=info["steps"],
        energy_used=env.energy_max - info["energy"],
        slip_events=slip_events,
        total_reward=total_reward,
        dist_final=info["dist_to_goal"],
    )


def run_episode_astar(env: RoverEnv) -> EpisodeStats:
    """
    A* oracle: plan optimal path on full known cost map, then execute it.
    Not subject to partial observability — this is the upper-bound baseline.
    """
    _, info = env.reset()
    start = (info["row"], info["col"])
    goal  = (info["goal_row"], info["goal_col"])

    try:
        result = plan(env.cost_map, start, goal)
    except (NoPathError, ValueError):
        # No path exists for this start/goal — count as failure
        return EpisodeStats(
            success=False, steps=0, energy_used=0,
            slip_events=0, total_reward=-50.0, dist_final=999,
        )

    # Execute planned path step by step
    total_reward = 0.0
    slip_events = 0
    prev = start

    for cell in result.path[1:]:
        dr = cell[0] - prev[0]
        dc = cell[1] - prev[1]
        # Map (dr, dc) back to action index
        from backend.env.rover_env import ACTIONS
        try:
            action = ACTIONS.index((dr, dc))
        except ValueError:
            action = 8  # STOP fallback
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info["slipped"]:
            slip_events += 1
            # On slip, replanning would be needed — for simplicity, accept it
        prev = (info["row"], info["col"])
        if terminated or truncated:
            break

    return EpisodeStats(
        success=info["dist_to_goal"] <= 1.5,
        steps=info["steps"],
        energy_used=env.energy_max - info["energy"],
        slip_events=slip_events,
        total_reward=total_reward,
        dist_final=info["dist_to_goal"],
    )


# ── Main ───────────────────────────────────────────────────────────────────

def evaluate(model_path: str | None, n_episodes: int = 30) -> None:
    # Find model
    if model_path is None:
        candidates = sorted(Path("backend/rl/models").glob("**/best_model.zip"))
        if not candidates:
            candidates = sorted(Path("backend/rl/models").glob("**/final_model.zip"))
        if not candidates:
            print("No trained model found. Run: python -m backend.rl.train")
            return
        model_path = str(candidates[-1])

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    # Shared env (reset between episodes, same terrain)
    env = RoverEnv(seed=7)

    print(f"Running {n_episodes} evaluation episodes per policy…")

    ppo_result    = EvalResult("PPO (trained)")
    random_result = EvalResult("Random policy (lower bound)")
    astar_result  = EvalResult("A* oracle (upper bound)")

    for ep in range(n_episodes):
        # Use same seed per episode so all policies face identical start/goal
        ep_seed = 1000 + ep
        env.np_random, _ = __import__('gymnasium').utils.seeding.np_random(ep_seed)

        ppo_result.episodes.append(run_episode_ppo(env, model))
        env.np_random, _ = __import__('gymnasium').utils.seeding.np_random(ep_seed)
        random_result.episodes.append(run_episode_random(env))
        env.np_random, _ = __import__('gymnasium').utils.seeding.np_random(ep_seed)
        astar_result.episodes.append(run_episode_astar(env))

        print(f"  ep {ep+1:3d}/{n_episodes}  PPO={'✓' if ppo_result.episodes[-1].success else '✗'}  "
              f"Rand={'✓' if random_result.episodes[-1].success else '✗'}  "
              f"A*={'✓' if astar_result.episodes[-1].success else '✗'}")

    # Print results
    for result in [ppo_result, random_result, astar_result]:
        result.print_summary()

    # Summary comparison table
    print(f"\n{'═'*50}")
    print(f"  COMPARISON SUMMARY  ({n_episodes} episodes)")
    print(f"{'═'*50}")
    print(f"  {'Metric':<20} {'PPO':>8} {'Random':>8} {'A*':>8}")
    print(f"  {'─'*44}")
    print(f"  {'Success rate %':<20} {ppo_result.success_rate:>7.1f}% "
          f"{random_result.success_rate:>7.1f}% {astar_result.success_rate:>7.1f}%")
    print(f"  {'Mean reward':<20} {ppo_result.mean_reward:>8.1f} "
          f"{random_result.mean_reward:>8.1f} {astar_result.mean_reward:>8.1f}")
    print(f"  {'Mean steps':<20} {ppo_result.mean_steps:>8.0f} "
          f"{random_result.mean_steps:>8.0f} {astar_result.mean_steps:>8.0f}")
    print(f"  {'Mean slips':<20} {ppo_result.mean_slips:>8.2f} "
          f"{random_result.mean_slips:>8.2f} {astar_result.mean_slips:>8.2f}")
    print(f"{'═'*50}")

    # Interpretation
    ppo_vs_random = ppo_result.success_rate - random_result.success_rate
    ppo_vs_astar  = ppo_result.success_rate / max(astar_result.success_rate, 1) * 100
    print(f"\n  PPO beats random by  : {ppo_vs_random:+.1f}% success rate")
    print(f"  PPO vs A* oracle     : {ppo_vs_astar:.0f}% of A* performance")
    print(f"\n  A trained model is good if:")
    print(f"    • Success rate significantly above random")
    print(f"    • Fewer slips than random (risk-awareness is working)")
    print(f"    • Reward gap to A* is closing with more training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, default=None)
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()
    evaluate(args.model, args.episodes)
