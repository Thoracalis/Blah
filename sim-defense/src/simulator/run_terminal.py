"""
run_terminal.py – Step 1: Run the simulator in terminal mode, no AI, no UI.

Usage:
    python -m simulator.run_terminal [--episodes N] [--seed S] [--config path]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# Allow running as   python -m simulator.run_terminal   from sim-defense/src
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from simulator.env import AirDefenseEnv

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    # Make graph_file path absolute relative to config file location
    cfg_dir = Path(path).parent
    sim_cfg = cfg.get("simulator", {})
    gf = sim_cfg.get("graph_file", "src/data/sweden_graph.json")
    if not Path(gf).is_absolute():
        sim_cfg["graph_file"] = str(cfg_dir / gf)
    return cfg


def run_episode(env: AirDefenseEnv, seed: int, verbose: bool = True) -> dict:
    """Run one episode with a random defender. Returns summary stats."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if verbose and steps % 20 == 0:
            env._render_terminal()

        if terminated or truncated:
            break

    return {
        "seed": seed,
        "steps": steps,
        "total_reward": total_reward,
        "final_defense_ratio": info["defense_ratio"],
        "last_stop_rate": info["last_stop_rate"],
        "final_active_threats": info["active_threats"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Air Defense Simulator – terminal mode")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).parent.parent.parent / "config.yaml"),
                        help="Path to config YAML")
    parser.add_argument("--verbose", action="store_true", help="Print step-level info")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = AirDefenseEnv(cfg, render_mode="human" if args.verbose else None)

    print("\n" + "═" * 70)
    print("  AIR DEFENSE RESILIENCE SIMULATOR  –  Terminal Mode  (Step 1)")
    print("  DISCLAIMER: All data fictional. Educational purposes only.")
    print("═" * 70)

    results = []
    episode_iter = range(args.episodes)
    if HAS_RICH and not args.verbose:
        episode_iter = track(episode_iter, description="Running episodes…")

    for i in episode_iter:
        seed = args.seed + i
        t0 = time.perf_counter()
        stats = run_episode(env, seed=seed, verbose=args.verbose)
        elapsed = time.perf_counter() - t0
        stats["elapsed_s"] = round(elapsed, 2)
        results.append(stats)
        if not args.verbose:
            print(f"  Episode {i + 1:3d} | seed={seed} | "
                  f"reward={stats['total_reward']:8.2f} | "
                  f"defense={stats['final_defense_ratio']:.3f} | "
                  f"stop_rate={stats['last_stop_rate']:.2f} | "
                  f"{elapsed:.2f}s")

    # Summary
    avg_reward = sum(r["total_reward"] for r in results) / len(results)
    avg_defense = sum(r["final_defense_ratio"] for r in results) / len(results)
    avg_stop = sum(r["last_stop_rate"] for r in results) / len(results)

    print("\n" + "─" * 70)
    print(f"  Summary over {len(results)} episodes:")
    print(f"    Avg reward:        {avg_reward:8.2f}")
    print(f"    Avg defense ratio: {avg_defense:.3f}")
    print(f"    Avg stop rate:     {avg_stop:.3f}")
    print("─" * 70 + "\n")


if __name__ == "__main__":
    main()
