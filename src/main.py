from __future__ import annotations

import argparse
import sys
from typing import cast

from src.app import SimulationApp
from src.config import QualityName, default_configs
from src.render.renderer import run_interactive


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous 3-joint IK throwing-arm demo")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic spawn and behavior")
    parser.add_argument(
        "--quality",
        type=str,
        default="balanced",
        choices=("low", "balanced", "high"),
        help="Rendering quality preset",
    )
    parser.add_argument("--headless", action="store_true", help="Run physics/controller loop without OpenGL window")
    parser.add_argument("--episodes", type=int, default=20, help="Number of autonomous episodes to run")
    parser.add_argument("--pixel-size", type=float, default=None, help="Override pixel quantization cell size")
    parser.add_argument("--show-hud", action="store_true", help="Show in-window telemetry HUD")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.episodes <= 0:
        print("`--episodes` must be a positive integer.")
        return 2

    sim_cfg, arm_cfg, spawn_cfg, planner_cfg, render_cfg, control_cfg = default_configs(
        quality=cast(QualityName, args.quality),
        pixel_size=args.pixel_size,
        show_hud=args.show_hud,
    )
    try:
        app = SimulationApp(
            sim_config=sim_cfg,
            arm_config=arm_cfg,
            spawn_config=spawn_cfg,
            planner_config=planner_cfg,
            control_config=control_cfg,
            seed=args.seed,
            max_episodes=args.episodes,
        )
    except RuntimeError as exc:
        print(f"Failed to initialize simulation backend: {exc}")
        print("Install `pybullet` in a Python 3.11 environment to run this project.")
        return 1

    if args.headless:
        stats = app.run_headless()
        app.shutdown()
        print("Headless run complete")
        print(f"  Episodes: {stats.episodes_total}")
        print(f"  Successes: {stats.successes}")
        print(f"  Failures: {stats.failures}")
        print(f"  Success rate: {stats.success_rate * 100.0:.1f}%")
        print(f"  Avg cycle time: {stats.average_cycle_time:.2f}s")
        return 0

    run_interactive(app, render_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
