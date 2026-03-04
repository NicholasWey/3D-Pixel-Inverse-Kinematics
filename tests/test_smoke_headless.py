from __future__ import annotations

import pytest

pytest.importorskip("pybullet")

from src.app import SimulationApp
from src.config import default_configs


def _run(seed: int, episodes: int) -> tuple[object, list[tuple[tuple[float, ...], tuple[float, ...], float]]]:
    sim_cfg, arm_cfg, spawn_cfg, planner_cfg, _render_cfg, control_cfg = default_configs()
    app = SimulationApp(
        sim_config=sim_cfg,
        arm_config=arm_cfg,
        spawn_config=spawn_cfg,
        planner_config=planner_cfg,
        control_config=control_cfg,
        seed=seed,
        max_episodes=episodes,
    )
    try:
        stats = app.run_headless()
        history = list(app.controller.spawn_history)
    finally:
        app.shutdown()
    return stats, history


def test_headless_smoke_multi_episode() -> None:
    stats, _ = _run(seed=17, episodes=8)
    assert stats.episodes_total == 8
    assert stats.successes + stats.failures == 8
    assert stats.successes >= 1


def test_spawn_determinism_with_fixed_seed() -> None:
    _, h1 = _run(seed=123, episodes=6)
    _, h2 = _run(seed=123, episodes=6)
    assert h1 == h2

