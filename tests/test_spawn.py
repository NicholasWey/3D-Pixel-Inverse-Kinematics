from __future__ import annotations

import numpy as np

from src.config import SpawnConfig
from src.physics.spawn import sample_episode_layout


def test_spawn_sampler_respects_constraints() -> None:
    cfg = SpawnConfig()
    rng = np.random.default_rng(7)
    for _ in range(200):
        block, target, radius = sample_episode_layout(rng, cfg)
        assert abs(block[0]) <= cfg.arena_extent_xy + 1e-8
        assert abs(block[1]) <= cfg.arena_extent_xy + 1e-8
        assert abs(target[0]) <= cfg.arena_extent_xy + 1e-8
        assert abs(target[1]) <= cfg.arena_extent_xy + 1e-8
        assert np.linalg.norm(block[:2] - target[:2]) >= cfg.min_block_target_distance - 1e-8
        assert radius >= 0.12
        assert np.isclose(block[2], cfg.block_half_extent)
        assert np.isclose(target[2], 0.0)

