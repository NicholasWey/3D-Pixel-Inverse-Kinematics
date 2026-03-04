from __future__ import annotations

import math

import numpy as np

from src.config import SpawnConfig


def _sample_annulus(rng: np.random.Generator, min_radius: float, max_radius: float) -> np.ndarray:
    angle = float(rng.uniform(0.0, math.tau))
    radius = float(rng.uniform(min_radius, max_radius))
    return np.array([math.cos(angle) * radius, math.sin(angle) * radius], dtype=np.float64)


def sample_episode_layout(
    rng: np.random.Generator,
    spawn_config: SpawnConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    for _ in range(512):
        block_xy = _sample_annulus(rng, spawn_config.block_annulus_min, spawn_config.block_annulus_max)
        target_xy = _sample_annulus(rng, spawn_config.target_annulus_min, spawn_config.target_annulus_max)
        if abs(block_xy[0]) > spawn_config.arena_extent_xy or abs(block_xy[1]) > spawn_config.arena_extent_xy:
            continue
        if abs(target_xy[0]) > spawn_config.arena_extent_xy or abs(target_xy[1]) > spawn_config.arena_extent_xy:
            continue
        if float(np.linalg.norm(block_xy - target_xy)) < spawn_config.min_block_target_distance:
            continue

        block = np.array([block_xy[0], block_xy[1], spawn_config.block_half_extent], dtype=np.float64)
        target = np.array([target_xy[0], target_xy[1], 0.0], dtype=np.float64)
        target_radius = float(
            spawn_config.target_radius_base
            + rng.uniform(-spawn_config.target_radius_jitter, spawn_config.target_radius_jitter)
        )
        target_radius = max(0.12, target_radius)
        return block, target, target_radius

    fallback_block = np.array(
        [spawn_config.block_annulus_max * 0.8, 0.0, spawn_config.block_half_extent],
        dtype=np.float64,
    )
    fallback_target = np.array([-spawn_config.target_annulus_max * 0.7, 0.0, 0.0], dtype=np.float64)
    return fallback_block, fallback_target, spawn_config.target_radius_base

