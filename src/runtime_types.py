from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class JointState:
    positions: np.ndarray
    velocities: np.ndarray


@dataclass(slots=True)
class EndEffectorState:
    position: np.ndarray
    velocity: np.ndarray


@dataclass(slots=True)
class ThrowPlan:
    release_joint_positions: np.ndarray
    release_joint_velocities: np.ndarray
    release_position: np.ndarray
    release_velocity: np.ndarray
    predicted_landing: np.ndarray
    flight_time: float
    cost: float


@dataclass(slots=True)
class EpisodeStats:
    episodes_total: int = 0
    successes: int = 0
    failures: int = 0
    average_cycle_time: float = 0.0
    last_cycle_time: float = 0.0
    success_rate: float = 0.0
    current_phase: str = "SPAWN"

    def register_episode(self, success: bool, cycle_time: float) -> None:
        self.episodes_total += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self.last_cycle_time = float(cycle_time)
        n = float(self.episodes_total)
        self.average_cycle_time = (
            cycle_time
            if self.episodes_total == 1
            else ((self.average_cycle_time * (n - 1.0)) + cycle_time) / n
        )
        self.success_rate = self.successes / n if n > 0 else 0.0


@dataclass(slots=True)
class WorldState:
    sim_time: float
    phase: str
    episode_index: int
    joint_state: JointState
    ee_state: EndEffectorState
    block_position: np.ndarray
    block_velocity: np.ndarray
    target_position: np.ndarray
    target_radius: float
    grasp_active: bool
    stats: EpisodeStats
    predicted_landing: np.ndarray | None = None
    render_capsule_a: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float32))
    render_capsule_b: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float32))
    render_capsule_r: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

