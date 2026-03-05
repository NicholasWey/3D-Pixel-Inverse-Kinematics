from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

QualityName = Literal["low", "balanced", "high"]

QUALITY_PRESETS: dict[QualityName, dict[str, float | int]] = {
    "low": {"max_steps": 96, "shadow_steps": 16, "pixel_size": 7.0},
    "balanced": {"max_steps": 128, "shadow_steps": 20, "pixel_size": 5.5},
    "high": {"max_steps": 170, "shadow_steps": 28, "pixel_size": 4.0},
}


@dataclass(slots=True)
class SimConfig:
    time_step: float = 1.0 / 240.0
    gravity: float = 9.81
    max_frame_time: float = 0.25


@dataclass(slots=True)
class ArmConfig:
    base_height: float = 0.66
    link1_length: float = 0.72
    link2_length: float = 0.58
    # The control-point offset from forearm joint to the tool link COM in the URDF.
    tool_length: float = 0.09
    joint_lower_limits: np.ndarray = field(
        default_factory=lambda: np.array(
            [-np.pi, -1.30, -2.15],
            dtype=np.float64,
        )
    )
    joint_upper_limits: np.ndarray = field(
        default_factory=lambda: np.array(
            [np.pi, 0.95, 0.15],
            dtype=np.float64,
        )
    )
    joint_velocity_limits: np.ndarray = field(
        default_factory=lambda: np.array([4.5, 4.5, 5.0], dtype=np.float64)
    )
    joint_force_limits: np.ndarray = field(
        default_factory=lambda: np.array([140.0, 130.0, 120.0], dtype=np.float64)
    )
    home_joint_positions: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.58, -0.82], dtype=np.float64)
    )
    position_gains: np.ndarray = field(
        default_factory=lambda: np.array([0.26, 0.28, 0.30], dtype=np.float64)
    )
    velocity_gains: np.ndarray = field(
        default_factory=lambda: np.array([0.92, 0.95, 0.96], dtype=np.float64)
    )
    disable_arm_floor_collision: bool = False


@dataclass(slots=True)
class SpawnConfig:
    arena_extent_xy: float = 1.75
    # Keep block spawns in the physically reachable low-pick ring for this 3-DOF arm.
    block_annulus_min: float = 1.10
    block_annulus_max: float = 1.35
    target_annulus_min: float = 0.70
    target_annulus_max: float = 1.45
    min_block_target_distance: float = 0.55
    block_half_extent: float = 0.06
    target_radius_base: float = 0.23
    target_radius_jitter: float = 0.05


@dataclass(slots=True)
class PlannerConfig:
    flight_time_min: float = 0.35
    flight_time_max: float = 1.15
    flight_time_samples: int = 13
    release_speed_max: float = 7.2
    release_height_min: float = 0.32
    yaw_samples: int = 9
    yaw_offset: float = 0.60
    shoulder_min: float = -0.20
    shoulder_max: float = 0.90
    shoulder_samples: int = 8
    elbow_min: float = -1.95
    elbow_max: float = -0.25
    elbow_samples: int = 8
    preferred_flight_time: float = 0.68
    joint_velocity_margin: float = 1.10


@dataclass(slots=True)
class RenderConfig:
    quality: QualityName = "balanced"
    max_steps: int = int(QUALITY_PRESETS["balanced"]["max_steps"])
    shadow_steps: int = int(QUALITY_PRESETS["balanced"]["shadow_steps"])
    pixel_size: float = float(QUALITY_PRESETS["balanced"]["pixel_size"])
    window_size: tuple[int, int] = (1600, 900)
    fov_y_degrees: float = 54.0
    floor_extent: float = 3.0
    show_hud: bool = False


@dataclass(slots=True)
class ControlConfig:
    pick_approach_height: float = 0.22
    pick_contact_height: float = 0.10
    waypoint_tolerance: float = 0.06
    release_pose_tolerance: float = 0.08
    grasp_distance_threshold: float = 0.13
    grasp_relative_speed_threshold: float = 1.8
    move_timeout_seconds: float = 3.8
    evaluate_timeout_seconds: float = 4.2
    release_windup_seconds: float = 0.075
    settle_speed_threshold: float = 0.12
    settle_height_threshold: float = 0.03
    max_grasp_retries: int = 1


def make_render_config(
    quality: QualityName = "balanced",
    pixel_size: float | None = None,
    show_hud: bool = False,
) -> RenderConfig:
    preset = QUALITY_PRESETS[quality]
    return RenderConfig(
        quality=quality,
        max_steps=int(preset["max_steps"]),
        shadow_steps=int(preset["shadow_steps"]),
        pixel_size=float(preset["pixel_size"] if pixel_size is None else pixel_size),
        show_hud=show_hud,
    )


def default_configs(
    quality: QualityName = "balanced",
    pixel_size: float | None = None,
    show_hud: bool = False,
) -> tuple[SimConfig, ArmConfig, SpawnConfig, PlannerConfig, RenderConfig, ControlConfig]:
    return (
        SimConfig(),
        ArmConfig(),
        SpawnConfig(),
        PlannerConfig(),
        make_render_config(quality=quality, pixel_size=pixel_size, show_hud=show_hud),
        ControlConfig(),
    )
