from __future__ import annotations

import math

import numpy as np

from src.config import ArmConfig, PlannerConfig
from src.robot.kinematics import ArmKinematics
from src.runtime_types import ThrowPlan


class BallisticThrowPlanner:
    def __init__(
        self,
        kinematics: ArmKinematics,
        arm_config: ArmConfig,
        planner_config: PlannerConfig,
        gravity: float,
    ) -> None:
        self.kinematics = kinematics
        self.arm_config = arm_config
        self.planner_config = planner_config
        self.gravity = float(gravity)

    def plan(self, current_q: np.ndarray, target_position: np.ndarray) -> ThrowPlan | None:
        target = np.asarray(target_position, dtype=np.float64)
        q_curr = np.asarray(current_q, dtype=np.float64)
        g_vec = np.array([0.0, 0.0, -self.gravity], dtype=np.float64)

        base_yaw = math.atan2(float(target[1]), float(target[0]))
        yaw_values = np.linspace(
            base_yaw - self.planner_config.yaw_offset,
            base_yaw + self.planner_config.yaw_offset,
            self.planner_config.yaw_samples,
            dtype=np.float64,
        )
        shoulder_values = np.linspace(
            self.planner_config.shoulder_min,
            self.planner_config.shoulder_max,
            self.planner_config.shoulder_samples,
            dtype=np.float64,
        )
        elbow_values = np.linspace(
            self.planner_config.elbow_min,
            self.planner_config.elbow_max,
            self.planner_config.elbow_samples,
            dtype=np.float64,
        )
        flight_times = np.linspace(
            self.planner_config.flight_time_min,
            self.planner_config.flight_time_max,
            self.planner_config.flight_time_samples,
            dtype=np.float64,
        )

        best_plan: ThrowPlan | None = None
        best_cost = float("inf")
        lower = self.arm_config.joint_lower_limits
        upper = self.arm_config.joint_upper_limits
        qdot_limit = self.arm_config.joint_velocity_limits * self.planner_config.joint_velocity_margin

        for yaw in yaw_values:
            for shoulder in shoulder_values:
                for elbow in elbow_values:
                    q_release = np.array([yaw, shoulder, elbow], dtype=np.float64)
                    q_release = np.clip(q_release, lower, upper)
                    release_pos = self.kinematics.forward_kinematics(q_release)
                    if release_pos[2] < self.planner_config.release_height_min:
                        continue

                    delta_target = target - release_pos
                    for flight_t in flight_times:
                        velocity = (delta_target - 0.5 * g_vec * (flight_t**2)) / flight_t
                        speed = float(np.linalg.norm(velocity))
                        if speed > self.planner_config.release_speed_max:
                            continue

                        jac = self.kinematics.jacobian(q_release)
                        qdot = np.linalg.pinv(jac, rcond=1e-3) @ velocity
                        if np.any(np.abs(qdot) > qdot_limit):
                            continue

                        predicted = release_pos + velocity * flight_t + 0.5 * g_vec * (flight_t**2)
                        landing_error = float(np.linalg.norm(predicted[:2] - target[:2]))
                        posture_cost = float(np.linalg.norm(q_release - q_curr))
                        flight_cost = abs(float(flight_t - self.planner_config.preferred_flight_time))
                        cost = (landing_error * 10.0) + (0.22 * speed) + (0.14 * posture_cost) + (0.25 * flight_cost)
                        if cost >= best_cost:
                            continue

                        best_cost = cost
                        best_plan = ThrowPlan(
                            release_joint_positions=q_release.copy(),
                            release_joint_velocities=qdot.astype(np.float64),
                            release_position=release_pos.astype(np.float64),
                            release_velocity=velocity.astype(np.float64),
                            predicted_landing=predicted.astype(np.float64),
                            flight_time=float(flight_t),
                            cost=float(cost),
                        )

        return best_plan

