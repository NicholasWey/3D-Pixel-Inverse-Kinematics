from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ArmGeometry:
    base_height: float
    link1_length: float
    link2_length: float
    tool_length: float

    @property
    def max_reach(self) -> float:
        return self.link1_length + self.link2_length + self.tool_length

    @property
    def min_reach(self) -> float:
        return abs(self.link1_length - (self.link2_length + self.tool_length))


class ArmKinematics:
    def __init__(self, geometry: ArmGeometry) -> None:
        self.geometry = geometry
        self._base = np.array([0.0, 0.0, self.geometry.base_height], dtype=np.float64)

    @property
    def base_position(self) -> np.ndarray:
        return self._base.copy()

    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        q0, q1, q2 = [float(v) for v in joint_positions]
        c0 = float(np.cos(q0))
        s0 = float(np.sin(q0))
        c1 = float(np.cos(q1))
        s1 = float(np.sin(q1))
        c12 = float(np.cos(q1 + q2))
        s12 = float(np.sin(q1 + q2))
        reach_tail = self.geometry.link2_length + self.geometry.tool_length

        local = np.array(
            [
                self.geometry.link1_length * c1 + reach_tail * c12,
                0.0,
                -(self.geometry.link1_length * s1 + reach_tail * s12),
            ],
            dtype=np.float64,
        )
        world = np.array(
            [
                self._base[0] + c0 * local[0],
                self._base[1] + s0 * local[0],
                self._base[2] + local[2],
            ],
            dtype=np.float64,
        )
        return world

    def joint_positions_world(self, joint_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q0, q1, q2 = [float(v) for v in joint_positions]
        c0 = float(np.cos(q0))
        s0 = float(np.sin(q0))
        c1 = float(np.cos(q1))
        s1 = float(np.sin(q1))
        c12 = float(np.cos(q1 + q2))
        s12 = float(np.sin(q1 + q2))

        shoulder = self._base.copy()
        elbow_local = np.array(
            [self.geometry.link1_length * c1, 0.0, -self.geometry.link1_length * s1],
            dtype=np.float64,
        )
        elbow = np.array(
            [
                shoulder[0] + c0 * elbow_local[0],
                shoulder[1] + s0 * elbow_local[0],
                shoulder[2] + elbow_local[2],
            ],
            dtype=np.float64,
        )
        wrist_local = np.array(
            [
                self.geometry.link1_length * c1 + self.geometry.link2_length * c12,
                0.0,
                -(self.geometry.link1_length * s1 + self.geometry.link2_length * s12),
            ],
            dtype=np.float64,
        )
        wrist = np.array(
            [
                shoulder[0] + c0 * wrist_local[0],
                shoulder[1] + s0 * wrist_local[0],
                shoulder[2] + wrist_local[2],
            ],
            dtype=np.float64,
        )
        ee = self.forward_kinematics(joint_positions)
        return shoulder, elbow, wrist, ee

    def jacobian(self, joint_positions: np.ndarray) -> np.ndarray:
        q0, q1, q2 = [float(v) for v in joint_positions]
        c0 = float(np.cos(q0))
        s0 = float(np.sin(q0))
        c1 = float(np.cos(q1))
        s1 = float(np.sin(q1))
        c12 = float(np.cos(q1 + q2))
        s12 = float(np.sin(q1 + q2))
        reach_tail = self.geometry.link2_length + self.geometry.tool_length

        local_x = self.geometry.link1_length * c1 + reach_tail * c12
        dlocal_dq1 = np.array(
            [
                -self.geometry.link1_length * s1 - reach_tail * s12,
                0.0,
                -(self.geometry.link1_length * c1 + reach_tail * c12),
            ],
            dtype=np.float64,
        )
        dlocal_dq2 = np.array(
            [
                -reach_tail * s12,
                0.0,
                -reach_tail * c12,
            ],
            dtype=np.float64,
        )

        dq0 = np.array([-s0 * local_x, c0 * local_x, 0.0], dtype=np.float64)
        dq1 = np.array([c0 * dlocal_dq1[0], s0 * dlocal_dq1[0], dlocal_dq1[2]], dtype=np.float64)
        dq2 = np.array([c0 * dlocal_dq2[0], s0 * dlocal_dq2[0], dlocal_dq2[2]], dtype=np.float64)

        return np.column_stack((dq0, dq1, dq2))

    def clamp_target_to_workspace(self, target: np.ndarray, margin: float = 1e-3) -> np.ndarray:
        delta = np.asarray(target, dtype=np.float64) - self._base
        norm = float(np.linalg.norm(delta))
        if norm < 1e-9:
            return self._base + np.array([self.geometry.min_reach + margin, 0.0, 0.0], dtype=np.float64)
        min_reach = max(self.geometry.min_reach + margin, 0.05)
        max_reach = self.geometry.max_reach - margin
        if norm > max_reach:
            delta *= max_reach / norm
        elif norm < min_reach:
            delta *= min_reach / norm
        return self._base + delta
