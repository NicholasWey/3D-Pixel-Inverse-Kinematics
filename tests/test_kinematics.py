from __future__ import annotations

import numpy as np

from src.config import ArmConfig
from src.robot.kinematics import ArmGeometry, ArmKinematics


def _kinematics() -> ArmKinematics:
    arm = ArmConfig()
    return ArmKinematics(
        ArmGeometry(
            base_height=arm.base_height,
            link1_length=arm.link1_length,
            link2_length=arm.link2_length,
            tool_length=arm.tool_length,
        )
    )


def test_forward_kinematics_zero_pose() -> None:
    arm = ArmConfig()
    kin = _kinematics()
    q = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    ee = kin.forward_kinematics(q)
    expected = np.array(
        [arm.link1_length + arm.link2_length + arm.tool_length, 0.0, arm.base_height],
        dtype=np.float64,
    )
    np.testing.assert_allclose(ee, expected, atol=1e-7)


def test_forward_kinematics_yaw_quarter_turn() -> None:
    arm = ArmConfig()
    kin = _kinematics()
    q = np.array([np.pi * 0.5, 0.0, 0.0], dtype=np.float64)
    ee = kin.forward_kinematics(q)
    expected = np.array(
        [0.0, arm.link1_length + arm.link2_length + arm.tool_length, arm.base_height],
        dtype=np.float64,
    )
    np.testing.assert_allclose(ee, expected, atol=1e-7)


def test_jacobian_matches_finite_difference() -> None:
    kin = _kinematics()
    q = np.array([0.42, 0.31, -0.55], dtype=np.float64)
    jac = kin.jacobian(q)
    eps = 1e-6
    jac_fd = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        dq = np.zeros(3, dtype=np.float64)
        dq[i] = eps
        plus = kin.forward_kinematics(q + dq)
        minus = kin.forward_kinematics(q - dq)
        jac_fd[:, i] = (plus - minus) / (2.0 * eps)
    np.testing.assert_allclose(jac, jac_fd, atol=1e-5, rtol=1e-4)

