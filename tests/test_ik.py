from __future__ import annotations

import numpy as np

from src.config import ArmConfig
from src.robot.ik import DampedLeastSquaresIK
from src.robot.kinematics import ArmGeometry, ArmKinematics


def _solver() -> tuple[ArmConfig, ArmKinematics, DampedLeastSquaresIK]:
    arm = ArmConfig()
    kin = ArmKinematics(
        ArmGeometry(
            base_height=arm.base_height,
            link1_length=arm.link1_length,
            link2_length=arm.link2_length,
            tool_length=arm.tool_length,
        )
    )
    return arm, kin, DampedLeastSquaresIK(kinematics=kin)


def test_ik_converges_reachable_target() -> None:
    arm, kin, solver = _solver()
    target_joint_pose = np.array([0.42, 0.58, -0.92], dtype=np.float64)
    target = kin.forward_kinematics(target_joint_pose)
    result = solver.solve(
        target=target,
        initial_guess=arm.home_joint_positions,
        lower_limits=arm.joint_lower_limits,
        upper_limits=arm.joint_upper_limits,
    )
    ee = kin.forward_kinematics(result.solution)
    err = float(np.linalg.norm(ee - result.clamped_target))
    assert err < 1e-2
    assert np.all(result.solution <= arm.joint_upper_limits + 1e-8)
    assert np.all(result.solution >= arm.joint_lower_limits - 1e-8)


def test_ik_handles_unreachable_target_without_nan() -> None:
    arm, _, solver = _solver()
    far_target = np.array([8.0, 0.0, 6.0], dtype=np.float64)
    result = solver.solve(
        target=far_target,
        initial_guess=arm.home_joint_positions,
        lower_limits=arm.joint_lower_limits,
        upper_limits=arm.joint_upper_limits,
    )
    assert np.all(np.isfinite(result.solution))
    assert np.all(result.solution <= arm.joint_upper_limits + 1e-8)
    assert np.all(result.solution >= arm.joint_lower_limits - 1e-8)
