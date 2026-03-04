from __future__ import annotations

import numpy as np

from src.config import ArmConfig, PlannerConfig, SimConfig
from src.planner.ballistic import BallisticThrowPlanner
from src.robot.kinematics import ArmGeometry, ArmKinematics


def test_ballistic_planner_hits_target_analytically() -> None:
    arm = ArmConfig()
    planner_cfg = PlannerConfig()
    sim_cfg = SimConfig()
    kin = ArmKinematics(
        ArmGeometry(
            base_height=arm.base_height,
            link1_length=arm.link1_length,
            link2_length=arm.link2_length,
            tool_length=arm.tool_length,
        )
    )
    planner = BallisticThrowPlanner(
        kinematics=kin,
        arm_config=arm,
        planner_config=planner_cfg,
        gravity=sim_cfg.gravity,
    )
    target = np.array([1.1, 0.35, 0.0], dtype=np.float64)
    plan = planner.plan(current_q=arm.home_joint_positions, target_position=target)
    assert plan is not None

    g_vec = np.array([0.0, 0.0, -sim_cfg.gravity], dtype=np.float64)
    landing = plan.release_position + plan.release_velocity * plan.flight_time + 0.5 * g_vec * (plan.flight_time**2)
    np.testing.assert_allclose(landing[:2], target[:2], atol=1e-6)

