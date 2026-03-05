from __future__ import annotations

from enum import Enum

import numpy as np

from src.config import ArmConfig, ControlConfig, PlannerConfig, SimConfig, SpawnConfig
from src.physics.spawn import sample_episode_layout
from src.physics.world import PhysicsWorld
from src.planner.ballistic import BallisticThrowPlanner
from src.robot.ik import DampedLeastSquaresIK
from src.runtime_types import EpisodeStats, WorldState


class ArmPhase(str, Enum):
    SPAWN = "SPAWN"
    PLAN_PICK = "PLAN_PICK"
    MOVE_PICK = "MOVE_PICK"
    GRASP = "GRASP"
    PLAN_THROW = "PLAN_THROW"
    MOVE_THROW = "MOVE_THROW"
    RELEASE = "RELEASE"
    EVALUATE = "EVALUATE"
    RESET = "RESET"


class ArmController:
    def __init__(
        self,
        world: PhysicsWorld,
        sim_config: SimConfig,
        arm_config: ArmConfig,
        spawn_config: SpawnConfig,
        planner_config: PlannerConfig,
        control_config: ControlConfig,
        rng: np.random.Generator,
        max_episodes: int | None,
    ) -> None:
        self.world = world
        self.sim_config = sim_config
        self.arm_config = arm_config
        self.spawn_config = spawn_config
        self.planner_config = planner_config
        self.control_config = control_config
        self.rng = rng
        self.max_episodes = max_episodes

        self.ik_solver = DampedLeastSquaresIK(self.world.kinematics)
        self.throw_planner = BallisticThrowPlanner(
            kinematics=self.world.kinematics,
            arm_config=self.arm_config,
            planner_config=self.planner_config,
            gravity=self.sim_config.gravity,
        )
        self.phase = ArmPhase.SPAWN
        self.phase_timer = 0.0
        self.episode_start_time = 0.0
        self.active_episode = 0
        self.pick_waypoints: list[np.ndarray] = []
        self.pick_stage = 0
        self.grasp_retries = 0
        self.release_time_left = 0.0
        self.throw_plan = None
        self.predicted_landing: np.ndarray | None = None
        self.scored_this_episode = False
        self.done = False
        self.stats = EpisodeStats()
        self.spawn_history: list[tuple[tuple[float, ...], tuple[float, ...], float]] = []

    def _transition(self, phase: ArmPhase) -> None:
        self.phase = phase
        self.phase_timer = 0.0

    def _begin_episode(self) -> None:
        block, target, target_radius = sample_episode_layout(self.rng, self.spawn_config)
        self.world.reset_episode(block, target, target_radius)
        self.active_episode = self.stats.episodes_total + 1
        self.episode_start_time = self.world.sim_time
        self.pick_waypoints = []
        self.pick_stage = 0
        self.grasp_retries = 0
        self.release_time_left = 0.0
        self.throw_plan = None
        self.predicted_landing = None
        self.scored_this_episode = False
        self.spawn_history.append(
            (
                tuple(float(v) for v in np.round(block, 6)),
                tuple(float(v) for v in np.round(target, 6)),
                float(round(target_radius, 6)),
            )
        )
        self._transition(ArmPhase.PLAN_PICK)

    def _end_episode(self, success: bool) -> None:
        cycle = max(0.0, self.world.sim_time - self.episode_start_time)
        self.stats.register_episode(success=success, cycle_time=cycle)
        self._transition(ArmPhase.RESET)

    def update(self, dt: float) -> None:
        if self.done:
            self.world.apply_joint_position_targets(self.arm_config.home_joint_positions)
            self.stats.current_phase = self.phase.value
            return

        self.phase_timer += dt

        joint_state = self.world.get_joint_state()
        ee_state = self.world.get_end_effector_state()
        block_pos, _ = self.world.get_block_state()

        if self.phase == ArmPhase.SPAWN:
            self._begin_episode()
        elif self.phase == ArmPhase.PLAN_PICK:
            approach = block_pos + np.array([0.0, 0.0, self.control_config.pick_approach_height], dtype=np.float64)
            contact = block_pos + np.array([0.0, 0.0, self.control_config.pick_contact_height], dtype=np.float64)
            self.pick_waypoints = [approach, contact]
            self.pick_stage = 0
            self._transition(ArmPhase.MOVE_PICK)
        elif self.phase == ArmPhase.MOVE_PICK:
            if not self.pick_waypoints:
                self._transition(ArmPhase.PLAN_PICK)
            else:
                target = self.pick_waypoints[self.pick_stage].copy()
                ik = self.ik_solver.solve(
                    target=target,
                    initial_guess=joint_state.positions,
                    lower_limits=self.arm_config.joint_lower_limits,
                    upper_limits=self.arm_config.joint_upper_limits,
                )
                self.world.apply_joint_position_targets(ik.solution)
                if float(np.linalg.norm(ee_state.position - target)) <= self.control_config.waypoint_tolerance:
                    if self.pick_stage < len(self.pick_waypoints) - 1:
                        self.pick_stage += 1
                    else:
                        self._transition(ArmPhase.GRASP)
                elif self.phase_timer > self.control_config.move_timeout_seconds:
                    self._end_episode(success=False)
        elif self.phase == ArmPhase.GRASP:
            self.world.apply_joint_position_targets(joint_state.positions)
            success = self.world.try_grasp(
                distance_threshold=self.control_config.grasp_distance_threshold,
                relative_speed_threshold=self.control_config.grasp_relative_speed_threshold,
            )
            if success:
                self._transition(ArmPhase.PLAN_THROW)
            elif self.phase_timer > 0.30:
                self.grasp_retries += 1
                if self.grasp_retries > self.control_config.max_grasp_retries:
                    self._end_episode(success=False)
                else:
                    self._transition(ArmPhase.PLAN_PICK)
        elif self.phase == ArmPhase.PLAN_THROW:
            self.throw_plan = self.throw_planner.plan(joint_state.positions, self.world.target_position)
            if self.throw_plan is None:
                self._end_episode(success=False)
            else:
                self.predicted_landing = self.throw_plan.predicted_landing.copy()
                self._transition(ArmPhase.MOVE_THROW)
        elif self.phase == ArmPhase.MOVE_THROW:
            if self.throw_plan is None:
                self._end_episode(success=False)
            else:
                self.world.apply_joint_position_targets(self.throw_plan.release_joint_positions)
                err = float(np.linalg.norm(joint_state.positions - self.throw_plan.release_joint_positions))
                if err <= self.control_config.release_pose_tolerance:
                    self.release_time_left = self.control_config.release_windup_seconds
                    self._transition(ArmPhase.RELEASE)
                elif self.phase_timer > self.control_config.move_timeout_seconds:
                    self._end_episode(success=False)
        elif self.phase == ArmPhase.RELEASE:
            if self.throw_plan is None:
                self._end_episode(success=False)
            else:
                self.world.apply_joint_velocity_targets(self.throw_plan.release_joint_velocities)
                self.release_time_left -= dt
                if self.release_time_left <= 0.0:
                    self.world.release_grasp()
                    self.world.set_block_velocity(self.throw_plan.release_velocity)
                    self.scored_this_episode = False
                    self._transition(ArmPhase.EVALUATE)
        elif self.phase == ArmPhase.EVALUATE:
            self.world.apply_joint_position_targets(self.arm_config.home_joint_positions)
            if self.world.is_block_in_target():
                self.scored_this_episode = True
            settled = self.world.block_has_settled(
                speed_threshold=self.control_config.settle_speed_threshold,
                height_threshold=self.control_config.settle_height_threshold,
            )
            if settled or self.phase_timer > self.control_config.evaluate_timeout_seconds:
                self._end_episode(success=self.scored_this_episode)
        elif self.phase == ArmPhase.RESET:
            self.world.release_grasp()
            self.world.apply_joint_position_targets(self.arm_config.home_joint_positions)
            if self.max_episodes is not None and self.stats.episodes_total >= self.max_episodes:
                self.done = True
            else:
                self._transition(ArmPhase.SPAWN)

        self.stats.current_phase = self.phase.value

    def snapshot(self) -> WorldState:
        joint_state = self.world.get_joint_state()
        ee_state = self.world.get_end_effector_state()
        block_pos, block_vel = self.world.get_block_state()
        capsule_a, capsule_b, capsule_r = self.world.render_capsules(joint_state.positions)
        stats_copy = EpisodeStats(
            episodes_total=self.stats.episodes_total,
            successes=self.stats.successes,
            failures=self.stats.failures,
            average_cycle_time=self.stats.average_cycle_time,
            last_cycle_time=self.stats.last_cycle_time,
            success_rate=self.stats.success_rate,
            current_phase=self.phase.value,
        )
        return WorldState(
            sim_time=self.world.sim_time,
            phase=self.phase.value,
            episode_index=self.active_episode,
            joint_state=joint_state,
            ee_state=ee_state,
            block_position=block_pos.astype(np.float64),
            block_velocity=block_vel.astype(np.float64),
            target_position=self.world.target_position.astype(np.float64),
            target_radius=float(self.world.target_radius),
            grasp_active=self.world.grasp_constraint_id is not None,
            stats=stats_copy,
            predicted_landing=None if self.predicted_landing is None else self.predicted_landing.copy(),
            render_capsule_a=capsule_a,
            render_capsule_b=capsule_b,
            render_capsule_r=capsule_r,
        )
