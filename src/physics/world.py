from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import ArmConfig, SimConfig, SpawnConfig
from src.robot.kinematics import ArmGeometry, ArmKinematics
from src.runtime_types import EndEffectorState, JointState

try:
    import pybullet as p
    import pybullet_data
except Exception as exc:  # pragma: no cover - runtime dependency guard
    p = None
    pybullet_data = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class PhysicsWorld:
    def __init__(self, sim_config: SimConfig, arm_config: ArmConfig, spawn_config: SpawnConfig) -> None:
        if p is None:
            raise RuntimeError("pybullet is required for PhysicsWorld") from _IMPORT_ERROR

        self.sim_config = sim_config
        self.arm_config = arm_config
        self.spawn_config = spawn_config
        self.client = p.connect(p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Failed to connect to pybullet in DIRECT mode")

        p.resetSimulation(physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0.0, 0.0, -self.sim_config.gravity, physicsClientId=self.client)
        p.setTimeStep(self.sim_config.time_step, physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            numSolverIterations=90,
            deterministicOverlappingPairs=1,
            physicsClientId=self.client,
        )
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)

        urdf_path = Path(__file__).resolve().parent / "assets" / "arm3.urdf"
        self.robot_id = p.loadURDF(
            str(urdf_path),
            basePosition=[0.0, 0.0, 0.0],
            useFixedBase=True,
            physicsClientId=self.client,
        )
        if self.arm_config.disable_arm_floor_collision:
            self._set_arm_floor_collision(enabled=False)

        self.control_joint_names = ("base_yaw", "shoulder_pitch", "elbow_pitch")
        self.joint_indices = self._resolve_joint_indices()
        self.ee_link_index = self._resolve_link_index("tool_fixed")
        self.kinematics = ArmKinematics(
            ArmGeometry(
                base_height=self.arm_config.base_height,
                link1_length=self.arm_config.link1_length,
                link2_length=self.arm_config.link2_length,
                tool_length=self.arm_config.tool_length,
            )
        )

        self.block_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.spawn_config.block_half_extent] * 3,
            physicsClientId=self.client,
        )
        self.block_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.spawn_config.block_half_extent] * 3,
            rgbaColor=[0.86, 0.78, 0.56, 1.0],
            physicsClientId=self.client,
        )
        self.block_id: int | None = None
        self.grasp_constraint_id: int | None = None
        self.target_position = np.zeros(3, dtype=np.float64)
        self.target_radius = self.spawn_config.target_radius_base
        self.sim_time = 0.0

        self.reset_arm_pose(self.arm_config.home_joint_positions)

    def _set_arm_floor_collision(self, enabled: bool) -> None:
        flag = 1 if enabled else 0
        joint_count = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        # Disable collisions between the plane and articulated links (keep base link unchanged).
        for link_index in range(joint_count):
            p.setCollisionFilterPair(
                self.robot_id,
                self.plane_id,
                link_index,
                -1,
                flag,
                physicsClientId=self.client,
            )

    def _resolve_joint_indices(self) -> list[int]:
        mapping: dict[str, int] = {}
        joint_count = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        for joint_idx in range(joint_count):
            info = p.getJointInfo(self.robot_id, joint_idx, physicsClientId=self.client)
            mapping[info[1].decode("utf-8")] = joint_idx
        return [mapping[name] for name in self.control_joint_names]

    def _resolve_link_index(self, joint_name: str) -> int:
        joint_count = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        for joint_idx in range(joint_count):
            info = p.getJointInfo(self.robot_id, joint_idx, physicsClientId=self.client)
            if info[1].decode("utf-8") == joint_name:
                return joint_idx
        raise KeyError(f"Joint for link index lookup was not found: {joint_name}")

    def reset_arm_pose(self, joint_positions: np.ndarray) -> None:
        q = np.asarray(joint_positions, dtype=np.float64)
        for idx, value in zip(self.joint_indices, q, strict=True):
            p.resetJointState(self.robot_id, idx, float(value), targetVelocity=0.0, physicsClientId=self.client)
        self.apply_joint_position_targets(q)

    def apply_joint_position_targets(self, joint_targets: np.ndarray) -> None:
        target = np.asarray(joint_targets, dtype=np.float64)
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.POSITION_CONTROL,
            targetPositions=[float(v) for v in target],
            targetVelocities=[0.0, 0.0, 0.0],
            forces=self.arm_config.joint_force_limits.tolist(),
            positionGains=self.arm_config.position_gains.tolist(),
            velocityGains=self.arm_config.velocity_gains.tolist(),
            physicsClientId=self.client,
        )

    def apply_joint_velocity_targets(self, joint_velocities: np.ndarray) -> None:
        vel = np.asarray(joint_velocities, dtype=np.float64)
        clipped = np.clip(vel, -self.arm_config.joint_velocity_limits, self.arm_config.joint_velocity_limits)
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.VELOCITY_CONTROL,
            targetVelocities=[float(v) for v in clipped],
            forces=self.arm_config.joint_force_limits.tolist(),
            physicsClientId=self.client,
        )

    def spawn_block(self, position: np.ndarray) -> None:
        if self.block_id is not None:
            p.removeBody(self.block_id, physicsClientId=self.client)
            self.block_id = None
        self.block_id = p.createMultiBody(
            baseMass=0.30,
            baseCollisionShapeIndex=self.block_collision,
            baseVisualShapeIndex=self.block_visual,
            basePosition=[float(v) for v in position],
            physicsClientId=self.client,
        )
        p.changeDynamics(
            self.block_id,
            -1,
            lateralFriction=0.9,
            rollingFriction=0.01,
            spinningFriction=0.01,
            restitution=0.22,
            linearDamping=0.03,
            angularDamping=0.05,
            physicsClientId=self.client,
        )

    def set_target(self, position: np.ndarray, radius: float) -> None:
        self.target_position = np.asarray(position, dtype=np.float64)
        self.target_radius = float(radius)

    def reset_episode(self, block_position: np.ndarray, target_position: np.ndarray, target_radius: float) -> None:
        self.release_grasp()
        self.reset_arm_pose(self.arm_config.home_joint_positions)
        self.spawn_block(block_position)
        self.set_target(target_position, target_radius)

    def step(self) -> None:
        p.stepSimulation(physicsClientId=self.client)
        self.sim_time += self.sim_config.time_step

    def get_joint_state(self) -> JointState:
        states = [p.getJointState(self.robot_id, idx, physicsClientId=self.client) for idx in self.joint_indices]
        positions = np.array([s[0] for s in states], dtype=np.float64)
        velocities = np.array([s[1] for s in states], dtype=np.float64)
        return JointState(positions=positions, velocities=velocities)

    def get_end_effector_state(self) -> EndEffectorState:
        link_state = p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            computeLinkVelocity=1,
            computeForwardKinematics=1,
            physicsClientId=self.client,
        )
        return EndEffectorState(
            position=np.array(link_state[0], dtype=np.float64),
            velocity=np.array(link_state[6], dtype=np.float64),
        )

    def get_block_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.block_id is None:
            return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        pos, _ = p.getBasePositionAndOrientation(self.block_id, physicsClientId=self.client)
        vel, _ = p.getBaseVelocity(self.block_id, physicsClientId=self.client)
        return np.array(pos, dtype=np.float64), np.array(vel, dtype=np.float64)

    def try_grasp(self, distance_threshold: float, relative_speed_threshold: float) -> bool:
        if self.block_id is None:
            return False
        if self.grasp_constraint_id is not None:
            return True

        ee = self.get_end_effector_state()
        block_pos, block_vel = self.get_block_state()
        distance = float(np.linalg.norm(ee.position - block_pos))
        relative_speed = float(np.linalg.norm(ee.velocity - block_vel))
        if distance > distance_threshold or relative_speed > relative_speed_threshold:
            return False

        ee_state = p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            computeForwardKinematics=1,
            physicsClientId=self.client,
        )
        ee_pos = ee_state[0]
        ee_orn = ee_state[1]
        block_pos_q, block_orn_q = p.getBasePositionAndOrientation(self.block_id, physicsClientId=self.client)
        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
        child_frame_pos, child_frame_orn = p.multiplyTransforms(
            inv_ee_pos,
            inv_ee_orn,
            block_pos_q,
            block_orn_q,
        )
        self.grasp_constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.ee_link_index,
            childBodyUniqueId=self.block_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=[0.0, 0.0, 0.0],
            childFramePosition=child_frame_pos,
            parentFrameOrientation=[0.0, 0.0, 0.0, 1.0],
            childFrameOrientation=child_frame_orn,
            physicsClientId=self.client,
        )
        p.changeConstraint(self.grasp_constraint_id, maxForce=550.0, physicsClientId=self.client)
        return True

    def release_grasp(self) -> None:
        if self.grasp_constraint_id is None:
            return
        p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client)
        self.grasp_constraint_id = None

    def set_block_velocity(self, linear_velocity: np.ndarray) -> None:
        if self.block_id is None:
            return
        p.resetBaseVelocity(
            self.block_id,
            linearVelocity=[float(v) for v in np.asarray(linear_velocity, dtype=np.float64)],
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self.client,
        )

    def is_block_in_target(self) -> bool:
        block_pos, _ = self.get_block_state()
        distance_xy = float(np.linalg.norm(block_pos[:2] - self.target_position[:2]))
        near_floor = block_pos[2] <= (self.spawn_config.block_half_extent + 0.08)
        return distance_xy <= self.target_radius and near_floor

    def block_has_settled(self, speed_threshold: float, height_threshold: float) -> bool:
        block_pos, block_vel = self.get_block_state()
        speed = float(np.linalg.norm(block_vel))
        near_floor = block_pos[2] <= (self.spawn_config.block_half_extent + height_threshold)
        return speed <= speed_threshold and near_floor

    def render_capsules(self, joint_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shoulder, elbow, _, ee = self.kinematics.joint_positions_world(joint_positions)
        tower_a = np.array([0.0, 0.0, 0.04], dtype=np.float64)
        tower_b = shoulder
        a = np.vstack((tower_a, shoulder, elbow)).astype(np.float32)
        b = np.vstack((tower_b, elbow, ee)).astype(np.float32)
        r = np.array([0.085, 0.055, 0.045], dtype=np.float32)
        return a, b, r

    def close(self) -> None:
        if self.client >= 0:
            p.disconnect(physicsClientId=self.client)
            self.client = -1
