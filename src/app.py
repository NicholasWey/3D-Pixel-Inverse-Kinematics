from __future__ import annotations

import numpy as np

from src.config import ArmConfig, ControlConfig, PlannerConfig, SimConfig, SpawnConfig
from src.controller.state_machine import ArmController
from src.physics.world import PhysicsWorld
from src.runtime_types import EpisodeStats, WorldState


class SimulationApp:
    def __init__(
        self,
        sim_config: SimConfig,
        arm_config: ArmConfig,
        spawn_config: SpawnConfig,
        planner_config: PlannerConfig,
        control_config: ControlConfig,
        seed: int | None,
        max_episodes: int | None,
    ) -> None:
        self.sim_config = sim_config
        self.world = PhysicsWorld(sim_config=sim_config, arm_config=arm_config, spawn_config=spawn_config)
        self.rng = np.random.default_rng(seed)
        self.controller = ArmController(
            world=self.world,
            sim_config=sim_config,
            arm_config=arm_config,
            spawn_config=spawn_config,
            planner_config=planner_config,
            control_config=control_config,
            rng=self.rng,
            max_episodes=max_episodes,
        )
        self.accumulator = 0.0
        self.latest_state: WorldState = self.controller.snapshot()

    def step_frame(self, frame_time: float) -> WorldState:
        dt = min(float(frame_time), self.sim_config.max_frame_time)
        self.accumulator += dt
        stepped = False
        while self.accumulator >= self.sim_config.time_step:
            if self.controller.done:
                self.accumulator = 0.0
                break
            self.controller.update(self.sim_config.time_step)
            self.world.step()
            self.accumulator -= self.sim_config.time_step
            stepped = True
        if stepped or self.latest_state is None:
            self.latest_state = self.controller.snapshot()
        else:
            self.latest_state = self.controller.snapshot()
        return self.latest_state

    def run_headless(self, max_steps: int | None = None) -> EpisodeStats:
        if max_steps is None:
            if self.controller.max_episodes is None:
                max_steps = int(120.0 / self.sim_config.time_step)
            else:
                max_steps = int((self.controller.max_episodes * 20.0) / self.sim_config.time_step)

        for _ in range(max_steps):
            if self.controller.done:
                break
            self.controller.update(self.sim_config.time_step)
            self.world.step()

        self.latest_state = self.controller.snapshot()
        return self.latest_state.stats

    def shutdown(self) -> None:
        self.world.close()

