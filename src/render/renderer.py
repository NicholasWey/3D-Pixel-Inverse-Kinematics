from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsInt, cast

import moderngl
import moderngl_window as mglw
import numpy as np
from moderngl_window.context.base import WindowConfig

from src.app import SimulationApp
from src.config import QUALITY_PRESETS, RenderConfig
from src.runtime_types import WorldState
from src.ui.hud import HudOverlay


def _normalize(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length < 1e-8:
        return v
    return v / length


@dataclass(slots=True)
class LaunchContext:
    sim_app: SimulationApp
    render_config: RenderConfig


_LAUNCH_CONTEXT: LaunchContext | None = None


def run_interactive(sim_app: SimulationApp, render_config: RenderConfig) -> None:
    global _LAUNCH_CONTEXT
    _LAUNCH_CONTEXT = LaunchContext(sim_app=sim_app, render_config=render_config)
    PixelArmWindow.window_size = render_config.window_size
    try:
        mglw.run_window_config(PixelArmWindow)
    finally:
        sim_app.shutdown()
        _LAUNCH_CONTEXT = None


class PixelArmWindow(WindowConfig):
    gl_version = (3, 3)
    title = "Autonomous 3-Joint IK Thrower"
    window_size = (1600, 900)
    resizable = True
    aspect_ratio = None
    vsync = True

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if _LAUNCH_CONTEXT is None:
            raise RuntimeError("Renderer launch context was not initialized")

        self.sim = _LAUNCH_CONTEXT.sim_app
        self.render_config = _LAUNCH_CONTEXT.render_config

        shader_dir = Path(__file__).resolve().parent / "shaders"
        self.program = self.ctx.program(
            vertex_shader=(shader_dir / "fullscreen.vert").read_text(encoding="utf-8"),
            fragment_shader=(shader_dir / "scene.frag").read_text(encoding="utf-8"),
        )
        self.quad = self.ctx.vertex_array(self.program, [])
        self.hud = HudOverlay() if self.render_config.show_hud else None

        self.target = np.array([0.0, 0.0, 0.4], dtype=np.float32)
        self.yaw = math.radians(42.0)
        # Start above the floor plane looking down to avoid spawning camera inside floor geometry.
        self.pitch = math.radians(24.0)
        self.radius = 3.9
        self.min_pitch = math.radians(5.0)
        self.max_pitch = math.radians(82.0)
        self.min_radius = 1.6
        self.max_radius = 14.0
        self.fov_y = math.radians(self.render_config.fov_y_degrees)
        self.current_quality = self.render_config.quality
        self.max_steps = int(self.render_config.max_steps)
        self.shadow_steps = int(self.render_config.shadow_steps)
        self.paused = False
        self._performance_samples: deque[float] = deque(maxlen=120)
        self._latest_state = self.sim.latest_state

        print("Controls:")
        print("  Left mouse drag: orbit camera")
        print("  Mouse wheel: zoom")
        print("  1/2/3: quality preset")
        print("  H: toggle HUD")
        print("  Space: pause simulation")
        print("  Esc: quit")

    def _set_uniform(self, name: str, value: Any) -> None:
        cast(Any, self.program[name]).value = value

    def _set_quality(self, quality: str) -> None:
        if quality not in QUALITY_PRESETS:
            return
        self.current_quality = cast(str, quality)
        preset = QUALITY_PRESETS[cast(Any, quality)]
        self.max_steps = int(preset["max_steps"])
        self.shadow_steps = int(preset["shadow_steps"])
        if self.render_config.pixel_size <= 0.0:
            self.render_config.pixel_size = float(preset["pixel_size"])
        print(f"Quality preset: {self.current_quality}")

    def _auto_quality_downshift(self) -> None:
        if len(self._performance_samples) < self._performance_samples.maxlen:
            return
        avg_dt = float(sum(self._performance_samples) / len(self._performance_samples))
        if avg_dt <= (1.0 / 53.0):
            return
        if self.current_quality == "high":
            self._set_quality("balanced")
            self._performance_samples.clear()
        elif self.current_quality == "balanced":
            self._set_quality("low")
            self._performance_samples.clear()

    def _camera_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        camera_pos = np.array(
            [
                self.target[0] + self.radius * cp * cy,
                self.target[1] + self.radius * cp * sy,
                self.target[2] + self.radius * sp,
            ],
            dtype=np.float32,
        )
        forward = _normalize(self.target - camera_pos)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = _normalize(np.cross(forward, world_up))
        if float(np.linalg.norm(right)) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        up = _normalize(np.cross(right, forward))
        return camera_pos, forward, right, up

    def on_render(self, time: float, frame_time: float) -> None:
        del time
        self._performance_samples.append(float(frame_time))
        self._auto_quality_downshift()
        self._latest_state = self.sim.step_frame(0.0 if self.paused else frame_time)
        state = self._latest_state
        self._draw_state(state)

    def _draw_state(self, state: WorldState) -> None:
        width, height = self.wnd.buffer_size
        if width <= 0 or height <= 0:
            return
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.disable(moderngl.DEPTH_TEST)

        camera_pos, forward, right, up = self._camera_vectors()
        zoom_t = float(
            np.clip(
                (self.radius - self.min_radius) / max(1e-6, (self.max_radius - self.min_radius)),
                0.0,
                1.0,
            )
        )
        self._set_uniform("u_resolution", (float(width), float(height)))
        self._set_uniform("u_time", float(state.sim_time))
        self._set_uniform("u_cam_pos", tuple(float(v) for v in camera_pos))
        self._set_uniform("u_cam_forward", tuple(float(v) for v in forward))
        self._set_uniform("u_cam_right", tuple(float(v) for v in right))
        self._set_uniform("u_cam_up", tuple(float(v) for v in up))
        dynamic_floor_extent = max(self.render_config.floor_extent, 2.4 + self.radius * 1.3)
        dynamic_far_distance = max(36.0, 18.0 + self.radius * 8.0)
        dynamic_hit_epsilon = float(np.clip(0.0011 + 0.00012 * self.radius, 0.0011, 0.0035))
        dynamic_max_steps = int(np.clip(math.ceil(self.max_steps * (1.0 + self.radius / 8.0)), self.max_steps, 320))
        dynamic_shadow_steps = int(
            np.clip(
                math.ceil(self.shadow_steps * (0.95 + self.radius / 14.0)),
                self.shadow_steps,
                72,
            )
        )

        self._set_uniform("u_fov_y", self.fov_y)
        self._set_uniform("u_floor_extent", dynamic_floor_extent)
        effective_pixel_size = float(
            np.clip(self.render_config.pixel_size * (1.0 - 0.45 * zoom_t), 1.0, self.render_config.pixel_size)
        )
        self._set_uniform("u_pixel_size", effective_pixel_size)
        self._set_uniform("u_max_steps", dynamic_max_steps)
        self._set_uniform("u_shadow_steps", dynamic_shadow_steps)
        self._set_uniform("u_far_distance", dynamic_far_distance)
        self._set_uniform("u_hit_epsilon", dynamic_hit_epsilon)
        self._set_uniform("u_zoom_t", zoom_t)
        self._set_uniform("u_exposure", float(1.08 + 0.24 * zoom_t))
        self._set_uniform("u_contrast", float(1.06 + 0.18 * zoom_t))

        self._set_uniform("u_capsule_a0", tuple(float(v) for v in state.render_capsule_a[0]))
        self._set_uniform("u_capsule_b0", tuple(float(v) for v in state.render_capsule_b[0]))
        self._set_uniform("u_capsule_a1", tuple(float(v) for v in state.render_capsule_a[1]))
        self._set_uniform("u_capsule_b1", tuple(float(v) for v in state.render_capsule_b[1]))
        self._set_uniform("u_capsule_a2", tuple(float(v) for v in state.render_capsule_a[2]))
        self._set_uniform("u_capsule_b2", tuple(float(v) for v in state.render_capsule_b[2]))
        self._set_uniform("u_capsule_r0", float(state.render_capsule_r[0]))
        self._set_uniform("u_capsule_r1", float(state.render_capsule_r[1]))
        self._set_uniform("u_capsule_r2", float(state.render_capsule_r[2]))

        self._set_uniform("u_block_pos", tuple(float(v) for v in state.block_position))
        self._set_uniform(
            "u_block_half",
            (
                float(self.sim.world.spawn_config.block_half_extent),
                float(self.sim.world.spawn_config.block_half_extent),
                float(self.sim.world.spawn_config.block_half_extent),
            ),
        )
        self._set_uniform("u_target_pos", tuple(float(v) for v in state.target_position))
        self._set_uniform("u_target_radius", float(state.target_radius))

        if state.predicted_landing is None:
            self._set_uniform("u_show_prediction", 0)
            self._set_uniform("u_predicted_landing", (0.0, 0.0, 0.0))
        else:
            self._set_uniform("u_show_prediction", 1)
            self._set_uniform("u_predicted_landing", tuple(float(v) for v in state.predicted_landing))

        self.quad.render(mode=moderngl.TRIANGLES, vertices=3)
        if self.hud is not None:
            self.hud.draw(self._hud_lines(state), viewport_height=height)

    def _hud_lines(self, state: WorldState) -> list[str]:
        stats = state.stats
        lines = [
            f"phase: {state.phase}",
            f"episode: {state.episode_index}   completed: {stats.episodes_total}",
            f"score: {stats.successes}   fail: {stats.failures}",
            f"success rate: {stats.success_rate * 100.0:5.1f}%",
            f"cycle avg: {stats.average_cycle_time:4.2f}s   last: {stats.last_cycle_time:4.2f}s",
            f"quality: {self.current_quality}   ray steps: {self.max_steps}",
            f"target radius: {state.target_radius:4.2f}",
        ]
        if state.predicted_landing is not None:
            lines.append(
                f"predicted landing: ({state.predicted_landing[0]:4.2f}, {state.predicted_landing[1]:4.2f})"
            )
        lines.append("controls: mouse orbit/zoom, [1][2][3] quality, [H] HUD, [Space] pause")
        return lines

    def _is_action_press(self, action: object) -> bool:
        keys = self.wnd.keys
        press = getattr(keys, "ACTION_PRESS", None)
        if action == press:
            return True
        if isinstance(action, str) and action.upper() == "ACTION_PRESS":
            return True
        if isinstance(action, int):
            return action == 1
        if isinstance(action, SupportsInt):
            try:
                return int(cast(SupportsInt, action)) == 1
            except (TypeError, ValueError):
                return False
        return False

    def _key_candidates(self, *names: str, fallback: int | None = None) -> tuple[object, ...]:
        keys = self.wnd.keys
        out: list[object] = []
        for name in names:
            value = getattr(keys, name, None)
            if value is None or value == "undefined":
                continue
            out.append(value)
        if fallback is not None:
            out.append(fallback)
        return tuple(out)

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int, *extra: object) -> None:
        del extra
        del x, y
        sensitivity = 0.004
        self.yaw += dx * sensitivity
        self.pitch = float(np.clip(self.pitch - dy * sensitivity, self.min_pitch, self.max_pitch))

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float, *extra: object) -> None:
        del extra
        del x_offset
        self.radius = float(np.clip(self.radius * math.exp(-y_offset * 0.12), self.min_radius, self.max_radius))

    def on_key_event(self, key: object, action: object, modifiers: object, *extra: object) -> None:
        del modifiers
        del extra
        if not self._is_action_press(action):
            return

        esc_keys = self._key_candidates("ESCAPE")
        if key in esc_keys:
            self.wnd.close()
            return

        h_keys = self._key_candidates("H", fallback=ord("h"))
        if key in h_keys:
            if self.hud is None:
                self.hud = HudOverlay()
            else:
                self.hud = None
            return

        space_keys = self._key_candidates("SPACE", fallback=32)
        if key in space_keys:
            self.paused = not self.paused
            return

        num1_keys = self._key_candidates("NUMBER_1", "NUMPAD_1", fallback=49)
        num2_keys = self._key_candidates("NUMBER_2", "NUMPAD_2", fallback=50)
        num3_keys = self._key_candidates("NUMBER_3", "NUMPAD_3", fallback=51)
        if key in num1_keys:
            self._set_quality("low")
        elif key in num2_keys:
            self._set_quality("balanced")
        elif key in num3_keys:
            self._set_quality("high")

    def render(self, time: float, frame_time: float) -> None:
        self.on_render(time, frame_time)

    def mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
        self.on_mouse_drag_event(x, y, dx, dy)

    def mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
        self.on_mouse_scroll_event(x_offset, y_offset)

    def key_event(self, key: int, action: int, modifiers: int) -> None:
        self.on_key_event(key, action, modifiers)
