from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("moderngl")

import moderngl


def test_shader_compile_and_draw_smoke() -> None:
    shader_dir = Path(__file__).resolve().parents[1] / "src" / "render" / "shaders"
    vert_src = (shader_dir / "fullscreen.vert").read_text(encoding="utf-8")
    frag_src = (shader_dir / "scene.frag").read_text(encoding="utf-8")

    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"standalone context unavailable: {exc}")
        return

    program = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
    vao = ctx.vertex_array(program, [])
    fbo = ctx.simple_framebuffer((64, 64))
    fbo.use()

    program["u_resolution"].value = (64.0, 64.0)
    program["u_time"].value = 0.0
    program["u_cam_pos"].value = (2.5, 2.2, 1.6)
    program["u_cam_forward"].value = (-0.65, -0.58, -0.48)
    program["u_cam_right"].value = (0.67, -0.74, 0.0)
    program["u_cam_up"].value = (-0.35, -0.32, 0.88)
    program["u_fov_y"].value = 0.95
    program["u_floor_extent"].value = 3.0
    program["u_pixel_size"].value = 5.0
    program["u_max_steps"].value = 96
    program["u_shadow_steps"].value = 16
    program["u_far_distance"].value = 46.0
    program["u_hit_epsilon"].value = 0.0015

    program["u_capsule_a0"].value = (0.0, 0.0, 0.04)
    program["u_capsule_b0"].value = (0.0, 0.0, 0.30)
    program["u_capsule_a1"].value = (0.0, 0.0, 0.30)
    program["u_capsule_b1"].value = (0.45, 0.0, 0.62)
    program["u_capsule_a2"].value = (0.45, 0.0, 0.62)
    program["u_capsule_b2"].value = (0.98, 0.0, 0.48)
    program["u_capsule_r0"].value = 0.08
    program["u_capsule_r1"].value = 0.05
    program["u_capsule_r2"].value = 0.04

    program["u_block_pos"].value = (0.9, -0.2, 0.06)
    program["u_block_half"].value = (0.06, 0.06, 0.06)
    program["u_target_pos"].value = (-1.0, 0.5, 0.0)
    program["u_target_radius"].value = 0.25
    program["u_show_prediction"].value = 1
    program["u_predicted_landing"].value = (-0.85, 0.6, 0.0)

    vao.render(mode=moderngl.TRIANGLES, vertices=3)
    data = fbo.read(components=3)
    assert len(data) == 64 * 64 * 3
