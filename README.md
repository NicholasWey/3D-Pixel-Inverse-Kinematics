# Autonomous 3-Joint IK Throwing Arm

Physics-based 3D robotic arm demo built with Python, PyBullet, and GLSL raymarch rendering.

The arm runs fully autonomously:
1. Randomly spawns a block and target zone.
2. Uses Jacobian damped-least-squares IK to pick the block.
3. Plans a ballistic throw and releases the block.
4. Scores success/failure, then resets and repeats.

The rendering style is pixel-art 3D: it looks 2D at a glance, but all geometry and motion are truly 3D.

## Requirements

- Python 3.11 recommended for full functionality (`pybullet` wheel availability)
- GPU/OpenGL support for interactive mode

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Interactive window:

```bash
python -m src.main --quality balanced --show-hud --episodes 20
```

Headless deterministic run:

```bash
python -m src.main --headless --episodes 20 --seed 42
```

CLI flags:

- `--seed <int>`
- `--quality <low|balanced|high>`
- `--headless`
- `--episodes <n>`
- `--pixel-size <float>`
- `--show-hud`

## Controls (Interactive)

- Left mouse drag: orbit camera
- Mouse wheel: zoom
- `1` / `2` / `3`: quality presets
- `H`: toggle HUD
- `Space`: pause simulation
- `Esc`: quit

## Project Layout

- `src/main.py`: CLI and app launch
- `src/app.py`: fixed-step simulation runtime
- `src/physics/world.py`: PyBullet world, arm/block setup, grasp/release
- `src/robot/kinematics.py`: FK + analytical Jacobian
- `src/robot/ik.py`: damped least-squares IK
- `src/planner/ballistic.py`: throw planner
- `src/controller/state_machine.py`: autonomous task state machine
- `src/render/renderer.py`: ModernGL window renderer
- `src/render/shaders/scene.frag`: pixel-art raymarch shader
- `src/ui/hud.py`: bitmap telemetry overlay
- `tests/`: unit + smoke coverage
