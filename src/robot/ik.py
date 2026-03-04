from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.robot.kinematics import ArmKinematics


@dataclass(slots=True)
class IKResult:
    solution: np.ndarray
    converged: bool
    iterations: int
    residual: float
    clamped_target: np.ndarray


class DampedLeastSquaresIK:
    def __init__(
        self,
        kinematics: ArmKinematics,
        damping: float = 0.08,
        max_step_norm: float = 0.17,
        max_iterations: int = 48,
        tolerance: float = 1e-3,
    ) -> None:
        self.kinematics = kinematics
        self.damping = damping
        self.max_step_norm = max_step_norm
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve(
        self,
        target: np.ndarray,
        initial_guess: np.ndarray,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
    ) -> IKResult:
        q = np.clip(np.asarray(initial_guess, dtype=np.float64), lower_limits, upper_limits)
        clamped_target = self.kinematics.clamp_target_to_workspace(np.asarray(target, dtype=np.float64))
        identity3 = np.eye(3, dtype=np.float64)

        best_q = q.copy()
        best_err = float("inf")
        converged = False
        residual = float("inf")
        it = 0

        for it in range(1, self.max_iterations + 1):
            ee = self.kinematics.forward_kinematics(q)
            error = clamped_target - ee
            residual = float(np.linalg.norm(error))
            if residual < best_err:
                best_err = residual
                best_q = q.copy()
            if residual <= self.tolerance:
                converged = True
                best_q = q.copy()
                break

            jacobian = self.kinematics.jacobian(q)
            jj_t = jacobian @ jacobian.T
            cond = float(np.linalg.cond(jj_t + 1e-7 * identity3))
            adaptive = self.damping * (1.0 + min(cond / 200.0, 3.0))
            delta = jacobian.T @ np.linalg.solve(jj_t + (adaptive**2) * identity3, error)
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > self.max_step_norm:
                delta *= self.max_step_norm / delta_norm

            q = np.clip(q + delta, lower_limits, upper_limits)

        return IKResult(
            solution=best_q,
            converged=converged,
            iterations=it,
            residual=best_err if converged else residual,
            clamped_target=clamped_target,
        )

