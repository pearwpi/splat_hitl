"""Training environment. The SAME pieces the flight loop uses.

WHY THIS LIVES HERE AND NOT IN THE RESEARCH TREE
------------------------------------------------
Students get `cf_vicon_stack` and `splat_hitl`, and nothing else. So the
environment they train in has to be in one of those two, and this is the one
that already owns the policy-facing half of the pipeline.

That constraint turns out to be the right architecture rather than a
compromise. Sim-to-real parity used to be TWO IMPLEMENTATIONS THAT MUST AGREE,
policed after the fact by comparing a fingerprint. Here the environment and the
runtime import the same `ObservationBuilder`, the same `VelocityIntegrator` and
the same `PolicyContract`, so a policy sees a byte-identical observation and its
action produces an identical velocity in both. Parity stops being enforced and
becomes structural: there is no second implementation to drift.

What is deliberately NOT here: the renderer itself. `SplatWorkerClient` speaks
to a worker process over JSON, and that worker drags torch, gsplat, nerfstudio
and CUDA. This package is numpy-only and should stay that way, because a
student who cannot import it cannot start.

WHAT IS SIMULATED, AND WHAT IS NOT
----------------------------------
The dynamics are a double integrator with a velocity envelope -- exactly what
`VelocityIntegrator` does in flight, because it IS that object. There is no
attitude loop, no drag, no rotor dynamics and no ground effect. That is the
honest boundary: this trains a policy that decides WHERE TO GO, and the real
drone's controller decides how. A policy that only works because it exploited
frictionless flight will show up as a large `divergence_ms` on the first HITL
run, which is what that column is for.

    env = SplatEnv(contract, renderer, course, esdf)
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(raw_action)

`raw_action` is the network's raw output -- the [-1, 1] box. Scaling, clipping
and the norm limit happen inside, through `action_from_raw`, so the policy is
trained against the same interpretation the runtime will apply.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .collision import ESDF, CollisionMonitor
from .commands import Limits, VelocityIntegrator, action_from_raw
from .contract import PolicyContract
from .gates import PASSED, GateCourse
from .observation import ObservationBuilder
from .renderer import RendererClient

__all__ = ["EnvConfig", "SplatEnv"]

#: The run outcomes, using the RUNTIME's vocabulary so a sim log and a flight
#: log drop into the same tooling. `recorder.TERMINATION_MAP` maps these to
#: metric-splat's names; adding a fifth reason here means adding it there.
COURSE_COMPLETE = "course_complete"
VIRTUAL_COLLISION = "virtual_collision"
VIRTUAL_OUTSIDE = "virtual_outside"
MAX_DURATION = "max_duration"


@dataclass
class EnvConfig:
    """Episode setup and the reward. Both are part of the task definition.

    The reward coefficients are here rather than as CLI flags because two
    students who tune them differently are not solving the same problem, and
    their success rates are then not comparable. Change them deliberately, in
    one place, and say so in the write-up.
    """
    start_position_m: Sequence[float] = (0.0, 0.0, 0.6)
    start_yaw_rad: float = 0.0
    #: uniform box jitter on the start position, metres
    start_jitter_m: float = 0.0
    #: uniform jitter on the start heading, radians. The policy only ever sees
    #: the scene from its STARTING heading (contract.action.yaw_mode == fixed),
    #: so this is what teaches it more than one view of the course.
    start_yaw_jitter_rad: float = 0.0
    max_steps: int = 500
    clearance_m: float = 0.10
    #: reward
    progress_weight: float = 1.0       # per metre closer to the next gate
    gate_reward: float = 10.0
    goal_reward: float = 100.0
    collision_penalty: float = -100.0
    outside_penalty: float = -100.0
    timeout_penalty: float = 0.0
    step_penalty: float = -0.05
    #: envelope. Defaults match the flight limits, which is the point.
    limits: Limits = field(default_factory=Limits)
    seed: int = 0


class SplatEnv:
    """Gymnasium-shaped, without depending on gymnasium.

    `reset`/`step` return the standard tuples, so `as_gym()` is a thin wrapper
    and the tests run in an environment that has only numpy. A student who
    wants stable-baselines3 installs it; a student reading the code does not
    have to.
    """

    def __init__(self, contract: PolicyContract, renderer: RendererClient,
                 course: GateCourse, esdf: Optional[ESDF] = None,
                 config: EnvConfig = EnvConfig()):
        if contract.observation.sensor.fingerprint() != renderer.sensor.fingerprint():
            raise ValueError(
                "the contract's sensor (%s) is not the renderer's (%s). Train "
                "through the same camera you will fly through, or the policy "
                "has never seen the world it is asked to fly in."
                % (contract.observation.sensor.fingerprint(),
                   renderer.sensor.fingerprint()))
        if contract.action.kind != "acceleration":
            raise ValueError(
                "this environment integrates accelerations; the contract emits "
                "%r. A velocity contract needs no integrator and a different "
                "env." % (contract.action.kind,))
        self.contract = contract
        self.renderer = renderer
        self.course = course
        self.cfg = config
        self.builder = ObservationBuilder(contract.observation)
        self.integrator = VelocityIntegrator(contract.action,
                                             contract.control.dt_s,
                                             config.limits)
        self.monitor = (None if esdf is None
                        else CollisionMonitor(esdf, config.clearance_m))
        self.rng = np.random.default_rng(config.seed)
        self.position_m = np.zeros(3)
        self.yaw_rad = 0.0
        self.steps = 0
        self._prev_gate_distance: Optional[float] = None
        self._done = True

    # -- shapes, for whoever is building a network -------------------------
    @property
    def observation_shape(self) -> tuple:
        return self.contract.observation.shape

    @property
    def action_shape(self) -> tuple:
        return (3,)

    # -- lifecycle ---------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.cfg
        p = np.asarray(c.start_position_m, dtype=float).reshape(3).copy()
        if c.start_jitter_m > 0:
            p = p + self.rng.uniform(-c.start_jitter_m, c.start_jitter_m, size=3)
        yaw = float(c.start_yaw_rad)
        if c.start_yaw_jitter_rad > 0:
            yaw += float(self.rng.uniform(-c.start_yaw_jitter_rad,
                                          c.start_yaw_jitter_rad))

        self.position_m = p
        self.yaw_rad = yaw
        self.steps = 0
        self._done = False
        self.builder.reset()
        # The episode heading is captured here for the same reason the runtime
        # captures it from the first pose: it is the only heading this episode
        # will ever be seen from.
        self.integrator.reset(episode_yaw_rad=yaw)
        self.course.reset()
        if self.monitor is not None:
            self.monitor.reset()
        self._prev_gate_distance = self.course.distance_to_next(self.position_m)
        obs = self._observe()
        return obs, self._info(None)

    def step(self, raw_action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError(
                "step() after the episode ended. reset() first -- continuing "
                "would carry the previous episode's velocity and heading into "
                "the next one, which trains on a state that never occurs.")
        dt = self.contract.control.dt_s
        action = action_from_raw(self.contract.action, raw_action)
        v_prev = self.integrator.velocity_world.copy()
        velocity_action, _ = self.integrator.step(action, current_yaw_rad=self.yaw_rad)
        v_now = np.asarray(velocity_action.vector, dtype=float)

        # Trapezoidal on the CLAMPED velocities. Equal to p + v*dt + a*dt^2/2
        # while the envelope is not binding, and correct when it is -- using
        # the raw acceleration there would move the drone further than the
        # velocity it is actually allowed to have.
        p_prev = self.position_m.copy()
        self.position_m = p_prev + 0.5 * (v_prev + v_now) * dt
        self.steps += 1

        reward = float(self.cfg.step_penalty)
        reason: Optional[str] = None

        if self.monitor is not None:
            ev = self.monitor.update(p_prev, self.position_m)
            if ev is not None:
                if ev.kind == "outside":
                    reward += self.cfg.outside_penalty
                    reason = VIRTUAL_OUTSIDE
                else:
                    reward += self.cfg.collision_penalty
                    reason = VIRTUAL_COLLISION

        if reason is None:
            for e in self.course.update(p_prev, self.position_m):
                if e.kind == PASSED:
                    reward += self.cfg.gate_reward
            d = self.course.distance_to_next(self.position_m)
            if d is not None and self._prev_gate_distance is not None:
                reward += self.cfg.progress_weight * (self._prev_gate_distance - d)
            self._prev_gate_distance = d
            if self.course.complete:
                reward += self.cfg.goal_reward
                reason = COURSE_COMPLETE

        truncated = False
        if reason is None and self.steps >= self.cfg.max_steps:
            reward += self.cfg.timeout_penalty
            reason = MAX_DURATION
            truncated = True

        terminated = reason is not None and not truncated
        self._done = terminated or truncated
        return self._observe(), float(reward), terminated, truncated, self._info(reason)

    def close(self) -> None:
        self.renderer.close()

    # -- internals ---------------------------------------------------------
    def _observe(self) -> np.ndarray:
        # Yaw only: the contract holds the heading fixed, and roll and pitch
        # are the real controller's business in flight, so simulating them here
        # would show the policy an attitude the HITL renderer never will.
        obs = self.renderer.render(self.position_m,
                                   (0.0, 0.0, self.integrator.episode_yaw_rad))
        return self.builder.push(obs.depth_m)

    def _info(self, reason: Optional[str]) -> Dict[str, Any]:
        return {
            "reason": reason,
            "steps": self.steps,
            "position_m": self.position_m.copy(),
            "velocity_world_m_s": self.integrator.velocity_world.copy(),
            "yaw_rad": self.yaw_rad,
            "episode_yaw_rad": self.integrator.episode_yaw_rad,
            "gates_passed": self.course.passed,
            "gate_distance_m": self.course.distance_to_next(self.position_m),
            "contract_fingerprint": self.contract.fingerprint(),
        }

    def as_gym(self):
        """Wrap as a `gymnasium.Env`. Imported here so the dependency is
        optional -- everything above works, and is tested, without it."""
        from .gym_env import GymSplatEnv          # noqa: F401  (lazy on purpose)
        return GymSplatEnv(self)

    def __repr__(self) -> str:
        return ("SplatEnv(obs %s, %d gate(s), %.0f Hz, %s)"
                % (self.observation_shape, len(self.course.gates),
                   self.contract.control.rate_hz, self.contract.fingerprint()))
