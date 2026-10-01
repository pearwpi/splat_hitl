"""Training environment. The SAME pieces the flight loop uses.

WHY THIS LIVES HERE AND NOT IN THE RESEARCH TREE
------------------------------------------------
Students get `cf_vicon_stack` and `splat_hitl`, and nothing else. So the
environment they train in has to be in one of those two, and this is the one
that already owns the policy-facing half of the pipeline.

That constraint turns out to be the right architecture rather than a
compromise. Sim-to-real parity used to be TWO IMPLEMENTATIONS THAT MUST AGREE,
policed after the fact by comparing a fingerprint. Here the environment and the
runtime import the same `ObservationBuilder`, the same action stage and the
same `PolicyContract`, so a policy sees a byte-identical observation and its
action produces an identical velocity in both. Parity stops being enforced and
becomes structural: there is no second implementation to drift.

What is deliberately NOT here: the renderer itself. `SplatWorkerClient` speaks
to a worker process over JSON, and that worker drags torch, gsplat, nerfstudio
and CUDA. This package is numpy-only and should stay that way, because a
student who cannot import it cannot start.

TWO KINDS OF ACTION
-------------------
`contract.action.kind` decides what the policy's three numbers mean, and the
env picks the matching stage at construction:

    "acceleration"  a double integrator, the RL default. The policy commands a
                    change in velocity and carries the velocity itself as
                    state, so its timestep is part of its behaviour.
    "velocity"      a passthrough. The policy IS the outer loop -- a hand-tuned
                    PID, say -- and its output is the velocity it wants. There
                    is no integrator and no state to wind up.

Both end at the same place: a clamped world-ENU velocity that steps position.
So a position controller tuned here and an RL policy trained here are flown by
identical code, and the gains you find in sim are the gains you fly.

WHAT IS SIMULATED, AND WHAT IS NOT
----------------------------------
The dynamics are a velocity envelope, integrated -- exactly what the flight
path applies, because it IS that object. There is no attitude loop, no drag,
no rotor dynamics and no ground effect, and a commanded velocity is reached
instantly. That is the honest boundary: this trains a policy that decides
WHERE TO GO, and the real drone's controller decides how. A controller that
only works because it exploited frictionless flight will show up as overshoot
on the first HITL run that sim never predicted -- for an acceleration contract
the `divergence_ms` column measures exactly that gap.

The altitude limits are flight's too. The runtime never commands a height
outside `limits.min_altitude_m`..`limits.max_altitude_m` (0.10-1.80 m above
the lab floor by default), so here the drone stops at those heights and the
episode carries on, as the flight does. `info["altitude_clamped"]` says when.
Height is measured from the lab floor, and only the scene's registration says
where that is, so a scanned scene needs `transform=bundle.transform()`.

    env = SplatEnv(contract, renderer, course, esdf,
                   transform=bundle.transform())
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(raw_action)

`raw_action` is the network's raw output -- the [-1, 1] box. Scaling, clipping
and the norm limit happen inside, through `action_from_raw`, so the policy is
trained against the same interpretation the runtime will apply. That is true
of a PID's output too: emit metres per second divided by `action.scale`, and
the envelope is applied for you.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .collision import ESDF, CollisionMonitor
from .commands import Limits, action_from_raw, make_action_stage
from .contract import PolicyContract
from .frames import SplatTransform
from .gates import PASSED, GateCourse
from .observation import ObservationBuilder
from .renderer import RendererClient, SplatWorkerClient

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
    #: envelope. Defaults match the flight limits, which is the point. That
    #: includes the altitude band, which the env keeps the drone inside.
    limits: Limits = field(default_factory=Limits)
    seed: int = 0


class SplatEnv:
    """Gymnasium-shaped, without depending on gymnasium.

    `reset`/`step` return the standard tuples, so `as_gym()` is a thin wrapper
    and the tests run in an environment that has only numpy. A student who
    wants stable-baselines3 installs it; a student reading the code does not
    have to.

    `transform` is the scene's registration, `bundle.transform()`. The env
    uses it for one thing: finding the lab floor, so the altitude limits are
    heights above the floor, as in flight. Leave it out only for
    `FakeRenderer`, whose room has its floor at z = 0.
    """

    def __init__(self, contract: PolicyContract, renderer: RendererClient,
                 course: GateCourse, esdf: Optional[ESDF] = None,
                 config: EnvConfig = EnvConfig(),
                 transform: Optional[SplatTransform] = None):
        if contract.observation.sensor.fingerprint() != renderer.sensor.fingerprint():
            raise ValueError(
                "the contract's sensor (%s) is not the renderer's (%s). Train "
                "through the same camera you will fly through, or the policy "
                "has never seen the world it is asked to fly in."
                % (contract.observation.sensor.fingerprint(),
                   renderer.sensor.fingerprint()))
        # Acceleration and velocity contracts both fly here; they differ only
        # in which action stage sits between the policy and the position
        # update. A position contract does not -- it commands a place, and
        # nothing in this env would then be simulating a controller.
        if contract.action.kind not in ("acceleration", "velocity"):
            raise ValueError(
                "this environment steps a velocity; the contract emits %r. A "
                "position contract needs a tracking controller in front of it, "
                "which is a different env." % (contract.action.kind,))
        # The lab's vertical and the lab origin, in scene metres. Scene metres
        # are R @ p_lab + t * metres_per_unit, so lab +z is R's third column
        # and the origin sits at t * metres_per_unit. A scene's own z axis
        # need not be the lab's, so the vertical comes from the registration.
        if transform is None:
            if isinstance(renderer, SplatWorkerClient):
                raise ValueError(
                    "a scanned scene needs its registration: pass "
                    "transform=bundle.transform(). The env keeps the drone "
                    "%.2f-%.2f m above the lab floor, as flight does, and "
                    "only the registration says where that floor is. (No "
                    "registration yet? "
                    "transform=SplatTransform.identity_metres(1.0) treats "
                    "scene z as the height.)"
                    % (config.limits.min_altitude_m,
                       config.limits.max_altitude_m))
            self._up = np.array([0.0, 0.0, 1.0])
            self._lab_origin_m = np.zeros(3)
        else:
            self._up = np.asarray(transform.R, dtype=float)[:, 2].copy()
            self._lab_origin_m = (np.asarray(transform.t, dtype=float)
                                  * transform.metres_per_unit)
        self.transform = transform
        lo, hi = config.limits.min_altitude_m, config.limits.max_altitude_m
        h0 = self._height_of(config.start_position_m)
        if not lo <= h0 <= hi:
            raise ValueError(
                "start_position_m is %.2f m above the floor, outside the "
                "flight limits (%.2f-%.2f m). %s"
                % (h0, lo, hi,
                   "It is in scene metres, not Vicon metres." if transform
                   is not None else
                   "Without a transform the floor is taken to be z = 0, "
                   "which is only true of FakeRenderer's room: for a scanned "
                   "scene, pass transform=bundle.transform()."))
        self.contract = contract
        self.renderer = renderer
        self.course = course
        self.cfg = config
        self.builder = ObservationBuilder(contract.observation)
        self.integrator = make_action_stage(contract.action,
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
        clamped = self._hold_altitude()       # jitter can reach past a limit
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
        return obs, self._info(None, clamped)

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

        # Trapezoidal on the CLAMPED velocities. Under an acceleration
        # contract this equals p + v*dt + a*dt^2/2 while the envelope is not
        # binding, and stays correct when it is -- using the raw acceleration
        # there would move the drone further than the velocity it is actually
        # allowed to have. Under a velocity contract it averages the previous
        # and current commands, which is the same rule and is why a step
        # command does not teleport the drone a full dt on its first tick.
        p_prev = self.position_m.copy()
        self.position_m = p_prev + 0.5 * (v_prev + v_now) * dt
        # Flight holds its altitude setpoint at the same limits and carries on,
        # so the drone stops at the limit here and the episode goes on.
        altitude_clamped = self._hold_altitude()
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
        return (self._observe(), float(reward), terminated, truncated,
                self._info(reason, altitude_clamped))

    def close(self) -> None:
        self.renderer.close()

    # -- the altitude band --------------------------------------------------
    @property
    def height_m(self) -> float:
        """Height above the lab floor: the number flight's limits apply to."""
        return self._height_of(self.position_m)

    def _height_of(self, p_m) -> float:
        p = np.asarray(p_m, dtype=float).reshape(3)
        return float(self._up @ (p - self._lab_origin_m))

    def _hold_altitude(self) -> bool:
        """Move the drone straight up or down into the altitude band.

        Only the position moves, not the velocity state, because flight does
        not touch the state either: a policy that keeps climbing at the
        ceiling keeps its upward velocity, here and in flight alike.
        """
        lim = self.cfg.limits
        h = self.height_m
        # a nanometre of slack, so rounding after a clamp is not a new clamp
        if lim.min_altitude_m - 1e-9 <= h <= lim.max_altitude_m + 1e-9:
            return False
        target = min(max(h, lim.min_altitude_m), lim.max_altitude_m)
        self.position_m = self.position_m + (target - h) * self._up
        return True

    # -- internals ---------------------------------------------------------
    def _observe(self) -> np.ndarray:
        # Yaw only: the contract holds the heading fixed, and roll and pitch
        # are the real controller's business in flight, so simulating them here
        # would show the policy an attitude the HITL renderer never will.
        obs = self.renderer.render(self.position_m,
                                   (0.0, 0.0, self.integrator.episode_yaw_rad))
        return self.builder.push(obs.depth_m, obs.rgb)

    def _info(self, reason: Optional[str],
              altitude_clamped: bool = False) -> Dict[str, Any]:
        return {
            "reason": reason,
            "steps": self.steps,
            "position_m": self.position_m.copy(),
            "height_m": self.height_m,
            "altitude_clamped": bool(altitude_clamped),
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
