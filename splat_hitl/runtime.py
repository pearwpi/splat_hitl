"""The HITL loop, and the state machine that decides what to do when it slips.

WHAT THIS OWNS
--------------
Pose in, render, policy, command out, plus scoring against gates and the ESDF.
It returns what should be published; it does not publish. That keeps ROS out of
the logic and makes every failure path testable on a laptop -- including the
ones you cannot safely produce in a lab, like a renderer that stalls for 400 ms
mid-flight or a policy that returns NaN.

THE TIMING CONTRACT
-------------------
The driver transmits from its own 50 Hz timer and stops the drone if no command
arrives for `COMMAND_TIMEOUT_S` (0.30 s); the firmware stops stabilising after
0.50 s. So this loop does NOT have to hit 50 Hz -- it has to refresh the command
often enough that the driver's blunt timeout never fires, and to notice trouble
before that happens.

Hence the states, in order of increasing pessimism:

    RUNNING   the policy is flying
    HOLDING   something is late or the pose is stale -- hover in place and
              wait a moment for it to clear
    LANDING   it did not clear -- descend under our own control rather than
              wait to be stopped by a timeout
    FINISHED  terminal, with a reason

Degrading on our own terms matters. A drone that is landed deliberately is an
experiment that ended; a drone stopped by a watchdog is a drone that falls.

WHAT IS DELIBERATELY FATAL
--------------------------
A policy that raises, or returns a non-finite action, ends the run. For a course
that is the honest verdict: an exception in flight is a failure, not a hiccup to
be smoothed over. The alternative -- catch and continue -- hides the bug and
scores the student on a controller they did not write.
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .collision import CollisionMonitor
from .commands import Action, Clamped, HoverCommand, Limits, to_hover
from .frames import matrix_to_quat, matrix_to_rpy, quat_to_matrix
from .gates import GateCourse
from .policy import Policy, PolicyState
from .renderer import Observation, RendererClient

__all__ = ["PoseSample", "PoseSource", "ScriptedPoseSource",
           "TransformedPoseSource", "RuntimeConfig", "TickResult", "Runtime",
           "RUNNING", "HOLDING", "LANDING", "FINISHED"]

RUNNING = "running"
HOLDING = "holding"
LANDING = "landing"
FINISHED = "finished"


@dataclass(frozen=True)
class PoseSample:
    t_s: float
    position_m: np.ndarray
    quat_xyzw: Tuple[float, float, float, float]

    @property
    def yaw_rad(self) -> float:
        return float(matrix_to_rpy(quat_to_matrix(*self.quat_xyzw))[2])

    @property
    def rpy(self) -> np.ndarray:
        return matrix_to_rpy(quat_to_matrix(*self.quat_xyzw))


class PoseSource(ABC):
    """Where the drone's pose comes from.

    The renderer does not care whether a pose was integrated or measured, which
    is the whole reason HITL is a substitution rather than a redesign. A Vicon
    implementation subscribing to /vicon/<body>/<body> satisfies this interface
    and needs nothing else to change.
    """

    @abstractmethod
    def latest(self) -> Optional[PoseSample]:
        """Most recent pose, or None if nothing has arrived yet."""


class ScriptedPoseSource(PoseSource):
    """Replays a fixed list of samples. For tests and dry runs."""

    def __init__(self, samples: Sequence[PoseSample]):
        self.samples = list(samples)
        self.i = -1

    def latest(self) -> Optional[PoseSample]:
        if self.i < 0:
            return None
        return self.samples[min(self.i, len(self.samples) - 1)]

    def advance(self) -> None:
        self.i += 1

    @property
    def exhausted(self) -> bool:
        return self.i >= len(self.samples) - 1


class TransformedPoseSource(PoseSource):
    """Vicon metres in, SCENE metres out.

    EVERYTHING DOWNSTREAM OF HERE IS IN THE SCENE FRAME -- the renderer, the
    gates, and the ESDF all describe the splat, not the room. Converting once,
    at the source, is what keeps them agreeing; applying the transform inside
    the renderer instead would leave the gates and the collision field still
    talking about Vicon coordinates, which is a silent and total mismatch.

    Commands still come out in the drone's BODY frame, which is physical and so
    is the same in either world. The yaw a body-frame conversion needs is scene
    yaw, and that is exactly what this emits.
    """

    def __init__(self, inner: PoseSource, transform):
        self.inner = inner
        self.transform = transform

    def latest(self) -> Optional[PoseSample]:
        s = self.inner.latest()
        if s is None:
            return None
        pos = self.transform.point_to_splat(s.position_m).reshape(3)
        R = self.transform.rotation_to_splat(quat_to_matrix(*s.quat_xyzw))
        return PoseSample(s.t_s, pos, matrix_to_quat(R))


@dataclass
class RuntimeConfig:
    hold_altitude_m: float = 0.60
    #: pose older than this -> HOLDING
    pose_stale_s: float = 0.15
    #: a tick slower than this counts as late
    tick_budget_s: float = 0.10
    #: consecutive late ticks before degrading
    late_ticks_to_hold: int = 3
    #: HOLDING for longer than this -> LANDING
    hold_before_land_s: float = 1.0
    #: descent rate once LANDING
    land_speed_ms: float = 0.25
    #: below this altitude, LANDING is complete
    land_complete_m: float = 0.12
    #: hard stop on run length
    max_duration_s: float = 120.0
    limits: Limits = field(default_factory=Limits)
    yaw_sign: int = 1
    #: refuse to fly a policy trained through a different sensor model
    enforce_sensor_match: bool = True


@dataclass
class TickResult:
    state: str
    command: Optional[HoverCommand]
    clamped: Optional[Clamped] = None
    events: List[str] = field(default_factory=list)
    pose_age_s: float = float("nan")
    render_s: float = 0.0
    policy_s: float = 0.0
    total_s: float = 0.0
    reason: Optional[str] = None

    @property
    def finished(self) -> bool:
        return self.state == FINISHED


class Runtime:
    """One flight. Construct, `step()` until finished, read `summary()`."""

    def __init__(self, pose_source: PoseSource, renderer: RendererClient,
                 policy: Policy, course: Optional[GateCourse] = None,
                 collision: Optional[CollisionMonitor] = None,
                 config: RuntimeConfig = RuntimeConfig()):
        if config.enforce_sensor_match and policy.sensor_fingerprint is not None:
            want = renderer.sensor.fingerprint()
            if policy.sensor_fingerprint != want:
                raise ValueError(
                    "policy %r was trained through sensor model %s but the "
                    "renderer is configured as %s. It has never seen this "
                    "world. Fix the sensor config rather than the policy."
                    % (policy.name, policy.sensor_fingerprint, want))
        self.poses = pose_source
        self.renderer = renderer
        self.policy = policy
        self.course = course
        self.collision = collision
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self.state = RUNNING
        self.reason: Optional[str] = None
        self.t0: Optional[float] = None
        self.steps = 0
        self.late_ticks = 0
        self.hold_started: Optional[float] = None
        self.prev: Optional[PoseSample] = None
        self.history: List[TickResult] = []
        self.policy.reset()
        if self.course is not None:
            self.course.reset()
        if self.collision is not None:
            self.collision.reset()

    # -- helpers -----------------------------------------------------------
    def _hold_command(self, altitude_m: float) -> Tuple[HoverCommand, Clamped]:
        return to_hover(Action("velocity", "body_flu", np.zeros(3)), 0.0,
                        altitude_m, self.cfg.limits, self.cfg.yaw_sign)

    def _descend_command(self, altitude_m: float) -> Tuple[HoverCommand, Clamped]:
        # Built directly rather than through to_hover(): the altitude clamp
        # would floor this at min_altitude_m, which is exactly wrong when the
        # whole point is to reach the ground.
        target = max(0.0, altitude_m - self.cfg.land_speed_ms * self.cfg.tick_budget_s)
        return HoverCommand(0.0, 0.0, 0.0, target), Clamped()

    def _finish(self, reason: str, events: List[str]) -> TickResult:
        self.state = FINISHED
        self.reason = reason
        events.append("FINISHED: %s" % reason)
        return TickResult(FINISHED, None, events=events, reason=reason)

    def _gate_state(self, pose: PoseSample):
        if self.course is None or self.course.complete:
            return None, None, self.course.passed if self.course else 0
        g = self.course.next_gate
        d = g.centre - pose.position_m
        dist = float(np.linalg.norm(d))
        bearing = float(math.atan2(d[1], d[0]) - pose.yaw_rad)
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        return dist, bearing, self.course.passed

    # -- the loop ----------------------------------------------------------
    def step(self, now: Optional[float] = None) -> TickResult:
        t_start = time.perf_counter()
        now = time.monotonic() if now is None else float(now)
        if self.t0 is None:
            self.t0 = now
        events: List[str] = []

        if self.state == FINISHED:
            return TickResult(FINISHED, None, reason=self.reason)

        if now - self.t0 > self.cfg.max_duration_s:
            return self._finish("max_duration", events)

        pose = self.poses.latest()
        if pose is None:
            return self._finish("no_pose", events)
        age = now - pose.t_s

        # --- scoring runs on the REAL pose, whatever the policy is doing ---
        if self.prev is not None:
            if self.collision is not None:
                ev = self.collision.update(self.prev.position_m, pose.position_m)
                if ev is not None:
                    events.append(str(ev))
                    return self._finish("virtual_" + ev.kind, events)
            if self.course is not None:
                for ge in self.course.update(self.prev.position_m, pose.position_m):
                    events.append(str(ge))
                if self.course.complete:
                    return self._finish("course_complete", events)

        # --- degrade on staleness before doing any work --------------------
        stale = age > self.cfg.pose_stale_s
        if stale:
            events.append("pose stale: %.0f ms" % (age * 1000.0))

        if self.state == LANDING:
            cmd, rep = self._descend_command(float(pose.position_m[2]))
            self.prev = pose
            self.steps += 1
            if pose.position_m[2] <= self.cfg.land_complete_m:
                return self._finish("landed", events)
            return self._record(TickResult(LANDING, cmd, rep, events, age,
                                           total_s=time.perf_counter() - t_start))

        if stale or self.late_ticks >= self.cfg.late_ticks_to_hold:
            if self.state != HOLDING:
                self.state = HOLDING
                self.hold_started = now
                events.append("HOLDING")
            elif self.hold_started is not None and \
                    now - self.hold_started > self.cfg.hold_before_land_s:
                self.state = LANDING
                events.append("LANDING: held for %.1f s without recovering"
                              % (now - self.hold_started))
            # Choose the command from the state we ENDED this tick in. Deciding
            # to land and then commanding a hold for one more tick leaves the
            # state and the command disagreeing, which is exactly the sort of
            # inconsistency that is unreadable in a log afterwards.
            if self.state == LANDING:
                cmd, rep = self._descend_command(float(pose.position_m[2]))
            else:
                cmd, rep = self._hold_command(self.cfg.hold_altitude_m)
            self.prev = pose
            self.steps += 1
            if not stale:
                self.late_ticks = max(0, self.late_ticks - 1)
            return self._record(TickResult(self.state, cmd, rep, events, age,
                                           total_s=time.perf_counter() - t_start))

        if self.state == HOLDING:
            self.state = RUNNING
            self.hold_started = None
            events.append("recovered")

        # --- render ---------------------------------------------------------
        try:
            obs = self.renderer.render(pose.position_m, pose.rpy)
        except Exception as exc:
            events.append("render failed: %r" % (exc,))
            return self._finish("render_error", events)

        # --- policy ---------------------------------------------------------
        dist, bearing, passed = self._gate_state(pose)
        vel = np.zeros(3)
        if self.prev is not None and pose.t_s > self.prev.t_s:
            vel = (pose.position_m - self.prev.position_m) / (pose.t_s - self.prev.t_s)
        pstate = PolicyState(now - self.t0, pose.position_m, vel, pose.yaw_rad,
                             passed, dist, bearing)
        t_pol = time.perf_counter()
        try:
            action = self.policy.act(obs, pstate)
        except Exception as exc:
            events.append("policy raised: %r" % (exc,))
            return self._finish("policy_error", events)
        policy_s = time.perf_counter() - t_pol

        if action is None or not np.all(np.isfinite(np.asarray(action.vector, float))) \
                or not math.isfinite(action.yaw_rate_rad_s):
            events.append("policy returned a non-finite action")
            return self._finish("policy_error", events)

        try:
            cmd, rep = to_hover(action, pose.yaw_rad, self.cfg.hold_altitude_m,
                                self.cfg.limits, self.cfg.yaw_sign)
        except ValueError as exc:
            events.append("action could not be mapped: %s" % exc)
            return self._finish("action_error", events)
        if rep.any:
            events.append(str(rep))

        total = time.perf_counter() - t_start
        if total > self.cfg.tick_budget_s:
            self.late_ticks += 1
            events.append("tick over budget: %.0f ms (%d in a row)"
                          % (total * 1000.0, self.late_ticks))
        else:
            self.late_ticks = 0

        self.prev = pose
        self.steps += 1
        return self._record(TickResult(RUNNING, cmd, rep, events, age,
                                       obs.render_s, policy_s, total))

    def _record(self, r: TickResult) -> TickResult:
        self.history.append(r)
        return r

    # -- reporting ---------------------------------------------------------
    def summary(self) -> str:
        L = ["  policy      : %s" % self.policy.name,
             "  steps       : %d" % self.steps,
             "  outcome     : %s" % (self.reason or self.state)]
        if self.course is not None:
            L.append(self.course.summary())
        if self.collision is not None:
            L.append(self.collision.summary())
        late = sum(1 for r in self.history if r.total_s > self.cfg.tick_budget_s)
        if late:
            worst = max(r.total_s for r in self.history)
            L.append("  timing      : %d/%d ticks over the %.0f ms budget, "
                     "worst %.0f ms" % (late, len(self.history),
                                        self.cfg.tick_budget_s * 1000,
                                        worst * 1000))
        return "\n".join(L)
