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
The driver transmits from its own 50 Hz timer and holds the drone where it is
if no command arrives for `COMMAND_TIMEOUT_S` (0.30 s); the firmware levels out
after 0.50 s and stops the motors after 2 s. So this loop does NOT have to hit
50 Hz -- it has to refresh the command often enough that the driver's blunt
timeout never fires, and to notice trouble before that happens.

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
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .collision import CollisionMonitor
from .commands import (Action, Clamped, HoverCommand, Limits,
                       VelocityIntegrator, make_action_stage, to_hover)
from .contract import PolicyContract
from .observation import ObservationBuilder
from .frames import matrix_to_quat, matrix_to_rpy, quat_to_matrix
from .gates import GateCourse
from .policy import Policy, PolicyState
from .renderer import RendererClient

__all__ = ["PoseSample", "PoseSource", "ScriptedPoseSource",
           "TransformedPoseSource", "RuntimeConfig", "TickResult", "Runtime",
           "RUNNING", "HOLDING", "LANDING", "FINISHED"]

RUNNING = "running"
HOLDING = "holding"
LANDING = "landing"
FINISHED = "finished"


@dataclass(frozen=True)
class PoseSample:
    """A pose, carrying TWO times because they answer different questions.

    t_s          the SOURCE's own clock, used for STALENESS. It must be the
                 same clock the caller passes to `Runtime.step()`. A local
                 monotonic receipt time is the right choice: it is immune to
                 clock skew between this machine and the Vicon PC, and skew is
                 exactly what a watchdog must not be fooled by.
    capture_t_s  optional epoch time the frame was CAPTURED, from the bridge's
                 capture-time stamping. For measuring LATENCY, never staleness.

    Conflating them is not hypothetical: comparing `time.monotonic()` against a
    ROS header stamp gives an age around -1.8e9 s, which silently disables the
    staleness check because a huge negative number never exceeds a threshold.
    """
    t_s: float
    position_m: np.ndarray
    quat_xyzw: Tuple[float, float, float, float]
    capture_t_s: Optional[float] = None
    #: height above the floor in the DRONE's frame. Set by a source whose
    #: position is not in that frame (TransformedPoseSource); None otherwise.
    altitude_m: Optional[float] = None

    @property
    def height_m(self) -> float:
        """Height above the floor in the frame the drone flies in -- what
        cmd_hover's z_distance means. Once a scene transform is in use,
        position_m[2] is a SCENE coordinate and is not this."""
        return (float(self.position_m[2]) if self.altitude_m is None
                else float(self.altitude_m))

    @property
    def yaw_rad(self) -> float:
        return float(matrix_to_rpy(quat_to_matrix(*self.quat_xyzw))[2])

    @property
    def rpy(self) -> np.ndarray:
        return matrix_to_rpy(quat_to_matrix(*self.quat_xyzw))

    def interval_since(self, prev: "PoseSample") -> float:
        """Seconds between two samples, on the best clock both of them carry.

        `t_s` is a RECEIPT time, so the gap between two of them is the true
        interval plus whatever jitter the network and the executor added. That
        is the right clock for staleness -- a watchdog must measure how long it
        has been since something arrived -- and the wrong one for a derivative:
        dividing a real position delta by a jittered interval turns arrival
        jitter into velocity error.

        `capture_t_s` is when the cameras actually saw the drone, so the gap
        between two of them is the physical interval. Roughly 2 ms of arrival
        jitter across a 67 ms control step is a 3% velocity error, which at
        racing speed is tens of millimetres per second on a number the whole
        sim-to-real comparison rests on.

        The two clocks are never mixed: capture times are epoch, receipt times
        are monotonic, and differencing one against the other gives about
        -1.8e9 s. Both samples must carry a capture time or neither is used.
        """
        if self.capture_t_s is not None and prev.capture_t_s is not None:
            dt = float(self.capture_t_s) - float(prev.capture_t_s)
            if dt > 0.0:
                return dt
        return float(self.t_s) - float(prev.t_s)


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

    METRES, NOT UNITS. `point_to_splat()` hands back NORMALISED splat units,
    and every consumer downstream wants scene metres: `ESDF.at()` divides by
    `metres_per_unit` itself, `SplatWorkerClient.render()` divides by the
    worker's `scale_to_metres` itself, `gates.json` is written in scene metres,
    and `SplatEnv` trains in them. This used to emit units -- about 3.3x too
    small on every scene -- so under `--transform` the collision monitor, the
    gate scoring and the renderer all looked at the wrong place, and nothing
    complained. `render_check.vicon_pose_to_worker` does the same conversion,
    and a test holds the two together.
    """

    def __init__(self, inner: PoseSource, transform):
        self.inner = inner
        self.transform = transform

    def latest(self) -> Optional[PoseSample]:
        s = self.inner.latest()
        if s is None:
            return None
        pos = (self.transform.point_to_splat(s.position_m).reshape(3)
               * self.transform.metres_per_unit)
        R = self.transform.rotation_to_splat(quat_to_matrix(*s.quat_xyzw))
        # capture_t_s must ride along. Dropping it here silently turned
        # end-to-end latency into NaN the moment a calibration transform was
        # supplied -- that is, in exactly the real HITL configuration, and
        # never in the fake-room dry runs this was tested with.
        return PoseSample(s.t_s, pos, matrix_to_quat(R),
                          capture_t_s=s.capture_t_s, altitude_m=s.height_m)


@dataclass
class RuntimeConfig:
    #: the altitude a run STARTS at. A policy's vertical velocity moves it from
    #: there, as SplatEnv does in simulation; HOLDING holds wherever it has got
    #: to rather than dropping back here, which could be into an obstacle.
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
    #: a pose further than this into the future means the clocks disagree
    max_clock_skew_s: float = 1.0
    #: the full train/fly contract. None keeps the pre-contract behaviour:
    #: raw metric depth to the policy and no integrator.
    contract: Optional[PolicyContract] = None
    #: with contract.action.yaw_mode == "fixed" the policy has only ever seen
    #: the scene from its starting heading, so drift past this makes every
    #: observation invalid and the runtime stops using the policy.
    max_yaw_drift_rad: float = math.radians(20.0)
    #: optional proportional hold on that heading. 0 = report drift, correct
    #: nothing; a controller nobody asked for is worse than a number nobody
    #: ignores.
    hold_yaw_gain: float = 0.0
    #: fraction the MEASURED step rate may differ from contract.control before
    #: the run is refused. A double integrator at the wrong rate is silently
    #: wrong, so this is checked against the clock rather than against config.
    rate_tolerance: float = 0.20
    #: steps to measure before that check fires
    rate_check_after: int = 30


@dataclass
class TickResult:
    state: str
    command: Optional[HoverCommand]
    clamped: Optional[Clamped] = None
    events: List[str] = field(default_factory=list)
    pose_age_s: float = float("nan")
    latency_s: float = float("nan")
    render_s: float = 0.0
    policy_s: float = 0.0
    total_s: float = 0.0
    reason: Optional[str] = None
    #: |open-loop velocity - measured velocity|, the sim-to-real gap per tick
    divergence_ms: float = float("nan")
    #: heading drift from the episode heading the policy assumes
    yaw_error_rad: float = float("nan")

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

        self.builder: Optional[ObservationBuilder] = None
        #: VelocityIntegrator or VelocityPassthrough, per contract.action.kind
        self.integrator: Optional[VelocityIntegrator] = None
        c = config.contract
        if c is not None:
            # The renderer must be configured as the contract describes, or the
            # observation builder is encoding an image the policy never saw.
            if c.observation.sensor.fingerprint() != renderer.sensor.fingerprint():
                raise ValueError(
                    "the contract's sensor (%s) is not the renderer's (%s).\n"
                    "Load one sensor model, not two."
                    % (c.observation.sensor.fingerprint(),
                       renderer.sensor.fingerprint()))
            declared = getattr(policy, "contract_fingerprint", None)
            if config.enforce_sensor_match and declared is not None \
                    and declared != c.fingerprint():
                raise ValueError(
                    "policy %r declares contract %s but this run is configured "
                    "as %s. Compare the two files rather than guessing which "
                    "field moved." % (policy.name, declared, c.fingerprint()))
            self.builder = ObservationBuilder(c.observation)
            # The same call the env makes, so an acceleration contract flies
            # through the integrator it trained with and a velocity contract
            # through the passthrough it trained with. A position contract
            # gets neither and goes straight to to_hover().
            if c.action.kind in ("acceleration", "velocity"):
                self.integrator = make_action_stage(
                    c.action, c.control.dt_s, config.limits,
                    hold_yaw_gain=config.hold_yaw_gain)
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
        #: heading captured from the first pose. With yaw_mode="fixed" this is
        #: the ONLY heading the policy has ever seen the scene from.
        self.episode_yaw_rad: Optional[float] = None
        self._step_times: List[float] = []
        self._rate_checked = False
        #: the altitude being commanded, in the drone's frame. Integrated from
        #: the policy's vertical velocity on every RUNNING tick.
        self.altitude_sp_m = float(self.cfg.hold_altitude_m)
        #: time of the last RUNNING tick, for that integration. Cleared by any
        #: tick that does not fly the policy, so recovering from a hold does
        #: not integrate the whole hold at once.
        self._last_run_t: Optional[float] = None
        if self.builder is not None:
            self.builder.reset()
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
        now_wall = time.time()
        self._step_times.append(now)
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
        if age < -self.cfg.max_clock_skew_s:
            # A pose from the future means `now` and `t_s` are on different
            # clocks -- almost always a ROS epoch header stamp compared against
            # time.monotonic(). Left alone this does not raise: it just makes
            # `age` enormously negative, so the staleness watchdog can never
            # fire. A safety check that can be silently switched off by a unit
            # error is worse than no check, so this is fatal and loud.
            events.append(
                "pose is %.0f s in the future -- the pose source's clock and "
                "the loop's clock disagree. `PoseSample.t_s` must be on the "
                "same clock you pass to step(); use capture_t_s for latency."
                % (-age,))
            return self._finish("clock_mismatch", events)
        latency = (now_wall - pose.capture_t_s
                   if (pose.capture_t_s is not None and now_wall is not None)
                   else float("nan"))

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
            self._last_run_t = None
            cmd, rep = self._descend_command(pose.height_m)
            self.prev = pose
            self.steps += 1
            if pose.height_m <= self.cfg.land_complete_m:
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
            self._last_run_t = None
            if self.state == LANDING:
                cmd, rep = self._descend_command(pose.height_m)
            else:
                cmd, rep = self._hold_command(self.altitude_sp_m)
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

        # --- the contract, if there is one -----------------------------------
        if self.episode_yaw_rad is None:
            # Captured from the FIRST pose of the run, not from config: with
            # yaw_mode="fixed" this is the heading the policy will assume for
            # the whole flight, so it has to be where the drone actually is.
            self.episode_yaw_rad = float(pose.yaw_rad)
            if self.integrator is not None:
                self.integrator.reset(self.episode_yaw_rad)
                events.append("episode heading %.1f deg"
                              % math.degrees(self.episode_yaw_rad))

        yaw_err = float(pose.yaw_rad) - self.episode_yaw_rad
        yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi

        c = self.cfg.contract
        if c is not None and not self._rate_checked \
                and len(self._step_times) > self.cfg.rate_check_after:
            gaps = sorted(self._step_times[i] - self._step_times[i - 1]
                          for i in range(1, len(self._step_times)))
            measured = gaps[len(gaps) // 2]
            want = c.control.dt_s
            self._rate_checked = True
            if measured > 0 and abs(measured - want) / want > self.cfg.rate_tolerance:
                events.append(
                    "measured step rate %.1f Hz, contract says %.1f Hz. A "
                    "policy that integrates its own action carries its "
                    "timestep inside its behaviour, so running it at a "
                    "different rate does not make it smoother -- it makes it "
                    "faster." % (1.0 / measured, c.control.rate_hz))
                return self._finish("rate_mismatch", events)

        if c is not None and c.action.yaw_mode == "fixed" \
                and abs(yaw_err) > self.cfg.max_yaw_drift_rad:
            # Every degree of drift points the camera somewhere this policy has
            # never looked. Hold rather than land: holding is recoverable, and
            # continuing to act on invalid observations is not.
            events.append("yaw drifted %.1f deg from the episode heading; the "
                          "policy's observations are no longer valid"
                          % math.degrees(yaw_err))
            self.state = HOLDING
            self.hold_started = now
            self._last_run_t = None
            cmd, rep = self._hold_command(self.altitude_sp_m)
            return self._record(TickResult(
                HOLDING, cmd, rep, events, age, latency,
                total_s=time.perf_counter() - t_start, yaw_error_rad=yaw_err))

        # --- render ---------------------------------------------------------
        try:
            obs = self.renderer.render(pose.position_m, pose.rpy)
        except Exception as exc:
            events.append("render failed: %r" % (exc,))
            return self._finish("render_error", events)

        if self.builder is not None:
            try:
                obs = replace(obs,
                              policy_input=self.builder.push(obs.depth_m,
                                                            obs.rgb))
            except Exception as exc:
                events.append("observation encoding failed: %r" % (exc,))
                return self._finish("observation_error", events)

        # --- policy ---------------------------------------------------------
        dist, bearing, passed = self._gate_state(pose)
        vel = np.zeros(3)
        if self.prev is not None:
            dt_v = pose.interval_since(self.prev)
            if dt_v > 0.0:
                vel = (pose.position_m - self.prev.position_m) / dt_v
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

        divergence = float("nan")
        if self.integrator is not None:
            try:
                action, ireport = self.integrator.step(
                    action, current_yaw_rad=float(pose.yaw_rad),
                    measured_velocity_world=vel)
            except Exception as exc:
                events.append("integrator refused the action: %s" % exc)
                return self._finish("action_error", events)
            divergence = ireport.divergence_ms
            if ireport.state_saturated:
                events.append("integrator state saturated")

        climb_dt = 0.0
        if self._last_run_t is not None:
            climb_dt = min(max(now - self._last_run_t, 0.0), self.cfg.tick_budget_s)
        try:
            cmd, rep = to_hover(action, pose.yaw_rad, self.altitude_sp_m,
                                self.cfg.limits, self.cfg.yaw_sign,
                                climb_dt_s=climb_dt)
        except ValueError as exc:
            events.append("action could not be mapped: %s" % exc)
            return self._finish("action_error", events)
        self.altitude_sp_m = cmd.z_distance
        self._last_run_t = now
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
        return self._record(TickResult(RUNNING, cmd, rep, events, age, latency,
                                       obs.render_s, policy_s, total,
                                       divergence_ms=divergence,
                                       yaw_error_rad=yaw_err))

    def _record(self, r: TickResult) -> TickResult:
        self.history.append(r)
        return r

    def stop(self, reason: str = "operator_stop") -> None:
        """End the run from outside -- Ctrl-C, a supervisor, a test.

        Without this the caller records one outcome and the runtime reports
        another, which is how a dry run ends with the log saying
        `operator_stop` and the summary still saying `running`.
        """
        if self.state != FINISHED:
            self.state = FINISHED
            self.reason = reason

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
