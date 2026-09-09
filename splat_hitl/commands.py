"""Turn a policy's action into a Crazyflie command, with the units made explicit.

WHY THIS MODULE EXISTS
----------------------
The driver reproduces Crazyswarm2's conventions exactly and deliberately, and
those conventions are internally inconsistent. From crazyflie_server.py:

    cmd_position         yaw        DEGREES,  world frame, metres
    cmd_hover            yaw_rate   RADIANS/s, AND SIGN-FLIPPED on the way out
                                    (`-1.0 * math.degrees(msg.yaw_rate)`),
                                    vx/vy are BODY frame, z_distance is an
                                    ABSOLUTE altitude, not a delta
    cmd_velocity_world   yaw_rate   DEGREES/s, world frame
    cmd_full_state       angular    DEGREES/s

Four messages, three yaw conventions and a sign flip. Meanwhile policies emit
whatever they were trained in: body-frame velocity, world acceleration, NED
with +z DOWN while the Crazyflie's +z is UP.

Every one of those mismatches produces a drone that flies confidently in the
wrong direction, and half of them are invisible while yaw is zero -- which is
exactly how every example in the surrounding repos is written.

So: one canonical internal convention, conversions that are named, and
refusals where a conversion would need state the caller has not supplied.

CANONICAL CONVENTION
--------------------
    world ENU, +z UP, yaw POSITIVE COUNTER-CLOCKWISE seen from above.

That matches Vicon and the flight stack. Anything else is converted at the
boundary, once, by a function whose name says what it is doing.

ON THE YAW SIGN
---------------
`yaw_sign` exists because the teleop has `--yaw-sign` for the same reason: the
sign of the CRTP yaw setpoint has changed across firmware releases. This module
does not pretend to know which one you are running. Verify it PROPS OFF, set it
once, and record it in the run log.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .frames import rpy_to_matrix

__all__ = [
    "Action", "Limits", "Clamped", "HoverCommand", "VelocityWorldCommand",
    "PositionCommand", "to_hover", "to_velocity_world", "to_position",
    "to_world_enu", "VelocityIntegrator", "IntegratorReport", "action_from_raw",
]

# kinds a policy can emit
KINDS = ("velocity", "acceleration", "position")
# frames it can emit them in
FRAMES = ("world_enu", "world_ned", "body_flu", "body_frd")


@dataclass(frozen=True)
class Action:
    """What the policy produced, with its frame stated rather than assumed.

    body_flu  x forward, y left,  z up      (ROS body convention)
    body_frd  x forward, y right, z down    (aerospace convention)
    world_enu x east,    y north, z up      (Vicon, and our canonical)
    world_ned x north,   y east,  z down    (VizFlyt2's dynamics and planners)

    `yaw_rate_rad_s` is always CCW-positive about world +z, whatever the frame
    of `vector`: a yaw rate has no frame beyond its sign convention.
    """
    kind: str
    frame: str
    vector: np.ndarray
    yaw_rate_rad_s: float = 0.0
    yaw_rad: Optional[float] = None          # only meaningful for kind="position"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError("kind %r not in %s" % (self.kind, KINDS))
        if self.frame not in FRAMES:
            raise ValueError("frame %r not in %s" % (self.frame, FRAMES))
        v = np.asarray(self.vector, dtype=float).reshape(-1)
        if v.shape != (3,):
            raise ValueError("vector must have 3 components, got %s" % (v.shape,))
        if not np.all(np.isfinite(v)):
            raise ValueError("vector contains non-finite values: %r" % (v,))
        if not math.isfinite(self.yaw_rate_rad_s):
            raise ValueError("yaw_rate_rad_s is not finite")
        object.__setattr__(self, "vector", v)


@dataclass(frozen=True)
class Limits:
    """Envelope applied to every command. These are NOT the safety core.

    cf_core's guards decide whether to keep flying. These decide what a
    well-behaved command looks like, and they report saturation, because a
    policy that constantly asks for 8 m/s and receives 1.5 is a finding, not a
    detail.
    """
    max_speed_ms: float = 1.5
    max_climb_ms: float = 0.6
    max_yaw_rate_rad_s: float = math.radians(90.0)
    min_altitude_m: float = 0.10
    max_altitude_m: float = 1.20

    def __post_init__(self):
        if not 0 < self.min_altitude_m < self.max_altitude_m:
            raise ValueError("need 0 < min_altitude_m < max_altitude_m")
        for n in ("max_speed_ms", "max_climb_ms", "max_yaw_rate_rad_s"):
            if not getattr(self, n) > 0:
                raise ValueError("%s must be positive" % n)


@dataclass
class Clamped:
    """What the envelope had to change. Log it."""
    speed: bool = False
    climb: bool = False
    yaw_rate: bool = False
    altitude: bool = False
    requested: dict = field(default_factory=dict)

    @property
    def any(self) -> bool:
        return self.speed or self.climb or self.yaw_rate or self.altitude

    def __str__(self) -> str:
        if not self.any:
            return "clamped: none"
        hit = [n for n in ("speed", "climb", "yaw_rate", "altitude") if getattr(self, n)]
        return "clamped: %s (requested %s)" % (
            ", ".join(hit),
            ", ".join("%s=%.3f" % kv for kv in sorted(self.requested.items())))


# --------------------------------------------------------------- frame maths
def to_world_enu(action: Action, yaw_rad: float) -> np.ndarray:
    """The action's vector expressed in world ENU.

    `yaw_rad` is the drone's CURRENT heading, CCW-positive about world +z. It
    is required for body frames and ignored for world frames -- passing it
    always keeps the call site honest about the fact that body-frame commands
    are only meaningful alongside an attitude.
    """
    v = action.vector
    if action.frame == "world_enu":
        return v.copy()
    if action.frame == "world_ned":
        # NED -> ENU: north,east,down -> east,north,up
        return np.array([v[1], v[0], -v[2]], dtype=float)
    if action.frame == "body_frd":
        v = np.array([v[0], -v[1], -v[2]], dtype=float)   # FRD -> FLU
    # body_flu -> world_enu is a yaw rotation; roll and pitch are the
    # controller's business, not the setpoint's.
    R = rpy_to_matrix(0.0, 0.0, yaw_rad)
    return R @ v


def _clamp_horizontal(v: np.ndarray, limits: Limits, rep: Clamped) -> np.ndarray:
    out = v.copy()
    speed = float(math.hypot(out[0], out[1]))
    if speed > limits.max_speed_ms:
        rep.speed = True
        rep.requested["speed_ms"] = speed
        out[0] *= limits.max_speed_ms / speed
        out[1] *= limits.max_speed_ms / speed
    if abs(out[2]) > limits.max_climb_ms:
        rep.climb = True
        rep.requested["climb_ms"] = float(out[2])
        out[2] = math.copysign(limits.max_climb_ms, out[2])
    return out


def _clamp_yaw_rate(w: float, limits: Limits, rep: Clamped) -> float:
    if abs(w) > limits.max_yaw_rate_rad_s:
        rep.yaw_rate = True
        rep.requested["yaw_rate_rad_s"] = float(w)
        return math.copysign(limits.max_yaw_rate_rad_s, w)
    return float(w)


def _clamp_altitude(z: float, limits: Limits, rep: Clamped) -> float:
    if z < limits.min_altitude_m or z > limits.max_altitude_m:
        rep.altitude = True
        rep.requested["altitude_m"] = float(z)
        return float(min(max(z, limits.min_altitude_m), limits.max_altitude_m))
    return float(z)


def _reject_acceleration(action: Action, what: str) -> None:
    if action.kind == "acceleration":
        raise ValueError(
            "cannot build %s from an acceleration action.\n"
            "Turning acceleration into velocity needs the current velocity and "
            "a timestep -- state this module does not have. Integrate it at the "
            "call site, where dt is known, and pass a velocity Action. Doing it "
            "silently here would hide the integrator that actually determines "
            "how the drone behaves." % what)


# ------------------------------------------------------- the integrator
@dataclass
class IntegratorReport:
    """What the integrator did, so the sim-to-real gap is visible per tick.

    `divergence_ms` is the whole point of carrying both velocities: it is the
    distance between what the policy THINKS its velocity is (the double
    integrator it was trained with) and what the drone is ACTUALLY doing. It
    starts near zero and grows with drag, thrust error and wind. Logged every
    tick, it is the sim-to-real gap plotted against time instead of argued
    about.
    """
    velocity_world_open_loop: np.ndarray
    velocity_world_measured: Optional[np.ndarray]
    divergence_ms: float
    mode: str
    yaw_error_rad: float
    state_saturated: bool

    def __str__(self) -> str:
        d = "n/a" if math.isnan(self.divergence_ms) else "%.3f m/s" % self.divergence_ms
        return ("integrator[%s] |v|=%.3f m/s, divergence %s, yaw err %+.1f deg%s"
                % (self.mode, float(np.linalg.norm(self.velocity_world_open_loop)),
                   d, math.degrees(self.yaw_error_rad),
                   ", STATE SATURATED" if self.state_saturated else ""))


def _wrap_pi(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def action_from_raw(spec, raw, yaw_rate_rad_s: float = 0.0) -> Action:
    """A policy network's raw output -> a physical Action, per the contract.

    Reproduces the trainer exactly:

        action = np.clip(action, -1.0, 1.0)
        acceleration_body = action * max_acceleration_m_s2
        if ||acceleration_body|| > max: rescale to max

    Note the last step is a NORM limit, not a per-axis clip. (1, 1, 1) scaled by
    1.5 has norm 2.6, and the trainer pulls it back to 1.5. Clipping per axis
    instead would leave the diagonal 73% too fast, which is a bias a policy
    trained under the norm limit never had to correct for.
    """
    v = np.asarray(raw, dtype=float).reshape(-1)
    if v.shape != (3,):
        raise ValueError("raw action must have 3 components, got %s" % (v.shape,))
    if spec.clip_unit:
        v = np.clip(v, -1.0, 1.0)
    v = v * float(spec.scale)
    if spec.limit_norm:
        n = float(np.linalg.norm(v))
        if n > float(spec.scale) and n > 0.0:
            v = v * (float(spec.scale) / n)
    return Action(kind=spec.kind, frame=spec.frame, vector=v,
                  yaw_rate_rad_s=float(yaw_rate_rad_s))


class VelocityIntegrator:
    """Acceleration action -> velocity Action, with the integrator made explicit.

    `to_hover` refuses an acceleration on purpose, and that refusal stays: the
    integrator is what actually decides how the drone behaves, so it is an
    object you construct and reset rather than a hidden conversion. This is the
    thing the refusal message tells you to build.

    THE FRAME IS THE PARITY-CRITICAL PART
    -------------------------------------
    The trainer captures `episode_basis` at reset and NEVER updates it, and
    holds `yaw_policy_rad = 0.0`. So its "body frame" is the heading the
    episode started at, frozen, and its velocity state lives in the scene
    frame. With `yaw_mode="fixed"` this reproduces that: body-frame
    accelerations are rotated by the EPISODE yaw, not the current yaw, and the
    emitted yaw rate is forced to zero.

    That also means the drone must actually hold that heading. It will drift,
    and every degree of drift points the camera somewhere the policy has never
    looked, so `yaw_error_rad` is reported every tick for the runtime to act
    on. No yaw correction is applied here by default (`hold_yaw_gain=0`),
    because a controller nobody asked for is worse than a number nobody
    ignores. Set a gain if you want the heading actively held.

    WHY THE STATE IS CLAMPED, NOT JUST THE OUTPUT
    ---------------------------------------------
    `Limits` clamps the command. If the internal velocity were left unclamped,
    a policy that keeps asking for +x would wind the state up to 10 m/s while
    the output sat at 1.5, and when it finally commanded a reversal nothing
    would happen for several seconds. That is textbook integrator windup with a
    drone attached, so the state is clamped by the same envelope.
    """

    def __init__(self, spec, dt_s: float, limits: Limits = Limits(),
                 hold_yaw_gain: float = 0.0):
        if spec.kind != "acceleration":
            raise ValueError("VelocityIntegrator is for acceleration actions; "
                             "this contract emits %r, which needs no "
                             "integrator." % (spec.kind,))
        if not dt_s > 0:
            raise ValueError("dt_s must be positive, got %r" % (dt_s,))
        if hold_yaw_gain < 0:
            raise ValueError("hold_yaw_gain must be >= 0")
        self.spec = spec
        self.dt_s = float(dt_s)
        self.limits = limits
        self.hold_yaw_gain = float(hold_yaw_gain)
        self.episode_yaw_rad = 0.0
        self.velocity_world = np.zeros(3, dtype=float)
        self._started = False

    def reset(self, episode_yaw_rad: float, velocity_world=(0.0, 0.0, 0.0)) -> None:
        """Start a run. `episode_yaw_rad` is the heading the policy will assume."""
        if not math.isfinite(episode_yaw_rad):
            raise ValueError("episode_yaw_rad is not finite")
        self.episode_yaw_rad = float(episode_yaw_rad)
        v = np.asarray(velocity_world, dtype=float).reshape(-1)
        if v.shape != (3,):
            raise ValueError("velocity_world must have 3 components")
        self.velocity_world = v.copy()
        self._started = True

    def yaw_error_rad(self, current_yaw_rad: float) -> float:
        """How far the drone has drifted from the heading the policy assumes."""
        return _wrap_pi(float(current_yaw_rad) - self.episode_yaw_rad)

    def _clamp_state(self, v: np.ndarray):
        out = v.copy()
        hit = False
        speed = float(math.hypot(out[0], out[1]))
        if speed > self.limits.max_speed_ms:
            out[0] *= self.limits.max_speed_ms / speed
            out[1] *= self.limits.max_speed_ms / speed
            hit = True
        if abs(out[2]) > self.limits.max_climb_ms:
            out[2] = math.copysign(self.limits.max_climb_ms, out[2])
            hit = True
        return out, hit

    def step(self, action: Action, current_yaw_rad: float,
             measured_velocity_world=None) -> Tuple[Action, IntegratorReport]:
        """One control step. Returns a WORLD-ENU velocity Action and a report."""
        if not self._started:
            raise RuntimeError(
                "VelocityIntegrator.step() before reset(). Without a reset the "
                "run inherits the previous run's velocity and heading, which "
                "is a silently wrong flight rather than a failed one.")
        if action.kind != "acceleration":
            raise ValueError("expected an acceleration action, got %r" % (action.kind,))

        yaw_for_frame = (self.episode_yaw_rad if self.spec.yaw_mode == "fixed"
                         else float(current_yaw_rad))
        accel_world = to_world_enu(action, yaw_for_frame)

        open_loop, hit = self._clamp_state(self.velocity_world + accel_world * self.dt_s)
        self.velocity_world = open_loop

        measured = None
        divergence = float("nan")
        if measured_velocity_world is not None:
            m = np.asarray(measured_velocity_world, dtype=float).reshape(-1)
            if m.shape != (3,):
                raise ValueError("measured_velocity_world must have 3 components")
            measured, _ = self._clamp_state(m + accel_world * self.dt_s)
            divergence = float(np.linalg.norm(open_loop - measured))

        if self.spec.integrator == "measured":
            if measured is None:
                raise ValueError(
                    "integrator='measured' needs measured_velocity_world every "
                    "step. Falling back to the open-loop value would quietly "
                    "change which controller is flying the drone.")
            emitted = measured
        else:
            emitted = open_loop

        err = self.yaw_error_rad(current_yaw_rad)
        if self.spec.yaw_mode == "fixed":
            yaw_rate = -self.hold_yaw_gain * err
        else:
            yaw_rate = action.yaw_rate_rad_s

        out = Action(kind="velocity", frame="world_enu", vector=emitted,
                     yaw_rate_rad_s=float(yaw_rate))
        return out, IntegratorReport(open_loop.copy(),
                                     None if measured is None else measured.copy(),
                                     divergence, self.spec.integrator, err, hit)


class VelocityPassthrough:
    """Velocity action -> velocity Action. The trivial case, made explicit.

    A velocity contract has no integrator. The policy's number IS the
    commanded velocity, so there is no state to wind up, nothing to reset but
    the heading, and no divergence between "what the policy thinks its
    velocity is" and "what it commanded" -- those are the same number.

    It exists so that SplatEnv has one seam instead of a branch: this class
    and VelocityIntegrator present the same three methods, and the env picks
    one at construction and never asks again.

    WHAT IT STILL DOES, AND WHY
    ---------------------------
    Three things, all of which happen in flight too, so leaving any of them
    out would make the simulator the easier world:

    * rotates a body-frame command into world ENU, because that is what
      `to_hover` does before it reaches the radio;
    * clamps to the same `Limits` envelope, so a student whose gains ask for
      4 m/s discovers the ceiling in sim rather than in the net;
    * forces the yaw rate to zero under `yaw_mode="fixed"`, because a fixed
      contract means the policy has only ever seen the course from its
      starting heading.

    WHAT IT DELIBERATELY DOES NOT DO
    --------------------------------
    No first-order lag. The commanded velocity is applied immediately, which
    says the inner loop tracks velocity perfectly -- it does not. That gap is
    real and it is the interesting part of an outer-loop tuning exercise, so
    it is left visible rather than approximated with a time constant nobody
    measured. Pass `measured_velocity_world` and the report carries the gap;
    in simulation there is nothing to measure and it reads n/a.
    """

    def __init__(self, spec, dt_s: float, limits: Limits = Limits(),
                 hold_yaw_gain: float = 0.0):
        if spec.kind != "velocity":
            raise ValueError("VelocityPassthrough is for velocity actions; "
                             "this contract emits %r. An acceleration action "
                             "needs VelocityIntegrator." % (spec.kind,))
        if not dt_s > 0:
            raise ValueError("dt_s must be positive, got %r" % (dt_s,))
        if hold_yaw_gain < 0:
            raise ValueError("hold_yaw_gain must be >= 0")
        self.spec = spec
        self.dt_s = float(dt_s)
        self.limits = limits
        self.hold_yaw_gain = float(hold_yaw_gain)
        self.episode_yaw_rad = 0.0
        self.velocity_world = np.zeros(3, dtype=float)
        self._started = False

    def reset(self, episode_yaw_rad: float, velocity_world=(0.0, 0.0, 0.0)) -> None:
        """Start a run. `episode_yaw_rad` is the heading the policy assumes.

        `velocity_world` is the velocity the drone is already carrying. There
        is no integrator state here, but the env needs a previous velocity to
        integrate position trapezoidally across the first step, and starting
        that at something other than the truth puts a step into tick one.
        """
        if not math.isfinite(episode_yaw_rad):
            raise ValueError("episode_yaw_rad is not finite")
        self.episode_yaw_rad = float(episode_yaw_rad)
        v = np.asarray(velocity_world, dtype=float).reshape(-1)
        if v.shape != (3,):
            raise ValueError("velocity_world must have 3 components")
        self.velocity_world = v.copy()
        self._started = True

    def yaw_error_rad(self, current_yaw_rad: float) -> float:
        """How far the drone has drifted from the heading the policy assumes."""
        return _wrap_pi(float(current_yaw_rad) - self.episode_yaw_rad)

    def _clamp(self, v: np.ndarray):
        out = v.copy()
        hit = False
        speed = float(math.hypot(out[0], out[1]))
        if speed > self.limits.max_speed_ms:
            out[0] *= self.limits.max_speed_ms / speed
            out[1] *= self.limits.max_speed_ms / speed
            hit = True
        if abs(out[2]) > self.limits.max_climb_ms:
            out[2] = math.copysign(self.limits.max_climb_ms, out[2])
            hit = True
        return out, hit

    def step(self, action: Action, current_yaw_rad: float,
             measured_velocity_world=None) -> Tuple[Action, IntegratorReport]:
        """One control step. Returns a WORLD-ENU velocity Action and a report."""
        if not self._started:
            raise RuntimeError(
                "VelocityPassthrough.step() before reset(). Without a reset "
                "the run inherits the previous run's heading, which is a "
                "silently wrong flight rather than a failed one.")
        if action.kind != "velocity":
            raise ValueError("expected a velocity action, got %r" % (action.kind,))

        # The CURRENT yaw, not the episode yaw: this is the rotation `to_hover`
        # applies in flight, and matching it is the whole point. Under
        # yaw_mode="fixed" the two agree anyway -- and when they stop agreeing,
        # the runtime has already dropped to HOLDING.
        commanded, hit = self._clamp(to_world_enu(action, float(current_yaw_rad)))
        self.velocity_world = commanded

        measured = None
        divergence = float("nan")
        if measured_velocity_world is not None:
            m = np.asarray(measured_velocity_world, dtype=float).reshape(-1)
            if m.shape != (3,):
                raise ValueError("measured_velocity_world must have 3 components")
            measured = m.copy()
            divergence = float(np.linalg.norm(commanded - measured))

        err = self.yaw_error_rad(current_yaw_rad)
        yaw_rate = (-self.hold_yaw_gain * err if self.spec.yaw_mode == "fixed"
                    else action.yaw_rate_rad_s)

        out = Action(kind="velocity", frame="world_enu", vector=commanded,
                     yaw_rate_rad_s=float(yaw_rate))
        return out, IntegratorReport(commanded.copy(), measured, divergence,
                                     "passthrough", err, hit)


def make_action_stage(spec, dt_s: float, limits: Limits = Limits(),
                      hold_yaw_gain: float = 0.0):
    """The right action->velocity stage for a contract's ActionSpec.

    One call site instead of a branch repeated in the env, the runtime and
    every test: an acceleration contract gets the double integrator it was
    trained with, a velocity contract gets the passthrough.
    """
    if spec.kind == "acceleration":
        return VelocityIntegrator(spec, dt_s, limits, hold_yaw_gain)
    if spec.kind == "velocity":
        return VelocityPassthrough(spec, dt_s, limits, hold_yaw_gain)
    raise ValueError(
        "no action stage for kind %r. A position contract commands a place, "
        "not a rate, and does not pass through this path." % (spec.kind,))


# ------------------------------------------------------------------ commands
@dataclass(frozen=True)
class HoverCommand:
    """Fields for crazyflie_interfaces/Hover, in the message's own units.

    vx, vy      BODY frame m/s
    yaw_rate    RADIANS/s as the message wants. The driver applies
                `-1.0 * math.degrees(...)` on the way to the radio; that flip
                is the driver's, not ours, and must not be pre-applied here.
    z_distance  ABSOLUTE altitude in metres, not a delta.
    """
    vx: float
    vy: float
    yaw_rate: float
    z_distance: float


@dataclass(frozen=True)
class VelocityWorldCommand:
    """crazyflie_interfaces/VelocityWorld. yaw_rate is DEGREES/s."""
    vx: float
    vy: float
    vz: float
    yaw_rate: float


@dataclass(frozen=True)
class PositionCommand:
    """crazyflie_interfaces/Position. yaw is DEGREES."""
    x: float
    y: float
    z: float
    yaw: float


def to_hover(action: Action, current_yaw_rad: float, target_altitude_m: float,
             limits: Limits = Limits(), yaw_sign: int = 1
             ) -> Tuple[HoverCommand, Clamped]:
    """Body-frame velocity + absolute altitude.

    The vector is converted to world ENU and back into the body frame, which is
    a no-op for `body_flu` and a real rotation for everything else. Doing it
    through world rather than special-casing keeps one code path.
    """
    _reject_acceleration(action, "a Hover command")
    if action.kind == "position":
        raise ValueError("Hover carries a velocity, not a position; use "
                         "to_position() or convert at the call site.")
    if yaw_sign not in (1, -1):
        raise ValueError("yaw_sign must be +1 or -1")

    rep = Clamped()
    world = _clamp_horizontal(to_world_enu(action, current_yaw_rad), limits, rep)
    body = rpy_to_matrix(0.0, 0.0, -current_yaw_rad) @ world     # ENU -> body FLU
    w = _clamp_yaw_rate(action.yaw_rate_rad_s, limits, rep)
    z = _clamp_altitude(target_altitude_m, limits, rep)
    return HoverCommand(float(body[0]), float(body[1]),
                        float(yaw_sign * w), z), rep


def to_velocity_world(action: Action, current_yaw_rad: float,
                      limits: Limits = Limits(), yaw_sign: int = 1
                      ) -> Tuple[VelocityWorldCommand, Clamped]:
    """World-frame velocity. yaw_rate leaves in DEGREES/s."""
    _reject_acceleration(action, "a VelocityWorld command")
    if action.kind == "position":
        raise ValueError("VelocityWorld carries a velocity, not a position.")
    if yaw_sign not in (1, -1):
        raise ValueError("yaw_sign must be +1 or -1")

    rep = Clamped()
    v = _clamp_horizontal(to_world_enu(action, current_yaw_rad), limits, rep)
    w = _clamp_yaw_rate(action.yaw_rate_rad_s, limits, rep)
    return VelocityWorldCommand(float(v[0]), float(v[1]), float(v[2]),
                                float(math.degrees(yaw_sign * w))), rep


def to_position(action: Action, current_yaw_rad: float,
                limits: Limits = Limits(), yaw_sign: int = 1
                ) -> Tuple[PositionCommand, Clamped]:
    """Absolute world position. yaw leaves in DEGREES.

    Only altitude is clamped: the speed limits describe velocities and mean
    nothing here. Bounding where a position setpoint may sit is the leash's job
    in cf_core, and duplicating it with different numbers would be worse than
    not doing it.
    """
    if action.kind != "position":
        raise ValueError("to_position() needs kind='position', got %r" % action.kind)
    if action.frame not in ("world_enu", "world_ned"):
        raise ValueError("a position setpoint must be given in a WORLD frame, "
                         "got %r -- a body-frame position is a displacement, "
                         "and turning it into a setpoint needs the current "
                         "position." % action.frame)
    if yaw_sign not in (1, -1):
        raise ValueError("yaw_sign must be +1 or -1")

    rep = Clamped()
    p = to_world_enu(action, current_yaw_rad)
    z = _clamp_altitude(float(p[2]), limits, rep)
    yaw = 0.0 if action.yaw_rad is None else float(action.yaw_rad)
    return PositionCommand(float(p[0]), float(p[1]), z,
                           float(math.degrees(yaw_sign * yaw))), rep
