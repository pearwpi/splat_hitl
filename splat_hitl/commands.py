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
    "to_world_enu",
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
