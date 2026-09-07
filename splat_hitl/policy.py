"""The slot a student's work goes into.

An RL gate-racer and a learned optical-flow module are the same shape from the
runtime's point of view: something that turns an observation plus a little state
into an `Action`. That is the entire interface, and keeping it that small is
what lets one runtime serve every assignment.

    class MyPolicy(Policy):
        def act(self, obs, state) -> Action:
            ...

THE FINGERPRINT IS NOT OPTIONAL DECORATION
------------------------------------------
`sensor_fingerprint` records the sensor model the policy was trained through.
The runtime refuses to fly a policy whose fingerprint does not match the
renderer's, because that failure is otherwise indistinguishable from "the policy
is bad" -- both present as flying into things -- and the check costs one string
comparison against a week of debugging the wrong layer.

Set it to None only for a policy that genuinely does not look at images.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .commands import Action
from .renderer import Observation

__all__ = ["PolicyState", "Policy", "HoverPolicy", "ForwardPolicy",
           "GateSeekPolicy"]


@dataclass(frozen=True)
class PolicyState:
    """Everything the policy is told besides the image.

    Deliberately small. A policy that needs the drone's absolute position to
    work has not learned to fly, it has learned the room.
    """
    t_s: float
    position_m: np.ndarray
    velocity_ms: np.ndarray
    yaw_rad: float
    gate_index: int = 0
    gate_distance_m: Optional[float] = None
    gate_bearing_rad: Optional[float] = None      # relative to current heading


class Policy(ABC):
    """Base class for anything that flies the drone."""

    #: fingerprint of the SensorModel this policy was trained through
    sensor_fingerprint: Optional[str] = None

    #: human-readable, goes into the run log
    name: str = "unnamed"

    @abstractmethod
    def act(self, obs: Observation, state: PolicyState) -> Action:
        """Return the next Action. Must not block for long: the runtime's
        budget is a fraction of the driver's 300 ms command timeout."""

    def reset(self) -> None:
        """Called once before a run. Clear any recurrent state here."""


# ------------------------------------------------------------- references
class HoverPolicy(Policy):
    """Does nothing, correctly. The control case for every experiment."""
    name = "hover"

    def act(self, obs: Observation, state: PolicyState) -> Action:
        return Action("velocity", "body_flu", np.zeros(3))


class ForwardPolicy(Policy):
    """Constant forward speed, no steering. The floor a policy must beat.

    It will fly into the first wall, which makes it a good smoke test for the
    collision monitor and a good demonstration of why depth matters.
    """
    name = "forward"

    def __init__(self, speed_ms: float = 0.5):
        self.speed_ms = float(speed_ms)

    def act(self, obs: Observation, state: PolicyState) -> Action:
        return Action("velocity", "body_flu", [self.speed_ms, 0.0, 0.0])


class GateSeekPolicy(Policy):
    """Steers toward the next gate and slows for close obstacles.

    Not a good racer -- it has no lookahead and will happily be trapped by
    geometry between it and the gate. It exists so the runtime can be tested
    end to end against something that actually completes a course, and as the
    baseline a student's policy has to beat.
    """
    name = "gate_seek"

    def __init__(self, speed_ms: float = 0.6, yaw_gain: float = 1.5,
                 brake_distance_m: float = 1.0):
        self.speed_ms = float(speed_ms)
        self.yaw_gain = float(yaw_gain)
        self.brake_distance_m = float(brake_distance_m)

    def act(self, obs: Observation, state: PolicyState) -> Action:
        bearing = state.gate_bearing_rad
        if bearing is None:
            return Action("velocity", "body_flu", np.zeros(3))

        # Slow down when the view ahead is short. The centre patch, not the
        # whole frame: the edges of a wide lens see walls that are not in the
        # way.
        h, w = obs.depth_m.shape[:2]
        patch = obs.depth_m[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
        ahead = float(np.percentile(patch, 10)) if patch.size else self.brake_distance_m
        throttle = max(0.0, min(1.0, ahead / self.brake_distance_m))

        # Turn toward the gate; go forward only when roughly facing it.
        facing = max(0.0, math.cos(bearing))
        return Action("velocity", "body_flu",
                      [self.speed_ms * throttle * facing, 0.0, 0.0],
                      yaw_rate_rad_s=self.yaw_gain * bearing)
