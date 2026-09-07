"""Racing gates: geometry, pass detection, and progress scoring.

WHAT A GATE IS HERE
-------------------
A rectangular aperture in a plane, with a direction you are required to fly
through. Not a waypoint sphere: a sphere cannot tell "went through" from "went
past", and cannot tell forwards from backwards, which are exactly the two
distinctions a race needs.

    centre    c   where the middle of the opening is
    normal    n   unit, the direction you must travel through it
    up        u   unit, in the plane, orthogonalised against n
    right     r   = n x u, completing a right-handed frame
    width     along r,  height along u

A point's position relative to the gate is then
    along  = (p - c) . n     signed distance through the plane
    across = (p - c) . r     horizontal offset in the opening
    lift   = (p - c) . u     vertical offset in the opening

CROSSINGS ARE DETECTED ON THE SEGMENT, NOT THE SAMPLE
-----------------------------------------------------
A policy running at 30 Hz at 3 m/s moves 10 cm per step, and a gate plane is
infinitely thin. Testing whether individual samples are "near" a gate misses
every fast pass. So the test is whether the SEGMENT between consecutive poses
crosses the plane -- the same reason `metric-splat`'s collision checker uses
swept segments rather than endpoints.

Crossing is defined as a state transition on `along >= 0`, not as a sign
product: `d0 * d1 < 0` silently misses the case where a sample lands exactly on
the plane, which is rare and infuriating.

MISSES ARE EVENTS TOO
---------------------
Crossing the plane OUTSIDE the aperture is reported as a MISS with the distance
by which it was missed, rather than as silence. "Flew past the gate 8 cm to the
left" and "never went near the gate" are different failures and should not look
identical in a log.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["Gate", "GateEvent", "GateCourse", "PASSED", "MISSED", "WRONG_WAY"]

PASSED = "passed"
MISSED = "missed"          # crossed the plane, outside the opening
WRONG_WAY = "wrong_way"    # crossed the opening against the required direction


def _unit(v, name: str) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.shape != (3,):
        raise ValueError("%s must have 3 components, got %s" % (name, v.shape))
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("%s has zero length" % name)
    return v / n


@dataclass(frozen=True)
class Gate:
    """One gate. All lengths in metres, in the scene frame.

    `up` need not be perpendicular to `normal`: it is orthogonalised, which is
    what you want when a human types an approximate "up" for a tilted gate. It
    must not be PARALLEL to the normal, because then the opening has no
    orientation and that is a specification error, not something to paper over.
    """
    name: str
    centre: np.ndarray
    normal: np.ndarray
    width_m: float
    height_m: float
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def __post_init__(self):
        c = np.asarray(self.centre, dtype=float).reshape(-1)
        if c.shape != (3,):
            raise ValueError("centre must have 3 components")
        n = _unit(self.normal, "normal")
        u = _unit(self.up, "up")
        if abs(float(np.dot(u, n))) > 1.0 - 1e-6:
            raise ValueError("up is parallel to normal for gate %r; the opening "
                             "has no orientation" % self.name)
        u = _unit(u - np.dot(u, n) * n, "up (orthogonalised)")
        if not (self.width_m > 0 and self.height_m > 0):
            raise ValueError("gate %r needs positive width and height" % self.name)
        object.__setattr__(self, "centre", c)
        object.__setattr__(self, "normal", n)
        object.__setattr__(self, "up", u)

    @property
    def right(self) -> np.ndarray:
        return np.cross(self.normal, self.up)

    def local(self, p) -> tuple:
        """(along, across, lift) for a point, in the gate's own frame."""
        d = np.asarray(p, dtype=float).reshape(3) - self.centre
        return (float(np.dot(d, self.normal)),
                float(np.dot(d, self.right)),
                float(np.dot(d, self.up)))

    def contains(self, across: float, lift: float) -> bool:
        return abs(across) <= self.width_m / 2.0 and abs(lift) <= self.height_m / 2.0

    def miss_distance(self, across: float, lift: float) -> float:
        """How far outside the opening, 0 if inside. Chebyshev-style per axis."""
        da = max(0.0, abs(across) - self.width_m / 2.0)
        dl = max(0.0, abs(lift) - self.height_m / 2.0)
        return float(math.hypot(da, dl))

    def to_dict(self) -> dict:
        return {"name": self.name, "centre": self.centre.tolist(),
                "normal": self.normal.tolist(), "up": self.up.tolist(),
                "width_m": self.width_m, "height_m": self.height_m}

    @classmethod
    def from_dict(cls, d: dict) -> "Gate":
        return cls(d["name"], d["centre"], d["normal"],
                   float(d["width_m"]), float(d["height_m"]),
                   d.get("up", [0.0, 0.0, 1.0]))


@dataclass(frozen=True)
class GateEvent:
    kind: str
    index: int
    name: str
    point: np.ndarray          # where the segment met the plane
    t: float                   # fraction along the segment, 0..1
    miss_m: float = 0.0        # >0 only for MISSED

    def __str__(self) -> str:
        s = "%s gate %d (%s)" % (self.kind.upper(), self.index, self.name)
        if self.kind == MISSED:
            s += " by %.0f cm" % (self.miss_m * 100.0)
        return s


class GateCourse:
    """An ordered course. Only the next gate is armed.

    Gate k+1 cannot be passed while gate k is pending: that is what makes it a
    course rather than a set. A drone that flies through a later gate first gets
    no credit and no event, which shows up as a stalled `passed` count -- the
    honest signal that it is off-course.
    """

    def __init__(self, gates: Sequence[Gate]):
        if not gates:
            raise ValueError("a course needs at least one gate")
        self.gates: List[Gate] = list(gates)
        self.reset()

    def reset(self) -> None:
        self.next_index = 0
        self.events: List[GateEvent] = []

    @property
    def complete(self) -> bool:
        return self.next_index >= len(self.gates)

    @property
    def passed(self) -> int:
        return self.next_index

    @property
    def next_gate(self) -> Optional[Gate]:
        return None if self.complete else self.gates[self.next_index]

    def distance_to_next(self, p) -> Optional[float]:
        g = self.next_gate
        if g is None:
            return None
        return float(np.linalg.norm(np.asarray(p, dtype=float).reshape(3) - g.centre))

    def update(self, p_prev, p_now) -> List[GateEvent]:
        """Advance the course over one segment. Returns the events it produced.

        At most one event per call: a segment long enough to cross two gate
        planes is a symptom worth seeing in the logs as a stalled count, not
        something to quietly credit.
        """
        if self.complete:
            return []
        g = self.gates[self.next_index]
        p0 = np.asarray(p_prev, dtype=float).reshape(3)
        p1 = np.asarray(p_now, dtype=float).reshape(3)
        d0 = float(np.dot(p0 - g.centre, g.normal))
        d1 = float(np.dot(p1 - g.centre, g.normal))

        # State transition on "at or past the plane", so a sample landing
        # exactly on it cannot be lost.
        forward = d0 < 0.0 <= d1
        backward = d1 < 0.0 <= d0
        if not (forward or backward):
            return []

        t = d0 / (d0 - d1) if d0 != d1 else 0.0
        hit = p0 + t * (p1 - p0)
        _, across, lift = g.local(hit)

        if not g.contains(across, lift):
            ev = GateEvent(MISSED, self.next_index, g.name, hit, float(t),
                           g.miss_distance(across, lift))
        elif backward:
            ev = GateEvent(WRONG_WAY, self.next_index, g.name, hit, float(t))
        else:
            ev = GateEvent(PASSED, self.next_index, g.name, hit, float(t))
            self.next_index += 1

        self.events.append(ev)
        return [ev]

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {"_comment": "gate centres and sizes in METRES, scene frame",
                "gates": [g.to_dict() for g in self.gates]}

    def save(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")

    @classmethod
    def load(cls, path) -> "GateCourse":
        with open(path) as fh:
            d = json.load(fh)
        return cls([Gate.from_dict(g) for g in d["gates"]])

    def summary(self) -> str:
        n = len(self.gates)
        misses = sum(1 for e in self.events if e.kind == MISSED)
        wrong = sum(1 for e in self.events if e.kind == WRONG_WAY)
        L = ["  gates passed : %d / %d%s" % (self.passed, n,
                                             "   COURSE COMPLETE" if self.complete else "")]
        if misses:
            worst = max(e.miss_m for e in self.events if e.kind == MISSED)
            L.append("  plane misses : %d (worst %.0f cm outside the opening)"
                     % (misses, worst * 100.0))
        if wrong:
            L.append("  wrong way    : %d" % wrong)
        if not self.complete:
            L.append("  stopped at   : gate %d (%s)"
                     % (self.next_index, self.gates[self.next_index].name))
        return "\n".join(L)
