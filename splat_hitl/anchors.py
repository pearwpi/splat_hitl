"""Room anchors, and the drift of the Vicon frame they measure.

A bundle's `vicon_transform` maps Vicon metres into the splat. It is fixed at
registration time, and it is only true while the Vicon frame it was fitted in
still exists. That frame is not as permanent as it looks: on 22 Sep 2026 the six
anchors of the PEAR room -- A4 sheets taped to the floor, untouched, with one
marker each -- read up to 64 mm differently from two days earlier, in a pattern
that is exactly one rigid 0.662 deg rotation. Nobody had recalibrated. Six
separate pieces of tape cannot move as a rigid body, so what moved was the
frame, and a bundle flown against the old transform would have put the drone
50 mm out at the far gate, a quarter of its clearance, with nothing printing a
warning.

So a bundle records the anchor positions it was registered against, a flight
session records the anchors again in ten seconds, and `drift()` compares them.
The result is either a few millimetres -- nothing to do -- or a rigid correction
that composes onto the stored transform, or a residual too large to be a frame
change at all, which means an anchor really has been moved and the registration
has to be redone.

numpy only.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from .frames import SplatTransform

__all__ = ["AnchorSet", "AnchorDrift", "drift"]

# A frame change is rigid. Anything left over after fitting a rotation and a
# translation is an anchor that has physically moved, or a mis-measured one.
RESIDUAL_LIMIT_M = 0.010


@dataclass(frozen=True)
class AnchorSet:
    """Named points that do not move, in Vicon metres.

    `positions[name]` is the marker centre as Vicon reported it. `recorded` and
    `source` say when and from what, because the whole point of the file is to
    be compared against a later reading of the same physical points.
    """

    positions: Dict[str, np.ndarray]
    recorded: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self):
        if len(self.positions) < 3:
            raise ValueError("an anchor set needs at least 3 points, got %d"
                             % len(self.positions))
        clean = {}
        for k, v in self.positions.items():
            p = np.asarray(v, dtype=float).reshape(-1)
            if p.shape != (3,):
                raise ValueError("anchor %r is not a 3-vector: %r" % (k, v))
            if not np.all(np.isfinite(p)):
                raise ValueError("anchor %r is not finite: %r" % (k, v))
            clean[str(k)] = p
        object.__setattr__(self, "positions", clean)
        P = self.as_array()
        if np.linalg.matrix_rank(P - P.mean(0), tol=1e-6) < 2:
            raise ValueError("the anchors are collinear; they cannot fix a "
                             "rotation about the line they lie on")

    @property
    def names(self):
        return sorted(self.positions)

    def as_array(self, names: Optional[Sequence[str]] = None) -> np.ndarray:
        return np.array([self.positions[n] for n in (names or self.names)], dtype=float)

    # -- io ----------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Mapping) -> "AnchorSet":
        pos = d.get("anchors", d.get("positions"))
        if not isinstance(pos, Mapping):
            raise ValueError("no 'anchors' mapping in the anchor file")
        return cls({k: v for k, v in pos.items()},
                   recorded=d.get("recorded"), source=d.get("source"),
                   notes=d.get("notes") or d.get("_comment"))

    @classmethod
    def load(cls, path) -> "AnchorSet":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict:
        out = {"_comment": self.notes or (
            "Room anchors in Vicon metres, as read when this scene was "
            "registered. Record them again at the start of a flight session "
            "and compare with splat_hitl.anchors.drift: the Vicon frame moves."),
            "anchors": {k: [float(x) for x in v] for k, v in sorted(self.positions.items())}}
        if self.recorded:
            out["recorded"] = self.recorded
        if self.source:
            out["source"] = self.source
        return out

    def save(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")

    def __repr__(self):
        return "AnchorSet(%d points: %s)" % (len(self.positions), ", ".join(self.names))


@dataclass(frozen=True)
class AnchorDrift:
    """How far the Vicon frame has moved since a bundle was registered.

    `R` and `t` take a point as the CURRENT frame reports it to what the frame
    at registration time would have called it, so that a transform fitted then
    can be used now: `p_then = R @ p_now + t`.
    """

    R: np.ndarray
    t: np.ndarray
    angle_deg: float
    translation_m: float
    per_anchor_m: Dict[str, float]
    residual_m: float
    scale: float
    names: Sequence[str] = field(default_factory=tuple)

    @property
    def rigid(self) -> bool:
        """True when a rotation and a translation explain the whole change.

        False means at least one anchor moved relative to the others, which no
        correction can repair: the registration has to be redone.
        """
        return self.residual_m <= RESIDUAL_LIMIT_M

    @property
    def moved_m(self) -> float:
        """The largest distance any anchor appears to have moved."""
        return max(self.per_anchor_m.values()) if self.per_anchor_m else 0.0

    def displacement_at(self, points) -> np.ndarray:
        """How far the correction moves each of `points`, in metres.

        A rotation shows up as nothing at the pivot and centimetres at the far
        end of the room, so ask about the places you care about -- the gates,
        the goal -- not about the origin.
        """
        P = np.asarray(points, dtype=float).reshape(-1, 3)
        return np.linalg.norm((P @ self.R.T + self.t) - P, axis=1)

    def apply(self, tf: SplatTransform) -> SplatTransform:
        """`tf` re-expressed so it takes CURRENT-frame Vicon metres.

        p_splat = scale*(R_tf @ (R @ p_now + t)) + t_tf, folded back into a
        single similarity. Refuses when the drift is not rigid, because then the
        thing it would be correcting for is not a frame change.
        """
        if not self.rigid:
            raise ValueError(
                "anchor residual %.1f mm after a rigid fit, over the %.1f mm "
                "limit: at least one anchor has moved relative to the others, "
                "so no frame correction can be right. Re-register the scene."
                % (1000 * self.residual_m, 1000 * RESIDUAL_LIMIT_M))
        return SplatTransform(tf.R @ self.R,
                              tf.scale * (tf.R @ self.t) + tf.t,
                              tf.scale,
                              rms_residual_units=tf.rms_residual_units,
                              n_points=tf.n_points)

    def __str__(self):
        head = ("anchor drift: %.3f deg, %.1f mm translation, worst anchor "
                "%.1f mm, residual %.1f mm over %d anchors"
                % (self.angle_deg, 1000 * self.translation_m,
                   1000 * self.moved_m, 1000 * self.residual_m, len(self.per_anchor_m)))
        if not self.rigid:
            head += "  -- NOT a rigid frame change; re-register"
        return head


def drift(reference: AnchorSet, measured: AnchorSet) -> AnchorDrift:
    """Compare a fresh anchor reading against the one a bundle was built on.

    Fits a rotation and a translation only. Scale is measured and reported but
    never applied: a volume whose scale has changed is a volume to recalibrate,
    not one to fudge, and folding a scale error into the correction would hide
    exactly the thing worth seeing.
    """
    names = [n for n in reference.names if n in measured.positions]
    if len(names) < 3:
        raise ValueError("only %d anchors in common (%s); need at least 3"
                         % (len(names), ", ".join(names) or "none"))
    A = measured.as_array(names)          # now
    B = reference.as_array(names)         # then
    ca, cb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    resid = (A @ R.T + t) - B
    scale = float((S * np.diag(D)).sum() / max(((A - ca) ** 2).sum(), 1e-12))
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))))
    return AnchorDrift(
        R=R, t=t, angle_deg=ang, translation_m=float(np.linalg.norm(t)),
        per_anchor_m={n: float(np.linalg.norm(A[i] - B[i])) for i, n in enumerate(names)},
        residual_m=float(np.sqrt((resid ** 2).sum(1).mean())),
        scale=scale, names=tuple(names))
