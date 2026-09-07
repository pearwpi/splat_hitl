"""Solve the Vicon -> splat similarity transform from measured correspondences.

WHY THIS IS THE RISKIEST STEP IN THE WHOLE PIPELINE
---------------------------------------------------
Everything else fails loudly. This fails silently: a 2 degree rotation error or
a 10 cm offset produces a system where every component reports healthy, the
render looks beautiful, and the policy is confidently wrong about where the
walls are. There is no runtime symptom -- the drone simply flies into things
for reasons that look like bad control.

So this module does three things beyond solving:

  1. Reports residuals, per point and RMS, in METRES. A solve without a
     residual is not an answer.
  2. Refuses degenerate geometry. Points along a line leave rotation about that
     line unconstrained; points on a plane leave reflection ambiguous. Both
     produce a confident, wrong transform -- exactly the failure mode that the
     marker-layout work in FINDINGS 1.7 hit from the other direction.
  3. Never returns a reflection. Umeyama's determinant correction is applied,
     and SplatTransform re-checks it.

HOW TO COLLECT THE DATA
-----------------------
You need N >= 4 points whose position you know in BOTH frames:

  Vicon side   carry the drone to the point, read the mocap position.
  Splat side   click the same physical feature in the cleaned point cloud
               (`scene_tools.py` already has pickers that return normalised
               coordinates).

Pick points that FILL the volume in all three axes. Four corners of the floor
is the classic mistake -- it is a plane, and `solve()` will refuse it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .frames import SplatTransform

__all__ = ["RegistrationResult", "solve", "check_geometry"]

# Degeneracy: a point set thinner than this in any direction cannot constrain
# rotation about that direction, as a fraction of the largest extent. Below
# this we REFUSE, because the solve would be confidently wrong.
MIN_EXTENT_RATIO = 0.08
# ...and an absolute floor in metres (these are per-axis standard deviations,
# not bounding-box sizes). A 10 cm sd is roughly a 35 cm spread.
MIN_EXTENT_M = 0.10

# Marginal: solvable, but the transform will extrapolate poorly outside the
# calibrated region. We WARN rather than refuse, because sometimes a cramped
# volume is all you have.
WARN_EXTENT_RATIO = 0.20
WARN_EXTENT_M = 0.30


@dataclass
class RegistrationResult:
    transform: Optional[SplatTransform]
    rms_m: float
    max_m: float
    per_point_m: np.ndarray
    condition: str                 # "ok" | "colinear" | "coplanar" | "too_few"
    detail: str

    @property
    def ok(self) -> bool:
        return self.transform is not None

    def report(self) -> str:
        L = []
        if not self.ok:
            L.append("  REGISTRATION REFUSED: %s" % self.detail)
            return "\n".join(L)
        L.append("  %r" % self.transform)
        L.append("  residual: rms %.1f mm, worst %.1f mm, over %d points"
                 % (self.rms_m * 1000.0, self.max_m * 1000.0, len(self.per_point_m)))
        worst = int(np.argmax(self.per_point_m))
        L.append("  worst point is #%d at %.1f mm" % (worst, self.per_point_m[worst] * 1000.0))
        if self.condition != "ok":
            L.append("  WARNING: %s" % self.detail)
        return "\n".join(L)


def check_geometry(points_m: np.ndarray) -> tuple:
    """Is this point set rich enough to pin down a rotation?

    Returns (condition, detail). Uses the singular values of the centred set:
    a near-zero third value means the points are coplanar, a near-zero second
    means colinear.
    """
    P = np.asarray(points_m, dtype=float).reshape(-1, 3)
    n = len(P)
    if n < 4:
        return "too_few", ("need at least 4 correspondences, got %d "
                           "(3 points fix a plane but leave the solution "
                           "fragile; 4 non-coplanar points are the minimum "
                           "that constrains all three axes)" % n)
    Q = P - P.mean(axis=0)
    sv = np.linalg.svd(Q, compute_uv=False) / max(1.0, np.sqrt(n))
    largest = float(sv[0])
    if largest <= 0:
        return "colinear", "all points coincide"
    ratio2, ratio3 = float(sv[1] / largest), float(sv[2] / largest)
    spread = "spread %.0f x %.0f x %.0f mm (per-axis sd)" % (
        sv[0] * 1000, sv[1] * 1000, sv[2] * 1000)
    if ratio2 < MIN_EXTENT_RATIO or sv[1] < MIN_EXTENT_M:
        return "colinear", ("points are nearly colinear, %s. Rotation about "
                            "that line is unconstrained and the solve would be "
                            "confidently wrong." % spread)
    if ratio3 < MIN_EXTENT_RATIO or sv[2] < MIN_EXTENT_M:
        return "coplanar", ("points are nearly coplanar, %s. Add points at "
                            "different HEIGHTS -- four corners of the floor is "
                            "the classic mistake." % spread)
    if ratio3 < WARN_EXTENT_RATIO or sv[2] < WARN_EXTENT_M:
        return "marginal", ("point set is thin in one direction, %s. The fit "
                            "will extrapolate poorly outside the calibrated "
                            "region; spread the points further apart if you "
                            "can." % spread)
    return "ok", ""


def solve(vicon_m: Sequence, splat_units: Sequence,
          allow_degenerate: bool = False) -> RegistrationResult:
    """Umeyama similarity fit: vicon metres -> splat normalised units.

    `allow_degenerate` exists for tests and for the deliberate case where you
    genuinely only have planar correspondences and accept the risk. It is not a
    default and it should not become one.
    """
    X = np.asarray(vicon_m, dtype=float).reshape(-1, 3)      # source
    Y = np.asarray(splat_units, dtype=float).reshape(-1, 3)  # target
    if X.shape != Y.shape:
        raise ValueError("correspondence count mismatch: %d vicon vs %d splat"
                         % (len(X), len(Y)))

    condition, detail = check_geometry(X)
    if condition in ("too_few", "colinear", "coplanar") and not allow_degenerate:
        return RegistrationResult(None, float("nan"), float("nan"),
                                  np.zeros(0), condition, detail)

    n = len(X)
    mu_x, mu_y = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - mu_x, Y - mu_y
    Sigma = (Yc.T @ Xc) / n
    U, D, Vt = np.linalg.svd(Sigma)

    # Umeyama's determinant correction. Without it a noisy fit can return a
    # REFLECTION, which renders a mirrored world that looks entirely normal.
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    var_x = float((Xc ** 2).sum()) / n
    if var_x <= 0:
        return RegistrationResult(None, float("nan"), float("nan"), np.zeros(0),
                                  "colinear", "source points have zero variance")
    scale = float(np.trace(np.diag(D) @ S) / var_x)
    t = mu_y - scale * (R @ mu_x)

    pred = (scale * (R @ X.T)).T + t
    err_units = np.linalg.norm(pred - Y, axis=1)
    metres_per_unit = 1.0 / scale
    err_m = err_units * metres_per_unit

    tf = SplatTransform(R, t, scale,
                        rms_residual_units=float(np.sqrt((err_units ** 2).mean())),
                        n_points=n)
    return RegistrationResult(tf, float(np.sqrt((err_m ** 2).mean())),
                              float(err_m.max()), err_m, condition,
                              detail or "")
