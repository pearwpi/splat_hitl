"""Virtual collision: did the drone hit a wall that only exists in the splat?

WHAT THIS IS AND IS NOT
-----------------------
The drone flies in an empty net. The scene is rendered. So there is nothing to
crash into -- which is exactly the problem: without this, "the trajectory looked
reasonable" is the only available verdict, and a course needs pass/fail.

This is the SCORING signal. It is not the safety layer:

    cf_core geofence      protects the real room. Lands or cuts motors.
    CollisionMonitor      scores the simulation. Ends the run.

They are different mechanisms with different consequences and you need both. A
policy that flies through a virtual wall is a failed run, not an emergency; a
drone leaving the physical volume is an emergency regardless of what the splat
says.

THREE THINGS THE ESDF WILL NOT TELL YOU, AND ONE IT LIES ABOUT
---------------------------------------------------------------
1. It is UNSIGNED. Inside an obstacle reads the same as just outside one.
2. It is TRUNCATED. A value at the cap means "at least this far", not "exactly
   this far" -- so comparing it against a threshold is only meaningful when the
   threshold is below the truncation. `Clearance.truncated` says when you are
   in that regime.
3. Outside the mapped volume there is NO answer. Unknown is not safe, and
   `Clearance.outside` is not something to treat as clear. A policy that leaves
   the mapped region has left the experiment.
4. Nearest-voxel lookup with a voxel COARSER than your clearance threshold
   gives confident nonsense. The constructor refuses that combination.

Everything here takes and returns METRES. The grid is stored in normalised
scene units, and the conversion happens once, on the way in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Clearance", "ESDF", "CollisionMonitor", "CollisionEvent",
           "synthetic_room"]


@dataclass(frozen=True)
class Clearance:
    """Distance to the nearest geometry, with its caveats attached."""
    metres: float
    truncated: bool = False        # value is at the cap: read as ">= metres"
    outside: bool = False          # query was outside the mapped volume

    def clear_of(self, threshold_m: float) -> bool:
        """True only if the point is KNOWN to be at least threshold_m away.

        Outside the map is not clear. That is the whole point of the flag.
        """
        if self.outside:
            return False
        return self.metres >= threshold_m

    def __str__(self) -> str:
        if self.outside:
            return "outside the mapped volume"
        return "%s%.3f m" % (">=" if self.truncated else "", self.metres)


#: Half the Crazyflie 2.1+ motor-to-motor diagonal (~65 mm) plus a little for
#: the propeller arc. `clearance_m` is measured from the drone's CENTRE, so a
#: clearance below this passes the check while the airframe is already inside
#: the surface. It is a sanity floor for judging an ESDF, not a limit anything
#: enforces -- a smaller margin is a legitimate choice you should make on
#: purpose.
AIRFRAME_RADIUS_M = 0.075


class ESDF:
    """Euclidean distance field over a scene, queried in metres."""

    def __init__(self, grid: np.ndarray, voxel_size: float, origin: Sequence[float],
                 truncation: float, metres_per_unit: float = 1.0):
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 3:
            raise ValueError("ESDF grid must be 3-D, got %d dims" % grid.ndim)
        if not voxel_size > 0:
            raise ValueError("voxel_size must be positive")
        if not truncation > 0:
            raise ValueError("truncation must be positive")
        if not metres_per_unit > 0:
            raise ValueError("metres_per_unit must be positive")
        self.grid = grid
        self.voxel_size = float(voxel_size)
        self.origin = np.asarray(origin, dtype=float).reshape(3)
        self.truncation = float(truncation)
        self.metres_per_unit = float(metres_per_unit)

    # -- derived, in metres ------------------------------------------------
    @property
    def voxel_size_m(self) -> float:
        return self.voxel_size * self.metres_per_unit

    @property
    def truncation_m(self) -> float:
        return self.truncation * self.metres_per_unit

    @property
    def max_clearance_m(self) -> float:
        """The largest clearance this field can both resolve and discriminate.

        A usable clearance must sit at or above one voxel (nearest-voxel lookup
        cannot resolve anything finer) and strictly below the truncation (at or
        past it, every free voxel reads as the cap and nothing ever fails). So
        the usable range is [voxel_size_m, truncation_m), and the largest
        practical value is one voxel below the cap.

        Returns 0.0 when the field admits no usable clearance at all, which
        happens when the truncation is not at least two voxels.
        """
        top = self.truncation_m - self.voxel_size_m
        return float(top) if top >= self.voxel_size_m else 0.0

    @property
    def bounds_m(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = self.origin * self.metres_per_unit
        hi = (self.origin + np.array(self.grid.shape) * self.voxel_size) * self.metres_per_unit
        return lo, hi

    # -- queries -----------------------------------------------------------
    def at(self, p_m) -> Clearance:
        """Clearance at a point given in metres."""
        p = np.asarray(p_m, dtype=float).reshape(3) / self.metres_per_unit
        idx = np.floor((p - self.origin) / self.voxel_size).astype(int)
        if np.any(idx < 0) or np.any(idx >= np.array(self.grid.shape)):
            return Clearance(float("nan"), outside=True)
        d = float(self.grid[idx[0], idx[1], idx[2]])
        return Clearance(d * self.metres_per_unit,
                         truncated=d >= self.truncation - 1e-12)

    def min_along(self, a_m, b_m, step_m: Optional[float] = None) -> Clearance:
        """Worst clearance along the SEGMENT a->b.

        Endpoint checks alone let a fast policy pass through a wall between
        samples. Sampling is at half a voxel by default, which is the finest
        spacing a nearest-voxel lookup can actually distinguish.
        """
        a = np.asarray(a_m, dtype=float).reshape(3)
        b = np.asarray(b_m, dtype=float).reshape(3)
        if step_m is None:
            step_m = 0.5 * self.voxel_size_m
        if not step_m > 0:
            raise ValueError("step_m must be positive")
        length = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(length / step_m)))
        worst = None
        for i in range(n + 1):
            c = self.at(a + (b - a) * (i / float(n)))
            if c.outside:
                return c                       # unknown dominates; stop here
            if worst is None or c.metres < worst.metres:
                worst = c
        return worst

    def __repr__(self) -> str:
        lo, hi = self.bounds_m
        return ("ESDF(%s voxels, %.0f mm each, volume %.2f x %.2f x %.2f m, "
                "truncated at %.2f m)"
                % ("x".join(str(s) for s in self.grid.shape),
                   self.voxel_size_m * 1000, *(hi - lo), self.truncation_m))

    # -- loading -----------------------------------------------------------
    @classmethod
    def load_npy(cls, path, metres_per_unit: Optional[float] = None) -> "ESDF":
        """Load the pickled-dict .npy that `scene_geometry.py esdf` writes.

        Keys: esdf, voxel_size, origin, truncation (+ scale_to_metres when the
        RAW-metre flags were used). NOTE this uses allow_pickle, so only load
        files you produced or were given by the course.
        """
        d = np.load(str(path), allow_pickle=True)
        d = d.item() if hasattr(d, "item") and d.dtype == object else d
        if not isinstance(d, dict):
            raise ValueError("%s does not contain an ESDF dict" % path)
        mpu = metres_per_unit
        if mpu is None:
            mpu = float(d.get("scale_to_metres", 1.0))
        return cls(d["esdf"], float(d["voxel_size"]), d["origin"],
                   float(d["truncation"]), mpu)


@dataclass(frozen=True)
class CollisionEvent:
    kind: str                    # "collision" | "outside"
    at_m: np.ndarray
    clearance_m: float
    step: int

    def __str__(self) -> str:
        if self.kind == "outside":
            return ("LEFT THE MAPPED VOLUME at step %d, (%.2f, %.2f, %.2f) m"
                    % (self.step, *self.at_m))
        return ("VIRTUAL COLLISION at step %d, clearance %.0f mm, "
                "(%.2f, %.2f, %.2f) m"
                % (self.step, self.clearance_m * 1000.0, *self.at_m))


class CollisionMonitor:
    """Latching monitor over a flight. One verdict per run."""

    def __init__(self, esdf: ESDF, clearance_m: float = 0.10,
                 outside_is_failure: bool = True):
        if not clearance_m > 0:
            raise ValueError("clearance_m must be positive")
        if clearance_m < esdf.voxel_size_m:
            raise ValueError(
                "clearance %.0f mm is finer than the ESDF voxel (%.0f mm). "
                "Nearest-voxel lookup cannot resolve it, so the check would "
                "return confident nonsense.\n"
                "This field supports %.0f to %.0f mm; below that, rebuild it "
                "with a smaller --voxel-size-m."
                % (clearance_m * 1000, esdf.voxel_size_m * 1000,
                   esdf.voxel_size_m * 1000, esdf.max_clearance_m * 1000))
        if clearance_m >= esdf.truncation_m:
            raise ValueError(
                "clearance %.3f m is at or beyond the truncation (%.3f m). "
                "Every free voxel reads as the cap, so nothing would ever "
                "fail.\n"
                "The largest clearance this field can express is %.3f m. For a "
                "bigger margin, rebuild the ESDF with a larger truncation -- "
                "0.30 to 0.50 m leaves room to choose."
                % (clearance_m, esdf.truncation_m, esdf.max_clearance_m))
        self.esdf = esdf
        self.clearance_m = float(clearance_m)
        self.outside_is_failure = bool(outside_is_failure)
        self.warnings: List[str] = []
        if clearance_m < 2.0 * esdf.voxel_size_m:
            self.warnings.append(
                "clearance %.0f mm is less than two voxels (%.0f mm); "
                "discretisation error is a large fraction of the margin"
                % (clearance_m * 1000, esdf.voxel_size_m * 1000))
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.event: Optional[CollisionEvent] = None
        self.closest_m = float("inf")
        self.outside_steps = 0

    @property
    def failed(self) -> bool:
        return self.event is not None

    def update(self, p_prev_m, p_now_m) -> Optional[CollisionEvent]:
        """Advance one segment. Latches: once failed, stays failed."""
        self.step += 1
        if self.failed:
            return None
        c = self.esdf.min_along(p_prev_m, p_now_m)
        if c.outside:
            self.outside_steps += 1
            if self.outside_is_failure:
                self.event = CollisionEvent(
                    "outside", np.asarray(p_now_m, dtype=float).reshape(3),
                    float("nan"), self.step)
                return self.event
            return None
        self.closest_m = min(self.closest_m, c.metres)
        if c.metres < self.clearance_m:
            self.event = CollisionEvent(
                "collision", np.asarray(p_now_m, dtype=float).reshape(3),
                c.metres, self.step)
            return self.event
        return None

    def summary(self) -> str:
        L = []
        for w in self.warnings:
            L.append("  WARNING: %s" % w)
        if self.failed:
            L.append("  %s" % self.event)
        else:
            L.append("  no virtual collision in %d steps" % self.step)
        if math.isfinite(self.closest_m):
            L.append("  closest approach: %.0f mm (threshold %.0f mm)"
                     % (self.closest_m * 1000, self.clearance_m * 1000))
        if self.outside_steps and not self.outside_is_failure:
            L.append("  %d step(s) outside the mapped volume, not counted as failure"
                     % self.outside_steps)
        return "\n".join(L)


# --------------------------------------------------------------------- helper
def synthetic_room(size_m=(4.0, 3.0, 2.5), voxel_m: float = 0.05,
                   truncation_m: float = 1.0,
                   obstacles: Sequence[dict] = ()) -> ESDF:
    """An exact ESDF for a rectangular room, for tests and for developing
    without a scene.

    Distances are computed analytically rather than from a point cloud, so the
    answers are exact and a test can assert against them. `obstacles` takes
    dicts of {"centre": (x,y,z), "radius": r} for spheres.

    The room's walls, floor and ceiling are the geometry; the interior is free.
    """
    size = np.asarray(size_m, dtype=float).reshape(3)
    dims = np.maximum(1, np.ceil(size / voxel_m).astype(int))
    # voxel centres
    ax = [(np.arange(dims[i]) + 0.5) * voxel_m for i in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    P = np.stack([X, Y, Z], axis=-1)
    # distance to the nearest wall of the box [0, size]
    to_wall = np.minimum(P, size - P).min(axis=-1)
    d = np.clip(to_wall, 0.0, truncation_m)
    for ob in obstacles:
        c = np.asarray(ob["centre"], dtype=float).reshape(3)
        r = float(ob["radius"])
        to_ob = np.linalg.norm(P - c, axis=-1) - r
        d = np.minimum(d, np.clip(to_ob, 0.0, truncation_m))
    return ESDF(d, voxel_m, np.zeros(3), truncation_m, metres_per_unit=1.0)
