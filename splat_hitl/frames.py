"""Frame conversions between the Vicon world and a Gaussian-splat scene.

THE PROBLEM THIS SOLVES
-----------------------
Three coordinate conventions meet in a HITL flight and none of them agree:

  Vicon world   metres, ENU (x east, y north, z up), orientation as a
                quaternion (x, y, z, w).
  Splat scene   NORMALISED units. Nerfstudio rescales during training and
                records the factor in dataparser_transforms.json; dividing by
                it recovers metres. Its axes are wherever the capture happened
                to put them -- there is no convention to rely on.
  Renderer      wants a camera pose in the splat frame.

A registration error here is SILENT. Every component reports healthy, the
renderer produces a beautiful image, and the policy is simply wrong about where
the walls are. That is why `SplatTransform` refuses to be constructed from
anything it cannot check, and why registration.py reports residuals rather than
just returning an answer.

CONVENTION, stated once
-----------------------
`SplatTransform` maps a point in VICON METRES to SPLAT NORMALISED units:

    p_splat = scale * (R @ p_vicon) + t

`scale` therefore carries both the metres->normalised factor and any residual
scale error in the registration. Use `metres_per_unit` to go back the other way
when you need to talk to the ESDF in raw metres.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "quat_to_matrix", "matrix_to_quat", "matrix_to_rpy", "rpy_to_matrix",
    "enu_to_ned", "ned_to_enu", "SplatTransform",
]


# --------------------------------------------------------------------- rotations
def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Rotation matrix from a quaternion in ROS order (x, y, z, w).

    ROS and Vicon both put w LAST. Several drone libraries put it first, and
    that mistake produces a rotation that looks plausible and is wrong, so the
    order is in the signature rather than in a comment.
    """
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=float)


def matrix_to_quat(R: np.ndarray) -> tuple:
    """Quaternion (x, y, z, w) from a rotation matrix. Shepperd's method."""
    R = np.asarray(R, dtype=float)
    t = float(np.trace(R))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / n, y / n, z / n, w / n)


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """Intrinsic Z-Y-X (yaw, pitch, roll) Euler angles, returned as (r, p, y).

    Gimbal lock at pitch = +-90 deg is handled by putting all the rotation into
    roll and zeroing yaw, which is the standard degenerate choice. It is a
    choice, not a fact: near-vertical camera mounts will see yaw jump.
    """
    R = np.asarray(R, dtype=float)
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    if abs(sy) > 1.0 - 1e-9:                       # gimbal lock
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw], dtype=float)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Inverse of matrix_to_rpy: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


# ----------------------------------------------------------------- ENU <-> NED
def enu_to_ned(v: Sequence[float]) -> np.ndarray:
    """ENU (east, north, up) -> NED (north, east, down). Its own inverse."""
    v = np.asarray(v, dtype=float)
    return np.array([v[1], v[0], -v[2]], dtype=float)


ned_to_enu = enu_to_ned          # the mapping is an involution


# ------------------------------------------------------------- the transform
@dataclass(frozen=True)
class SplatTransform:
    """Similarity transform: Vicon metres -> splat normalised units.

        p_splat = scale * (R @ p_vicon) + t

    Construct it with `SplatTransform.identity_metres(scale_to_metres)` when the
    splat frame and the Vicon frame are already aligned (they never are), or
    from `registration.solve()` against measured correspondences.
    """
    R: np.ndarray
    t: np.ndarray
    scale: float
    rms_residual_units: float = float("nan")
    n_points: int = 0

    def __post_init__(self):
        R = np.asarray(self.R, dtype=float)
        if R.shape != (3, 3):
            raise ValueError("R must be 3x3, got %s" % (R.shape,))
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
            raise ValueError("R is not orthonormal; R @ R.T deviates by %.2e"
                             % float(np.abs(R @ R.T - np.eye(3)).max()))
        if not math.isclose(float(np.linalg.det(R)), 1.0, abs_tol=1e-6):
            raise ValueError("det(R) = %.6f, not +1 -- this is a reflection, "
                             "not a rotation. A mirrored scene renders "
                             "plausibly and is wrong." % float(np.linalg.det(R)))
        if not (self.scale > 0.0) or not math.isfinite(self.scale):
            raise ValueError("scale must be finite and positive, got %r" % (self.scale,))
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", np.asarray(self.t, dtype=float).reshape(3))

    # -- construction ------------------------------------------------------
    @classmethod
    def identity_metres(cls, scale_to_metres: float) -> "SplatTransform":
        """Axes already aligned; only the metres<->normalised factor differs.

        `scale_to_metres` is the pipeline's own quantity: normalised * this =
        metres. So the forward (metres -> normalised) scale is its reciprocal.
        """
        if not scale_to_metres > 0:
            raise ValueError("scale_to_metres must be positive")
        return cls(np.eye(3), np.zeros(3), 1.0 / float(scale_to_metres))

    # -- properties --------------------------------------------------------
    @property
    def metres_per_unit(self) -> float:
        """Normalised units -> metres. The pipeline's `scale_to_metres`."""
        return 1.0 / self.scale

    # -- application -------------------------------------------------------
    def point_to_splat(self, p_vicon_m) -> np.ndarray:
        p = np.asarray(p_vicon_m, dtype=float)
        return (self.scale * (self.R @ p.reshape(-1, 3).T)).T + self.t

    def point_to_vicon(self, p_splat) -> np.ndarray:
        p = np.asarray(p_splat, dtype=float).reshape(-1, 3)
        return (self.R.T @ ((p - self.t) / self.scale).T).T

    def rotation_to_splat(self, R_body_vicon: np.ndarray) -> np.ndarray:
        """Body orientation expressed in the splat frame."""
        return self.R @ np.asarray(R_body_vicon, dtype=float)

    def pose_to_splat(self, p_vicon_m, quat_xyzw) -> tuple:
        """(position, rpy) in the splat frame, ready for the renderer."""
        R_body = quat_to_matrix(*quat_xyzw)
        pos = self.point_to_splat(p_vicon_m).reshape(3)
        rpy = matrix_to_rpy(self.rotation_to_splat(R_body))
        return pos, rpy

    def inverse(self) -> "SplatTransform":
        return SplatTransform(self.R.T, -(self.R.T @ self.t) / self.scale,
                              1.0 / self.scale)

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "_comment": "Vicon metres -> splat normalised: p_splat = scale*(R@p)+t",
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "scale": float(self.scale),
            "metres_per_unit": float(self.metres_per_unit),
            "rms_residual_units": float(self.rms_residual_units),
            "rms_residual_m": float(self.rms_residual_units * self.metres_per_unit),
            "n_points": int(self.n_points),
        }

    def save(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")

    @classmethod
    def load(cls, path) -> "SplatTransform":
        with open(path) as fh:
            d = json.load(fh)
        return cls(np.array(d["R"], dtype=float), np.array(d["t"], dtype=float),
                   float(d["scale"]), float(d.get("rms_residual_units", float("nan"))),
                   int(d.get("n_points", 0)))

    def __repr__(self) -> str:
        rpy = np.degrees(matrix_to_rpy(self.R))
        return ("SplatTransform(rpy=[%.2f %.2f %.2f] deg, t=[%.3f %.3f %.3f], "
                "1 unit = %.4f m, rms=%.4f m over %d pts)"
                % (rpy[0], rpy[1], rpy[2], self.t[0], self.t[1], self.t[2],
                   self.metres_per_unit,
                   self.rms_residual_units * self.metres_per_unit, self.n_points))
