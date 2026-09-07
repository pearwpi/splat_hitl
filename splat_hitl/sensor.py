"""The sensor model: the one contract training and HITL must agree on.

WHY THIS IS A FILE AND NOT A PILE OF ARGUMENTS
-----------------------------------------------
A policy learns the camera it was trained through. Change the field of view,
the resolution, the mount angle or the depth encoding between training and
flight, and the policy is looking at a world it has never seen -- while every
component still reports healthy.

That failure is indistinguishable from "the policy is bad". Both present as
flying into things. So the sensor model is versioned, hashed, and recorded into
every run log on both sides; comparing two hashes answers "did this policy ever
see this world?" in one step, instead of a week of debugging the wrong layer.

Depth encoding deserves particular care. The renderer emits METRIC depth in
scene units. Published policies each want something different -- DiffPhysDrone
wants `3.0 / clip(d, 0.3, 24.0) - 0.6`, MAVRL wants a downsampled raw metric
image, a potential-field planner wants an inverted uint8 where LARGE MEANS
CLOSE. Encoding is part of the contract, not a preprocessing detail, because a
per-frame normalisation makes a threshold mean nothing in metres.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Optional

__all__ = ["SensorModel", "DepthEncoding"]

_ENCODINGS = ("metric", "inverse_diffphys", "inverted_uint8", "normalized_uint8")


@dataclass(frozen=True)
class DepthEncoding:
    """How metric depth becomes what the policy actually sees.

    kind:
      metric            raw metres, float32. The honest default.
      inverse_diffphys  3/clip(d, near, far) - 0.6, the released DiffPhysDrone
                        preprocessing.
      inverted_uint8    255 * (far - clip(d)) / (far - near), LARGE = CLOSE.
                        Fixed metric endpoints, unlike normalized_uint8.
      normalized_uint8  per-frame min/max scaling. Included because existing
                        code does it; a threshold in this space has NO fixed
                        meaning in metres and moves with the farthest visible
                        surface. Prefer inverted_uint8.
    """
    kind: str = "metric"
    near_m: float = 0.3
    far_m: float = 24.0
    empty_depth_m: float = 12.0        # what a pixel with no Gaussian hit gets

    def __post_init__(self):
        if self.kind not in _ENCODINGS:
            raise ValueError("depth encoding %r not in %s" % (self.kind, _ENCODINGS))
        if not 0 < self.near_m < self.far_m:
            raise ValueError("need 0 < near_m < far_m, got %r, %r"
                             % (self.near_m, self.far_m))
        if not self.near_m <= self.empty_depth_m <= self.far_m:
            raise ValueError("empty_depth_m %r must lie within [near, far]"
                             % (self.empty_depth_m,))


@dataclass(frozen=True)
class SensorModel:
    """Everything that decides what the policy sees, in one hashable object."""
    name: str
    width: int
    height: int
    fov_x_deg: float
    mount_pitch_deg: float = 0.0       # +ve = camera tilted DOWN from level
    mount_yaw_deg: float = 0.0
    stereo_baseline_m: Optional[float] = None      # None = monocular
    depth: DepthEncoding = field(default_factory=DepthEncoding)
    rate_hz: float = 30.0
    notes: str = ""

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if not 0.0 < self.fov_x_deg < 180.0:
            raise ValueError("fov_x_deg must lie in (0, 180), got %r" % (self.fov_x_deg,))
        if not self.rate_hz > 0:
            raise ValueError("rate_hz must be positive")
        if self.stereo_baseline_m is not None and not self.stereo_baseline_m > 0:
            raise ValueError("stereo_baseline_m must be positive or None")

    # -- derived ----------------------------------------------------------
    @property
    def aspect(self) -> float:
        return self.width / float(self.height)

    @property
    def fov_y_deg(self) -> float:
        """Vertical FOV implied by the horizontal FOV and the aspect ratio."""
        return math.degrees(2.0 * math.atan(
            math.tan(math.radians(self.fov_x_deg) / 2.0) / self.aspect))

    @property
    def is_stereo(self) -> bool:
        return self.stereo_baseline_m is not None

    def intrinsics(self) -> dict:
        """Pinhole intrinsics in pixels. fy is derived from fx and the aspect
        ratio, i.e. square pixels are assumed."""
        fx = (self.width / 2.0) / math.tan(math.radians(self.fov_x_deg) / 2.0)
        return {"fx": fx, "fy": fx, "cx": self.width / 2.0, "cy": self.height / 2.0}

    # -- identity ---------------------------------------------------------
    def fingerprint(self) -> str:
        """Stable 12-char hash of everything that changes what the policy sees.

        `name` and `notes` are excluded on purpose: renaming a config must not
        invalidate a trained policy, and a comment is not a sensor property.
        """
        d = asdict(self)
        d.pop("name", None)
        d.pop("notes", None)
        blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def assert_compatible(self, other: "SensorModel") -> None:
        """Raise unless two models would show a policy the same world."""
        if self.fingerprint() != other.fingerprint():
            raise ValueError(
                "sensor model mismatch: %s (%s) vs %s (%s).\n"
                "A policy trained through one of these has never seen the "
                "other. Fix the config rather than the policy."
                % (self.name, self.fingerprint(), other.name, other.fingerprint()))

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        d["derived"] = {"fov_y_deg": self.fov_y_deg, "aspect": self.aspect,
                        "intrinsics_px": self.intrinsics()}
        return d

    def save(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")

    @classmethod
    def from_dict(cls, d: dict) -> "SensorModel":
        d = dict(d)
        d.pop("fingerprint", None)
        d.pop("derived", None)
        depth = d.pop("depth", None) or {}
        return cls(depth=DepthEncoding(**depth), **d)

    @classmethod
    def load(cls, path) -> "SensorModel":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def __repr__(self) -> str:
        return ("SensorModel(%s, %dx%d, %.1f deg H / %.1f deg V, pitch %+.1f, %s, "
                "%s, %.0f Hz, %s)"
                % (self.name, self.width, self.height, self.fov_x_deg,
                   self.fov_y_deg, self.mount_pitch_deg,
                   "stereo %.0f mm" % (self.stereo_baseline_m * 1000)
                   if self.is_stereo else "mono",
                   self.depth.kind, self.rate_hz, self.fingerprint()))
