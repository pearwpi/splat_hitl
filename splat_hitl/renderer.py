"""Rendering, behind an interface that takes and returns metres.

WHY AN INTERFACE AT ALL
-----------------------
The real renderer needs a GPU, a CUDA gsplat build, a trained splat, and its own
conda environment. Putting that in the flight loop's import path would mean the
whole runtime is only testable in one place, on one machine. Instead everything
downstream talks to `RendererClient`, and `FakeRenderer` -- pure numpy, no
assets -- stands in for it. The loop, the scoring, the safety logic and the
command mapping are then all provable on a laptop.

UNITS, WHICH THE WORKER GETS ASYMMETRICALLY WRONG
--------------------------------------------------
`metric-splat`'s `splat_rendering.py` worker wants POSITIONS IN NORMALISED
SCENE UNITS and returns DEPTH IN METRES. That asymmetry is real, it is easy to
get wrong, and it is the kind of thing an interface exists to absorb:

    RendererClient.render(position_m, body_rpy)  ->  depth in METRES

`SplatWorkerClient` divides by `scale_to_metres` on the way in. Nothing above
this layer sees a normalised coordinate.

THE MOUNT IS APPLIED HERE, ONCE
-------------------------------
A camera pitched 10 degrees down is part of the sensor model, not part of the
policy's business. The client composes body attitude with the mount rotation so
that callers pass the DRONE's pose and get the CAMERA's view, and nobody
downstream has to remember which of the two they are holding.
"""
from __future__ import annotations

import base64
import json
import math
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .frames import matrix_to_rpy, rpy_to_matrix
from .sensor import SensorModel

__all__ = ["Observation", "RendererClient", "FakeRenderer", "SplatWorkerClient"]

# Camera axes (OpenCV: x right, y down, z forward) expressed in body FLU.
_CAM_TO_BODY = np.array([[0.0, 0.0, 1.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0]])


@dataclass(frozen=True)
class Observation:
    """One rendered view. `depth_m` is always METRES, whatever produced it.

    `policy_input` is what a CONTRACTED policy actually reads: depth clipped,
    normalised and stacked into a history by `ObservationBuilder`, exactly as
    the trainer did it. It is None when no contract is in force, which is why
    the reference policies read `depth_m` and a trained one must read
    `policy_input` -- and get a clear failure rather than a subtly wrong image
    if it is missing.
    """
    depth_m: np.ndarray
    rgb: Optional[np.ndarray]
    render_s: float
    sensor_fingerprint: str
    policy_input: Optional[np.ndarray] = None

    @property
    def shape(self) -> tuple:
        return tuple(self.depth_m.shape)


class RendererClient(ABC):
    """Pose in metres -> depth in metres, at the sensor model's geometry."""

    def __init__(self, sensor: SensorModel):
        self.sensor = sensor
        self._mount = rpy_to_matrix(0.0,
                                    math.radians(sensor.mount_pitch_deg),
                                    math.radians(sensor.mount_yaw_deg))

    def camera_rpy(self, body_rpy: Sequence[float]) -> np.ndarray:
        """Body attitude composed with the fixed mount rotation.

        `mount_pitch_deg` is POSITIVE DOWN, which is how people describe a
        camera mount and the opposite of the right-handed pitch sign, so the
        rotation is built from the negated angle.
        """
        r, p, y = (float(v) for v in body_rpy)
        R_body = rpy_to_matrix(r, p, y)
        R_mount = rpy_to_matrix(0.0, -math.radians(self.sensor.mount_pitch_deg),
                                math.radians(self.sensor.mount_yaw_deg))
        return matrix_to_rpy(R_body @ R_mount)

    @abstractmethod
    def render(self, position_m, body_rpy) -> Observation:
        ...

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------- fake
class FakeRenderer(RendererClient):
    """Analytic ray-cast of a box room with spherical obstacles. No GPU.

    Deliberately the SAME world description that `collision.synthetic_room()`
    takes, so the depth a policy sees and the field that scores it describe one
    consistent toy scene. A student can fly the entire loop, hit a virtual wall,
    and see the depth image that should have warned them -- before any splat
    exists.

    Returns RANGE along each ray, not perpendicular z-depth. Stated because
    renderers differ and the difference is a few percent at the edges of a wide
    field of view.

    IT ALSO RENDERS COLOUR, flat per surface: the six walls get six distinct
    colours and each obstacle gets its own. That is enough to develop a
    segmentation or optical-flow loop end to end -- the shapes move correctly,
    the colours are consistent between frames -- without a GPU or a scene.

    It is NOT enough to TRAIN a perception network on. The colours are flat,
    the labels are exact and there is no texture, lighting or noise, so a
    network fitted here learns a problem that does not exist. Train on real or
    Blender-rendered imagery; use this to check that your loop is wired up.
    """

    #: index 0 is "hit nothing", 1..6 are the six walls in the order
    #: x-, x+, y-, y+, z- (floor), z+ (ceiling), then obstacles.
    PALETTE = np.array([
        [0.00, 0.00, 0.00],
        [0.75, 0.25, 0.25], [0.25, 0.55, 0.85],
        [0.85, 0.65, 0.20], [0.35, 0.70, 0.35],
        [0.45, 0.45, 0.50], [0.90, 0.90, 0.88],
    ], dtype=np.float32)
    OBSTACLE_COLOURS = np.array([
        [0.85, 0.35, 0.75], [0.30, 0.80, 0.80], [0.95, 0.55, 0.15],
        [0.55, 0.35, 0.85], [0.20, 0.85, 0.45],
    ], dtype=np.float32)

    def __init__(self, sensor: SensorModel, room_size_m=(4.0, 3.0, 2.5),
                 obstacles: Sequence[dict] = (), far_m: float = 24.0):
        super().__init__(sensor)
        self.room = np.asarray(room_size_m, dtype=float).reshape(3)
        self.obstacles = [
            (np.asarray(o["centre"], dtype=float).reshape(3), float(o["radius"]))
            for o in obstacles
        ]
        # An obstacle may name its own colour, so a coloured sphere can stand in
        # for a landmark you intend to detect.
        colours = [self.OBSTACLE_COLOURS[i % len(self.OBSTACLE_COLOURS)]
                   for i in range(len(self.obstacles))]
        for i, o in enumerate(obstacles):
            if o.get("colour") is not None:
                colours[i] = np.asarray(o["colour"], dtype=np.float32).reshape(3)
        self.palette = (np.vstack([self.PALETTE] + [np.asarray(colours,
                                                               dtype=np.float32)])
                        if colours else self.PALETTE.copy())
        self.far_m = float(far_m)
        self.scale_to_metres = 1.0
        self._dirs_cam = self._pixel_rays()

    def _pixel_rays(self) -> np.ndarray:
        k = self.sensor.intrinsics()
        u = np.arange(self.sensor.width) + 0.5
        v = np.arange(self.sensor.height) + 0.5
        U, V = np.meshgrid(u, v)                      # (H, W)
        d = np.stack([(U - k["cx"]) / k["fx"],
                      (V - k["cy"]) / k["fy"],
                      np.ones_like(U)], axis=-1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def render(self, position_m, body_rpy) -> Observation:
        t0 = time.perf_counter()
        p = np.asarray(position_m, dtype=float).reshape(3)
        R = rpy_to_matrix(*self.camera_rpy(body_rpy)) @ _CAM_TO_BODY
        dirs = self._dirs_cam @ R.T                   # (H, W, 3) world directions

        # Exit distance from the axis-aligned room, slab method. The camera is
        # assumed to be inside; if it is not, the room contributes nothing.
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lo = (0.0 - p) / dirs
            t_hi = (self.room - p) / dirs
        # A ray parallel to a slab gives +-inf, which correctly means "this axis
        # never constrains it"; 0/0 gives nan and must become +inf, not 0.
        t_exit = np.maximum(t_lo, t_hi)
        t_exit = np.where(np.isnan(t_exit), np.inf, t_exit)
        depth = np.min(t_exit, axis=-1)

        # Which of the six faces the ray leaves through: the axis that
        # constrains it, and the sign of the ray along that axis.
        axis = np.argmin(t_exit, axis=-1)
        along = np.take_along_axis(dirs, axis[..., None], axis=-1)[..., 0]
        surface = 1 + axis * 2 + (along > 0).astype(np.int64)
        surface = np.where(depth > 0, surface, 0)
        depth = np.where(depth > 0, depth, self.far_m)

        for i, (c, r) in enumerate(self.obstacles):
            oc = p - c
            b = 2.0 * np.einsum("ijk,k->ij", dirs, oc)
            cc = float(np.dot(oc, oc)) - r * r
            disc = b * b - 4.0 * cc
            hit = disc > 0
            if not np.any(hit):
                continue
            sq = np.sqrt(np.where(hit, disc, 0.0))
            t1 = (-b - sq) / 2.0
            t2 = (-b + sq) / 2.0
            t = np.where(t1 > 1e-6, t1, np.where(t2 > 1e-6, t2, np.inf))
            t = np.where(hit, t, np.inf)
            surface = np.where(t < depth, len(self.PALETTE) + i, surface)
            depth = np.minimum(depth, t)

        depth = np.clip(depth, 0.0, self.far_m).astype(np.float32)
        rgb = self.palette[np.clip(surface, 0, len(self.palette) - 1)]
        return Observation(depth, rgb, time.perf_counter() - t0,
                           self.sensor.fingerprint())


# ------------------------------------------------------------------- worker
class SplatWorkerClient(RendererClient):
    """Client for `metric-splat`'s `splat_rendering.py` subprocess worker.

    The worker runs in its own interpreter because Nerfstudio, gsplat and the
    policy stacks cannot share a dependency tree. It speaks newline-delimited
    JSON over stdin/stdout with base64 float32 payloads:

        handshake  {"ready": true, "scale_to_metres": ..., "backend": ...}
        request    {"position": [...normalised...], "orientation_rpy": [...],
                    "image_width": W, "image_height": H,
                    "fov_x_half_tan": tan(fov_x / 2)}
        response   {"ok": true, "shape": [H, W], "depth_b64": ...,
                    "rgb_shape": [...], "rgb_b64": ...}
        shutdown   {"cmd": "close"}

    NOT EXERCISED AGAINST A LIVE WORKER YET -- it needs a GPU and a scene. The
    protocol is transcribed from `splat_rendering.py` rather than assumed, but
    treat the first run on pear-2 as an integration test, not a formality.
    """

    def __init__(self, sensor: SensorModel, worker_cmd: Sequence[str],
                 env: Optional[dict] = None, timeout_s: float = 30.0):
        super().__init__(sensor)
        self.timeout_s = float(timeout_s)
        self.proc = subprocess.Popen(list(worker_cmd), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True,
                                     bufsize=1, env=env)
        ready = self._read()
        if not ready.get("ready"):
            raise RuntimeError("render worker did not start: %r" % (ready,))
        self.scale_to_metres = float(ready["scale_to_metres"])
        self.backend = ready.get("backend")

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("render worker closed its output unexpectedly "
                               "(exit code %r)" % (self.proc.poll(),))
        return json.loads(line)

    def render(self, position_m, body_rpy) -> Observation:
        t0 = time.perf_counter()
        p = np.asarray(position_m, dtype=float).reshape(3) / self.scale_to_metres
        rpy = self.camera_rpy(body_rpy)
        req = {"position": p.tolist(),
               "orientation_rpy": [float(v) for v in rpy],
               "image_width": int(self.sensor.width),
               "image_height": int(self.sensor.height),
               "fov_x_half_tan": float(math.tan(math.radians(self.sensor.fov_x_deg) / 2.0))}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        resp = self._read()
        if not resp.get("ok"):
            raise RuntimeError("render failed: %s" % resp.get("error"))
        h, w = resp["shape"]
        depth = np.frombuffer(base64.b64decode(resp["depth_b64"]),
                              dtype=np.float32).reshape(h, w)
        rgb = None
        if resp.get("rgb_b64"):
            rgb = np.frombuffer(base64.b64decode(resp["rgb_b64"]),
                                dtype=np.float32).reshape(*resp["rgb_shape"])
        return Observation(depth, rgb, time.perf_counter() - t0,
                           self.sensor.fingerprint())

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
