"""Does the rendered view agree with the room the tape measured?

WHY THIS EXISTS
---------------
Everything between a Vicon pose and a rendered image is a convention: the
registration, the metric scale, the units the worker wants, the rotation order,
the mount. Each one is individually plausible and silently wrong. A rendered
picture of a room is not evidence that it is a picture of the RIGHT PLACE --
it will look like the room either way.

So this does not ask "did a frame come back". It parks the camera a known
distance in front of a box whose size came off a tape measure, renders, and
reads the depth at the centre of the image. If that number is the standoff, then
the registration, the scale, the pose convention and the renderer are all
consistent AT ONCE. If it is not, the size of the error says which:

    ~1.4% out everywhere .... the worker is using the capture's dataparser
                              scale instead of the measured one
    right magnitude, wrong
    place, or empty ......... a rotation convention (order, sign, or the
                              base roll the visualiser applies and the worker
                              does not)
    all pixels at the empty
    depth ................... the camera is outside the splat, or pointed away

    python3 -m splat_hitl.render_check --bundle scenes/net_2026-09-16 \\
        --splat-rendering ../MihirBhat/scripts/splat_rendering.py \\
        --out /tmp/render_check

Needs a CUDA gsplat, so it runs on the render machine, not on a laptop.

THE SECOND OPINION
------------------
The expected depth is also computed by sphere-tracing the bundle's own ESDF
along the same ray. Tape, distance field and renderer are three independent
paths to one number; agreement between all three is the claim worth making.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import zlib
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .blockmap import BlockMap
from .bundle import SceneBundle
from .frames import SplatTransform, matrix_to_rpy, rpy_to_matrix
from .renderer import SplatWorkerClient
from .sensor import SensorModel

__all__ = ["write_png", "sphere_trace", "standoff_poses", "centre_depth_m",
           "mirror_error_m", "vicon_pose_to_worker"]


def centre_depth_m(obs, half: int = 2) -> float:
    """Median depth over a small patch at the middle of the frame.

    A single centre pixel on a 96 x 64 image is one 0.6-degree sample and a
    splat is speckly; the median of a patch is the same measurement without the
    coin flip. Reads `Observation.depth_m` -- always metres, whatever rendered
    it -- and exists as a function so that the field name is pinned by a test
    rather than by the first run on a GPU machine.
    """
    d = np.asarray(obs.depth_m, dtype=float)
    h, w = d.shape[:2]
    patch = d[max(h // 2 - half, 0):h // 2 + half + 1,
              max(w // 2 - half, 0):w // 2 + half + 1]
    return float(np.median(patch))


# ----------------------------------------------------------------- PNG output
def write_png(path: str, rgb) -> None:
    """Write an 8-bit RGB PNG with the standard library only.

    Pillow is not a dependency of this package and is not going to become one
    for the sake of saving a picture.
    """
    a = np.asarray(rgb)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    h, w = a.shape[:2]
    raw = b"".join(b"\x00" + a[y, :, :3].tobytes() for y in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        fh.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        fh.write(chunk(b"IEND", b""))


def depth_to_rgb(depth, near_m: float, far_m: float):
    """Near is bright. Pixels at or past `far_m` render black, so 'saw
    nothing' and 'saw something far away' do not look alike."""
    d = np.asarray(depth, dtype=float)
    v = 1.0 - np.clip((d - near_m) / max(far_m - near_m, 1e-6), 0.0, 1.0)
    v[~np.isfinite(d)] = 0.0
    v[d >= far_m] = 0.0
    return np.repeat(v[:, :, None], 3, axis=2)


# ------------------------------------------------------------- the geometry
def sphere_trace(esdf, origin_m, direction, max_m: float = 8.0,
                 hit_m: Optional[float] = None) -> float:
    """Distance along a ray to the first surface, from an unsigned ESDF.

    Marching by the clearance is exact for an unsigned field and needs no
    gradient: a step that size cannot pass through anything. `hit_m` therefore
    defaults to one voxel -- the finest a nearest-voxel lookup can resolve, and
    asking for less would march forever against a surface it cannot see.

    Returns inf when the ray runs out or leaves the mapped volume. Those are
    not the same as "clear", and the caller has to be able to tell.

    One case it cannot resolve: a surface that IS the edge of the mapped volume,
    where leaving the map and hitting the wall happen at the same point. Real
    scenes pad the field past the geometry, so this only bites toy rooms whose
    walls are their own bounding box.
    """
    hit = float(esdf.voxel_size_m if hit_m is None else hit_m)
    d = np.asarray(direction, dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    p = np.asarray(origin_m, dtype=float).reshape(3)
    t = 0.0
    for _ in range(4096):
        c = esdf.at(p + t * d)
        if c.outside:
            return float("inf")
        if c.metres <= hit:
            return t
        t += c.metres
        if t > max_m:
            return float("inf")
    return float("inf")


def standoff_poses(bmap: BlockMap, standoff_m: float = 1.0
                   ) -> List[Tuple[str, np.ndarray, float, float]]:
    """One pose per block: back off its -x face, look along +x at its centre.

    Chosen rather than sampled so the expected reading is arithmetic -- the
    standoff -- instead of something else the renderer also has to be right
    about.
    """
    out = []
    for i, b in enumerate(bmap.blocks):
        cam = np.array([b.lo[0] - standoff_m,
                        (b.lo[1] + b.hi[1]) / 2.0,
                        (b.lo[2] + b.hi[2]) / 2.0])
        if not bool(bmap.inside_boundary([cam])[0]):
            continue
        out.append(("block%d" % i, cam, 0.0, float(standoff_m)))
    return out


def mirror_error_m(obs, esdf, tf, cam_vicon_m, yaw_rad, sensor,
                   columns: int = 9) -> Tuple[float, float]:
    """(error as rendered, error if left-right mirrored), in metres.

    A mirrored frame reads the SAME depth at the centre of the image, and keeps
    the floor at the bottom, so neither the standoff check nor a human looking
    at the picture can rule it out. `splat_rendering.py` carries --flip-lr
    because someone has been caught by it before.

    So trace a fan of rays through the middle row of pixels and compare each
    against the depth actually rendered there. An unmirrored image matches; a
    mirrored one matches its own reflection instead, and the two numbers say
    which by a wide margin.
    """
    from .renderer import _CAM_TO_BODY
    k = sensor.intrinsics()
    d = np.asarray(obs.depth_m, dtype=float)
    h, w = d.shape[:2]
    row = d[h // 2]
    R = tf.rotation_to_splat(rpy_to_matrix(0.0, 0.0, float(yaw_rad))) @ _CAM_TO_BODY
    origin = tf.point_to_splat(cam_vicon_m).reshape(3) * tf.metres_per_unit
    us = np.linspace(w * 0.12, w * 0.88, columns)
    traced, seen = [], []
    for u in us:
        ray = np.array([(u - k["cx"]) / k["fx"], 0.0, 1.0])
        t = sphere_trace(esdf, origin, R @ ray)
        if not np.isfinite(t):
            continue
        # sphere_trace walks along the ray, so its answer is a RANGE; the
        # renderer's depth is a range too (gsplat returns distance along z of
        # the normalised ray), so no cos correction is applied here.
        traced.append(t)
        seen.append(int(round(u)))
    if len(traced) < 3:
        return float("nan"), float("nan")
    got = np.array([row[i] for i in seen], dtype=float)
    flipped = np.array([row[w - 1 - i] for i in seen], dtype=float)
    traced = np.array(traced)
    return (float(np.mean(np.abs(got - traced))),
            float(np.mean(np.abs(flipped - traced))))


def vicon_pose_to_worker(tf: SplatTransform, p_vicon_m, yaw_vicon_rad):
    """(position in SPLAT METRES, body rpy in the SPLAT frame).

    Two conversions, and missing either one is silent.

    POSITION: `SplatWorkerClient.render` takes metres and divides by
    `scale_to_metres` itself, while `SplatTransform.point_to_splat` hands back
    NORMALISED units. The factor between them is 3.3 on this scene.

    HEADING: the splat frame is rotated from the lab frame -- 61.7 degrees on
    this scene, which is more than the camera's whole field of view. A yaw
    passed through unconverted points the camera at something beside the
    direction of travel, in every frame, by exactly the registration angle.

    Public, and the only implementation, because the first thing that wrote its
    own copy of this converted the position and forgot the heading.
    """
    R_body = rpy_to_matrix(0.0, 0.0, float(yaw_vicon_rad))
    pos_units = tf.point_to_splat(np.asarray(p_vicon_m, dtype=float)).reshape(3)
    rpy = matrix_to_rpy(tf.rotation_to_splat(R_body))
    return pos_units * tf.metres_per_unit, rpy


# ------------------------------------------------------------------ the check
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--splat-rendering", required=True,
                    help="path to metric-splat's splat_rendering.py")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for the worker; it needs CUDA gsplat")
    ap.add_argument("--out", default=None, help="directory for the PNGs")
    ap.add_argument("--standoff-m", type=float, default=1.0)
    ap.add_argument("--tolerance-m", type=float, default=0.10)
    a = ap.parse_args(argv)

    b = SceneBundle.load(a.bundle)
    rep = b.check()
    if not rep.ok:
        print("the bundle does not check out; fix that first\n%s" % rep)
        return 1
    tf = b.transform()
    if tf is None:
        print("this bundle has no vicon_transform, so there is no lab frame to "
              "render from")
        return 1
    bmap = BlockMap.load(b.path("map"))
    esdf = b.esdf()
    contract = b.contract()
    sensor = contract.observation.sensor

    cmd = [a.python, str(a.splat_rendering),
           "--backend", "cleaned-splat",
           "--splat", b.path("splat"),
           "--splat-config", os.path.join(b.root, "nerfstudio_config.yml"),
           "--transforms-json", os.path.join(b.root, "capture_transforms.json"),
           "--scale-to-metres", "%.6f" % tf.metres_per_unit]
    print("worker: %s\n" % " ".join(cmd))
    client = SplatWorkerClient(sensor, cmd)
    print("handshake: backend=%s scale_to_metres=%.6f (bundle says %.6f)\n"
          % (client.backend, client.scale_to_metres, tf.metres_per_unit))
    if abs(client.scale_to_metres - tf.metres_per_unit) > 1e-4:
        print("  !! the worker is rendering at a different scale from the "
              "registration. Every pose is displaced by %.1f%%.\n"
              % (100.0 * abs(client.scale_to_metres / tf.metres_per_unit - 1.0)))

    if a.out:
        os.makedirs(a.out, exist_ok=True)

    rows, bad, mirrored = [], 0, []
    try:
        for name, cam, yaw, expect in standoff_poses(bmap, a.standoff_m):
            pos_m, rpy = vicon_pose_to_worker(tf, cam, yaw)
            obs = client.render(pos_m, rpy)
            d = obs.depth_m
            centre = centre_depth_m(obs)
            fwd = tf.R @ np.array([math.cos(yaw), math.sin(yaw), 0.0])
            traced = sphere_trace(
                esdf, tf.point_to_splat(cam).reshape(3) * tf.metres_per_unit, fwd)
            err = centre - expect
            asis, flip = mirror_error_m(obs, esdf, tf, cam, yaw, sensor)
            mirrored.append((asis, flip))
            rows.append((name, expect, centre, traced, err, obs.render_s))
            if abs(err) > a.tolerance_m:
                bad += 1
            if a.out:
                write_png(os.path.join(a.out, "%s_rgb.png" % name), obs.rgb)
                write_png(os.path.join(a.out, "%s_depth.png" % name),
                          depth_to_rgb(d, 0.2, sensor.depth.far_m))
    finally:
        client.close()

    print("%-9s %-10s %-12s %-12s %-10s %s"
          % ("pose", "tape (m)", "rendered", "esdf ray", "error", "ms"))
    for name, expect, centre, traced, err, lat in rows:
        print("%-9s %9.3f %11.3f %11.3f %+9.0f mm %6.0f"
              % (name, expect, centre, traced, err * 1000.0, lat * 1000.0))
    print("\n%d of %d poses within %.0f mm of the tape"
          % (len(rows) - bad, len(rows), a.tolerance_m * 1000))

    good = [(x, y) for x, y in mirrored if np.isfinite(x) and np.isfinite(y)]
    if good:
        asis = float(np.mean([x for x, _ in good]))
        flip = float(np.mean([y for _, y in good]))
        print("\nleft-right: as rendered %.0f mm from the ESDF across the row, "
              "mirrored %.0f mm" % (asis * 1000, flip * 1000))
        if flip < asis:
            print("  !! the MIRRORED image fits the scene better. The frame is "
                  "flipped left-right; the worker takes --flip-lr.")
            bad += 1
        else:
            print("  the image as rendered fits, by %.1fx. Not mirrored."
                  % (flip / max(asis, 1e-6)))
    if a.out:
        print("frames written to %s" % a.out)
    return 1 if bad else 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())


#: Kept so older callers do not break; the public name is the one to use.
_vicon_pose_to_worker = vicon_pose_to_worker
