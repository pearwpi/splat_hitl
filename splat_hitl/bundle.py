"""What "a cleaned metric splat" actually is, as a checkable set of files.

WHY A BUNDLE AND NOT A FOLDER OF FILES
--------------------------------------
Handing a student "the splat" is underspecified. Rendering needs the `.splat`;
collision needs the ESDF; the task needs the gates; the units need
`scale_to_metres`; flying it needs the Vicon registration; and the policy needs
the contract. Six artefacts that must describe ONE scene in ONE frame at ONE
scale, produced by different tools on different days.

The failure mode is not a missing file -- that is loud. It is an ESDF built
from a different export than the splat, or a gate placed outside the mapped
volume, or a contract whose sensor is not the one the scene was captured with.
Every one of those runs perfectly and is wrong, which is the same class of
failure as everything else this project has spent its time on.

So a bundle carries a manifest, and `check()` verifies the parts against each
other rather than merely counting them:

    python3 -m splat_hitl.bundle scenes/playTunnels

ONE THING THAT IS A WARNING AND NOT AN ERROR
-------------------------------------------
The metric scale. `dataparser_transforms.json` is the CAPTURE'S CLAIM about how
big the room is; the Vicon registration is a MEASUREMENT of the same room. A
LiDAR or VIO capture is good to about a percent, so they disagree, and failing
a bundle for that would fail every honestly registered scene. Below
`DATAPARSER_SCALE_TOL` the gap is reported with both numbers and the measured
value is kept. Above it, it is two different exports, and still an error.

WHAT IS DELIBERATELY NOT VALIDATED
----------------------------------
The `.splat` itself. Parsing it needs the renderer's dependencies, and this
package is numpy-only so that it imports on the flight machine. The manifest
records its size and SHA-256 instead, which catches a truncated copy -- the
realistic failure for a 6 MB binary moved by hand -- without pulling gsplat in.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .blockmap import BlockMap
from .collision import AIRFRAME_RADIUS_M, ESDF
from .contract import PolicyContract
from .anchors import AnchorDrift, AnchorSet
from .anchors import drift as anchor_drift
from .frames import SplatTransform
from .gates import GateCourse

__all__ = ["SceneBundle", "BundleReport", "MANIFEST_NAME"]

MANIFEST_NAME = "manifest.json"

#: How far the capture's own scale may sit from the measured one before it
#: stops being capture error and starts being a mismatched export. A metric
#: VIO or LiDAR capture lands within about a percent of a tape measure.
DATAPARSER_SCALE_TOL = 0.03

#: Keys the manifest may name, and whether a bundle is unusable without them.
#: Several are optional because a scene is useful before it is finished:
#: `vicon_transform` is missing until someone has been in the lab, and `gates`
#: only exist for a RACE. A planning assignment's task is a start and a goal,
#: not a course, and requiring gates would have forced every such scene to
#: invent some. `check()` says which of these are absent rather than failing.
_FILES = {
    "splat": True,             # rendering. Not parsed here -- see the docstring
    "esdf": True,              # collision
    "contract": True,          # what the policy assumes
    "gates": False,            # the task, IF the task is a race
    "map": False,              # the boundary/block file the student is handed
    "task": False,             # start/goal pairs, for a planning assignment
    "pointcloud": False,       # reference geometry, for visualisation
    "constraints": False,      # motion planes, from the scene tools
    "scene_calibration": False,  # render orientation
    "dataparser_transforms": False,  # scale_to_metres provenance, ARKit/Polycam
    "splat_frame": False,      # scale_to_metres provenance, COLMAP. See check()
    "vicon_transform": False,  # Vicon -> splat registration. Needed to FLY.
    "anchors": False,          # the fixed room points that registration was fitted to
}


@dataclass
class BundleReport:
    """Findings, split by whether they stop you."""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        out = []
        for tag, items in (("ERROR", self.errors), ("WARN ", self.warnings),
                           ("note ", self.notes)):
            out += ["  %s %s" % (tag, m) for m in items]
        return "\n".join(out) if out else "  (nothing to report)"


class SceneBundle:
    """One scene, loadable and checkable.

    Paths in the manifest are relative to the bundle directory, so a bundle can
    be moved or renamed without editing it -- which it will be, because it is
    going to be copied onto a dozen student machines.
    """

    def __init__(self, root, manifest: Dict[str, Any]):
        self.root = str(root)
        self.manifest = manifest
        self.name = str(manifest.get("name") or os.path.basename(self.root))

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, root) -> "SceneBundle":
        path = os.path.join(str(root), MANIFEST_NAME)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "%s has no %s. A directory of files is not a bundle: without a "
                "manifest nothing records which ESDF belongs to which splat."
                % (root, MANIFEST_NAME))
        with open(path) as fh:
            return cls(root, json.load(fh))

    def path(self, key: str) -> Optional[str]:
        """Absolute path for a manifest key, or None if it is not declared."""
        files = self.manifest.get("files") or {}
        rel = files.get(key)
        return None if not rel else os.path.join(self.root, rel)

    def _require(self, key: str) -> str:
        p = self.path(key)
        if p is None:
            raise KeyError("bundle %s declares no %r" % (self.name, key))
        return p

    def contract(self) -> PolicyContract:
        return PolicyContract.load(self._require("contract"))

    def gates(self) -> GateCourse:
        return GateCourse.load(self._require("gates"))

    def blockmap(self) -> BlockMap:
        return BlockMap.load(self._require("map"))

    def pointcloud_xyz(self, max_points=None) -> np.ndarray:
        """Vertex positions of the declared point cloud, in SPLAT units.

        Splat units, not metres: it is the same frame the splat and the ESDF
        are in, and converting here would hide which frame the caller is
        holding. `vicon_transform` turns it into the lab frame.
        """
        from .pointcloud import read_xyz
        return read_xyz(self._require("pointcloud"), max_points=max_points)

    def task(self) -> dict:
        with open(self._require("task")) as fh:
            return json.load(fh)

    def esdf(self) -> ESDF:
        return ESDF.load_npy(self._require("esdf"))

    def transform(self) -> Optional[SplatTransform]:
        p = self.path("vicon_transform")
        return None if p is None else SplatTransform.load(p)

    def anchors(self) -> Optional[AnchorSet]:
        """The fixed room points this scene's registration was fitted to."""
        p = self.path("anchors")
        return None if p is None else AnchorSet.load(p)

    def drift(self, measured) -> AnchorDrift:
        """Compare a fresh anchor reading against the registration's.

        `measured` is an AnchorSet, or anything AnchorSet takes: the same
        physical points as Vicon reports them today. The result says how far
        the world frame has moved since this bundle was built, and
        `.apply(bundle.transform())` gives the transform to fly with.
        """
        ref = self.anchors()
        if ref is None:
            raise ValueError(
                "%s declares no anchors, so a frame change cannot be detected. "
                "Add the anchor positions the registration was fitted to."
                % self.name)
        if not isinstance(measured, AnchorSet):
            measured = (AnchorSet.from_dict(measured)
                        if isinstance(measured, dict) and "anchors" in measured
                        else AnchorSet(dict(measured)))
        return anchor_drift(ref, measured)

    def transform_for(self, measured) -> SplatTransform:
        """This scene's transform, corrected for where the Vicon frame is now."""
        tf = self.transform()
        if tf is None:
            raise ValueError("%s has no vicon_transform to correct" % self.name)
        return self.drift(measured).apply(tf)

    # -- the part that matters ---------------------------------------------
    def check(self, clearance_m: float = 0.10) -> BundleReport:
        """Verify the parts against EACH OTHER, not just that they exist."""
        r = BundleReport()
        files = self.manifest.get("files") or {}

        for key, required in _FILES.items():
            rel = files.get(key)
            if not rel:
                if required:
                    r.errors.append("manifest declares no %r" % key)
                elif key == "vicon_transform":
                    r.notes.append(
                        "no vicon_transform: fine for training, but this bundle "
                        "cannot be FLOWN until the registration exists")
                elif key == "anchors" and files.get("vicon_transform"):
                    r.notes.append(
                        "no anchors: this bundle cannot tell whether the Vicon "
                        "frame has moved since it was registered, and a frame "
                        "that moves is silent -- see splat_hitl.anchors")
                elif key == "gates":
                    r.notes.append(
                        "no gates: this scene is not a race course. Its task, "
                        "if it declares one, is the start/goal pairs in 'task'")
                continue
            p = os.path.join(self.root, rel)
            if not os.path.exists(p):
                r.errors.append("%s declared as %r but missing" % (rel, key))

        if r.errors:
            return r                       # nothing below can be trusted

        # -- the splat, by size and hash, since we will not parse it --------
        want = (self.manifest.get("checksums") or {}).get("splat")
        sp = self.path("splat")
        if want:
            got = _sha256(sp)
            if got != want:
                r.errors.append(
                    "splat checksum mismatch: manifest says %s, file is %s. A "
                    "truncated or half-copied splat renders a partial scene "
                    "that looks plausible." % (want[:12], got[:12]))
        else:
            r.warnings.append("manifest records no splat checksum, so a "
                              "truncated copy would go unnoticed")

        # -- loadable? -------------------------------------------------------
        try:
            contract = self.contract()
        except Exception as exc:
            r.errors.append("contract will not load: %r" % (exc,))
            return r
        try:
            esdf = self.esdf()
        except Exception as exc:
            r.errors.append("ESDF will not load: %r" % (exc,))
            return r
        course = None
        if files.get("gates"):
            try:
                course = self.gates()
            except Exception as exc:
                r.errors.append("gates will not load: %r" % (exc,))
                return r

        # -- fingerprint, if the manifest claims one -------------------------
        claimed = self.manifest.get("contract_fingerprint")
        if claimed and claimed != contract.fingerprint():
            r.errors.append(
                "manifest says contract %s, the file is %s. Someone edited one "
                "without the other." % (claimed, contract.fingerprint()))

        # -- what margin can this field actually express? --------------------
        # Answered here, at scene-check time, rather than by a refusal from
        # CollisionMonitor when the student is already at the drone.
        usable = esdf.max_clearance_m
        if usable <= 0.0:
            r.errors.append(
                "the ESDF admits no usable clearance: truncation %.3f m is not "
                "at least two voxels (%.0f mm). Rebuild it with a larger "
                "truncation." % (esdf.truncation_m, esdf.voxel_size_m * 1000))
        elif usable < AIRFRAME_RADIUS_M:
            r.warnings.append(
                "the largest clearance this ESDF can express is %.0f mm, under "
                "the %.0f mm airframe radius -- a passing collision check does "
                "not mean the propellers cleared. Rebuild with a truncation of "
                "0.30-0.50 m."
                % (usable * 1000, AIRFRAME_RADIUS_M * 1000))

        if clearance_m < esdf.voxel_size_m:
            r.errors.append(
                "clearance %.0f mm is finer than the ESDF voxel (%.0f mm); the "
                "collision check cannot resolve it. This field supports "
                "%.0f to %.0f mm."
                % (clearance_m * 1000, esdf.voxel_size_m * 1000,
                   esdf.voxel_size_m * 1000, usable * 1000))
        elif clearance_m < 2.0 * esdf.voxel_size_m:
            r.warnings.append(
                "clearance %.0f mm is under two voxels (%.0f mm); "
                "discretisation is a large part of the margin"
                % (clearance_m * 1000, esdf.voxel_size_m * 1000))
        if clearance_m >= esdf.truncation_m:
            r.errors.append(
                "clearance %.3f m is at or past the truncation (%.3f m), so "
                "nothing would ever fail the check. The largest this field can "
                "express is %.3f m."
                % (clearance_m, esdf.truncation_m, usable))

        # -- do the gates lie in the mapped volume? --------------------------
        lo, hi = esdf.bounds_m
        for g in (course.gates if course is not None else ()):
            for corner, label in _gate_corners(g):
                if np.any(corner < lo) or np.any(corner > hi):
                    r.errors.append(
                        "gate %r has its %s outside the ESDF volume "
                        "(%.2f, %.2f, %.2f not within %s..%s). A course the "
                        "collision field cannot see is not scoreable."
                        % (g.name, label, *corner,
                           np.round(lo, 2).tolist(), np.round(hi, 2).tolist()))
                    break
            c = esdf.at(g.centre)
            if not c.outside and c.metres < clearance_m:
                r.warnings.append(
                    "gate %r has its centre %.0f mm from geometry, inside the "
                    "%.0f mm clearance: flying it correctly registers as a "
                    "collision" % (g.name, c.metres * 1000, clearance_m * 1000))

        # -- does the ESDF hold the cloud it was built from? -------------------
        # The scale test below is a PROXY for "these two came from different
        # exports". When the bundle carries the crop the field was built from,
        # that question can be measured instead of inferred: every splat in the
        # crop must read as geometry. A bundle that passes this has matching
        # exports whatever its capture believes about scale -- which matters,
        # because a3_test's capture is 4.6% out and is not mismatched at all.
        holds_cloud = None
        if files.get("pointcloud"):
            try:
                xyz = self.pointcloud_xyz() * esdf.metres_per_unit
                worst = max(esdf.at(q).metres for q in xyz[::max(1, len(xyz) // 5000)])
            except Exception as exc:
                r.warnings.append("point cloud unreadable: %r" % (exc,))
            else:
                limit = math.sqrt(3.0) * esdf.voxel_size_m + 1e-3   # one voxel diagonal
                holds_cloud = worst <= limit
                if not holds_cloud:
                    r.errors.append(
                        "the ESDF does not hold its own point cloud: a sampled "
                        "splat reads %.0f mm from geometry, past the %.0f mm one "
                        "voxel diagonal allows. They were built from different "
                        "exports." % (worst * 1000.0, limit * 1000.0))

        # -- one scale, and which source wins ---------------------------------
        # The dataparser scale is the CAPTURE'S CLAIM about how big the room is.
        # The Vicon registration is a MEASUREMENT of the same room against a
        # tape. On a LiDAR/VIO capture they disagree by around a percent, because
        # that is what such a capture is worth -- so a small gap here is a
        # finding, reported with both numbers, and the measured value wins.
        # Past DATAPARSER_SCALE_TOL it is no longer capture error: it is two
        # different exports, which is the failure this whole file exists to
        # catch, and that stays an error.
        dp = self.path("dataparser_transforms")
        if dp and os.path.exists(dp):
            try:
                with open(dp) as fh:
                    scale = float(json.load(fh)["scale"])
            except Exception as exc:
                r.warnings.append("dataparser_transforms unreadable: %r" % (exc,))
            else:
                mpu = 1.0 / scale if scale else float("nan")
                off = abs(mpu / esdf.metres_per_unit - 1.0)
                if off > DATAPARSER_SCALE_TOL and not holds_cloud:
                    r.errors.append(
                        "scale disagreement: dataparser_transforms implies "
                        "%.6f m per unit, the ESDF carries %.6f -- %.1f%% apart, "
                        "well past the %.0f%% that a capture's own scale error "
                        "explains. One was built from a different export."
                        % (mpu, esdf.metres_per_unit, off * 100.0,
                           DATAPARSER_SCALE_TOL * 100.0))
                elif off > DATAPARSER_SCALE_TOL:
                    r.warnings.append(
                        "the capture claims %.6f m per unit and the ESDF carries "
                        "%.6f -- %.1f%% apart, past the %.0f%% a capture's own "
                        "scale error usually explains. It IS capture error and "
                        "not two exports: the ESDF holds every sampled splat of "
                        "its own point cloud. The measured value is the one to "
                        "trust, and this capture is simply worse at guessing its "
                        "own size than the others."
                        % (mpu, esdf.metres_per_unit, off * 100.0,
                           DATAPARSER_SCALE_TOL * 100.0))
                elif off > 1e-3:
                    r.warnings.append(
                        "the capture claims %.6f m per unit, the ESDF was built "
                        "at %.6f -- %.1f%% apart. That gap IS the capture's "
                        "scale error, and the ESDF carries the value measured "
                        "against a tape, which is the one to trust."
                        % (mpu, esdf.metres_per_unit, off * 100.0))
        elif self.path("splat_frame") and os.path.exists(self.path("splat_frame")):
            # A COLMAP-solved capture has a dataparser scale, but it is in COLMAP's
            # arbitrary units and claims nothing about the room -- it can be tens of
            # percent from the truth without anything being wrong. Such a scene ships
            # splat_frame.json instead, which records where level and scale actually
            # came from. That is BETTER provenance than a dataparser scale, not worse,
            # so it is a note.
            try:
                with open(self.path("splat_frame")) as fh:
                    sf = json.load(fh)
                # `metres_per_splat_unit` is the corrected figure, after the pole-top
                # fit was folded back into the frame. Frames written before that
                # iteration existed carry only the raw `..._arkit` claim.
                mpu = float(sf["scale"].get("metres_per_splat_unit",
                                            sf["scale"].get("metres_per_splat_unit_arkit")))
                uperr = float(sf["gravity"]["nerfstudio_up_error_deg"])
            except Exception as exc:
                r.warnings.append("splat_frame unreadable: %r" % (exc,))
            else:
                off = abs(mpu / esdf.metres_per_unit - 1.0)
                r.notes.append(
                    "no dataparser_transforms, and correctly so: this capture was "
                    "solved by COLMAP, whose world is neither level nor metric "
                    "(nerfstudio's own 'up' is %.1f deg from gravity here), so its "
                    "dataparser scale is in arbitrary units and makes no claim about "
                    "the room. splat_frame.json carries the real provenance: %.6f m "
                    "per unit, recovered from the capture and corrected against the "
                    "surveyed pole tops, %.1f%% from the %.6f the registration "
                    "measured."
                    % (uperr, mpu, off * 100.0, esdf.metres_per_unit))
        else:
            r.warnings.append(
                "no dataparser_transforms: the metric scale has no provenance "
                "beyond whatever is baked into the ESDF")

        tf = self.transform()
        if tf is not None and not np.isclose(tf.metres_per_unit,
                                             esdf.metres_per_unit, rtol=1e-3):
            r.errors.append(
                "the Vicon registration is %.6f m per unit and the ESDF is "
                "%.6f. The drone would be scored against a differently sized "
                "copy of the scene." % (tf.metres_per_unit, esdf.metres_per_unit))

        # -- the map, and the task it poses ----------------------------------
        self._check_map_and_task(r, esdf, tf, clearance_m)

        r.notes.append("%s: %s, %s, contract %s"
                       % (self.name,
                          "%d gate(s)" % len(course.gates) if course is not None
                          else "no gates",
                          esdf, contract.fingerprint()))
        return r

    # -- the map, and the task it poses ------------------------------------
    def _check_map_and_task(self, r, esdf, tf, clearance_m) -> None:
        """Does the file the student is handed describe the room they fly in?

        Two descriptions of one room, kept in step by nothing. Move a box and
        edit the map without recapturing -- or recapture without editing the
        map -- and every verified-collision-free plan flies into something,
        while both files still load and both still look right.
        """
        mp = self.path("map")
        bmap = None
        if mp and os.path.exists(mp):
            try:
                bmap = BlockMap.load(mp)
            except Exception as exc:
                r.errors.append("map will not load: %r" % (exc,))
                return
            r.notes.append("map: %s" % (bmap,))

        if bmap is not None and tf is None:
            r.notes.append(
                "a map is declared but there is no vicon_transform, so it "
                "cannot be checked against the ESDF -- they are written in "
                "different frames")
        elif bmap is not None:
            self._compare_map_to_esdf(r, bmap, esdf, tf, clearance_m)

        tp = self.path("task")
        if not (tp and os.path.exists(tp)):
            return
        try:
            task = self.task()
        except Exception as exc:
            r.errors.append("task will not load: %r" % (exc,))
            return
        pairs = task.get("pairs") or []
        if not pairs:
            r.warnings.append("task declares no pairs, so it poses no problem")
        for pair in pairs:
            for end in ("start", "goal"):
                p = np.asarray(pair.get(end, []), dtype=float).reshape(-1)
                nm = "%s %s" % (pair.get("name", "?"), end)
                if p.shape != (3,):
                    r.errors.append("task %s is not three numbers" % nm)
                    continue
                if bmap is not None:
                    if not bool(bmap.inside_boundary(p)[0]):
                        r.errors.append(
                            "task %s (%.2f, %.2f, %.2f) is outside the map "
                            "boundary, so no legal plan reaches it"
                            % (nm, *p))
                        continue
                    if bool(bmap.inside_block(p)[0]):
                        r.errors.append(
                            "task %s (%.2f, %.2f, %.2f) is inside a block"
                            % (nm, *p))
                        continue
                if tf is None:
                    continue
                c = esdf.at(tf.point_to_splat(p).reshape(3) * tf.metres_per_unit)
                if c.outside:
                    r.errors.append(
                        "task %s (%.2f, %.2f, %.2f) is outside the ESDF volume"
                        % (nm, *p))
                elif c.metres < clearance_m:
                    r.errors.append(
                        "task %s (%.2f, %.2f, %.2f) sits %.0f mm from geometry, "
                        "inside the %.0f mm clearance -- the endpoint itself "
                        "scores as a collision"
                        % (nm, *p, c.metres * 1000, clearance_m * 1000))

    @staticmethod
    def _compare_map_to_esdf(r, bmap, esdf, tf, clearance_m) -> None:
        mpu = tf.metres_per_unit

        def clear_m(P):
            P = np.asarray(P, dtype=float).reshape(-1, 3)
            S = tf.point_to_splat(P) * mpu
            return np.array([esdf.at(s).metres for s in S])

        lo, hi = esdf.bounds_m
        S = tf.point_to_splat(bmap.corners()) * mpu
        outside = int(np.sum(np.any((S < lo) | (S > hi), axis=1)))
        if outside:
            r.errors.append(
                "%d of the map's 8 boundary corners fall outside the ESDF "
                "volume: part of the space students may plan through has no "
                "collision data at all" % outside)

        # obstacles the map does not know about
        G = bmap.sample_free(0.10)
        G = G[np.all((tf.point_to_splat(G) * mpu >= lo)
                     & (tf.point_to_splat(G) * mpu <= hi), axis=1)]
        if not len(G):
            return
        c = clear_m(G)
        far = bmap.distance_to_obstacle(G)
        hidden = (c < clearance_m) & (far > 0.25)
        frac = float(hidden.mean())
        if hidden.any():
            w = G[hidden][int(np.argmin(c[hidden]))]
            msg = ("%d of %d free-space samples (%.2f%%) are within %.0f mm of "
                   "geometry the map calls clear; worst at (%.2f, %.2f, %.2f). "
                   % (int(hidden.sum()), len(G), frac * 100.0,
                      clearance_m * 1000, *w))
            if frac > 0.02:
                r.errors.append(msg + "That is an obstacle missing from the map, "
                                "and every plan through it collides.")
            else:
                r.warnings.append(msg + "At this scale it is reconstruction "
                                  "speckle rather than a missing obstacle, but "
                                  "it does inflate the collision field there.")

        # blocks the map claims that were never built
        for i, b in enumerate(bmap.blocks):
            ctr = (b.lo + b.hi) / 2.0
            P = np.array([ctr, (ctr + b.lo) / 2.0, (ctr + b.hi) / 2.0])
            P = P[np.all((tf.point_to_splat(P) * mpu >= lo)
                         & (tf.point_to_splat(P) * mpu <= hi), axis=1)]
            if not len(P):
                continue
            if float(np.median(clear_m(P))) > 0.25:
                r.errors.append(
                    "block %d (%.2f, %.2f, %.2f)..(%.2f, %.2f, %.2f) is open "
                    "space in the ESDF. The map and the capture disagree about "
                    "what is in the room -- one of them is stale."
                    % (i, *b.lo, *b.hi))

    def __repr__(self) -> str:
        return "SceneBundle(%s, %s)" % (self.name, self.root)


def _gate_corners(g):
    half_w, half_h = g.width_m / 2.0, g.height_m / 2.0
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            yield (g.centre + sx * half_w * g.right + sy * half_h * g.up,
                   "corner (%+d, %+d)" % (sx, sy))


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":                                   # pragma: no cover
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 -m splat_hitl.bundle <bundle dir> "
                         "[clearance_m]")
    b = SceneBundle.load(sys.argv[1])
    rep = b.check(float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
    print("\n  %s\n%s\n" % (b, rep))
    raise SystemExit(0 if rep.ok else 1)
