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
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .collision import ESDF
from .contract import PolicyContract
from .frames import SplatTransform
from .gates import GateCourse

__all__ = ["SceneBundle", "BundleReport", "MANIFEST_NAME"]

MANIFEST_NAME = "manifest.json"

#: Keys the manifest may name, and whether a bundle is unusable without them.
#: `vicon_transform` is optional because a bundle is useful for TRAINING before
#: anyone has been in the lab; `check()` says so rather than failing.
_FILES = {
    "splat": True,             # rendering. Not parsed here -- see the docstring
    "esdf": True,              # collision
    "contract": True,          # what the policy assumes
    "gates": True,             # the task
    "pointcloud": False,       # reference geometry, for visualisation
    "constraints": False,      # motion planes, from the scene tools
    "scene_calibration": False,  # render orientation
    "dataparser_transforms": False,  # scale_to_metres provenance
    "vicon_transform": False,  # Vicon -> splat registration. Needed to FLY.
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

    def esdf(self) -> ESDF:
        return ESDF.load_npy(self._require("esdf"))

    def transform(self) -> Optional[SplatTransform]:
        p = self.path("vicon_transform")
        return None if p is None else SplatTransform.load(p)

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

        # -- clearance against the voxel size --------------------------------
        if clearance_m < esdf.voxel_size_m:
            r.errors.append(
                "clearance %.0f mm is finer than the ESDF voxel (%.0f mm); the "
                "collision check cannot resolve it"
                % (clearance_m * 1000, esdf.voxel_size_m * 1000))
        elif clearance_m < 2.0 * esdf.voxel_size_m:
            r.warnings.append(
                "clearance %.0f mm is under two voxels (%.0f mm); "
                "discretisation is a large part of the margin"
                % (clearance_m * 1000, esdf.voxel_size_m * 1000))
        if clearance_m >= esdf.truncation_m:
            r.errors.append(
                "clearance %.2f m is at or past the truncation (%.2f m), so "
                "nothing would ever fail the check"
                % (clearance_m, esdf.truncation_m))

        # -- do the gates lie in the mapped volume? --------------------------
        lo, hi = esdf.bounds_m
        for g in course.gates:
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

        # -- one scale, from one source --------------------------------------
        dp = self.path("dataparser_transforms")
        if dp and os.path.exists(dp):
            try:
                with open(dp) as fh:
                    scale = float(json.load(fh)["scale"])
            except Exception as exc:
                r.warnings.append("dataparser_transforms unreadable: %r" % (exc,))
            else:
                mpu = 1.0 / scale if scale else float("nan")
                if not np.isclose(mpu, esdf.metres_per_unit, rtol=1e-3):
                    r.errors.append(
                        "scale disagreement: dataparser_transforms implies "
                        "%.6f m per unit, the ESDF carries %.6f. One of them "
                        "was built from a different export."
                        % (mpu, esdf.metres_per_unit))
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

        r.notes.append("%s: %d gate(s), %s, contract %s"
                       % (self.name, len(course.gates), esdf, contract.fingerprint()))
        return r

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
