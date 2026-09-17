"""Reading the point cloud a bundle declares.

WHY THIS IS HERE
----------------
A bundle has a `pointcloud` key described as "reference geometry, for
visualisation", and until now nothing in this package could open one. That made
the key a promise we did not keep: a student asked to plot their trajectory over
the scene had to install open3d or write a PLY parser, neither of which is the
assignment.

THIS IS NOT A RULE AGAINST open3d
---------------------------------
Students are free to `pip install open3d` and use it in their own code; for a
point cloud with a path through it, it is the nicer tool. The constraint is on
THIS PACKAGE, which has to import on every student laptop, on the flight
machine -- a laptop with a radio and no GPU -- and in CI. open3d is a 400 MB
wheel that pins numpy, and this session has twice lost a round trip to a module
that could not be imported because of it. Thirty lines here buys the bundle's
promise without spending that.

Only x/y/z is read. The file is a Gaussian splat export with sixty-odd
properties per vertex -- spherical harmonics, opacity, scales, quaternions --
and none of them are geometry. Anything that wants those wants a renderer.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["read_xyz", "PLY_DTYPES"]

#: PLY scalar type names -> numpy dtypes.
PLY_DTYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def read_xyz(path, max_points: Optional[int] = None, seed: int = 0) -> np.ndarray:
    """(N, 3) float64 vertex positions from a binary little-endian PLY.

    `max_points` subsamples uniformly at random, which is what a plot wants: a
    cleaned room is a third of a million Gaussians and matplotlib will not draw
    that twice.

    ASCII and big-endian PLYs are refused rather than guessed at. Every splat
    export is binary little-endian, so a file that is not one did not come from
    this pipeline and reading it wrong would be worse than saying so.
    """
    with open(path, "rb") as fh:
        header = b""
        while b"end_header" not in header:
            chunk = fh.read(8192)
            if not chunk:
                raise ValueError("%s ends before its PLY header does" % (path,))
            header += chunk
    offset = header.index(b"\n", header.index(b"end_header")) + 1
    text = header[:offset].decode("ascii", "replace")

    if "format binary_little_endian" not in text:
        fmt = next((l for l in text.splitlines() if l.startswith("format")),
                   "no format line")
        raise ValueError(
            "%s is %r; only binary little-endian PLY is read here. Re-export "
            "it, or use a full PLY library." % (path, fmt.strip()))

    fields, count, in_vertex, seen = [], None, False, False
    for line in text.splitlines():
        w = line.split()
        if not w:
            continue
        if w[0] == "element":
            in_vertex = w[1] == "vertex"
            if in_vertex:
                if seen:
                    raise ValueError("%s declares two vertex elements" % (path,))
                seen, count = True, int(w[2])
            elif seen:
                raise ValueError(
                    "%s has elements after its vertices; this reader stops at "
                    "the vertex block and would read the wrong bytes" % (path,))
        elif w[0] == "property" and in_vertex:
            if w[1] == "list":
                raise ValueError("%s has a list property on its vertices" % (path,))
            if w[1] not in PLY_DTYPES:
                raise ValueError("%s uses the property type %r" % (path, w[1]))
            fields.append((w[-1], PLY_DTYPES[w[1]]))

    names = [f[0] for f in fields]
    if count is None:
        raise ValueError("%s has no vertex element" % (path,))
    if not {"x", "y", "z"} <= set(names):
        raise ValueError("%s has no x/y/z vertex properties" % (path,))

    data = np.fromfile(path, dtype=np.dtype(fields), count=count, offset=offset)
    if len(data) != count:
        raise ValueError("%s claims %d vertices and holds %d -- truncated copy?"
                         % (path, count, len(data)))
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    if max_points is not None and len(xyz) > max_points:
        idx = np.random.default_rng(seed).choice(len(xyz), int(max_points),
                                                 replace=False)
        xyz = xyz[np.sort(idx)]
    return xyz
