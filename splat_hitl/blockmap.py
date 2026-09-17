"""The map the student is handed: a boundary and axis-aligned blocks.

WHY THIS IS IN HERE AT ALL
--------------------------
The planning assignment gives out a text file and asks for a collision-free
path through it. The student writes their own parser -- that is part of the
work -- so this one exists for a different reason: to check the file against
the scene it claims to describe.

Those are two different descriptions of one room. The map says where the
obstacles are; the ESDF says where the geometry is. Nothing keeps them in step.
Move a box 20 cm and update the map without recapturing, or recapture without
editing the map, and every student's verified-collision-free plan flies into
something. Both files still load. Both look right. `SceneBundle.check()` is
where that gets caught, and this is what it needs to do the catching.

THE FORMAT
----------
Whitespace-delimited floats in metres, `#` starts a comment::

    boundary xmin ymin zmin xmax ymax zmax
    block    xmin ymin zmin xmax ymax zmax r g b

Blocks are axis-aligned and may overlap. Colours are 0-255 and carry no
meaning beyond drawing. There is no line for the floor, which is exactly why a
boundary's `zmin` has to be the lowest a plan may go rather than the height of
the physical ground -- see the map file's own header.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Block", "BlockMap"]


@dataclass(frozen=True)
class Block:
    """One axis-aligned obstacle, in metres."""
    lo: np.ndarray
    hi: np.ndarray
    rgb: Tuple[int, int, int] = (128, 128, 128)

    def distance_to(self, p_m) -> np.ndarray:
        """Distance from each point to this block's surface, 0 inside it."""
        p = np.asarray(p_m, dtype=float).reshape(-1, 3)
        return np.linalg.norm(np.maximum(np.maximum(self.lo - p, p - self.hi),
                                         0.0), axis=1)


class BlockMap:
    """A parsed map file. All lengths in the frame the file is written in."""

    def __init__(self, lo, hi, blocks: Sequence[Block], source: str = ""):
        self.lo = np.asarray(lo, dtype=float).reshape(3)
        self.hi = np.asarray(hi, dtype=float).reshape(3)
        if not np.all(self.hi > self.lo):
            raise ValueError("boundary is empty or inverted: %s .. %s"
                             % (self.lo, self.hi))
        self.blocks = list(blocks)
        self.source = str(source)

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, path) -> "BlockMap":
        lo = hi = None
        blocks: List[Block] = []
        with open(path) as fh:
            for n, raw in enumerate(fh, 1):
                w = raw.split("#", 1)[0].split()
                if not w:
                    continue
                kind, rest = w[0], w[1:]
                if kind == "boundary":
                    if len(rest) < 6:
                        raise ValueError("%s:%d boundary needs 6 numbers, got %d"
                                         % (path, n, len(rest)))
                    if lo is not None:
                        raise ValueError("%s:%d a second boundary line; a map "
                                         "has exactly one" % (path, n))
                    v = [float(x) for x in rest[:6]]
                    lo, hi = v[:3], v[3:]
                elif kind == "block":
                    if len(rest) < 6:
                        raise ValueError("%s:%d block needs at least 6 numbers, "
                                         "got %d" % (path, n, len(rest)))
                    v = [float(x) for x in rest[:6]]
                    rgb = tuple(int(float(x)) for x in rest[6:9]) if len(rest) >= 9 \
                        else (128, 128, 128)
                    blocks.append(Block(np.array(v[:3]), np.array(v[3:]), rgb))
                else:
                    raise ValueError("%s:%d unknown line type %r -- a map holds "
                                     "'boundary' and 'block' lines only"
                                     % (path, n, kind))
        if lo is None:
            raise ValueError("%s has no boundary line" % path)
        return cls(lo, hi, blocks, source=str(path))

    # -- queries -----------------------------------------------------------
    @property
    def size_m(self) -> np.ndarray:
        return self.hi - self.lo

    def corners(self) -> np.ndarray:
        """The boundary's eight corners, for containment checks."""
        return np.array([[x, y, z] for x in (self.lo[0], self.hi[0])
                         for y in (self.lo[1], self.hi[1])
                         for z in (self.lo[2], self.hi[2])])

    def inside_boundary(self, p_m) -> np.ndarray:
        p = np.asarray(p_m, dtype=float).reshape(-1, 3)
        return np.all((p >= self.lo) & (p <= self.hi), axis=1)

    def inside_block(self, p_m) -> np.ndarray:
        p = np.asarray(p_m, dtype=float).reshape(-1, 3)
        hit = np.zeros(len(p), dtype=bool)
        for b in self.blocks:
            hit |= np.all((p >= b.lo) & (p <= b.hi), axis=1)
        return hit

    def distance_to_obstacle(self, p_m) -> np.ndarray:
        """Distance to the nearest BLOCK surface. Inside a block reads 0.

        The boundary is not an obstacle here: leaving it is a different failure
        from hitting something, and conflating them hides which one happened.
        """
        p = np.asarray(p_m, dtype=float).reshape(-1, 3)
        if not self.blocks:
            return np.full(len(p), np.inf)
        d = np.full(len(p), np.inf)
        for b in self.blocks:
            d = np.minimum(d, b.distance_to(p))
        return d

    def sample_free(self, spacing_m: float = 0.10) -> np.ndarray:
        """A grid over the free space inside the boundary.

        Used to compare the map against a distance field. Spacing is a real
        choice: too fine and the check takes minutes, too coarse and a thin
        obstacle slips between samples.
        """
        # clipped, because arange's stop is exclusive-ish in floating point and
        # the last step lands a few ulp PAST hi -- which then reads as outside
        # the boundary it was built from.
        axes = [np.clip(np.arange(self.lo[i], self.hi[i] + 1e-9, spacing_m),
                        self.lo[i], self.hi[i]) for i in range(3)]
        g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        return g[~self.inside_block(g)]

    def __repr__(self) -> str:
        return ("BlockMap(%.2f x %.2f x %.2f m, %d block(s), z %.2f..%.2f)"
                % (self.size_m[0], self.size_m[1], self.size_m[2],
                   len(self.blocks), self.lo[2], self.hi[2]))
