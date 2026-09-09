"""Metric depth from the renderer -> the array the policy was trained on.

WHY THIS IS NOT INLINE IN THE RUNTIME
-------------------------------------
The conversion from metres to policy input is four steps -- fill non-finite
pixels, clip, normalise, stack a history -- and every one of them is a place
where training and flight can silently disagree. Written inline at the call
site it is four lines nobody reads. Written here, driven by `ObservationSpec`,
it is one implementation whose behaviour is pinned by the same fingerprint the
policy is checked against.

THE HISTORY IS THE SUBTLE PART
------------------------------
The trainer stacks the last four frames and, on the first frame of an episode,
fills all four slots with a COPY OF THAT FRAME rather than with zeros:

    if not self.depth_history:
        for _ in range(DEPTH_HISTORY_LENGTH):
            self.depth_history.append(depth_frame.copy())

Zeros would mean "every pixel is touching an obstacle", so a zero-primed first
observation shows the policy a wall and it reacts to it. That is a real
difference in behaviour for the first few control steps of every flight --
which is exactly when the drone is closest to a person -- so `prime` is a
contract field and not a detail.

`reset()` must be called at the start of every run. The runtime does it; if you
drive this yourself and forget, the first observation of run N carries frames
from run N-1.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from .contract import ObservationSpec

__all__ = ["ObservationBuilder"]


class ObservationBuilder:
    """Stateful because the history is state. One per run."""

    def __init__(self, spec: ObservationSpec):
        self.spec = spec
        self._hist: deque = deque(maxlen=int(spec.history))

    # -- lifecycle --------------------------------------------------------
    def reset(self) -> None:
        self._hist.clear()

    @property
    def primed(self) -> bool:
        return len(self._hist) == self._hist.maxlen

    # -- the conversion ---------------------------------------------------
    def _encode_depth(self, depth_m: np.ndarray) -> np.ndarray:
        s = self.spec
        d = np.asarray(depth_m, dtype=np.float64)
        if d.shape != (s.sensor.height, s.sensor.width):
            raise ValueError(
                "renderer returned depth %s but the contract says %s (h, w).\n"
                "A policy trained at one resolution has not seen the other; "
                "resizing here would hide that, so it is refused."
                % (d.shape, (s.sensor.height, s.sensor.width)))
        nan_fill = s.clip_far_m if s.nan_fill_m is None else s.nan_fill_m
        if nan_fill is None:
            nan_fill = float(s.sensor.depth.far_m)
        d = np.nan_to_num(d, nan=float(nan_fill), posinf=float(nan_fill),
                          neginf=float(s.neginf_fill_m))
        if s.clip_far_m is not None:
            d = np.clip(d, 0.0, float(s.clip_far_m))
        if s.normalize == "clip_unit":
            d = d / float(s.clip_far_m)
        return d.astype(np.float32)[None]                    # (1, H, W)

    def _encode_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) from the renderer -> (3, H, W) in [0, 1].

        Channels-last in, channels-first out, because every renderer hands back
        an image and every network wants a tensor. Doing it here rather than at
        the call site means there is one place for the transpose to be wrong.
        """
        s = self.spec
        c = np.asarray(rgb, dtype=np.float64)
        want = (s.sensor.height, s.sensor.width, 3)
        if c.shape != want:
            raise ValueError(
                "renderer returned rgb %s but the contract says %s (h, w, 3)."
                % (c.shape, want))
        c = np.nan_to_num(c, nan=0.0, posinf=float(s.rgb_range_max), neginf=0.0)
        c = np.clip(c / float(s.rgb_range_max), 0.0, 1.0)
        return np.ascontiguousarray(c.transpose(2, 0, 1)).astype(np.float32)

    def encode_frame(self, depth_m=None, rgb=None) -> np.ndarray:
        """One render -> one frame, (channels_per_frame, H, W) float32.

        Channel order within a frame is [R, G, B, D], whichever of those the
        contract asks for. Separate from `push` so it can be tested, and so a
        caller wanting the single-frame view for logging or display can have it
        without disturbing the history.
        """
        s = self.spec
        parts = []
        if s.wants_rgb:
            if rgb is None:
                raise ValueError(
                    "the contract asks for %r but the renderer returned no "
                    "colour. FakeRenderer produces RGB; a splat worker only "
                    "does if it was started with an RGB backend."
                    % (s.channels,))
            parts.append(self._encode_rgb(rgb))
        if s.wants_depth:
            if depth_m is None:
                raise ValueError("the contract asks for %r but no depth was "
                                 "supplied" % (s.channels,))
            parts.append(self._encode_depth(depth_m))
        return np.concatenate(parts, axis=0)

    def push(self, depth_m=None, rgb=None) -> np.ndarray:
        """Add a frame and return the stacked observation, (C, H, W) float32."""
        frame = self.encode_frame(depth_m, rgb)
        if not self._hist:
            if self.spec.prime == "repeat_first":
                for _ in range(self._hist.maxlen):
                    self._hist.append(frame.copy())
            else:                                    # "zeros"
                for _ in range(self._hist.maxlen - 1):
                    self._hist.append(np.zeros_like(frame))
                self._hist.append(frame.copy())
        else:
            self._hist.append(frame.copy())

        frames = list(self._hist)
        if self.spec.history_order == "newest_first":
            frames = frames[::-1]
        # concatenate, not stack: each frame is already (C_per_frame, H, W), so
        # the history lands on the channel axis and a depth-only contract keeps
        # exactly the (history, H, W) it has always had.
        obs = np.concatenate(frames, axis=0).astype(np.float32)
        if obs.shape != self.spec.shape:
            raise AssertionError("built %s, contract says %s"
                                 % (obs.shape, self.spec.shape))
        return obs

    def __repr__(self) -> str:
        return ("ObservationBuilder(%s, %d/%d frames held)"
                % (self.spec.shape, len(self._hist), self._hist.maxlen))
