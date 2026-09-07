"""Collect the Vicon<->splat correspondences, and solve for the transform.

THE WORKFLOW THIS IS SHAPED AROUND
----------------------------------
You are alone in the lab with a drone. You cannot click a point cloud and hold a
drone at the same time. So it is two commands, run at two different times:

  1. IN THE LAB, carrying the drone:

        python3 -m splat_hitl.calibrate collect \\
            --topic /vicon/crazyflie2/crazyflie2 --out vicon_points.json

     Put the drone on a recognisable feature -- a table corner, a floor marking,
     a taped cross -- type a label, press enter, and it averages a burst of
     samples. Repeat for at least four points that FILL the volume, including
     different heights.

  2. AT A DESK, in the point cloud:

     Click the same features with `scene_tools.py` and save them with matching
     labels, then:

        python3 -m splat_hitl.calibrate solve \\
            --vicon vicon_points.json --splat splat_points.json \\
            --out scene_a/vicon_to_splat.json

WHY THE CAPTURE IS FUSSY
------------------------
A correspondence taken while the drone was drifting, or while a marker was
occluded, is a wrong number that looks exactly like a right one -- and the
resulting transform is silently wrong everywhere. So a burst is REJECTED if:

  * the position wandered more than `--max-sd` during the burst,
  * any sample was the SDK's occluded-segment sentinel,
  * fewer than `--min-samples` arrived.

Rejected points are reported and re-taken, not quietly averaged in.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import registration
from .frames import SplatTransform

__all__ = ["CapturedPoint", "summarise_burst", "pair_by_label", "solve_files"]

OCCLUDED_SENTINEL = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)   # x y z qx qy qz qw


@dataclass
class CapturedPoint:
    label: str
    position_m: List[float]
    sd_m: float
    n_samples: int

    def to_dict(self) -> dict:
        return asdict(self)


def is_occluded_sentinel(sample: Sequence[float]) -> bool:
    """The all-zero position with an identity-ish quaternion the SDK emits for
    an occluded segment. Averaging one of these drags a point toward the origin."""
    if len(sample) < 7:
        return False
    return all(abs(a - b) < 1e-12 for a, b in zip(sample[:7], OCCLUDED_SENTINEL))


def summarise_burst(label: str, samples: Sequence[Sequence[float]],
                    max_sd_m: float = 0.005, min_samples: int = 30
                    ) -> Tuple[Optional[CapturedPoint], str]:
    """Average a burst, or explain why it is not usable.

    Returns (point or None, message). The message is always worth printing:
    on success it carries the spread, which is the only evidence that the
    drone was actually still.
    """
    if len(samples) < min_samples:
        return None, ("only %d samples (need %d) -- is the bridge running and "
                      "the body tracked?" % (len(samples), min_samples))
    occluded = sum(1 for s in samples if is_occluded_sentinel(s))
    if occluded:
        return None, ("%d of %d samples were the occluded-segment sentinel; the "
                      "body was not tracked for part of the burst"
                      % (occluded, len(samples)))
    P = np.asarray([s[:3] for s in samples], dtype=float)
    mean = P.mean(axis=0)
    sd = float(np.linalg.norm(P.std(axis=0)))
    if sd > max_sd_m:
        return None, ("moved %.1f mm during the burst (limit %.1f mm) -- let it "
                      "settle, or rest it on the feature rather than holding it"
                      % (sd * 1000, max_sd_m * 1000))
    return (CapturedPoint(label, [float(v) for v in mean], sd, len(samples)),
            "captured at (%+.3f, %+.3f, %+.3f) m, spread %.1f mm over %d samples"
            % (mean[0], mean[1], mean[2], sd * 1000, len(samples)))


def pair_by_label(vicon: Sequence[dict], splat: Sequence[dict]
                  ) -> Tuple[List[str], np.ndarray, np.ndarray, List[str]]:
    """Match the two files on their labels. Unmatched labels are reported, not
    silently dropped: a typo that halves your correspondences is otherwise
    invisible until the residual looks odd."""
    v = {d["label"]: d for d in vicon}
    s = {d["label"]: d for d in splat}
    common = sorted(set(v) & set(s))
    notes = []
    for missing in sorted(set(v) - set(s)):
        notes.append("no splat point labelled %r" % missing)
    for missing in sorted(set(s) - set(v)):
        notes.append("no vicon point labelled %r" % missing)
    V = np.array([v[k]["position_m"] for k in common], dtype=float) if common else np.zeros((0, 3))
    S = np.array([s[k].get("position_norm", s[k].get("position")) for k in common],
                 dtype=float) if common else np.zeros((0, 3))
    return common, V, S, notes


def solve_files(vicon_path, splat_path, out_path=None,
                allow_degenerate: bool = False) -> Tuple[Optional[SplatTransform], str]:
    with open(vicon_path) as fh:
        vicon = json.load(fh)["points"]
    with open(splat_path) as fh:
        raw = json.load(fh)
    splat = raw["points"] if isinstance(raw, dict) else raw

    labels, V, S, notes = pair_by_label(vicon, splat)
    L = []
    for n in notes:
        L.append("  WARNING: %s" % n)
    L.append("  paired %d point(s): %s" % (len(labels), ", ".join(labels) or "none"))
    if len(labels) < 4:
        L.append("  REFUSED: need at least 4 paired points, got %d" % len(labels))
        return None, "\n".join(L)

    res = registration.solve(V, S, allow_degenerate=allow_degenerate)
    L.append(res.report())
    if res.ok:
        # Name the worst point: it is almost always one bad correspondence
        # rather than a diffuse error, and knowing which one saves re-taking
        # all of them.
        worst = int(np.argmax(res.per_point_m))
        L.append("  worst correspondence is %r at %.0f mm"
                 % (labels[worst], res.per_point_m[worst] * 1000))
        if out_path:
            res.transform.save(out_path)
            L.append("  wrote %s" % out_path)
    return res.transform, "\n".join(L)


# --------------------------------------------------------------------- capture
def _collect_samples(topic: str, seconds: float) -> List[List[float]]:
    """Burst of poses from the Vicon topic. rclpy is imported HERE so that the
    rest of this module -- and its tests -- run on a laptop with no ROS."""
    import rclpy                                     # noqa: PLC0415
    from rclpy.node import Node
    from rosidl_runtime_py.utilities import get_message

    started = rclpy.ok()
    if not started:
        rclpy.init()
    node = Node("splat_hitl_calibrate")
    mtype = None
    for _ in range(50):
        for name, types in node.get_topic_names_and_types():
            if name == topic and types:
                mtype = types[0]
                break
        if mtype:
            break
        time.sleep(0.1)
    if not mtype:
        node.destroy_node()
        raise RuntimeError("topic %s not found -- is the bridge running?" % topic)

    out: List[List[float]] = []

    def cb(msg):
        p = getattr(msg, "pose", None)
        if p is not None and hasattr(p, "position"):
            o, q = p.position, p.orientation
        else:
            t = getattr(msg, "transform", None)
            if t is None:
                return
            o, q = t.translation, t.rotation
        out.append([o.x, o.y, o.z, q.x, q.y, q.z, q.w])

    node.create_subscription(get_message(mtype), topic, cb, 50)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.02)
    node.destroy_node()
    if not started:
        rclpy.shutdown()
    return out


def cmd_collect(args) -> int:
    print("\n  Collecting Vicon correspondences from %s" % args.topic)
    print("  Rest the drone ON each feature -- holding it by hand rarely gets")
    print("  under the %.0f mm spread limit. Blank label to finish.\n"
          % (args.max_sd * 1000))
    points: List[CapturedPoint] = []
    while True:
        label = input("  label (blank to finish): ").strip()
        if not label:
            break
        if any(p.label == label for p in points):
            print("    already have %r; use a different label" % label)
            continue
        try:
            samples = _collect_samples(args.topic, args.seconds)
        except Exception as exc:
            print("    capture failed: %s" % exc)
            continue
        pt, msg = summarise_burst(label, samples, args.max_sd, args.min_samples)
        print("    %s" % msg)
        if pt is not None:
            points.append(pt)
            print("    %d point(s) so far" % len(points))

    if len(points) < 4:
        print("\n  Only %d point(s). The solver needs at least 4, not coplanar."
              % len(points))
    with open(args.out, "w") as fh:
        json.dump({"_comment": "Vicon positions in METRES; label these to match "
                               "the points you click in the point cloud",
                   "topic": args.topic,
                   "points": [p.to_dict() for p in points]}, fh, indent=2)
        fh.write("\n")
    print("\n  wrote %s" % args.out)
    return 0


def cmd_solve(args) -> int:
    tf, report = solve_files(args.vicon, args.splat, args.out,
                             allow_degenerate=args.allow_degenerate)
    print("\n" + report + "\n")
    return 0 if tf is not None else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="capture Vicon points in the lab")
    c.add_argument("--topic", required=True)
    c.add_argument("--out", default="vicon_points.json")
    c.add_argument("--seconds", type=float, default=2.0)
    c.add_argument("--max-sd", type=float, default=0.005,
                   help="reject a burst that wandered more than this, metres")
    c.add_argument("--min-samples", type=int, default=30)
    c.set_defaults(func=cmd_collect)

    s = sub.add_parser("solve", help="pair, solve and save the transform")
    s.add_argument("--vicon", required=True)
    s.add_argument("--splat", required=True)
    s.add_argument("--out", default=None)
    s.add_argument("--allow-degenerate", action="store_true")
    s.set_defaults(func=cmd_solve)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
