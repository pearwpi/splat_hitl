"""Run logs, in a shape `metric-splat`'s existing tooling already understands.

WHY THE SCHEMA IS BORROWED RATHER THAN INVENTED
------------------------------------------------
`benchmark_splat_policy_comparison.py` already produces per-rollout batch logs,
`scene_visualization.py` already draws trajectory overlays from them, and
`success_rates.txt` already tabulates them. If a HITL run emits the same shape,
a real flight and a simulated rollout land in the same figure and the same
table, and "did the policy that worked in sim work on the drone" becomes a
diff rather than a research project.

So terminations are mapped onto that vocabulary:

    course_complete   -> goal
    virtual_collision -> collision
    virtual_outside   -> out_of_bounds
    max_duration      -> timeout
    everything else   -> other      (the precise reason is kept alongside)

WHAT ELSE GOES IN, AND WHY
--------------------------
The sensor fingerprint and the calibration residual. Six months from now the
question about any surprising result will be "was this flown through the sensor
model it was trained on, and how good was the registration" -- and if the answer
is not in the file, it is gone.
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["RunRecorder", "TERMINATION_MAP"]

TERMINATION_MAP = {
    "course_complete": "goal",
    "virtual_collision": "collision",
    "virtual_outside": "out_of_bounds",
    "max_duration": "timeout",
}

_CSV_FIELDS = ["t_s", "x_m", "y_m", "z_m", "yaw_deg", "state",
               "cmd_vx", "cmd_vy", "cmd_yaw_rate", "cmd_z",
               "clamped", "gates_passed", "pose_age_ms", "latency_ms",
               "render_ms", "policy_ms", "tick_ms"]


class RunRecorder:
    """Accumulates one run and writes it out.

    Everything is stored in RAW METRES. The normalised trajectory is emitted
    alongside only when a transform is supplied, so a log is never ambiguous
    about which frame a number is in -- the failure that the whole pipeline's
    unit discipline exists to prevent.
    """

    def __init__(self, run_id: str, policy_name: str,
                 sensor_fingerprint: Optional[str] = None,
                 transform: Optional[Any] = None,
                 scene: Optional[str] = None,
                 extra: Optional[Dict[str, Any]] = None):
        self.run_id = run_id
        self.policy_name = policy_name
        self.sensor_fingerprint = sensor_fingerprint
        self.transform = transform
        self.scene = scene
        self.extra = dict(extra or {})
        self.started_at = time.time()
        self.rows: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.reason: Optional[str] = None

    # -- accumulation ------------------------------------------------------
    def add(self, t_s: float, pose, tick) -> None:
        """One step. `pose` is a PoseSample, `tick` a TickResult."""
        c = tick.command
        self.rows.append({
            "t_s": float(t_s),
            "x_m": float(pose.position_m[0]),
            "y_m": float(pose.position_m[1]),
            "z_m": float(pose.position_m[2]),
            "yaw_deg": float(math.degrees(pose.yaw_rad)),
            "state": tick.state,
            "cmd_vx": None if c is None else float(c.vx),
            "cmd_vy": None if c is None else float(c.vy),
            "cmd_yaw_rate": None if c is None else float(c.yaw_rate),
            "cmd_z": None if c is None else float(c.z_distance),
            "clamped": bool(tick.clamped.any) if tick.clamped else False,
            "gates_passed": int(self.extra.get("_gates_passed", 0)),
            "pose_age_ms": None if not math.isfinite(tick.pose_age_s)
                           else round(tick.pose_age_s * 1000.0, 2),
            "latency_ms": None if not math.isfinite(getattr(tick, "latency_s", float("nan")))
                          else round(tick.latency_s * 1000.0, 2),
            "render_ms": round(tick.render_s * 1000.0, 2),
            "policy_ms": round(tick.policy_s * 1000.0, 2),
            "tick_ms": round(tick.total_s * 1000.0, 2),
        })
        for e in tick.events:
            self.events.append({"t_s": float(t_s), "step": len(self.rows), "text": e})

    def note_gates(self, passed: int) -> None:
        self.extra["_gates_passed"] = int(passed)

    def finish(self, reason: str) -> None:
        self.reason = reason

    # -- derived -----------------------------------------------------------
    @property
    def termination(self) -> str:
        return TERMINATION_MAP.get(self.reason or "", "other")

    def _timing(self) -> Dict[str, float]:
        if not self.rows:
            return {}
        t = [r["tick_ms"] for r in self.rows]
        r_ = [r["render_ms"] for r in self.rows]
        p = [r["policy_ms"] for r in self.rows]
        q = lambda v, f: float(np.percentile(v, f)) if v else 0.0
        return {"tick_ms_median": q(t, 50), "tick_ms_p95": q(t, 95),
                "tick_ms_max": max(t),
                "render_ms_median": q(r_, 50), "render_ms_p95": q(r_, 95),
                "policy_ms_median": q(p, 50), "policy_ms_p95": q(p, 95)}

    def to_dict(self) -> Dict[str, Any]:
        traj_m = [[r["x_m"], r["y_m"], r["z_m"]] for r in self.rows]
        d: Dict[str, Any] = {
            "_comment": "HITL run. Positions in RAW METRES unless the key says "
                        "otherwise. `termination` uses metric-splat's vocabulary "
                        "so this drops into its tooling; `reason` is the exact one.",
            "run_id": self.run_id,
            "started_at": self.started_at,
            "policy": self.policy_name,
            "scene": self.scene,
            "sensor_fingerprint": self.sensor_fingerprint,
            "termination": self.termination,
            "reason": self.reason,
            "steps": len(self.rows),
            "duration_s": (self.rows[-1]["t_s"] - self.rows[0]["t_s"]) if self.rows else 0.0,
            "gates_passed": int(self.extra.get("_gates_passed", 0)),
            "trajectory_raw_m": traj_m,
            "timing": self._timing(),
            "events": self.events,
        }
        if self.transform is not None:
            d["calibration"] = {
                "metres_per_unit": float(self.transform.metres_per_unit),
                "rms_residual_m": float(self.transform.rms_residual_units
                                        * self.transform.metres_per_unit),
                "n_points": int(self.transform.n_points),
            }
            try:
                pts = np.asarray(traj_m, dtype=float)
                if len(pts):
                    d["trajectory_norm"] = self.transform.point_to_splat(pts).tolist()
            except Exception:
                pass
        for k, v in self.extra.items():
            if not k.startswith("_"):
                d[k] = v
        return d

    # -- output ------------------------------------------------------------
    def save_json(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")

    def save_csv(self, path) -> None:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
            w.writeheader()
            for r in self.rows:
                w.writerow({k: r.get(k) for k in _CSV_FIELDS})

    def summary(self) -> str:
        t = self._timing()
        L = ["  run %s: %s (%s)" % (self.run_id, self.reason or "unfinished",
                                    self.termination),
             "  policy %s, %d steps over %.1f s"
             % (self.policy_name, len(self.rows),
                self.to_dict()["duration_s"])]
        if self.sensor_fingerprint:
            L.append("  sensor %s" % self.sensor_fingerprint)
        if self.transform is not None:
            L.append("  calibration residual %.0f mm over %d points"
                     % (self.transform.rms_residual_units
                        * self.transform.metres_per_unit * 1000,
                        self.transform.n_points))
        if t:
            L.append("  tick median %.1f ms, p95 %.1f ms, max %.1f ms"
                     % (t["tick_ms_median"], t["tick_ms_p95"], t["tick_ms_max"]))
        clamped = sum(1 for r in self.rows if r["clamped"])
        if clamped:
            L.append("  %d/%d commands were clamped -- the policy is asking for "
                     "more than the envelope allows" % (clamped, len(self.rows)))
        return "\n".join(L)
