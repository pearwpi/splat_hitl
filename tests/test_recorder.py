import csv
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splat_hitl.collision import CollisionMonitor, synthetic_room
from splat_hitl.commands import Clamped, HoverCommand
from splat_hitl.frames import SplatTransform
from splat_hitl.gates import Gate, GateCourse
from splat_hitl.policy import GateSeekPolicy
from splat_hitl.recorder import TERMINATION_MAP, RunRecorder
from splat_hitl.renderer import FakeRenderer
from splat_hitl.runtime import (PoseSample, Runtime, RuntimeConfig,
                                ScriptedPoseSource, TickResult)
from splat_hitl.sensor import SensorModel


def pose(t, x, z=1.25):
    return PoseSample(t, np.array([x, 1.5, z], float), (0.0, 0.0, 0.0, 1.0))


def tick(state="running", vx=0.5, clamped=False):
    rep = Clamped(); rep.speed = clamped
    return TickResult(state, HoverCommand(vx, 0.0, 0.1, 0.6), rep,
                      events=["hello"], pose_age_s=0.004,
                      render_s=0.005, policy_s=0.002, total_s=0.009)


# ------------------------------------------------------------- terminations
@pytest.mark.parametrize("reason,expect", [
    ("course_complete", "goal"),
    ("virtual_collision", "collision"),
    ("virtual_outside", "out_of_bounds"),
    ("max_duration", "timeout"),
    ("policy_error", "other"),
    ("landed", "other"),
    ("render_error", "other"),
])
def test_termination_vocabulary_matches_metric_splat(reason, expect):
    r = RunRecorder("r1", "p")
    r.finish(reason)
    assert r.termination == expect
    assert r.to_dict()["reason"] == reason        # exact reason is preserved


def test_unfinished_run_is_other():
    assert RunRecorder("r1", "p").termination == "other"


# -------------------------------------------------------------- accumulation
def test_rows_capture_pose_and_command():
    r = RunRecorder("r1", "p")
    r.add(0.0, pose(0.0, 1.0), tick(vx=0.7))
    row = r.rows[0]
    assert row["x_m"] == 1.0 and row["z_m"] == 1.25
    assert row["cmd_vx"] == 0.7 and row["state"] == "running"
    assert row["render_ms"] == 5.0 and row["tick_ms"] == 9.0


def test_a_tick_with_no_command_records_nulls():
    r = RunRecorder("r1", "p")
    r.add(0.0, pose(0.0, 1.0), TickResult("finished", None))
    assert r.rows[0]["cmd_vx"] is None


def test_events_carry_their_step_and_time():
    r = RunRecorder("r1", "p")
    r.add(1.5, pose(1.5, 1.0), tick())
    assert r.events[0] == {"t_s": 1.5, "step": 1, "text": "hello"}


def test_clamping_is_recorded():
    r = RunRecorder("r1", "p")
    r.add(0.0, pose(0.0, 1.0), tick(clamped=True))
    r.add(0.1, pose(0.1, 1.1), tick(clamped=False))
    assert [x["clamped"] for x in r.rows] == [True, False]
    assert "clamped" in r.summary()


def test_gates_passed_is_carried_into_rows():
    r = RunRecorder("r1", "p")
    r.note_gates(2)
    r.add(0.0, pose(0.0, 1.0), tick())
    assert r.rows[0]["gates_passed"] == 2
    assert r.to_dict()["gates_passed"] == 2


# ------------------------------------------------------------------- timing
def test_timing_percentiles_are_reported():
    r = RunRecorder("r1", "p")
    for i in range(20):
        t = tick(); t.total_s = 0.001 * (i + 1)
        r.add(i * 0.1, pose(i * 0.1, 1.0), t)
    d = r.to_dict()["timing"]
    assert d["tick_ms_max"] == 20.0
    assert d["tick_ms_median"] > 0 and d["tick_ms_p95"] >= d["tick_ms_median"]
    assert "tick median" in r.summary()


def test_timing_is_empty_for_an_empty_run():
    assert RunRecorder("r1", "p").to_dict()["timing"] == {}


# ----------------------------------------------------------------- provenance
def test_sensor_fingerprint_is_recorded():
    s = SensorModel(name="x", width=64, height=48, fov_x_deg=90.0)
    r = RunRecorder("r1", "p", sensor_fingerprint=s.fingerprint())
    assert r.to_dict()["sensor_fingerprint"] == s.fingerprint()
    assert s.fingerprint() in r.summary()


def test_calibration_quality_is_recorded_with_the_run():
    tf = SplatTransform(np.eye(3), np.zeros(3), 4.0,
                        rms_residual_units=0.04, n_points=6)
    r = RunRecorder("r1", "p", transform=tf)
    cal = r.to_dict()["calibration"]
    assert math.isclose(cal["metres_per_unit"], 0.25)
    assert math.isclose(cal["rms_residual_m"], 0.01)     # 0.04 units * 0.25
    assert cal["n_points"] == 6
    assert "calibration residual 10 mm" in r.summary()


def test_normalised_trajectory_is_emitted_only_with_a_transform():
    r = RunRecorder("r1", "p")
    r.add(0.0, pose(0.0, 1.0), tick())
    assert "trajectory_norm" not in r.to_dict()

    tf = SplatTransform(np.eye(3), np.zeros(3), 2.0)
    r2 = RunRecorder("r1", "p", transform=tf)
    r2.add(0.0, pose(0.0, 1.0), tick())
    d = r2.to_dict()
    assert np.allclose(d["trajectory_norm"][0], [2.0, 3.0, 2.5])
    assert np.allclose(d["trajectory_raw_m"][0], [1.0, 1.5, 1.25])


# --------------------------------------------------------------------- files
def test_json_round_trips(tmp_path):
    r = RunRecorder("run7", "gate_seek", scene="arena_a")
    r.add(0.0, pose(0.0, 1.0), tick())
    r.finish("course_complete")
    p = tmp_path / "run.json"
    r.save_json(p)
    d = json.loads(p.read_text())
    assert d["run_id"] == "run7" and d["termination"] == "goal"
    assert d["scene"] == "arena_a" and len(d["trajectory_raw_m"]) == 1


def test_csv_has_a_header_and_one_row_per_step(tmp_path):
    r = RunRecorder("r1", "p")
    for i in range(3):
        r.add(i * 0.1, pose(i * 0.1, 1.0 + i), tick())
    p = tmp_path / "run.csv"
    r.save_csv(p)
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 3
    assert rows[1]["x_m"] == "2.0" and "tick_ms" in rows[0]


def test_extra_fields_pass_through_but_private_ones_do_not():
    r = RunRecorder("r1", "p", extra={"battery_v": 4.1})
    r.note_gates(3)
    d = r.to_dict()
    assert d["battery_v"] == 4.1
    assert "_gates_passed" not in d


# ---------------------------------------------------------------- integration
def test_records_a_real_runtime_flight(tmp_path):
    """Drive the actual loop and check the log describes what happened."""
    sensor = SensorModel(name="t", width=32, height=24, fov_x_deg=90.0)
    course = GateCourse([Gate("a", np.array([2.0, 1.5, 1.25]),
                              np.array([1.0, 0, 0]), 1.2, 1.2)])
    esdf = synthetic_room(size_m=(4.0, 3.0, 2.5), voxel_m=0.05)
    samples = [pose(i * 0.1, 1.0 + i * 0.05) for i in range(30)]
    src = ScriptedPoseSource(samples)
    rt = Runtime(src, FakeRenderer(sensor, (4.0, 3.0, 2.5)), GateSeekPolicy(),
                 course, CollisionMonitor(esdf, 0.10), RuntimeConfig())
    rec = RunRecorder("live", "gate_seek", sensor.fingerprint())

    for _ in range(30):
        src.advance()
        s = src.samples[src.i]
        r = rt.step(s.t_s)
        rec.note_gates(course.passed)
        rec.add(s.t_s, s, r)
        if r.finished:
            rec.finish(r.reason)
            break

    assert rec.reason == "course_complete"
    assert rec.termination == "goal"
    assert rec.to_dict()["gates_passed"] == 1
    assert len(rec.rows) > 3
    rec.save_json(tmp_path / "r.json")
    rec.save_csv(tmp_path / "r.csv")
    assert (tmp_path / "r.json").exists() and (tmp_path / "r.csv").exists()
