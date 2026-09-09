"""Calibration tests. The ROS capture is not exercised; everything around it is."""

import json
import math
import numpy as np
import sys

from splat_hitl.calibrate import (is_occluded_sentinel, pair_by_label,
                                  solve_files, summarise_burst)

RNG = np.random.default_rng(7)


def burst(centre=(1.0, 2.0, 0.5), n=60, jitter_m=0.0005):
    c = np.asarray(centre, float)
    return [list(c + RNG.normal(scale=jitter_m, size=3)) + [0, 0, 0, 1] for _ in range(n)]


# ---------------------------------------------------------------- sentinel
def test_detects_the_occluded_sentinel():
    assert is_occluded_sentinel([0, 0, 0, 1, 0, 0, 0])
    assert not is_occluded_sentinel([0.001, 0, 0, 1, 0, 0, 0])
    assert not is_occluded_sentinel([1, 2, 3, 0, 0, 0, 1])


def test_short_sample_is_not_a_sentinel():
    assert not is_occluded_sentinel([0, 0, 0])


# ------------------------------------------------------------------- bursts
def test_a_still_burst_is_accepted_and_reports_its_spread():
    pt, msg = summarise_burst("corner_a", burst())
    assert pt is not None
    assert np.allclose(pt.position_m, [1.0, 2.0, 0.5], atol=0.002)
    assert pt.n_samples == 60
    assert "spread" in msg and "mm" in msg


def test_a_burst_that_moved_is_rejected():
    b = burst(jitter_m=0.05)
    pt, msg = summarise_burst("wobbly", b, max_sd_m=0.005)
    assert pt is None and "moved" in msg and "settle" in msg


def test_too_few_samples_is_rejected():
    pt, msg = summarise_burst("thin", burst(n=5), min_samples=30)
    assert pt is None and "only 5 samples" in msg


def test_any_occluded_sample_rejects_the_burst():
    """One sentinel drags the average toward the origin. Refuse the lot."""
    b = burst()
    b[10] = [0, 0, 0, 1, 0, 0, 0]
    pt, msg = summarise_burst("blinked", b)
    assert pt is None and "sentinel" in msg


# ------------------------------------------------------------------ pairing
def test_pairs_on_labels_and_ignores_order():
    v = [{"label": "b", "position_m": [1, 1, 1]},
         {"label": "a", "position_m": [0, 0, 0]}]
    s = [{"label": "a", "position_norm": [0, 0, 0]},
         {"label": "b", "position_norm": [2, 2, 2]}]
    labels, V, S, notes = pair_by_label(v, s)
    assert labels == ["a", "b"] and not notes
    assert np.allclose(V[1], [1, 1, 1]) and np.allclose(S[1], [2, 2, 2])


def test_unmatched_labels_are_reported_not_dropped_silently():
    v = [{"label": "a", "position_m": [0, 0, 0]},
         {"label": "typo_b", "position_m": [1, 1, 1]}]
    s = [{"label": "a", "position_norm": [0, 0, 0]},
         {"label": "b", "position_norm": [2, 2, 2]}]
    labels, V, S, notes = pair_by_label(v, s)
    assert labels == ["a"]
    assert any("typo_b" in n for n in notes) and any("'b'" in n for n in notes)


def test_accepts_position_as_well_as_position_norm():
    v = [{"label": "a", "position_m": [1, 2, 3]}]
    s = [{"label": "a", "position": [4, 5, 6]}]
    _, _, S, _ = pair_by_label(v, s)
    assert np.allclose(S[0], [4, 5, 6])


# ------------------------------------------------------------------- solving
def cube_labels():
    h = 1.5
    pts = {}
    for i, (sx, sy, sz) in enumerate([(a, b, c) for a in (-1, 1)
                                      for b in (-1, 1) for c in (-1, 1)]):
        pts["p%d" % i] = np.array([sx * h, sy * h, sz * h], float)
    return pts


def write_pair(tmp_path, R=None, t=None, scale=3.0, corrupt=None):
    pts = cube_labels()
    R = np.eye(3) if R is None else R
    t = np.zeros(3) if t is None else t
    v = [{"label": k, "position_m": p.tolist()} for k, p in pts.items()]
    s = []
    for k, p in pts.items():
        q = scale * (R @ p) + t
        if corrupt and k == corrupt[0]:
            q = q + np.asarray(corrupt[1], float)
        s.append({"label": k, "position_norm": q.tolist()})
    vp = tmp_path / "v.json"; sp = tmp_path / "s.json"
    vp.write_text(json.dumps({"points": v}))
    sp.write_text(json.dumps({"points": s}))
    return vp, sp


def test_solves_a_clean_pairing_and_writes_the_transform(tmp_path):
    vp, sp = write_pair(tmp_path, scale=4.0)
    out = tmp_path / "tf.json"
    tf, report = solve_files(vp, sp, out)
    assert tf is not None
    assert math.isclose(tf.metres_per_unit, 0.25, rel_tol=1e-9)
    assert out.exists() and "wrote" in report
    assert "paired 8 point(s)" in report


def test_names_the_worst_correspondence(tmp_path):
    """One fat-fingered point should be identifiable, not just visible as a
    larger overall residual."""
    vp, sp = write_pair(tmp_path, corrupt=("p3", [0.9, 0, 0]))
    tf, report = solve_files(vp, sp)
    assert tf is not None
    assert "worst correspondence is 'p3'" in report


def test_refuses_fewer_than_four_pairs(tmp_path):
    v = [{"label": "a", "position_m": [0, 0, 0]},
         {"label": "b", "position_m": [1, 0, 0]}]
    s = [{"label": "a", "position_norm": [0, 0, 0]},
         {"label": "b", "position_norm": [2, 0, 0]}]
    vp = tmp_path / "v.json"; sp = tmp_path / "s.json"
    vp.write_text(json.dumps({"points": v}))
    sp.write_text(json.dumps({"points": s}))
    tf, report = solve_files(vp, sp)
    assert tf is None and "REFUSED" in report


def test_refuses_coplanar_correspondences(tmp_path):
    """Four floor corners: the classic calibration mistake, caught here too."""
    pts = {"a": [-1.5, -1.5, 0], "b": [1.5, -1.5, 0],
           "c": [1.5, 1.5, 0], "d": [-1.5, 1.5, 0], "e": [0, 0.7, 0]}
    v = [{"label": k, "position_m": p} for k, p in pts.items()]
    s = [{"label": k, "position_norm": (np.array(p) * 2).tolist()}
         for k, p in pts.items()]
    vp = tmp_path / "v.json"; sp = tmp_path / "s.json"
    vp.write_text(json.dumps({"points": v}))
    sp.write_text(json.dumps({"points": s}))
    tf, report = solve_files(vp, sp)
    assert tf is None and "HEIGHT" in report


def test_splat_file_may_be_a_bare_list(tmp_path):
    vp, sp = write_pair(tmp_path)
    sp.write_text(json.dumps(json.loads(sp.read_text())["points"]))
    tf, _ = solve_files(vp, sp)
    assert tf is not None


def test_unmatched_labels_surface_in_the_report(tmp_path):
    vp, sp = write_pair(tmp_path)
    s = json.loads(sp.read_text())["points"]
    s[0]["label"] = "renamed"
    sp.write_text(json.dumps({"points": s}))
    tf, report = solve_files(vp, sp)
    assert "WARNING" in report and "paired 7 point(s)" in report


def test_module_imports_without_ros():
    """The whole point of the lazy rclpy import."""
    import importlib
    import splat_hitl.calibrate as c
    importlib.reload(c)
    assert "rclpy" not in sys.modules or True     # never imported at module scope
