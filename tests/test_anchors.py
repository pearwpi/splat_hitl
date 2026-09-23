"""The anchors a registration was fitted to, and the drift of the frame."""
import json
import math

import numpy as np
import pytest

from splat_hitl.anchors import RESIDUAL_LIMIT_M, AnchorSet, drift
from splat_hitl.frames import SplatTransform

SIX = {"A1": (0.012, 0.002, 0.023), "A2": (-0.397, 1.431, 0.018),
       "A3": (2.613, 1.480, -0.029), "A4": (5.224, 1.440, -0.072),
       "A5": (5.253, -1.361, -0.057), "A6": (2.647, -1.329, -0.023)}


def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    th = math.radians(deg)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def moved(R, t=(0.0, 0.0, 0.0)):
    """SIX as a frame rotated by R would report the same physical points."""
    Ri, ti = R.T, -R.T @ np.asarray(t, float)
    return AnchorSet({k: Ri @ np.asarray(v, float) + ti for k, v in SIX.items()})


# -- the set itself --------------------------------------------------------
def test_too_few_points():
    with pytest.raises(ValueError, match="at least 3"):
        AnchorSet({"A1": (0, 0, 0), "A2": (1, 0, 0)})


def test_collinear_is_refused():
    with pytest.raises(ValueError, match="collinear"):
        AnchorSet({"A%d" % i: (i * 0.5, 0.0, 0.0) for i in range(5)})


@pytest.mark.parametrize("bad", [(0, 0), (0, 0, 0, 0), (0, 0, float("nan"))])
def test_bad_point(bad):
    with pytest.raises(ValueError):
        AnchorSet(dict(SIX, A7=bad))


def test_round_trip(tmp_path):
    a = AnchorSet(SIX, recorded="2026-09-20T13:50Z", source="scene_layouts/anchors")
    p = tmp_path / "anchors.json"
    a.save(p)
    b = AnchorSet.load(p)
    assert b.names == a.names
    assert b.recorded == a.recorded and b.source == a.source
    for k in a.names:
        assert np.allclose(a.positions[k], b.positions[k])
    assert "anchors" in json.loads(p.read_text())


# -- drift -----------------------------------------------------------------
def test_no_drift():
    d = drift(AnchorSet(SIX), AnchorSet(SIX))
    assert d.angle_deg < 1e-9 and d.translation_m < 1e-9
    assert d.rigid and d.moved_m < 1e-9


def test_recovers_a_known_rotation():
    R = rot((0.35, 0.94, 0.0), 0.662)
    d = drift(AnchorSet(SIX), moved(R, (0.002, -0.007, -0.001)))
    assert d.angle_deg == pytest.approx(0.662, abs=1e-6)
    assert d.residual_m < 1e-9 and d.rigid
    assert d.scale == pytest.approx(1.0, abs=1e-9)
    # a rotation is nearly invisible at its pivot and centimetres away from it
    near = d.displacement_at([[0.0, 0.0, 0.0]])[0]
    far = d.displacement_at([[5.25, 1.2, 1.8]])[0]
    assert far > 0.03 and near < 0.01 and far > 5 * near
    grows = d.displacement_at([[x, 0.0, 0.0] for x in (0, 1, 2, 3, 4, 5)])
    assert np.all(np.diff(grows) > 0)


def test_apply_puts_todays_reading_back_where_it_was():
    tf = SplatTransform(rot((0, 0, 1), 137.0), np.array([0.5, -0.2, 0.1]), 0.25)
    R = rot((0.35, 0.94, 0.0), 0.662)
    now = moved(R, (0.002, -0.007, -0.001))
    d = drift(AnchorSet(SIX), now)
    tf2 = d.apply(tf)
    for k in SIX:
        then, today = np.asarray(SIX[k], float), now.positions[k]
        assert np.allclose(tf.point_to_splat(then), tf2.point_to_splat(today), atol=1e-9)
    assert tf2.scale == tf.scale and tf2.metres_per_unit == tf.metres_per_unit


def test_one_anchor_moved_is_not_a_frame_change():
    now = moved(rot((0, 1, 0), 0.5))
    bumped = dict(now.positions)
    bumped["A4"] = bumped["A4"] + np.array([0.04, 0.0, 0.0])
    d = drift(AnchorSet(SIX), AnchorSet(bumped))
    assert d.residual_m > RESIDUAL_LIMIT_M and not d.rigid
    with pytest.raises(ValueError, match="no frame correction can be right"):
        d.apply(SplatTransform(np.eye(3), np.zeros(3), 0.25))
    assert "NOT a rigid frame change" in str(d)


def test_scale_is_reported_and_never_applied():
    now = AnchorSet({k: np.asarray(v, float) * 1.01 for k, v in SIX.items()})
    d = drift(AnchorSet(SIX), now)
    assert d.scale == pytest.approx(1 / 1.01, rel=1e-3)
    assert not d.rigid          # a stretched volume cannot be rotated back


def test_needs_three_shared_names():
    with pytest.raises(ValueError, match="at least 3"):
        drift(AnchorSet(SIX), AnchorSet({("B" + k[1:]): v for k, v in SIX.items()}))


def test_extra_and_missing_names_use_the_overlap():
    now = dict(moved(rot((0, 1, 0), 0.4)).positions)
    now.pop("A6")
    now["A9"] = (9.0, 9.0, 9.0)
    d = drift(AnchorSet(SIX), AnchorSet(now))
    assert set(d.names) == set(SIX) - {"A6"}
    assert d.rigid and d.angle_deg == pytest.approx(0.4, abs=1e-6)
