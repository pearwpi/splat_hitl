"""Frame and registration tests.

These run with no GPU, no ROS, no drone and no splat. That is deliberate: this
is the layer where a mistake is silent in flight, so it has to be provable on a
laptop.
"""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splat_hitl.frames import (SplatTransform, enu_to_ned, matrix_to_quat,
                               matrix_to_rpy, ned_to_enu, quat_to_matrix,
                               rpy_to_matrix)
from splat_hitl import registration

RNG = np.random.default_rng(20260904)


def random_rotation(rng):
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


# ------------------------------------------------------------------ rotations
def test_quat_matrix_round_trip():
    for _ in range(200):
        R = random_rotation(RNG)
        q = matrix_to_quat(R)
        assert np.allclose(quat_to_matrix(*q), R, atol=1e-9)


def test_quat_normalises_input():
    R1 = quat_to_matrix(0.0, 0.0, 0.3826834, 0.9238795)
    R2 = quat_to_matrix(0.0, 0.0, 3.826834, 9.238795)      # same, unnormalised
    assert np.allclose(R1, R2, atol=1e-9)


def test_quat_w_last_not_first():
    """A 90 deg yaw must rotate +x onto +y. Catches a w-first mix-up."""
    R = quat_to_matrix(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    assert np.allclose(R @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-9)


def test_zero_quaternion_rejected():
    with pytest.raises(ValueError):
        quat_to_matrix(0.0, 0.0, 0.0, 0.0)


def test_rpy_round_trip():
    for _ in range(200):
        r = RNG.uniform(-math.pi, math.pi)
        p = RNG.uniform(-1.3, 1.3)              # stay clear of gimbal lock
        y = RNG.uniform(-math.pi, math.pi)
        back = matrix_to_rpy(rpy_to_matrix(r, p, y))
        assert np.allclose(back, [r, p, y], atol=1e-9)


def test_rpy_gimbal_lock_is_finite():
    for pitch in (math.pi / 2, -math.pi / 2):
        rpy = matrix_to_rpy(rpy_to_matrix(0.4, pitch, 1.1))
        assert np.all(np.isfinite(rpy))
        assert abs(abs(rpy[1]) - math.pi / 2) < 1e-6


def test_enu_ned_involution():
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(ned_to_enu(enu_to_ned(v)), v)
    assert np.allclose(enu_to_ned(v), [2.0, 1.0, -3.0])


# ------------------------------------------------------------ SplatTransform
def test_identity_metres_scale_semantics():
    """scale_to_metres is normalised->metres; the transform's scale is forward."""
    tf = SplatTransform.identity_metres(0.25)      # 1 unit = 0.25 m
    assert math.isclose(tf.metres_per_unit, 0.25)
    assert np.allclose(tf.point_to_splat([1.0, 0, 0]).reshape(3), [4.0, 0, 0])


def test_point_round_trip():
    R, t, s = random_rotation(RNG), RNG.normal(size=3), 3.7
    tf = SplatTransform(R, t, s)
    p = RNG.normal(size=(50, 3))
    assert np.allclose(tf.point_to_vicon(tf.point_to_splat(p)), p, atol=1e-9)


def test_inverse_is_inverse():
    tf = SplatTransform(random_rotation(RNG), RNG.normal(size=3), 2.5)
    inv = tf.inverse()
    p = RNG.normal(size=(20, 3))
    assert np.allclose(inv.point_to_splat(tf.point_to_splat(p)), p, atol=1e-9)


def test_rejects_non_orthonormal_rotation():
    with pytest.raises(ValueError, match="orthonormal"):
        SplatTransform(np.eye(3) * 1.5, np.zeros(3), 1.0)


def test_rejects_reflection():
    """A mirrored scene renders plausibly and is wrong. Refuse it."""
    R = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="reflection"):
        SplatTransform(R, np.zeros(3), 1.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_bad_scale(bad):
    with pytest.raises(ValueError):
        SplatTransform(np.eye(3), np.zeros(3), bad)


def test_pose_to_splat_composes_orientation():
    """Yawing the drone 90 deg in a frame yawed 90 deg gives 180 deg total."""
    R_scene = rpy_to_matrix(0, 0, math.pi / 2)
    tf = SplatTransform(R_scene, np.zeros(3), 1.0)
    q = matrix_to_quat(rpy_to_matrix(0, 0, math.pi / 2))
    _, rpy = tf.pose_to_splat([0, 0, 0], q)
    assert abs(abs(rpy[2]) - math.pi) < 1e-9


def test_save_load_round_trip(tmp_path):
    tf = SplatTransform(random_rotation(RNG), RNG.normal(size=3), 1.9,
                        rms_residual_units=0.01, n_points=7)
    p = tmp_path / "tf.json"
    tf.save(p)
    back = SplatTransform.load(p)
    assert np.allclose(back.R, tf.R) and np.allclose(back.t, tf.t)
    assert math.isclose(back.scale, tf.scale)
    assert json.loads(p.read_text())["n_points"] == 7


# ------------------------------------------------------------- registration
def volume_points(n, rng, extent=3.0):
    """Points filling a cube -- the geometry a good calibration collects.

    Corners first, then interior samples. Purely random points are a poor model
    of a real calibration: you deliberately visit the extremes of the volume,
    and with few samples pure uniform draws are often accidentally thin in one
    axis.
    """
    h = extent / 2.0
    corners = np.array([[sx * h, sy * h, sz * h]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                       dtype=float)
    if n <= len(corners):
        return corners[:n]
    return np.vstack([corners, rng.uniform(-h, h, size=(n - len(corners), 3))])


def test_recovers_known_transform_exactly():
    for _ in range(50):
        R, t, s = random_rotation(RNG), RNG.normal(size=3) * 2, RNG.uniform(0.2, 5.0)
        X = volume_points(12, RNG)
        Y = (s * (R @ X.T)).T + t
        res = registration.solve(X, Y)
        assert res.ok, res.detail
        assert np.allclose(res.transform.R, R, atol=1e-8)
        assert np.allclose(res.transform.t, t, atol=1e-8)
        assert math.isclose(res.transform.scale, s, rel_tol=1e-8)
        assert res.rms_m < 1e-8


def test_residual_is_reported_in_metres():
    """A known 1 cm perturbation must surface as ~1 cm, whatever the scale."""
    R, t, s = random_rotation(RNG), RNG.normal(size=3), 4.0   # 1 unit = 0.25 m
    X = volume_points(40, RNG)
    Y = (s * (R @ X.T)).T + t
    noise_m = 0.01
    Y += RNG.normal(scale=noise_m * s, size=Y.shape)          # 1 cm, in units
    res = registration.solve(X, Y)
    assert res.ok
    assert 0.5 * noise_m < res.rms_m < 2.5 * noise_m, res.rms_m


def test_refuses_colinear_points():
    d = np.array([1.0, 0.5, -0.2]); d /= np.linalg.norm(d)
    X = np.outer(np.linspace(-2, 2, 8), d)
    Y = X * 2.0
    res = registration.solve(X, Y)
    assert not res.ok and res.condition == "colinear"
    assert "unconstrained" in res.detail


def test_refuses_coplanar_points():
    """Four corners of the floor -- the classic calibration mistake."""
    X = np.array([[-1.5, -1.5, 0.0], [1.5, -1.5, 0.0],
                  [1.5, 1.5, 0.0], [-1.5, 1.5, 0.0],
                  [0.0, 0.8, 0.0], [-0.7, 0.0, 0.0]])
    res = registration.solve(X, X * 3.0)
    assert not res.ok and res.condition == "coplanar"
    assert "HEIGHT" in res.detail


def test_refuses_too_few_points():
    X = volume_points(3, RNG)
    res = registration.solve(X, X)
    assert not res.ok and res.condition == "too_few"


def test_allow_degenerate_escape_hatch():
    X = np.array([[-1.5, -1.5, 0.0], [1.5, -1.5, 0.0],
                  [1.5, 1.5, 0.0], [-1.5, 1.5, 0.0]])
    assert registration.solve(X, X * 3.0, allow_degenerate=True).ok


def test_never_returns_a_reflection():
    """Mirrored correspondences must not yield a mirrored 'solution'."""
    X = volume_points(30, RNG)
    Y = X.copy()
    Y[:, 2] *= -1.0                       # deliberate mirror
    res = registration.solve(X, Y)
    assert res.ok, res.detail
    assert np.linalg.det(res.transform.R) > 0
    assert res.rms_m > 0.1, "a mirror should NOT fit well"


def test_mismatched_counts_raise():
    with pytest.raises(ValueError, match="mismatch"):
        registration.solve(volume_points(5, RNG), volume_points(4, RNG))


def test_report_is_readable():
    X = volume_points(10, RNG)
    R, t, s = random_rotation(RNG), np.zeros(3), 2.0
    Y = (s * (R @ X.T)).T + t
    txt = registration.solve(X, Y).report()
    assert "residual" in txt and "mm" in txt
    bad = registration.solve(volume_points(3, RNG), volume_points(3, RNG)).report()
    assert "REFUSED" in bad


def test_marginal_geometry_solves_but_warns():
    """Thin in z but not degenerate: solvable, and the user must be told."""
    h = 1.5
    X = np.array([[sx * h, sy * h, sz * 0.22]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                 dtype=float)
    res = registration.solve(X, X * 2.0)
    assert res.ok, res.detail
    assert res.condition == "marginal"
    assert "extrapolate poorly" in res.detail
    assert "WARNING" in res.report()


def test_condition_bands_are_ordered():
    """As the point set flattens: ok -> marginal -> coplanar."""
    seen = []
    for z in (1.5, 0.25, 0.02):
        h = 1.5
        X = np.array([[sx * h, sy * h, sz * z]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                     dtype=float)
        seen.append(registration.check_geometry(X)[0])
    assert seen == ["ok", "marginal", "coplanar"], seen
