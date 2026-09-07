import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splat_hitl.sensor import DepthEncoding, SensorModel


def model(**kw):
    base = dict(name="test", width=160, height=120, fov_x_deg=90.0)
    base.update(kw)
    return SensorModel(**base)


# ------------------------------------------------------------------ geometry
def test_fov_y_from_aspect():
    m = model(width=160, height=120, fov_x_deg=90.0)
    assert math.isclose(m.aspect, 4 / 3)
    # tan(fovy/2) = tan(45 deg) / (4/3)
    expect = math.degrees(2 * math.atan(math.tan(math.radians(45)) / (4 / 3)))
    assert math.isclose(m.fov_y_deg, expect, rel_tol=1e-12)


def test_square_image_has_equal_fovs():
    m = model(width=200, height=200, fov_x_deg=70.0)
    assert math.isclose(m.fov_y_deg, 70.0, rel_tol=1e-12)


def test_intrinsics_match_fov():
    m = model(width=640, height=480, fov_x_deg=90.0)
    k = m.intrinsics()
    assert math.isclose(k["fx"], 320.0, rel_tol=1e-9)   # tan(45)=1
    assert math.isclose(k["cx"], 320.0) and math.isclose(k["cy"], 240.0)
    assert math.isclose(k["fx"], k["fy"])               # square pixels


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize("kw", [
    dict(width=0), dict(height=-1), dict(fov_x_deg=0.0),
    dict(fov_x_deg=180.0), dict(rate_hz=0.0), dict(stereo_baseline_m=0.0),
])
def test_rejects_impossible_configs(kw):
    with pytest.raises(ValueError):
        model(**kw)


def test_rejects_unknown_depth_encoding():
    with pytest.raises(ValueError, match="not in"):
        DepthEncoding(kind="whatever_the_student_wrote")


def test_rejects_inverted_depth_range():
    with pytest.raises(ValueError, match="near_m < far_m"):
        DepthEncoding(near_m=10.0, far_m=1.0)


def test_empty_depth_must_lie_in_range():
    with pytest.raises(ValueError, match="within"):
        DepthEncoding(near_m=0.3, far_m=10.0, empty_depth_m=24.0)


# --------------------------------------------------------------- fingerprint
def test_fingerprint_is_stable_across_instances():
    assert model().fingerprint() == model().fingerprint()


def test_fingerprint_ignores_name_and_notes():
    a = model(name="training", notes="built on the cluster")
    b = model(name="hitl", notes="lab pc")
    assert a.fingerprint() == b.fingerprint()
    a.assert_compatible(b)


@pytest.mark.parametrize("kw", [
    dict(width=161), dict(height=121), dict(fov_x_deg=89.0),
    dict(mount_pitch_deg=10.0), dict(mount_yaw_deg=5.0),
    dict(stereo_baseline_m=0.065), dict(rate_hz=15.0),
    dict(depth=DepthEncoding(kind="inverted_uint8")),
    dict(depth=DepthEncoding(near_m=0.5)),
    dict(depth=DepthEncoding(far_m=20.0)),
    dict(depth=DepthEncoding(empty_depth_m=10.0)),
])
def test_every_visible_property_changes_the_fingerprint(kw):
    """If a property can change what the policy sees, it must change the hash."""
    assert model(**kw).fingerprint() != model().fingerprint()


def test_incompatible_models_raise_with_a_useful_message():
    a = model(name="train", fov_x_deg=90.0)
    b = model(name="fly", fov_x_deg=75.0)
    with pytest.raises(ValueError) as e:
        a.assert_compatible(b)
    msg = str(e.value)
    assert "train" in msg and "fly" in msg
    assert "never seen" in msg


# -------------------------------------------------------------- persistence
def test_save_load_round_trip(tmp_path):
    m = model(name="course_default", mount_pitch_deg=10.0,
              stereo_baseline_m=0.065,
              depth=DepthEncoding(kind="inverse_diffphys", near_m=0.3, far_m=24.0))
    p = tmp_path / "sensor.json"
    m.save(p)
    back = SensorModel.load(p)
    assert back == m
    assert back.fingerprint() == m.fingerprint()


def test_saved_file_carries_derived_values_for_humans(tmp_path):
    p = tmp_path / "s.json"
    model().save(p)
    d = json.loads(p.read_text())
    assert "fingerprint" in d
    assert "fov_y_deg" in d["derived"] and "intrinsics_px" in d["derived"]


def test_load_tolerates_derived_fields(tmp_path):
    """Round-tripping through the human-readable form must not break."""
    p = tmp_path / "s.json"
    model().save(p)
    d = json.loads(p.read_text())
    assert SensorModel.from_dict(d).fingerprint() == model().fingerprint()


def test_stereo_flag():
    assert not model().is_stereo
    assert model(stereo_baseline_m=0.065).is_stereo
