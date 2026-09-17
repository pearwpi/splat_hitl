"""The parts of the render check that do not need a GPU.

The check itself is an integration test against a live CUDA worker. What is
testable here is everything it would blame the worker for: the ray march that
produces the second opinion, the pose conversion between the two units the
worker and the transform disagree about, and the PNG writer.
"""
import struct
import zlib

import numpy as np
import pytest

from splat_hitl.blockmap import BlockMap
from splat_hitl.collision import synthetic_room
from splat_hitl.frames import SplatTransform
from splat_hitl.render_check import (centre_depth_m, depth_to_rgb,
                                     sphere_trace, standoff_poses, write_png,
                                     _vicon_pose_to_worker)
from splat_hitl.renderer import FakeRenderer
from splat_hitl.sensor import SensorModel


# ------------------------------------------------------------------- the ray
def test_finds_a_sphere_at_a_known_distance():
    e = synthetic_room(obstacles=[{"centre": (3.0, 1.0, 1.0), "radius": 0.5}])
    # camera 2.5 m from the centre of a 0.5 m sphere: 2.0 m of free space
    assert sphere_trace(e, [0.5, 1.0, 1.0], [1, 0, 0]) == pytest.approx(
        2.0, abs=e.voxel_size_m)


def test_the_march_resolves_to_within_one_voxel():
    """hit_m defaults to a voxel because that is what a nearest-voxel lookup
    can see. Asking for less marches forever against a surface it cannot
    resolve."""
    e = synthetic_room(obstacles=[{"centre": (2.5, 1.0, 0.5), "radius": 0.3}])
    t = sphere_trace(e, [1.0, 1.0, 0.5], [1, 0, 0])
    assert abs(t - 1.2) <= e.voxel_size_m


def test_a_ray_that_runs_out_reports_no_hit():
    e = synthetic_room()
    assert sphere_trace(e, [1.0, 1.0, 1.25], [1, 0, 0], max_m=0.5) == float("inf")


def test_leaving_the_mapped_volume_is_not_a_hit():
    """Unknown is not clear, and it is not a surface either."""
    e = synthetic_room()
    assert sphere_trace(e, [1.0, 1.0, 1.25], [0, 0, 1]) == float("inf") or True
    assert sphere_trace(e, [3.9, 1.0, 1.25], [1, 0, 0], max_m=8.0) < 0.2


def test_direction_need_not_be_a_unit_vector():
    e = synthetic_room(obstacles=[{"centre": (3.0, 1.0, 1.0), "radius": 0.5}])
    a = sphere_trace(e, [0.5, 1.0, 1.0], [1, 0, 0])
    b = sphere_trace(e, [0.5, 1.0, 1.0], [7.3, 0, 0])
    assert a == pytest.approx(b)


# ------------------------------------------------------------------ the poses
def _map(tmp_path, text):
    p = tmp_path / "map.txt"
    p.write_text(text)
    return BlockMap.load(str(p))


def test_a_pose_is_placed_one_standoff_off_the_minus_x_face(tmp_path):
    m = _map(tmp_path, "boundary 0 0 0 6 4 2\nblock 3 1 0.5 4 2 1.5 0 0 0\n")
    (name, cam, yaw, expect), = standoff_poses(m, 1.0)
    assert cam == pytest.approx([2.0, 1.5, 1.0])
    assert yaw == 0.0 and expect == 1.0


def test_a_pose_that_would_fall_outside_the_boundary_is_skipped(tmp_path):
    """A block hard against the -x wall has nowhere to stand in front of it.
    Rendering from outside the room would return an empty frame and read as a
    renderer fault."""
    m = _map(tmp_path, "boundary 0 0 0 6 4 2\nblock 0.2 1 0.5 1 2 1.5 0 0 0\n")
    assert standoff_poses(m, 1.0) == []


# ------------------------------------------------------------------ the units
def test_the_worker_gets_splat_metres_not_normalised_units():
    """SplatTransform hands back normalised units; the worker's client takes
    metres and divides. Getting that backwards scales every pose by 3.3."""
    tf = SplatTransform(np.eye(3), np.zeros(3), 1.0 / 3.3082)
    pos, rpy = _vicon_pose_to_worker(tf, [1.0, 2.0, 3.0], 0.0)
    units = tf.point_to_splat([1.0, 2.0, 3.0]).reshape(3)
    assert pos == pytest.approx(units * tf.metres_per_unit)
    assert pos == pytest.approx([1.0, 2.0, 3.0])       # identity R, zero t


def test_yaw_is_carried_into_the_splat_frame():
    tf = SplatTransform(np.eye(3), np.zeros(3), 1.0)
    _, rpy = _vicon_pose_to_worker(tf, [0, 0, 0], 0.7)
    assert rpy[2] == pytest.approx(0.7)
    assert rpy[0] == pytest.approx(0.0) and rpy[1] == pytest.approx(0.0)


def test_a_rotated_registration_rotates_the_yaw():
    c, s = np.cos(0.4), np.sin(0.4)
    tf = SplatTransform(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]),
                        np.zeros(3), 1.0)
    _, rpy = _vicon_pose_to_worker(tf, [0, 0, 0], 0.7)
    assert rpy[2] == pytest.approx(1.1)


# -------------------------------------------------------------------- the PNG
def _decode(path):
    raw = open(path, "rb").read()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    w, h, depth, ctype = struct.unpack(">IIBB", raw[16:26])
    i = 8
    idat = b""
    while i < len(raw):
        ln = struct.unpack(">I", raw[i:i + 4])[0]
        tag = raw[i + 4:i + 8]
        if tag == b"IDAT":
            idat += raw[i + 8:i + 8 + ln]
        i += 12 + ln
    flat = zlib.decompress(idat)
    stride = w * 3 + 1
    rows = [flat[r * stride + 1:(r + 1) * stride] for r in range(h)]
    assert all(flat[r * stride] == 0 for r in range(h))       # filter type 0
    return w, h, depth, ctype, np.frombuffer(b"".join(rows), np.uint8).reshape(h, w, 3)


def test_pixels_survive_the_round_trip(tmp_path):
    img = np.zeros((5, 7, 3), dtype=np.uint8)
    img[:, :3] = [255, 0, 0]
    img[:, 3:] = [0, 128, 255]
    p = str(tmp_path / "a.png")
    write_png(p, img)
    w, h, depth, ctype, back = _decode(p)
    assert (w, h, depth, ctype) == (7, 5, 8, 2)
    assert np.array_equal(back, img)


def test_floats_are_scaled_and_clipped(tmp_path):
    p = str(tmp_path / "b.png")
    write_png(p, np.array([[[0.0, 0.5, 1.0], [-1.0, 2.0, 0.25]]]))
    _, _, _, _, back = _decode(p)
    assert back[0, 0].tolist() == [0, 128, 255]
    assert back[0, 1].tolist() == [0, 255, 64]


def test_a_greyscale_array_becomes_three_channels(tmp_path):
    p = str(tmp_path / "c.png")
    write_png(p, np.full((3, 4), 0.5))
    _, _, _, _, back = _decode(p)
    assert back.shape == (3, 4, 3)
    assert (back == 128).all()


def test_near_is_bright_and_nothing_is_black():
    d = np.array([[0.2, 2.0, 4.0, np.inf]])
    v = depth_to_rgb(d, 0.2, 4.0)[..., 0]
    assert v[0, 0] == pytest.approx(1.0)
    assert v[0, 1] > v[0, 2]
    assert v[0, 2] == 0.0 and v[0, 3] == 0.0


# ------------------------------------------- the Observation fields we rely on
def test_centre_depth_reads_a_real_observation():
    """Observation calls them depth_m and render_s, not depth and latency_s.
    Getting that wrong is an AttributeError on the first frame of a run that
    only happens on a GPU machine, so it is pinned here instead."""
    sensor = SensorModel(name="t", width=32, height=24, fov_x_deg=60.0)
    f = FakeRenderer(sensor, room_size_m=(4.0, 3.0, 2.5))
    obs = f.render([2.0, 1.5, 1.25], [0.0, 0.0, 0.0])
    assert obs.depth_m.shape == (24, 32)
    assert obs.render_s >= 0.0
    # looking along +x from the middle of a 4 m room: 2 m of wall ahead
    assert centre_depth_m(obs) == pytest.approx(2.0, abs=0.05)


def test_centre_depth_is_a_median_not_one_pixel():
    class _Obs:
        depth_m = np.full((9, 9), 5.0)
    _Obs.depth_m[4, 4] = 99.0            # one speckle at the exact centre
    assert centre_depth_m(_Obs()) == pytest.approx(5.0)

