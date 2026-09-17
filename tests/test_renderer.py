"""Renderer tests. The fake is checked against geometry we can compute by hand."""
import math

import numpy as np
import pytest

from splat_hitl.renderer import FakeRenderer, RendererClient
from splat_hitl.sensor import SensorModel


def sensor(**kw):
    base = dict(name="t", width=64, height=48, fov_x_deg=90.0)
    base.update(kw)
    return SensorModel(**base)


ROOM = (4.0, 3.0, 2.5)
MID = [2.0, 1.5, 1.25]


# ------------------------------------------------------------------- shapes
def test_depth_has_the_sensor_shape_and_dtype():
    r = FakeRenderer(sensor(width=64, height=48), ROOM)
    o = r.render(MID, [0, 0, 0])
    assert o.shape == (48, 64)
    assert o.depth_m.dtype == np.float32
    assert o.sensor_fingerprint == r.sensor.fingerprint()
    assert o.render_s >= 0.0


def test_resolution_follows_the_sensor_model():
    for w, h in ((32, 24), (160, 120)):
        o = FakeRenderer(sensor(width=w, height=h), ROOM).render(MID, [0, 0, 0])
        assert o.shape == (h, w)


# ---------------------------------------------------------------- geometry
def test_centre_pixel_sees_the_wall_straight_ahead():
    """At the middle of a 4 m room facing +x, the wall is 2 m away."""
    r = FakeRenderer(sensor(), ROOM)
    o = r.render(MID, [0, 0, 0])
    centre = o.depth_m[o.shape[0] // 2, o.shape[1] // 2]
    assert abs(float(centre) - 2.0) < 0.05


def test_turning_ninety_degrees_sees_the_nearer_wall():
    """Facing +y (north), the wall is 1.5 m away, not 2 m."""
    r = FakeRenderer(sensor(), ROOM)
    o = r.render(MID, [0, 0, math.pi / 2])
    centre = o.depth_m[o.shape[0] // 2, o.shape[1] // 2]
    assert abs(float(centre) - 1.5) < 0.05


def test_moving_closer_shortens_the_depth():
    r = FakeRenderer(sensor(), ROOM)
    far = r.render([1.0, 1.5, 1.25], [0, 0, 0]).depth_m
    near = r.render([3.0, 1.5, 1.25], [0, 0, 0]).depth_m
    c = (far.shape[0] // 2, far.shape[1] // 2)
    assert abs(float(far[c]) - 3.0) < 0.05
    assert abs(float(near[c]) - 1.0) < 0.05


def test_edge_pixels_are_further_than_the_centre():
    """Range, not z-depth: an oblique ray to the same wall is longer."""
    o = FakeRenderer(sensor(fov_x_deg=90.0), ROOM).render(MID, [0, 0, 0])
    row = o.depth_m[o.shape[0] // 2]
    assert row[0] > row[len(row) // 2]


def test_a_sphere_obstacle_occludes_the_wall():
    obs = [{"centre": (3.0, 1.5, 1.25), "radius": 0.3}]
    clear = FakeRenderer(sensor(), ROOM).render(MID, [0, 0, 0]).depth_m
    blocked = FakeRenderer(sensor(), ROOM, obs).render(MID, [0, 0, 0]).depth_m
    c = (clear.shape[0] // 2, clear.shape[1] // 2)
    assert abs(float(blocked[c]) - 0.7) < 0.05        # 1.0 m to centre - 0.3 r
    assert float(blocked[c]) < float(clear[c])


def test_everything_is_positive_and_bounded():
    obs = [{"centre": (3.0, 1.5, 1.25), "radius": 0.3}]
    o = FakeRenderer(sensor(fov_x_deg=120.0), ROOM, obs, far_m=24.0).render(MID, [0, 0, 0])
    assert np.all(o.depth_m > 0) and np.all(o.depth_m <= 24.0)
    assert np.all(np.isfinite(o.depth_m))


def test_looking_up_sees_the_ceiling():
    r = FakeRenderer(sensor(), ROOM)
    o = r.render(MID, [0, -math.pi / 2, 0])           # pitch up
    centre = o.depth_m[o.shape[0] // 2, o.shape[1] // 2]
    assert abs(float(centre) - 1.25) < 0.05


# ------------------------------------------------------------------- mount
#: OFF the mid-height of the room ON PURPOSE. At MID the floor and the ceiling
#: are both 1.25 m away, so a mount pitched up and one pitched down give the
#: same reading and the sign is invisible. From 0.50 m the floor is 0.50 m
#: below and the ceiling 2.00 m above, and only one of them can be right.
LOW = [2.0, 1.5, 0.5]


def test_mount_pitch_is_positive_down():
    """A camera pitched down must see the FLOOR sooner than a level one."""
    level = FakeRenderer(sensor(mount_pitch_deg=0.0), ROOM)
    tilted = FakeRenderer(sensor(mount_pitch_deg=45.0), ROOM)
    c = (24, 32)
    d_level = float(level.render(LOW, [0, 0, 0]).depth_m[c])
    d_tilt = float(tilted.render(LOW, [0, 0, 0]).depth_m[c])
    assert d_tilt < d_level
    # 45 deg down from 0.50 m up: range to the floor is 0.50 / sin(45)
    assert abs(d_tilt - 0.5 * math.sqrt(2)) < 0.05
    # and emphatically NOT the ceiling, 2.00 m above
    assert abs(d_tilt - 2.0 * math.sqrt(2)) > 1.0


def test_mount_pitch_of_ninety_looks_straight_down():
    r = FakeRenderer(sensor(mount_pitch_deg=90.0), ROOM)
    centre = float(r.render(LOW, [0, 0, 0]).depth_m[24, 32])
    assert abs(centre - 0.5) < 0.02          # the floor, not the 2.00 m ceiling


def test_a_negative_mount_pitch_looks_up():
    r = FakeRenderer(sensor(mount_pitch_deg=-90.0), ROOM)
    centre = float(r.render(LOW, [0, 0, 0]).depth_m[24, 32])
    assert abs(centre - 2.0) < 0.02


def test_camera_rpy_composes_body_and_mount():
    r = FakeRenderer(sensor(mount_pitch_deg=10.0), ROOM)
    rpy = r.camera_rpy([0.0, 0.0, 0.0])
    assert abs(math.degrees(rpy[1]) - 10.0) < 1e-6    # FLU: pitch down is +ve
    assert abs(rpy[0]) < 1e-9 and abs(rpy[2]) < 1e-9


def test_mount_yaw_offsets_the_view():
    straight = FakeRenderer(sensor(mount_yaw_deg=0.0), ROOM)
    turned = FakeRenderer(sensor(mount_yaw_deg=90.0), ROOM)
    c = (24, 32)
    assert abs(float(straight.render(MID, [0, 0, 0]).depth_m[c]) - 2.0) < 0.05
    assert abs(float(turned.render(MID, [0, 0, 0]).depth_m[c]) - 1.5) < 0.05


def test_body_yaw_and_mount_yaw_are_equivalent():
    """Turning the drone 90 deg == mounting the camera 90 deg round."""
    a = FakeRenderer(sensor(mount_yaw_deg=90.0), ROOM).render(MID, [0, 0, 0])
    b = FakeRenderer(sensor(mount_yaw_deg=0.0), ROOM).render(MID, [0, 0, math.pi / 2])
    assert np.allclose(a.depth_m, b.depth_m, atol=1e-4)


# ------------------------------------------------------------------ interface
def test_context_manager_closes():
    with FakeRenderer(sensor(), ROOM) as r:
        assert r.render(MID, [0, 0, 0]) is not None


def test_is_a_renderer_client():
    assert isinstance(FakeRenderer(sensor(), ROOM), RendererClient)


def test_cannot_instantiate_the_abstract_base():
    with pytest.raises(TypeError):
        RendererClient(sensor())


def test_fake_and_collision_field_describe_the_same_world():
    """The depth a policy sees and the field that scores it must agree."""
    from splat_hitl.collision import synthetic_room
    obs = [{"centre": (3.0, 1.5, 1.25), "radius": 0.3}]
    esdf = synthetic_room(size_m=ROOM, voxel_m=0.05, obstacles=obs)
    r = FakeRenderer(sensor(), ROOM, obs)
    # standing 1 m short of the sphere: depth ahead ~0.7, clearance ~0.7
    p = [2.0, 1.5, 1.25]
    ahead = float(r.render(p, [0, 0, 0]).depth_m[24, 32])
    clear = esdf.at(p).metres
    assert abs(ahead - 0.7) < 0.06
    assert abs(clear - 0.7) < 0.08
