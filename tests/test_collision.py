"""Virtual-collision tests against an analytically exact synthetic room."""
import math

import numpy as np
import pytest

from splat_hitl.collision import (Clearance, CollisionMonitor, ESDF,
                                  synthetic_room)


def room(**kw):
    base = dict(size_m=(4.0, 3.0, 2.5), voxel_m=0.05, truncation_m=1.0)
    base.update(kw)
    return synthetic_room(**base)


# ---------------------------------------------------------------- Clearance
def test_outside_is_not_clear():
    """Unknown must never read as safe. The whole flag exists for this."""
    assert not Clearance(float("nan"), outside=True).clear_of(0.1)


def test_clear_of_threshold():
    assert Clearance(0.5).clear_of(0.1)
    assert not Clearance(0.05).clear_of(0.1)


def test_truncated_str_says_at_least():
    assert str(Clearance(1.0, truncated=True)).startswith(">=")
    assert "outside" in str(Clearance(float("nan"), outside=True))


# --------------------------------------------------------------------- ESDF
@pytest.mark.parametrize("kw", [
    dict(voxel_size=0.0), dict(truncation=-1.0), dict(metres_per_unit=0.0),
])
def test_rejects_impossible_grids(kw):
    base = dict(grid=np.ones((4, 4, 4)), voxel_size=0.1, origin=(0, 0, 0),
                truncation=1.0, metres_per_unit=1.0)
    base.update(kw)
    with pytest.raises(ValueError):
        ESDF(**base)


def test_rejects_non_3d_grid():
    with pytest.raises(ValueError, match="3-D"):
        ESDF(np.ones((4, 4)), 0.1, (0, 0, 0), 1.0)


def test_centre_of_the_room_is_far_from_everything():
    e = room()
    c = e.at([2.0, 1.5, 1.25])
    assert not c.outside
    assert c.metres > 1.0 - 1e-9 and c.truncated       # capped at truncation


def test_near_a_wall_reads_the_wall_distance():
    e = room(voxel_m=0.05)
    c = e.at([0.10, 1.5, 1.25])                        # 10 cm from x=0
    assert not c.outside and not c.truncated
    assert abs(c.metres - 0.10) <= 0.05                # within one voxel


def test_outside_the_grid_is_reported_as_outside():
    e = room()
    for p in ([-0.5, 1.5, 1.25], [9.0, 1.5, 1.25], [2.0, 1.5, 9.0]):
        assert e.at(p).outside


def test_a_sphere_obstacle_shows_up():
    e = room(obstacles=[{"centre": (2.0, 1.5, 1.25), "radius": 0.3}])
    assert e.at([2.0, 1.5, 1.25]).metres < 1e-9        # inside the sphere
    c = e.at([2.5, 1.5, 1.25])                         # 20 cm outside it
    assert abs(c.metres - 0.20) <= 0.06


def test_metres_per_unit_scales_the_query():
    """Same grid, declared as 1 unit = 0.5 m: everything halves in metres."""
    g = room(voxel_m=0.05).grid
    a = ESDF(g, 0.05, np.zeros(3), 1.0, metres_per_unit=1.0)
    b = ESDF(g, 0.05, np.zeros(3), 1.0, metres_per_unit=0.5)
    assert math.isclose(b.voxel_size_m, a.voxel_size_m * 0.5)
    assert math.isclose(b.at([1.0, 0.75, 0.6]).metres,
                        a.at([2.0, 1.5, 1.2]).metres * 0.5, rel_tol=1e-9)


# ------------------------------------------------------------ swept segments
def test_segment_catches_a_wall_the_endpoints_miss():
    """Both ends clear, the middle grazes an obstacle. Endpoint checks fail."""
    e = room(obstacles=[{"centre": (2.0, 1.5, 1.25), "radius": 0.4}])
    a, b = [1.0, 1.5, 1.25], [3.0, 1.5, 1.25]
    assert e.at(a).metres > 0.3 and e.at(b).metres > 0.3
    assert e.min_along(a, b).metres < 1e-6


def test_min_along_returns_outside_if_the_path_leaves_the_map():
    e = room()
    assert e.min_along([2.0, 1.5, 1.25], [9.0, 1.5, 1.25]).outside


def test_min_along_rejects_bad_step():
    e = room()
    with pytest.raises(ValueError, match="step_m"):
        e.min_along([1, 1, 1], [2, 1, 1], step_m=0.0)


# ------------------------------------------------------------------ monitor
def test_constructor_refuses_clearance_finer_than_a_voxel():
    e = room(voxel_m=0.20)
    with pytest.raises(ValueError, match="finer than the ESDF voxel"):
        CollisionMonitor(e, clearance_m=0.05)


def test_constructor_refuses_clearance_beyond_truncation():
    e = room(truncation_m=0.30, voxel_m=0.05)
    with pytest.raises(ValueError, match="truncation"):
        CollisionMonitor(e, clearance_m=0.50)


def test_warns_when_clearance_is_under_two_voxels():
    m = CollisionMonitor(room(voxel_m=0.08), clearance_m=0.10)
    assert any("two voxels" in w for w in m.warnings)
    assert "WARNING" in m.summary()


def test_clean_flight_down_the_middle():
    e = room()
    m = CollisionMonitor(e, 0.10)
    p = np.array([1.0, 1.5, 1.25])
    for x in np.arange(1.1, 3.0, 0.1):
        q = np.array([x, 1.5, 1.25])
        assert m.update(p, q) is None
        p = q
    assert not m.failed
    assert "no virtual collision" in m.summary()


def test_flying_into_a_wall_fails():
    e = room()
    m = CollisionMonitor(e, 0.10)
    ev = None
    p = np.array([2.0, 1.5, 1.25])
    for x in np.arange(2.1, 4.2, 0.1):
        ev = m.update(p, [x, 1.5, 1.25]) or ev
        p = np.array([x, 1.5, 1.25])
        if ev:
            break
    assert ev is not None and ev.kind == "collision"
    assert m.failed and "VIRTUAL COLLISION" in str(ev)


def test_a_fast_step_cannot_tunnel_through_an_obstacle():
    e = room(obstacles=[{"centre": (2.0, 1.5, 1.25), "radius": 0.4}])
    m = CollisionMonitor(e, 0.10)
    ev = m.update([1.0, 1.5, 1.25], [3.0, 1.5, 1.25])   # 2 m in one step
    assert ev is not None and ev.kind == "collision"


def test_failure_latches():
    e = room()
    m = CollisionMonitor(e, 0.10)
    m.update([2.0, 1.5, 1.25], [3.99, 1.5, 1.25])
    assert m.failed
    first = m.event
    assert m.update([2.0, 1.5, 1.25], [2.1, 1.5, 1.25]) is None
    assert m.event is first                              # unchanged


def test_leaving_the_map_is_a_failure_by_default():
    m = CollisionMonitor(room(), 0.10)
    ev = m.update([2.0, 1.5, 1.25], [9.0, 1.5, 1.25])
    assert ev is not None and ev.kind == "outside"
    assert "LEFT THE MAPPED VOLUME" in str(ev)


def test_leaving_the_map_can_be_made_non_fatal():
    m = CollisionMonitor(room(), 0.10, outside_is_failure=False)
    assert m.update([2.0, 1.5, 1.25], [9.0, 1.5, 1.25]) is None
    assert not m.failed and m.outside_steps == 1
    assert "not counted as failure" in m.summary()


def test_closest_approach_is_tracked():
    e = room(obstacles=[{"centre": (2.0, 1.5, 1.25), "radius": 0.3}])
    m = CollisionMonitor(e, 0.10)
    m.update([1.0, 1.5, 1.25], [1.5, 1.5, 1.25])         # closes to ~0.2 m
    assert 0.1 < m.closest_m < 0.3
    assert "closest approach" in m.summary()


def test_reset_clears_the_verdict():
    m = CollisionMonitor(room(), 0.10)
    m.update([2.0, 1.5, 1.25], [3.99, 1.5, 1.25])
    assert m.failed
    m.reset()
    assert not m.failed and m.step == 0 and not math.isfinite(m.closest_m)


def test_rejects_non_positive_clearance():
    with pytest.raises(ValueError, match="clearance_m"):
        CollisionMonitor(room(), 0.0)


# ------------------------------------------------------------------- helper
def test_synthetic_room_is_analytically_right():
    e = synthetic_room(size_m=(2.0, 2.0, 2.0), voxel_m=0.1, truncation_m=5.0)
    # centre of a 2 m cube is 1 m from every wall
    assert abs(e.at([1.0, 1.0, 1.0]).metres - 1.0) <= 0.1
    # 30 cm in from one face
    assert abs(e.at([0.3, 1.0, 1.0]).metres - 0.3) <= 0.1


def test_repr_is_informative():
    r = repr(room())
    assert "voxels" in r and "truncated" in r
