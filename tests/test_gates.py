"""Gate tests, written as flight situations rather than as assertions on maths."""
import json
import math

import numpy as np
import pytest

from splat_hitl.gates import (MISSED, PASSED, WRONG_WAY, Gate, GateCourse)


def gate(name="g", centre=(0, 0, 1.0), normal=(1, 0, 0), w=1.0, h=1.0, up=(0, 0, 1)):
    return Gate(name, np.array(centre, float), np.array(normal, float), w, h,
                np.array(up, float))


# ------------------------------------------------------------------ geometry
def test_rejects_zero_normal():
    with pytest.raises(ValueError, match="zero length"):
        gate(normal=(0, 0, 0))


def test_rejects_up_parallel_to_normal():
    with pytest.raises(ValueError, match="parallel"):
        gate(normal=(0, 0, 1), up=(0, 0, 1))


@pytest.mark.parametrize("kw", [dict(w=0.0), dict(h=-1.0)])
def test_rejects_non_positive_aperture(kw):
    with pytest.raises(ValueError, match="positive"):
        gate(**kw)


def test_up_is_orthogonalised_not_rejected():
    """A human typing an approximate 'up' for a tilted gate must be tolerated."""
    g = gate(normal=(1, 0, 0), up=(0.2, 0.0, 1.0))
    assert abs(float(np.dot(g.up, g.normal))) < 1e-12
    assert math.isclose(float(np.linalg.norm(g.up)), 1.0)


def test_frame_is_right_handed():
    g = gate(normal=(1, 0, 0), up=(0, 0, 1))
    assert np.allclose(np.cross(g.normal, g.up), g.right)
    assert math.isclose(float(np.linalg.norm(g.right)), 1.0)


def test_local_coordinates():
    g = gate(centre=(2, 0, 1), normal=(1, 0, 0), up=(0, 0, 1))
    along, across, lift = g.local([2.5, 0.3, 1.4])
    assert math.isclose(along, 0.5)
    assert math.isclose(lift, 0.4)
    assert math.isclose(abs(across), 0.3)


def test_miss_distance_is_zero_inside():
    g = gate(w=1.0, h=0.8)
    assert g.miss_distance(0.4, 0.3) == 0.0
    assert g.miss_distance(0.6, 0.0) > 0.0


# --------------------------------------------------------------- basic passes
def test_straight_through_the_middle_passes():
    c = GateCourse([gate(centre=(1, 0, 1))])
    evs = c.update([0.5, 0, 1.0], [1.5, 0, 1.0])
    assert len(evs) == 1 and evs[0].kind == PASSED
    assert c.passed == 1 and c.complete


def test_approaching_but_not_reaching_produces_nothing():
    c = GateCourse([gate(centre=(1, 0, 1))])
    assert c.update([0.0, 0, 1.0], [0.9, 0, 1.0]) == []
    assert c.passed == 0


def test_crossing_backwards_is_wrong_way_and_does_not_advance():
    c = GateCourse([gate(centre=(1, 0, 1))])
    evs = c.update([1.5, 0, 1.0], [0.5, 0, 1.0])
    assert evs[0].kind == WRONG_WAY
    assert c.passed == 0


def test_crossing_the_plane_outside_the_opening_is_a_miss():
    c = GateCourse([gate(centre=(1, 0, 1), w=1.0, h=1.0)])
    evs = c.update([0.5, 1.2, 1.0], [1.5, 1.2, 1.0])     # 1.2 m off to the side
    assert evs[0].kind == MISSED
    assert math.isclose(evs[0].miss_m, 0.7, abs_tol=1e-9)   # 1.2 - 0.5
    assert c.passed == 0


def test_miss_above_the_opening():
    c = GateCourse([gate(centre=(1, 0, 1), w=1.0, h=0.6)])
    evs = c.update([0.5, 0, 1.9], [1.5, 0, 1.9])
    assert evs[0].kind == MISSED
    assert math.isclose(evs[0].miss_m, 0.6, abs_tol=1e-9)   # 0.9 - 0.3


def test_exactly_on_the_edge_counts_as_through():
    g = gate(centre=(1, 0, 1), w=1.0, h=1.0)
    c = GateCourse([g])
    evs = c.update([0.5, 0.5, 1.0], [1.5, 0.5, 1.0])       # exactly w/2
    assert evs[0].kind == PASSED


# ------------------------------------------------------------------ tunneling
def test_a_fast_segment_cannot_tunnel_through_a_gate():
    """3 m in one step, gate 1.2 m in. Sample-based tests miss this entirely."""
    c = GateCourse([gate(centre=(1.2, 0, 1))])
    evs = c.update([0.0, 0, 1.0], [3.0, 0, 1.0])
    assert evs[0].kind == PASSED
    assert 0.0 < evs[0].t < 1.0
    assert np.allclose(evs[0].point, [1.2, 0, 1.0], atol=1e-9)


def test_sample_landing_exactly_on_the_plane_is_not_lost():
    """d1 == 0 exactly: a sign-product test would silently miss this."""
    c = GateCourse([gate(centre=(1, 0, 1))])
    assert c.update([0.5, 0, 1.0], [1.0, 0, 1.0])[0].kind == PASSED


def test_travelling_along_the_plane_is_not_a_crossing():
    c = GateCourse([gate(centre=(1, 0, 1))])
    assert c.update([1.0, -0.4, 1.0], [1.0, 0.4, 1.0]) == []


# -------------------------------------------------------------------- courses
def course3():
    return GateCourse([gate("a", (1, 0, 1)), gate("b", (2, 0, 1)), gate("c", (3, 0, 1))])


def test_gates_must_be_taken_in_order():
    c = course3()
    # jump straight past a and b to c: no credit, no event
    assert c.update([0.0, 0, 1.0], [2.5, 0, 1.0])[0].kind == PASSED  # crosses a
    assert c.passed == 1                                             # only a


def test_a_later_gate_is_not_armed():
    c = course3()
    # fly through gate c's plane only; gate a is still the armed one
    assert c.update([2.5, 0, 1.0], [3.5, 0, 1.0]) == []
    assert c.passed == 0


def test_full_course_completes_in_order():
    c = course3()
    p = np.array([0.0, 0.0, 1.0])
    for x in np.arange(0.1, 3.6, 0.1):
        q = np.array([x, 0.0, 1.0])
        c.update(p, q)
        p = q
    assert c.complete and c.passed == 3
    assert [e.name for e in c.events] == ["a", "b", "c"]


def test_no_events_once_complete():
    c = GateCourse([gate(centre=(1, 0, 1))])
    c.update([0.5, 0, 1.0], [1.5, 0, 1.0])
    assert c.complete
    assert c.update([1.5, 0, 1.0], [0.5, 0, 1.0]) == []


def test_reset_rearms_the_course():
    c = course3()
    c.update([0.0, 0, 1.0], [1.5, 0, 1.0])
    assert c.passed == 1
    c.reset()
    assert c.passed == 0 and c.events == []


def test_distance_to_next():
    c = course3()
    assert math.isclose(c.distance_to_next([0.0, 0, 1.0]), 1.0)
    c.update([0.0, 0, 1.0], [1.5, 0, 1.0])
    assert math.isclose(c.distance_to_next([1.5, 0, 1.0]), 0.5)


def test_distance_is_none_when_complete():
    c = GateCourse([gate(centre=(1, 0, 1))])
    c.update([0.5, 0, 1.0], [1.5, 0, 1.0])
    assert c.distance_to_next([0, 0, 0]) is None


def test_empty_course_rejected():
    with pytest.raises(ValueError, match="at least one gate"):
        GateCourse([])


# ------------------------------------------------------------- tilted gates
def test_a_tilted_gate_works_the_same():
    n = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
    c = GateCourse([Gate("diag", np.array([1.0, 1.0, 1.0]), n, 1.0, 1.0,
                         np.array([0.0, 0.0, 1.0]))])
    before = np.array([1.0, 1.0, 1.0]) - n * 0.5
    after = np.array([1.0, 1.0, 1.0]) + n * 0.5
    assert c.update(before, after)[0].kind == PASSED


# ------------------------------------------------------------- persistence
def test_save_load_round_trip(tmp_path):
    c = course3()
    p = tmp_path / "course.json"
    c.save(p)
    back = GateCourse.load(p)
    assert [g.name for g in back.gates] == ["a", "b", "c"]
    assert np.allclose(back.gates[1].centre, c.gates[1].centre)
    assert np.allclose(back.gates[1].normal, c.gates[1].normal)
    assert json.loads(p.read_text())["gates"][0]["width_m"] == 1.0


# ----------------------------------------------------------------- summary
def test_summary_reports_progress_and_misses():
    c = course3()
    c.update([0.5, 1.2, 1.0], [1.5, 1.2, 1.0])      # miss gate a
    txt = c.summary()
    assert "0 / 3" in txt and "plane misses" in txt and "stopped at" in txt


def test_summary_marks_completion():
    c = GateCourse([gate(centre=(1, 0, 1))])
    c.update([0.5, 0, 1.0], [1.5, 0, 1.0])
    assert "COURSE COMPLETE" in c.summary()


def test_event_str_is_readable():
    c = GateCourse([gate("hoop", (1, 0, 1), w=1.0, h=1.0)])
    ev = c.update([0.5, 1.2, 1.0], [1.5, 1.2, 1.0])[0]
    assert "MISSED" in str(ev) and "hoop" in str(ev) and "cm" in str(ev)
