"""The mocap solver-flip filter.

Written against a measured failure: on 2026-09-17 the `crazyflie_a2` rigid
body was recorded jumping 93 deg in a single 240 Hz frame, at rest, with all
8 markers visible -- both by the driver's own filter and by spin_test.py, at
0%, 30% and 37.5% throttle. This is the code that stops that reaching a
velocity policy, which would otherwise rotate its world-frame command into a
body frame that is 90 deg wrong and fly sideways at full commanded speed.
"""
from splat_hitl.ros_node import (FLIP_GAP_CAP_S, MAX_YAW_RATE_DPS,
                                 MIN_FLIP_DEG, is_solver_flip)

FRAME = 1.0 / 240.0


def test_first_sample_is_never_a_flip():
    assert is_solver_flip(0.0, None, None, 0.0)[0] is False


def test_ordinary_jitter_passes():
    # spin_test measured yaw sd of about 1.05 deg with peaks near 3 deg.
    for step in (0.5, 1.0, 2.0, 3.0, 10.0):
        flip, d, _ = is_solver_flip(step, 0.0, 0.0, FRAME)
        assert flip is False, "%.1f deg rejected as a flip" % d


def test_the_measured_93_degree_jump_is_caught():
    flip, d, rate = is_solver_flip(93.0, 0.0, 0.0, FRAME)
    assert flip is True
    assert abs(d - 93.0) < 1e-9
    assert rate > MAX_YAW_RATE_DPS


def test_wrap_is_shortest_arc():
    # -179 -> +179 is 2 deg apart, not 358.
    flip, d, _ = is_solver_flip(179.0, -179.0, 0.0, FRAME)
    assert flip is False
    assert abs(d - 2.0) < 1e-9


def test_latched_flip_stays_rejected_as_the_gap_grows():
    """The reason the rate divisor is capped.

    On a rejection the caller does not advance last_yaw_t, so the gap keeps
    growing. Uncapped, rate = d / gap would fall under the threshold and a
    solver that stayed in the wrong solution would be quietly accepted after a
    second or two -- which is the exact case the filter exists for.
    """
    for gap in (FRAME, 0.05, 0.5, 2.0, 30.0):
        flip, _, _ = is_solver_flip(93.0, 0.0, 0.0, gap)
        assert flip is True, "latched flip accepted after %.2f s" % gap


def test_step_test_is_the_binding_one_at_the_default_rate():
    """Documented as requiring BOTH conditions; at the default it is one.

    With the divisor capped at FLIP_GAP_CAP_S, any step past MIN_FLIP_DEG
    implies at least MIN_FLIP_DEG / FLIP_GAP_CAP_S = 900 deg/s, which already
    exceeds the 720 deg/s default. So the rate test cannot be the binding
    condition unless the limit is raised above 900. Pinned here so that if
    someone retunes either constant, this says so out loud.
    """
    implied = MIN_FLIP_DEG / FLIP_GAP_CAP_S
    assert implied > MAX_YAW_RATE_DPS
    flip, _, rate = is_solver_flip(MIN_FLIP_DEG + 0.1, 0.0, 0.0, 10.0)
    assert flip is True and rate >= implied


def test_just_under_the_step_threshold_passes():
    flip, _, _ = is_solver_flip(MIN_FLIP_DEG - 0.1, 0.0, 0.0, FRAME)
    assert flip is False
