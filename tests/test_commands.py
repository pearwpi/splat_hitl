"""Action -> command tests.

This is the module where being wrong flies a drone into the floor, so the tests
are written as physical statements: "commanded forward while facing north, the
drone must go north".
"""
import math

import numpy as np
import pytest

from splat_hitl.commands import (Action, Clamped, Limits, VelocityIntegrator,
                                 VelocityPassthrough, make_action_stage,
                                 to_hover, to_position, to_velocity_world,
                                 to_world_enu)
from splat_hitl.contract import ActionSpec

N = math.pi / 2          # facing north (ENU +y) when yaw is +90 deg
FWD = np.array([1.0, 0.0, 0.0])


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize("kw", [
    dict(kind="thrust"), dict(frame="world_xyz"),
])
def test_rejects_unknown_kind_or_frame(kw):
    base = dict(kind="velocity", frame="body_flu", vector=FWD)
    base.update(kw)
    with pytest.raises(ValueError):
        Action(**base)


def test_rejects_bad_vector():
    with pytest.raises(ValueError, match="3 components"):
        Action("velocity", "body_flu", [1.0, 2.0])
    with pytest.raises(ValueError, match="non-finite"):
        Action("velocity", "body_flu", [1.0, float("nan"), 0.0])


def test_rejects_non_finite_yaw_rate():
    with pytest.raises(ValueError, match="not finite"):
        Action("velocity", "body_flu", FWD, yaw_rate_rad_s=float("inf"))


# ---------------------------------------------------------------- frame maths
def test_body_forward_at_zero_yaw_is_east():
    v = to_world_enu(Action("velocity", "body_flu", FWD), 0.0)
    assert np.allclose(v, [1, 0, 0], atol=1e-12)


def test_body_forward_facing_north_is_north():
    """The bug every example in the surrounding repos hides by keeping yaw=0."""
    v = to_world_enu(Action("velocity", "body_flu", FWD), N)
    assert np.allclose(v, [0, 1, 0], atol=1e-12)


def test_body_left_facing_north_is_west():
    v = to_world_enu(Action("velocity", "body_flu", [0, 1, 0]), N)
    assert np.allclose(v, [-1, 0, 0], atol=1e-12)


def test_ned_to_enu_swaps_and_flips_z():
    """VizFlyt2 emits NED with +z DOWN; the Crazyflie's +z is UP."""
    v = to_world_enu(Action("velocity", "world_ned", [1.0, 2.0, 3.0]), 0.0)
    assert np.allclose(v, [2.0, 1.0, -3.0])


def test_ned_descend_becomes_negative_climb():
    """+z in NED is DOWN. Getting this backwards flies into the floor."""
    v = to_world_enu(Action("velocity", "world_ned", [0, 0, 1.0]), 0.0)
    assert v[2] < 0


def test_frd_flips_y_and_z():
    v = to_world_enu(Action("velocity", "body_frd", [1.0, 1.0, 1.0]), 0.0)
    assert np.allclose(v, [1.0, -1.0, -1.0])


def test_world_frames_ignore_yaw():
    a = Action("velocity", "world_enu", [1.0, 2.0, 3.0])
    for yaw in (0.0, N, -1.2, 3.0):
        assert np.allclose(to_world_enu(a, yaw), [1.0, 2.0, 3.0])


# -------------------------------------------------------------------- hover
def test_hover_body_command_survives_round_trip():
    """A body-frame action must come back out unchanged, at any heading."""
    a = Action("velocity", "body_flu", [0.8, -0.3, 0.2])
    for yaw in (0.0, N, -2.1, 1.0):
        cmd, rep = to_hover(a, yaw, 0.6)
        assert math.isclose(cmd.vx, 0.8, abs_tol=1e-9)
        assert math.isclose(cmd.vy, -0.3, abs_tol=1e-9)
        assert not rep.any


def test_hover_converts_a_world_action_into_body():
    """Commanded east while facing north, the drone must move to its RIGHT."""
    cmd, _ = to_hover(Action("velocity", "world_enu", [1.0, 0, 0]), N, 0.6)
    assert math.isclose(cmd.vx, 0.0, abs_tol=1e-9)
    assert math.isclose(cmd.vy, -1.0, abs_tol=1e-9)      # right = -y in FLU


def test_hover_yaw_rate_stays_in_radians():
    """The driver applies the degrees conversion AND the sign flip. Not us."""
    w = 0.7
    cmd, _ = to_hover(Action("velocity", "body_flu", FWD, yaw_rate_rad_s=w), 0.0, 0.6)
    assert math.isclose(cmd.yaw_rate, w, abs_tol=1e-12)
    assert abs(cmd.yaw_rate) < math.pi          # not degrees


def test_hover_z_is_absolute_altitude():
    cmd, _ = to_hover(Action("velocity", "body_flu", FWD), 0.0, 0.75)
    assert math.isclose(cmd.z_distance, 0.75)


# ------------------------------------------------- the vertical channel
def test_hover_integrates_a_climb_onto_the_altitude():
    up = Action("velocity", "body_flu", [0.0, 0.0, 0.3])
    cmd, rep = to_hover(up, 0.0, 0.6, climb_dt_s=0.5)
    assert math.isclose(cmd.z_distance, 0.75) and not rep.any


def test_hover_without_a_climb_interval_holds_the_altitude():
    up = Action("velocity", "body_flu", [0.0, 0.0, 0.3])
    cmd, _ = to_hover(up, 0.0, 0.6)
    assert math.isclose(cmd.z_distance, 0.6)


def test_hover_descends_for_a_down_command_in_frd():
    down = Action("velocity", "body_frd", [0.0, 0.0, 0.2])   # +z is DOWN in FRD
    cmd, _ = to_hover(down, 0.0, 0.6, climb_dt_s=0.5)
    assert math.isclose(cmd.z_distance, 0.5)


def test_hover_climb_is_clamped_before_it_is_integrated():
    up = Action("velocity", "body_flu", [0.0, 0.0, 5.0])
    lim = Limits(max_climb_ms=0.6)
    cmd, rep = to_hover(up, 0.0, 0.6, lim, climb_dt_s=0.1)
    assert rep.climb and math.isclose(cmd.z_distance, 0.66)


def test_hover_altitude_stops_at_the_ceiling():
    up = Action("velocity", "body_flu", [0.0, 0.0, 0.6])
    cmd, rep = to_hover(up, 0.0, 1.78, Limits(max_altitude_m=1.80), climb_dt_s=0.1)
    assert rep.altitude and math.isclose(cmd.z_distance, 1.80)


def test_hover_refuses_a_negative_climb_interval():
    with pytest.raises(ValueError):
        to_hover(Action("velocity", "body_flu", FWD), 0.0, 0.6, climb_dt_s=-0.1)


def test_hover_rejects_acceleration_with_a_useful_reason():
    a = Action("acceleration", "body_flu", [1.0, 0, 0])
    with pytest.raises(ValueError) as e:
        to_hover(a, 0.0, 0.6)
    assert "current velocity" in str(e.value) and "dt" in str(e.value)


def test_hover_rejects_position():
    with pytest.raises(ValueError, match="velocity, not a position"):
        to_hover(Action("position", "world_enu", [1, 1, 1]), 0.0, 0.6)


# ------------------------------------------------------------ velocity_world
def test_velocity_world_yaw_rate_is_degrees():
    w = math.radians(45.0)
    cmd, _ = to_velocity_world(Action("velocity", "world_enu", [0, 0, 0],
                                      yaw_rate_rad_s=w), 0.0)
    assert math.isclose(cmd.yaw_rate, 45.0, abs_tol=1e-9)


def test_velocity_world_passes_world_vector_through():
    cmd, rep = to_velocity_world(Action("velocity", "world_enu", [0.5, -0.4, 0.2]), 1.3)
    assert (cmd.vx, cmd.vy, cmd.vz) == (0.5, -0.4, 0.2)
    assert not rep.any


def test_velocity_world_rotates_a_body_action():
    cmd, _ = to_velocity_world(Action("velocity", "body_flu", FWD), N)
    assert math.isclose(cmd.vx, 0.0, abs_tol=1e-9)
    assert math.isclose(cmd.vy, 1.0, abs_tol=1e-9)


# ------------------------------------------------------------------ position
def test_position_yaw_is_degrees():
    cmd, _ = to_position(Action("position", "world_enu", [1, 2, 0.5],
                                yaw_rad=math.radians(30.0)), 0.0)
    assert math.isclose(cmd.yaw, 30.0, abs_tol=1e-9)


def test_position_rejects_body_frame():
    with pytest.raises(ValueError, match="WORLD frame"):
        to_position(Action("position", "body_flu", [1, 0, 0]), 0.0)


def test_position_rejects_wrong_kind():
    with pytest.raises(ValueError, match="kind='position'"):
        to_position(Action("velocity", "world_enu", [1, 0, 0]), 0.0)


def test_position_from_ned_flips_altitude():
    """NED z=-0.6 is 0.6 m UP."""
    cmd, rep = to_position(Action("position", "world_ned", [1.0, 2.0, -0.6]), 0.0)
    assert math.isclose(cmd.x, 2.0) and math.isclose(cmd.y, 1.0)
    assert math.isclose(cmd.z, 0.6)
    assert not rep.altitude


# -------------------------------------------------------------------- limits
def test_speed_clamp_preserves_direction():
    a = Action("velocity", "world_enu", [6.0, 8.0, 0.0])      # 10 m/s
    cmd, rep = to_velocity_world(a, 0.0, Limits(max_speed_ms=1.5))
    assert rep.speed and math.isclose(rep.requested["speed_ms"], 10.0)
    assert math.isclose(math.hypot(cmd.vx, cmd.vy), 1.5, abs_tol=1e-9)
    assert math.isclose(cmd.vy / cmd.vx, 8.0 / 6.0, rel_tol=1e-9)   # same heading


def test_climb_clamp_keeps_sign():
    cmd, rep = to_velocity_world(Action("velocity", "world_enu", [0, 0, -4.0]),
                                 0.0, Limits(max_climb_ms=0.6))
    assert rep.climb and math.isclose(cmd.vz, -0.6)


def test_yaw_rate_clamp_keeps_sign():
    lim = Limits(max_yaw_rate_rad_s=math.radians(90))
    cmd, rep = to_hover(Action("velocity", "body_flu", [0, 0, 0],
                               yaw_rate_rad_s=-math.radians(400)), 0.0, 0.6, lim)
    assert rep.yaw_rate and math.isclose(cmd.yaw_rate, -math.radians(90))


def test_altitude_clamped_into_the_envelope():
    lim = Limits(min_altitude_m=0.1, max_altitude_m=1.2)
    hi, rep_hi = to_hover(Action("velocity", "body_flu", FWD), 0.0, 3.0, lim)
    lo, rep_lo = to_hover(Action("velocity", "body_flu", FWD), 0.0, -0.5, lim)
    assert rep_hi.altitude and math.isclose(hi.z_distance, 1.2)
    assert rep_lo.altitude and math.isclose(lo.z_distance, 0.1)


def test_clamp_report_reads_well():
    a = Action("velocity", "world_enu", [9.0, 0, 5.0], yaw_rate_rad_s=9.0)
    _, rep = to_velocity_world(a, 0.0, Limits())
    s = str(rep)
    assert "speed" in s and "climb" in s and "yaw_rate" in s and "requested" in s
    assert str(Clamped()) == "clamped: none"
    assert not Clamped().any


def test_limits_reject_impossible_envelopes():
    with pytest.raises(ValueError):
        Limits(min_altitude_m=2.0, max_altitude_m=1.0)
    with pytest.raises(ValueError):
        Limits(max_speed_ms=0.0)


# ------------------------------------------------------------------ yaw sign
def test_yaw_sign_flips_only_the_yaw():
    a = Action("velocity", "body_flu", [1.0, 0, 0], yaw_rate_rad_s=0.5)
    pos, _ = to_hover(a, 0.0, 0.6, Limits(), yaw_sign=1)
    neg, _ = to_hover(a, 0.0, 0.6, Limits(), yaw_sign=-1)
    assert math.isclose(pos.vx, neg.vx) and math.isclose(pos.vy, neg.vy)
    assert math.isclose(pos.yaw_rate, -neg.yaw_rate)


@pytest.mark.parametrize("bad", [0, 2, -2, 0.5])
def test_yaw_sign_must_be_plus_or_minus_one(bad):
    with pytest.raises(ValueError, match="yaw_sign"):
        to_hover(Action("velocity", "body_flu", FWD), 0.0, 0.6, Limits(), bad)


@pytest.mark.parametrize("ok", [1, -1, 1.0, -1.0])
def test_yaw_sign_accepts_the_float_spellings(ok):
    """1.0 == 1 in Python and multiplies correctly, so it is not an error."""
    a = Action("velocity", "body_flu", FWD, yaw_rate_rad_s=0.5)
    cmd, _ = to_hover(a, 0.0, 0.6, Limits(), ok)
    assert math.isclose(cmd.yaw_rate, ok * 0.5)


# ------------------------------------------------------- velocity passthrough
def _vel_spec(**kw):
    kw.setdefault("kind", "velocity")
    kw.setdefault("frame", "body_flu")
    kw.setdefault("yaw_mode", "fixed")
    return ActionSpec(**kw)


def test_passthrough_refuses_an_acceleration_spec():
    with pytest.raises(ValueError, match="needs VelocityIntegrator"):
        VelocityPassthrough(ActionSpec(kind="acceleration"), 0.1)


def test_passthrough_refuses_an_acceleration_action():
    p = VelocityPassthrough(_vel_spec(), 0.1)
    p.reset(0.0)
    with pytest.raises(ValueError, match="expected a velocity action"):
        p.step(Action(kind="acceleration", frame="body_flu",
                      vector=np.array([1.0, 0.0, 0.0])), current_yaw_rad=0.0)


def test_passthrough_refuses_to_step_before_reset():
    p = VelocityPassthrough(_vel_spec(), 0.1)
    with pytest.raises(RuntimeError, match="before reset"):
        p.step(Action(kind="velocity", frame="body_flu",
                      vector=np.array([1.0, 0.0, 0.0])), current_yaw_rad=0.0)


def test_passthrough_does_not_accumulate():
    """The property that separates it from the integrator."""
    p = VelocityPassthrough(_vel_spec(), 0.1, Limits(max_speed_ms=9.0))
    p.reset(0.0)
    a = Action(kind="velocity", frame="world_enu", vector=np.array([1.0, 0.0, 0.0]))
    for _ in range(10):
        out, _ = p.step(a, current_yaw_rad=0.0)
    assert np.allclose(out.vector, [1.0, 0.0, 0.0])


def test_passthrough_clamps_state_and_reports_it():
    p = VelocityPassthrough(_vel_spec(), 0.1,
                            Limits(max_speed_ms=0.5, max_climb_ms=0.25))
    p.reset(0.0)
    out, rep = p.step(Action(kind="velocity", frame="world_enu",
                             vector=np.array([3.0, 4.0, 2.0])),
                      current_yaw_rad=0.0)
    assert rep.state_saturated
    assert math.hypot(out.vector[0], out.vector[1]) == pytest.approx(0.5)
    assert out.vector[2] == pytest.approx(0.25)


def test_passthrough_rotates_body_into_world():
    p = VelocityPassthrough(_vel_spec(), 0.1, Limits(max_speed_ms=9.0))
    p.reset(math.pi / 2.0)
    out, _ = p.step(Action(kind="velocity", frame="body_flu",
                           vector=np.array([1.0, 0.0, 0.0])),
                    current_yaw_rad=math.pi / 2.0)
    assert out.frame == "world_enu"
    assert out.vector[0] == pytest.approx(0.0, abs=1e-12)
    assert out.vector[1] == pytest.approx(1.0)


def test_passthrough_divergence_is_commanded_minus_measured():
    p = VelocityPassthrough(_vel_spec(), 0.1, Limits(max_speed_ms=9.0))
    p.reset(0.0)
    a = Action(kind="velocity", frame="world_enu", vector=np.array([1.0, 0.0, 0.0]))
    _, rep = p.step(a, current_yaw_rad=0.0)
    assert math.isnan(rep.divergence_ms)          # nothing measured
    _, rep = p.step(a, current_yaw_rad=0.0,
                    measured_velocity_world=(0.7, 0.0, 0.0))
    assert rep.divergence_ms == pytest.approx(0.3)
    assert rep.mode == "passthrough"


def test_passthrough_reports_yaw_error_against_the_episode_heading():
    p = VelocityPassthrough(_vel_spec(), 0.1)
    p.reset(0.0)
    assert math.degrees(p.yaw_error_rad(math.radians(30.0))) == pytest.approx(30.0)


def test_make_action_stage_picks_by_kind():
    assert isinstance(make_action_stage(ActionSpec(kind="acceleration"), 0.1),
                      VelocityIntegrator)
    assert isinstance(make_action_stage(_vel_spec(), 0.1), VelocityPassthrough)
    with pytest.raises(ValueError, match="commands a place"):
        make_action_stage(ActionSpec(kind="position", frame="world_enu"), 0.1)
