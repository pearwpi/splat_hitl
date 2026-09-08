"""The contract, the observation builder, and the integrator.

The most valuable test here is `test_matches_trainer_integration`, which
reimplements the trainer's own arithmetic and asserts our path produces the
same numbers. Every other test guards a field; that one guards the claim.
"""
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from splat_hitl.commands import (Action, Limits, VelocityIntegrator,
                                 action_from_raw, to_hover)
from splat_hitl.contract import (ActionSpec, ControlSpec, ObservationSpec,
                                 PolicyContract, metric_splat_depth_ppo_v1)
from splat_hitl.observation import ObservationBuilder
from splat_hitl.sensor import SensorModel

FOV = 69.4


def C():
    return metric_splat_depth_ppo_v1(FOV)


# ------------------------------------------------------------------ contract
def test_fingerprint_ignores_name_and_notes():
    a = metric_splat_depth_ppo_v1(FOV, name="a", notes="one")
    b = metric_splat_depth_ppo_v1(FOV, name="b", notes="two")
    assert a.fingerprint() == b.fingerprint()
    a.assert_compatible(b)


@pytest.mark.parametrize("mutate", [
    lambda c: ObservationSpec(**{**c.observation.__dict__, "history": 2}),
    lambda c: ObservationSpec(**{**c.observation.__dict__, "clip_far_m": 8.0}),
    lambda c: ObservationSpec(**{**c.observation.__dict__, "prime": "zeros"}),
    lambda c: ObservationSpec(**{**c.observation.__dict__,
                                 "history_order": "newest_first"}),
])
def test_observation_changes_are_visible(mutate):
    """Every one of these was invisible to SensorModel.fingerprint()."""
    base = C()
    other = PolicyContract(name="x", observation=mutate(base), action=base.action,
                           control=base.control)
    assert other.fingerprint() != base.fingerprint()


def test_control_rate_changes_the_fingerprint():
    """15 Hz vs 30 Hz was the silent one: a double integrator at 2x rate."""
    base = C()
    other = PolicyContract(name="x", observation=base.observation,
                           action=base.action, control=ControlSpec(rate_hz=30.0))
    assert other.fingerprint() != base.fingerprint()
    with pytest.raises(ValueError, match="control.rate_hz"):
        base.assert_compatible(other)


def test_field_of_view_changes_the_fingerprint():
    """The FOV leaks in from the capture camera unless it is stated."""
    assert metric_splat_depth_ppo_v1(69.4).fingerprint() != \
           metric_splat_depth_ppo_v1(90.0).fingerprint()


def test_mismatch_names_the_field_not_just_the_hash():
    base = C()
    other = PolicyContract(name="x", observation=base.observation,
                           action=ActionSpec(**{**base.action.__dict__,
                                                "scale": 3.0}),
                           control=base.control)
    with pytest.raises(ValueError) as e:
        base.assert_compatible(other)
    assert "action.scale: 1.5 != 3.0" in str(e.value)


def test_round_trip_survives_comment_keys():
    """The shipped example config carries `_comment`; it must load back."""
    c = C()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "contract.json"
        c.save(p)
        raw = json.loads(p.read_text())
        raw["_comment"] = "humans read this file"
        raw["observation"]["_comment"] = "and this part of it"
        raw["observation"]["sensor"]["_comment"] = "and this"
        p.write_text(json.dumps(raw))
        back = PolicyContract.load(p)
    assert back.fingerprint() == c.fingerprint()
    c.assert_compatible(back)


def test_clip_unit_requires_a_clip():
    s = SensorModel(name="s", width=8, height=8, fov_x_deg=60.0)
    with pytest.raises(ValueError, match="clip_far_m"):
        ObservationSpec(sensor=s, normalize="clip_unit", clip_far_m=None)


def test_fov_half_tan():
    assert C().observation.fov_x_half_tan == pytest.approx(
        math.tan(math.radians(FOV) / 2.0))


# --------------------------------------------------------------- observation
def test_encoding_matches_the_trainer_exactly():
    """nan/+inf -> clip, -inf -> 0, clip then divide. Read off _observe()."""
    b = ObservationBuilder(C().observation)
    b.reset()
    d = np.full((64, 96), 2.0)
    d[0, 0], d[0, 1], d[0, 2], d[0, 3] = np.nan, np.inf, -np.inf, 99.0
    o = b.push(d)
    assert o.shape == (4, 64, 96) and o.dtype == np.float32
    assert o[0, 5, 5] == pytest.approx(0.5)      # 2 m of 4 m
    assert o[0, 0, 0] == pytest.approx(1.0)      # nan  -> far
    assert o[0, 0, 1] == pytest.approx(1.0)      # +inf -> far
    assert o[0, 0, 2] == pytest.approx(0.0)      # -inf -> touching
    assert o[0, 0, 3] == pytest.approx(1.0)      # clipped


def test_first_frame_primes_by_repetition_not_zeros():
    """Zeros would show the policy a wall in its face on the first tick."""
    b = ObservationBuilder(C().observation)
    b.reset()
    o = b.push(np.full((64, 96), 2.0))
    assert np.allclose(o[0], o[3])
    assert o.min() > 0.0


def test_zero_priming_is_available_and_different():
    spec = ObservationSpec(**{**C().observation.__dict__, "prime": "zeros"})
    b = ObservationBuilder(spec)
    b.reset()
    o = b.push(np.full((64, 96), 2.0))
    assert np.allclose(o[:3], 0.0) and np.allclose(o[3], 0.5)


def test_history_is_oldest_first():
    b = ObservationBuilder(C().observation)
    b.reset()
    b.push(np.full((64, 96), 2.0))
    o = b.push(np.full((64, 96), 1.0))
    assert o[0, 0, 0] == pytest.approx(0.5)      # oldest
    assert o[3, 0, 0] == pytest.approx(0.25)     # newest


def test_newest_first_reverses_it():
    spec = ObservationSpec(**{**C().observation.__dict__,
                              "history_order": "newest_first"})
    b = ObservationBuilder(spec)
    b.reset()
    b.push(np.full((64, 96), 2.0))
    o = b.push(np.full((64, 96), 1.0))
    assert o[0, 0, 0] == pytest.approx(0.25)


def test_wrong_resolution_is_refused_not_resized():
    b = ObservationBuilder(C().observation)
    b.reset()
    with pytest.raises(ValueError, match="has not seen the other"):
        b.push(np.zeros((32, 32)))


def test_reset_clears_the_history():
    b = ObservationBuilder(C().observation)
    b.reset()
    b.push(np.full((64, 96), 1.0))
    b.reset()
    o = b.push(np.full((64, 96), 3.0))
    assert np.allclose(o[0], o[3]) and o[0, 0, 0] == pytest.approx(0.75)


# ---------------------------------------------------------------- raw action
def test_action_from_raw_clips_scales_and_limits_the_norm():
    spec = C().action
    a = action_from_raw(spec, [5.0, 0.0, 0.0])       # clipped to 1 -> 1.5
    assert a.vector[0] == pytest.approx(1.5)
    a = action_from_raw(spec, [1.0, 1.0, 1.0])       # norm 2.598 -> 1.5
    assert float(np.linalg.norm(a.vector)) == pytest.approx(1.5)
    assert a.vector[0] == pytest.approx(1.5 / math.sqrt(3.0))


def test_norm_limit_is_not_per_axis_clipping():
    """Per-axis clipping would leave the diagonal 73% too fast."""
    a = action_from_raw(C().action, [1.0, 1.0, 0.0])
    assert float(np.linalg.norm(a.vector)) == pytest.approx(1.5)
    assert a.vector[0] < 1.5


# ---------------------------------------------------------------- integrator
def test_step_before_reset_refuses():
    it = VelocityIntegrator(C().action, dt_s=1 / 15.0)
    with pytest.raises(RuntimeError, match="before reset"):
        it.step(action_from_raw(C().action, [1, 0, 0]), 0.0)


def test_matches_trainer_integration():
    """Our path against the trainer's own arithmetic, step for step.

    Trainer (splat_rl_env.step):
        action              = clip(a, -1, 1)
        acceleration_body   = action * max_acceleration_m_s2
        (norm-limited to max_acceleration_m_s2)
        acceleration_scene  = episode_basis @ acceleration_body
        velocity_raw        = previous_velocity + acceleration_scene * dt

    `episode_basis` has columns (forward, lateral, up) for a fixed episode
    heading, which for a level drone is exactly Rz(yaw) -- so our body_flu
    rotation by the EPISODE yaw must reproduce it.
    """
    c = C()
    dt, yaw = c.control.dt_s, math.radians(37.0)
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    lat = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    episode_basis = np.column_stack([fwd, lat, up])

    it = VelocityIntegrator(c.action, dt_s=dt, limits=Limits(max_speed_ms=99.0,
                                                             max_climb_ms=99.0))
    it.reset(episode_yaw_rad=yaw)

    v_trainer = np.zeros(3)
    rng = np.random.default_rng(0)
    for _ in range(25):
        raw = rng.uniform(-1.2, 1.2, size=3)
        # -- trainer
        a_body = np.clip(raw, -1.0, 1.0) * c.action.scale
        n = np.linalg.norm(a_body)
        if n > c.action.scale:
            a_body = a_body * (c.action.scale / n)
        v_trainer = v_trainer + (episode_basis @ a_body) * dt
        # -- ours, with the drone drifting in yaw the whole time. The trainer
        # never rotates its basis, so a correct implementation must ignore this
        # drift entirely -- and a test where current_yaw == episode_yaw could
        # not tell the two apart.
        drifting_yaw = yaw + math.radians(rng.uniform(-45.0, 45.0))
        act, rep = it.step(action_from_raw(c.action, raw),
                           current_yaw_rad=drifting_yaw)
        assert np.allclose(act.vector, v_trainer, atol=1e-12)
        assert np.allclose(rep.velocity_world_open_loop, v_trainer, atol=1e-12)


def test_fixed_yaw_uses_the_episode_heading_not_the_current_one():
    """The policy never saw the scene from any other heading."""
    c = C()
    it = VelocityIntegrator(c.action, dt_s=1.0)
    it.reset(episode_yaw_rad=0.0)
    act, rep = it.step(action_from_raw(c.action, [1, 0, 0]),
                       current_yaw_rad=math.radians(90.0))
    assert act.vector[0] == pytest.approx(1.5)      # still +x, not +y
    assert act.vector[1] == pytest.approx(0.0, abs=1e-12)
    assert rep.yaw_error_rad == pytest.approx(math.radians(90.0))


def test_free_yaw_follows_the_drone():
    spec = ActionSpec(**{**C().action.__dict__, "yaw_mode": "free"})
    it = VelocityIntegrator(spec, dt_s=1.0)
    it.reset(episode_yaw_rad=0.0)
    act, _ = it.step(action_from_raw(spec, [1, 0, 0]),
                     current_yaw_rad=math.radians(90.0))
    assert act.vector[1] == pytest.approx(1.5)      # rotated into +y


def test_fixed_yaw_emits_zero_yaw_rate_by_default():
    c = C()
    it = VelocityIntegrator(c.action, dt_s=1.0)
    it.reset(episode_yaw_rad=0.0)
    act, _ = it.step(action_from_raw(c.action, [0, 0, 0], yaw_rate_rad_s=2.0),
                     current_yaw_rad=math.radians(20.0))
    assert act.yaw_rate_rad_s == pytest.approx(0.0)


def test_hold_yaw_gain_corrects_toward_the_episode_heading():
    c = C()
    it = VelocityIntegrator(c.action, dt_s=1.0, hold_yaw_gain=2.0)
    it.reset(episode_yaw_rad=0.0)
    act, _ = it.step(action_from_raw(c.action, [0, 0, 0]),
                     current_yaw_rad=math.radians(30.0))
    assert act.yaw_rate_rad_s == pytest.approx(-2.0 * math.radians(30.0))


def test_state_is_clamped_so_the_integrator_cannot_wind_up():
    """Without this, a reversal does nothing for seconds."""
    c = C()
    it = VelocityIntegrator(c.action, dt_s=1.0, limits=Limits(max_speed_ms=1.0))
    it.reset(episode_yaw_rad=0.0)
    for _ in range(20):
        _, rep = it.step(action_from_raw(c.action, [1, 0, 0]), 0.0)
    assert float(np.linalg.norm(it.velocity_world[:2])) == pytest.approx(1.0)
    assert rep.state_saturated
    # one reversal must show up immediately, not after unwinding 30 m/s
    act, _ = it.step(action_from_raw(c.action, [-1, 0, 0]), 0.0)
    assert act.vector[0] < 1.0


def test_divergence_is_reported_when_a_measurement_is_supplied():
    c = C()
    # A generous envelope on purpose: the default 1.5 m/s cap would clamp both
    # velocities to the same value and hide the divergence this is testing.
    it = VelocityIntegrator(c.action, dt_s=1.0, limits=Limits(max_speed_ms=9.0))
    it.reset(episode_yaw_rad=0.0)
    _, rep = it.step(action_from_raw(c.action, [1, 0, 0]), 0.0,
                     measured_velocity_world=[0.5, 0.0, 0.0])
    assert rep.velocity_world_open_loop[0] == pytest.approx(1.5)
    assert rep.velocity_world_measured[0] == pytest.approx(2.0)
    assert rep.divergence_ms == pytest.approx(0.5)


def test_measured_mode_refuses_to_run_blind():
    spec = ActionSpec(**{**C().action.__dict__, "integrator": "measured"})
    it = VelocityIntegrator(spec, dt_s=1.0)
    it.reset(episode_yaw_rad=0.0)
    with pytest.raises(ValueError, match="which controller is flying"):
        it.step(action_from_raw(spec, [1, 0, 0]), 0.0)


def test_velocity_output_is_accepted_by_to_hover():
    """The whole point: the integrator's output must be flyable."""
    c = C()
    it = VelocityIntegrator(c.action, dt_s=c.control.dt_s)
    it.reset(episode_yaw_rad=0.0)
    act, _ = it.step(action_from_raw(c.action, [1, 0, 0]), 0.0)
    cmd, clamped = to_hover(act, current_yaw_rad=0.0, target_altitude_m=0.6)
    assert cmd.vx > 0.0 and cmd.z_distance == pytest.approx(0.6)


def test_acceleration_is_still_refused_by_to_hover_directly():
    """The refusal stays. The integrator is the answer, not a bypass."""
    with pytest.raises(ValueError, match="cannot build a Hover command"):
        to_hover(action_from_raw(C().action, [1, 0, 0]), 0.0, 0.6)


def test_integrator_refuses_a_velocity_contract():
    spec = ActionSpec(kind="velocity", frame="body_flu", scale=1.0)
    with pytest.raises(ValueError, match="needs no"):
        VelocityIntegrator(spec, dt_s=1 / 15.0)


def test_fov_recovered_from_transforms_is_resolution_independent():
    """The worker's fallback is the capture camera's FOV, nothing else."""
    from splat_hitl.contract import fov_x_deg_from_transforms
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "transforms.json"
        p.write_text(json.dumps({"fl_x": 600.0, "fl_y": 600.0, "w": 1280, "h": 720}))
        fov = fov_x_deg_from_transforms(p)
    assert fov == pytest.approx(math.degrees(2 * math.atan(1280 / 1200.0)))
    # the render width never enters the formula
    assert fov == pytest.approx(93.6, abs=0.1)


def test_transforms_without_focal_length_is_refused():
    from splat_hitl.contract import fov_x_deg_from_transforms
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "transforms.json"
        p.write_text(json.dumps({"w": 1280}))
        with pytest.raises(ValueError, match="fl_x"):
            fov_x_deg_from_transforms(p)


def test_the_shipped_example_config_loads():
    """Reads OUR OWN INPUT, not our own output.

    `SensorModel.from_dict` once rejected the example config this repo ships,
    because eleven tests round-tripped generated objects and none of them ever
    opened the file a student would actually be handed. This test opens it.
    """
    here = Path(__file__).resolve().parent.parent
    c = PolicyContract.load(here / "config" / "policy_contract.example.json")
    assert c.observation.shape == (4, 64, 96)
    assert c.action.kind == "acceleration" and c.action.yaw_mode == "fixed"
    assert c.control.rate_hz == 15.0
    assert c.trainer_action_semantics == \
        "body_acceleration_fraction_double_integrator_v1"
    # and it must be usable, not merely parseable
    b = ObservationBuilder(c.observation)
    b.reset()
    assert b.push(np.full((64, 96), 2.0)).shape == (4, 64, 96)
    it = VelocityIntegrator(c.action, dt_s=c.control.dt_s)
    it.reset(episode_yaw_rad=0.0)
    act, _ = it.step(action_from_raw(c.action, [1, 0, 0]), 0.0)
    to_hover(act, current_yaw_rad=0.0, target_altitude_m=0.6)
