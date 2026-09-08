"""The contract, in force during an actual run.

`test_an_acceleration_policy_can_now_fly` is the one that matters: before the
integrator existed, a policy trained by `splat_rl_env.py` raised inside
`to_hover` on its first tick and could not be flown at all.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splat_hitl.commands import Action, action_from_raw
from splat_hitl.contract import (ActionSpec, ControlSpec, ObservationSpec,
                                 PolicyContract, metric_splat_depth_ppo_v1)
from splat_hitl.policy import HoverPolicy, Policy
from splat_hitl.renderer import FakeRenderer
from splat_hitl.runtime import (FINISHED, HOLDING, RUNNING, PoseSample, Runtime,
                                RuntimeConfig, ScriptedPoseSource)
from splat_hitl.sensor import DepthEncoding, SensorModel

ROOM = (6.0, 5.0, 2.5)


def contract(rate_hz=15.0, yaw_mode="fixed", width=32, height=24):
    """A small-image version of the real thing, so tests stay fast."""
    c = metric_splat_depth_ppo_v1(90.0)
    sensor = SensorModel(name="t", width=width, height=height, fov_x_deg=90.0,
                         depth=DepthEncoding(kind="metric", near_m=0.05,
                                             far_m=4.0, empty_depth_m=4.0),
                         rate_hz=rate_hz)
    obs = ObservationSpec(**{**c.observation.__dict__, "sensor": sensor})
    act = ActionSpec(**{**c.action.__dict__, "yaw_mode": yaw_mode})
    return PolicyContract(name="t", observation=obs, action=act,
                          control=ControlSpec(rate_hz=rate_hz))


def yaw_quat(yaw_rad):
    return (0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0))


def pose(t, x=2.0, y=2.0, z=0.6, yaw=0.0):
    return PoseSample(t, np.array([x, y, z], float), yaw_quat(yaw))


def feed(n, dt, yaw=lambda i: 0.0, x=lambda i: 2.0):
    """One pose per tick, stamped on the SAME clock the loop is stepped with.

    ScriptedPoseSource repeats its last sample forever, so a short list plus a
    long run looks exactly like a bridge that died -- the staleness watchdog
    fires and the test measures the watchdog instead of what it meant to.
    """
    return [pose(i * dt, x=x(i), yaw=yaw(i)) for i in range(n)]


class AccelPolicy(Policy):
    """Emits a raw unit action, exactly as a trained network head would."""
    name = "accel"

    def __init__(self, raw=(1.0, 0.0, 0.0), spec=None):
        self.raw, self.spec = raw, spec
        self.seen = []

    def act(self, obs, state):
        self.seen.append(obs.policy_input)
        return action_from_raw(self.spec, self.raw)


def build(c, policy=None, samples=None, cfg=None, n=64):
    src = ScriptedPoseSource(samples if samples is not None
                             else feed(n, c.control.dt_s))
    p = policy or AccelPolicy(spec=c.action)
    cfg = cfg or RuntimeConfig(contract=c, max_duration_s=1e6)
    return Runtime(src, FakeRenderer(c.observation.sensor, ROOM), p,
                   config=cfg), src


def run(rt, src, n, dt):
    out = []
    for i in range(n):
        src.advance()
        out.append(rt.step(now=i * dt))
    return out


# ----------------------------------------------------------- construction
def test_contract_sensor_must_be_the_renderer_sensor():
    c = contract()
    other = SensorModel(name="o", width=64, height=48, fov_x_deg=90.0)
    with pytest.raises(ValueError, match="not the renderer's"):
        Runtime(ScriptedPoseSource([pose(0)]), FakeRenderer(other, ROOM),
                HoverPolicy(), config=RuntimeConfig(contract=c))


def test_policy_declaring_a_different_contract_is_refused():
    c = contract()
    p = HoverPolicy()
    p.contract_fingerprint = "deadbeefcafe"
    with pytest.raises(ValueError, match="declares contract"):
        Runtime(ScriptedPoseSource([pose(0)]), FakeRenderer(c.observation.sensor, ROOM),
                p, config=RuntimeConfig(contract=c))


def test_matching_contract_fingerprint_is_accepted():
    c = contract()
    p = HoverPolicy()
    p.contract_fingerprint = c.fingerprint()
    Runtime(ScriptedPoseSource([pose(0)]), FakeRenderer(c.observation.sensor, ROOM),
            p, config=RuntimeConfig(contract=c))


# ------------------------------------------------------------ observation
def test_policy_input_is_built_to_the_contract_shape():
    c = contract()
    p = AccelPolicy(spec=c.action)
    rt, src = build(c, policy=p)
    run(rt, src, 3, c.control.dt_s)
    assert p.seen and p.seen[0].shape == c.observation.shape
    assert p.seen[0].dtype == np.float32
    assert 0.0 <= p.seen[0].min() and p.seen[0].max() <= 1.0


def test_without_a_contract_policy_input_stays_none():
    """The reference policies read depth_m; nothing silently changes for them."""
    c = contract()
    p = AccelPolicy(spec=c.action)
    rt = Runtime(ScriptedPoseSource([pose(0)]), FakeRenderer(c.observation.sensor, ROOM),
                 p, config=RuntimeConfig())
    rt.poses.advance()
    r = rt.step(now=0.0)
    assert p.seen[0] is None
    # no contract means no integrator, so to_hover still refuses an
    # acceleration -- the refusal this work routes around, not removes
    assert r.reason == "action_error"
    assert any("could not be mapped" in e for e in r.events)


# ------------------------------------------------------------- the action
def test_an_acceleration_policy_can_now_fly():
    """Before the integrator this raised inside to_hover on the first tick."""
    c = contract()
    rt, src = build(c)
    rs = run(rt, src, 5, c.control.dt_s)
    assert all(r.state == RUNNING for r in rs), [r.events for r in rs]
    assert rs[-1].command is not None
    # +x body acceleration at zero yaw accumulates into +x body velocity
    assert rs[-1].command.vx > rs[0].command.vx > 0.0


def test_velocity_grows_by_a_times_dt():
    c = contract()
    rt, src = build(c)
    rs = run(rt, src, 3, c.control.dt_s)
    step = c.action.scale * c.control.dt_s
    assert rs[0].command.vx == pytest.approx(step, rel=1e-6)
    assert rs[1].command.vx == pytest.approx(2 * step, rel=1e-6)


def test_episode_heading_is_captured_from_the_first_pose():
    c = contract()
    rt, src = build(c, samples=feed(2, c.control.dt_s,
                                    yaw=lambda i: math.radians(40.0)))
    run(rt, src, 1, c.control.dt_s)
    assert rt.episode_yaw_rad == pytest.approx(math.radians(40.0))
    assert any("episode heading 40.0 deg" in e for e in rt.history[0].events)


def test_divergence_against_the_measured_velocity_is_recorded():
    c = contract()
    dt = c.control.dt_s
    rt, src = build(c, samples=feed(3, dt, x=lambda i: 2.0 + 0.2 * i))
    rs = run(rt, src, 3, c.control.dt_s)
    assert math.isfinite(rs[-1].divergence_ms)
    assert rs[-1].divergence_ms > 0.0        # the drone is moving, the model is not


# ---------------------------------------------------------------- the yaw
def test_yaw_drift_stops_the_policy_flying():
    """Past the threshold every observation is of a view never trained on."""
    c = contract(yaw_mode="fixed")
    rt, src = build(c, samples=feed(2, c.control.dt_s,
                                    yaw=lambda i: math.radians(45.0) * i))
    rs = run(rt, src, 2, c.control.dt_s)
    assert rs[0].state == RUNNING
    assert rs[1].state == HOLDING
    assert any("no longer valid" in e for e in rs[1].events)
    assert rs[1].yaw_error_rad == pytest.approx(math.radians(45.0))


def test_free_yaw_does_not_trigger_the_drift_guard():
    c = contract(yaw_mode="free")
    rt, src = build(c, samples=feed(2, c.control.dt_s,
                                    yaw=lambda i: math.radians(80.0) * i))
    rs = run(rt, src, 2, c.control.dt_s)
    assert all(r.state == RUNNING for r in rs)


# --------------------------------------------------------------- the rate
def test_a_wrong_step_rate_stops_the_run():
    """15 Hz contract driven at 30 Hz: silently twice the velocity."""
    c = contract(rate_hz=15.0)
    rt, src = build(c, samples=feed(64, 1.0 / 30.0))
    rs = run(rt, src, 40, 1.0 / 30.0)
    # the check fires once, mid-run -- the last tick is just the finished state
    fired = [r for r in rs if r.reason == "rate_mismatch"]
    assert fired, [r.reason for r in rs]
    assert any("makes it faster" in e for e in fired[0].events)
    assert rs[-1].state == FINISHED


def test_the_right_rate_runs_clean():
    c = contract(rate_hz=15.0)
    rt, src = build(c)
    rs = run(rt, src, 40, c.control.dt_s)
    assert all(r.state == RUNNING for r in rs)


def test_a_rate_inside_tolerance_is_accepted():
    c = contract(rate_hz=15.0)
    slow = (1.0 / 15.0) * 1.1
    rt, src = build(c, samples=feed(64, slow),
                    cfg=RuntimeConfig(contract=c, max_duration_s=1e6,
                                      rate_tolerance=0.20))
    rs = run(rt, src, 40, slow)      # 10% slow
    assert all(r.state == RUNNING for r in rs)
