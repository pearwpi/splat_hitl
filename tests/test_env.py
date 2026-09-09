"""The training environment, and its parity with the flight runtime.

`test_env_and_runtime_agree` is the one that earns this module's existence: it
drives both with the same actions and asserts the policy sees the same numbers
and the drone is asked for the same velocity. If that ever fails, sim-to-real
parity has been broken by an edit, not discovered on a drone.
"""
import math

import numpy as np
import pytest

from splat_hitl.collision import synthetic_room
from splat_hitl.commands import Limits, action_from_raw
from splat_hitl.contract import (ActionSpec, ControlSpec, ObservationSpec,
                                 PolicyContract, metric_splat_depth_ppo_v1)
from splat_hitl.env import (COURSE_COMPLETE, MAX_DURATION, VIRTUAL_COLLISION,
                            VIRTUAL_OUTSIDE, EnvConfig, SplatEnv)
from splat_hitl.gates import Gate, GateCourse
from splat_hitl.policy import Policy
from splat_hitl.recorder import TERMINATION_MAP
from splat_hitl.renderer import FakeRenderer
from splat_hitl.runtime import (PoseSample, Runtime, RuntimeConfig,
                                ScriptedPoseSource)
from splat_hitl.sensor import DepthEncoding, SensorModel

ROOM = (6.0, 5.0, 2.5)
START = (1.0, 2.5, 0.6)
BIG = Limits(max_speed_ms=9.0, max_climb_ms=9.0)


def contract(rate_hz=15.0):
    base = metric_splat_depth_ppo_v1(90.0)
    sensor = SensorModel(name="t", width=32, height=24, fov_x_deg=90.0,
                         depth=DepthEncoding(kind="metric", near_m=0.05,
                                             far_m=4.0, empty_depth_m=4.0),
                         rate_hz=rate_hz)
    return PolicyContract(
        name="t",
        observation=ObservationSpec(**{**base.observation.__dict__,
                                       "sensor": sensor}),
        action=base.action, control=ControlSpec(rate_hz=rate_hz))


def course(x=3.0):
    return GateCourse([Gate("g1", np.array([x, 2.5, 0.6]),
                            np.array([1.0, 0.0, 0.0]), 1.2, 1.2)])


def make(c=None, cfg=None, esdf=None, gate_x=3.0):
    c = c or contract()
    return SplatEnv(c, FakeRenderer(c.observation.sensor, ROOM), course(gate_x),
                    esdf, cfg or EnvConfig(start_position_m=START, limits=BIG,
                                           max_steps=400))


# ------------------------------------------------------------- construction
def test_sensor_must_match_the_renderer():
    c = contract()
    other = SensorModel(name="o", width=64, height=48, fov_x_deg=90.0)
    with pytest.raises(ValueError, match="never seen the world"):
        SplatEnv(c, FakeRenderer(other, ROOM), course())


def test_a_velocity_contract_is_refused():
    c = contract()
    vel = PolicyContract(name="v", observation=c.observation,
                         action=ActionSpec(kind="velocity", frame="body_flu"),
                         control=c.control)
    with pytest.raises(ValueError, match="needs no integrator"):
        SplatEnv(vel, FakeRenderer(c.observation.sensor, ROOM), course())


# ------------------------------------------------------------------- basics
def test_reset_returns_a_primed_observation():
    env = make()
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_shape == (4, 24, 32)
    assert obs.dtype == np.float32
    assert np.allclose(obs[0], obs[-1])          # primed by repetition
    assert info["steps"] == 0 and info["gates_passed"] == 0


def test_step_before_reset_or_after_done_is_refused():
    env = make()
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0.0, 0.0, 0.0])


def test_position_follows_the_trapezoidal_rule():
    c = contract()
    env = make(c)
    env.reset(seed=0)
    dt, scale = c.control.dt_s, c.action.scale
    env.step([1.0, 0.0, 0.0])
    # v goes 0 -> a*dt, so the first step moves 0.5*a*dt^2
    assert env.position_m[0] - START[0] == pytest.approx(0.5 * scale * dt * dt,
                                                         rel=1e-9)
    env.step([1.0, 0.0, 0.0])
    assert env.position_m[0] - START[0] == pytest.approx(2.0 * scale * dt * dt,
                                                         rel=1e-9)


def test_flying_through_the_gate_completes_the_course():
    env = make()
    env.reset(seed=0)
    for _ in range(400):
        obs, r, term, trunc, info = env.step([1.0, 0.0, 0.0])
        if term or trunc:
            break
    assert term and info["reason"] == COURSE_COMPLETE
    assert info["gates_passed"] == 1
    assert r > 50.0                              # goal reward dominates


def test_sitting_still_times_out():
    env = make(cfg=EnvConfig(start_position_m=START, limits=BIG, max_steps=20))
    env.reset(seed=0)
    for _ in range(20):
        obs, r, term, trunc, info = env.step([0.0, 0.0, 0.0])
    assert trunc and not term and info["reason"] == MAX_DURATION


def _fly_forward(env, n=400):
    for _ in range(n):
        obs, r, term, trunc, info = env.step([1.0, 0.0, 0.0])
        if term or trunc:
            return r, term, trunc, info
    return r, term, trunc, info


def test_flying_into_an_obstacle_is_a_collision():
    esdf = synthetic_room(ROOM, voxel_m=0.05, truncation_m=1.0,
                          obstacles=[{"centre": (3.0, 2.5, 0.6), "radius": 0.3}])
    env = make(esdf=esdf, gate_x=5.9)
    env.reset(seed=0)
    r, term, trunc, info = _fly_forward(env)
    assert term and info["reason"] == VIRTUAL_COLLISION
    assert r < -50.0


def test_leaving_the_mapped_volume_is_its_own_outcome():
    """Not the same failure as hitting something, and not scored as one."""
    esdf = synthetic_room(ROOM, voxel_m=0.05, truncation_m=1.0)
    env = make(esdf=esdf, gate_x=5.9)
    env.reset(seed=0)
    r, term, trunc, info = _fly_forward(env)
    assert term and info["reason"] == VIRTUAL_OUTSIDE


def test_every_reason_is_in_the_runtime_vocabulary():
    """One termination vocabulary, or the two logs cannot be compared."""
    for reason in (COURSE_COMPLETE, VIRTUAL_COLLISION, VIRTUAL_OUTSIDE,
                   MAX_DURATION):
        assert reason in TERMINATION_MAP


def test_progress_toward_the_gate_is_rewarded():
    env = make()
    env.reset(seed=0)
    _, forward, _, _, _ = env.step([1.0, 0.0, 0.0])
    env.reset(seed=0)
    _, backward, _, _, _ = env.step([-1.0, 0.0, 0.0])
    assert forward > backward


def test_yaw_jitter_changes_the_episode_heading():
    cfg = EnvConfig(start_position_m=START, limits=BIG,
                    start_yaw_jitter_rad=math.radians(30.0))
    env = make(cfg=cfg)
    ys = set()
    for s in range(5):
        _, info = env.reset(seed=s)
        ys.add(round(info["episode_yaw_rad"], 6))
    assert len(ys) > 1
    assert all(abs(y) <= math.radians(30.0) + 1e-9 for y in ys)


# -------------------------------------------------------------------- parity
class Replay(Policy):
    """Emits a fixed raw action, the way a frozen network would."""
    name = "replay"

    def __init__(self, spec, raw):
        self.spec, self.raw = spec, raw
        self.inputs = []

    def act(self, obs, state):
        self.inputs.append(obs.policy_input.copy())
        return action_from_raw(self.spec, self.raw)


def test_env_and_runtime_agree():
    """Same actions in, same observations out, same velocity commanded.

    Not a tautology: the two drive different code paths -- the env integrates
    and moves the drone itself, the runtime takes poses from outside and only
    converts. If an edit ever changes one, this fails on a laptop instead of on
    a drone.
    """
    c = contract()
    raw = [0.6, 0.3, 0.0]
    n = 12

    env = make(c)
    env.reset(seed=0)
    positions = [env.position_m.copy()]
    env_obs = []
    env_vel = []
    for _ in range(n):
        obs, _, term, trunc, _ = env.step(raw)
        assert not (term or trunc)
        positions.append(env.position_m.copy())
        env_obs.append(obs)
        env_vel.append(env.integrator.velocity_world.copy())

    # replay those poses through the flight runtime
    yaw = env.integrator.episode_yaw_rad
    q = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    dt = c.control.dt_s
    samples = [PoseSample(i * dt, p, q) for i, p in enumerate(positions)]
    pol = Replay(c.action, raw)
    rt = Runtime(ScriptedPoseSource(samples),
                 FakeRenderer(c.observation.sensor, ROOM), pol,
                 config=RuntimeConfig(contract=c, limits=BIG,
                                      max_duration_s=1e6, rate_check_after=10**6))
    rt_vel = []
    for i in range(n + 1):
        rt.poses.advance()
        r = rt.step(now=i * dt)
        assert r.command is not None, r.events
        rt_vel.append(rt.integrator.velocity_world.copy())

    # observation at the same position must be byte-identical
    # encode_frame now returns (channels_per_frame, H, W), so priming the
    # history is a repeat along the channel axis rather than a new one
    first = env.builder.encode_frame(
        rt.renderer.render(positions[0], (0.0, 0.0, yaw)).depth_m)
    assert np.array_equal(pol.inputs[0],
                          np.repeat(first, c.observation.history, axis=0))
    for k in range(n):
        assert np.array_equal(pol.inputs[k + 1], env_obs[k]), k

    # and the velocity the drone is asked for must match, step for step
    for k in range(n):
        assert np.allclose(rt_vel[k], env_vel[k], atol=1e-12), k
