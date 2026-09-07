"""Runtime and state-machine tests.

Every degradation path is exercised here, including the ones that are unsafe or
impossible to produce deliberately in a lab: a renderer that throws mid-flight,
a policy that returns NaN, a pose feed that goes stale at altitude.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splat_hitl.collision import CollisionMonitor, synthetic_room
from splat_hitl.commands import Action
from splat_hitl.gates import Gate, GateCourse
from splat_hitl.policy import ForwardPolicy, GateSeekPolicy, HoverPolicy, Policy
from splat_hitl.renderer import FakeRenderer
from splat_hitl.runtime import (FINISHED, HOLDING, LANDING, RUNNING, PoseSample,
                                Runtime, RuntimeConfig, ScriptedPoseSource)
from splat_hitl.sensor import SensorModel

ROOM = (4.0, 3.0, 2.5)
SENSOR = SensorModel(name="t", width=32, height=24, fov_x_deg=90.0)
LEVEL = (0.0, 0.0, 0.0, 1.0)


def pose(t, x, y=1.5, z=0.6, q=LEVEL):
    return PoseSample(t, np.array([x, y, z], float), q)


def build(policy=None, samples=None, course=None, collision=None, cfg=None,
          renderer=None):
    src = ScriptedPoseSource(samples or [pose(0.0, 1.0)])
    return Runtime(src, renderer or FakeRenderer(SENSOR, ROOM),
                   policy or HoverPolicy(), course, collision,
                   cfg or RuntimeConfig()), src


# ------------------------------------------------------------- sensor match
def test_refuses_a_policy_trained_through_a_different_sensor():
    p = HoverPolicy()
    p.sensor_fingerprint = "deadbeefcafe"
    with pytest.raises(ValueError) as e:
        Runtime(ScriptedPoseSource([pose(0, 1)]), FakeRenderer(SENSOR, ROOM), p)
    assert "never seen this world" in str(e.value)


def test_accepts_a_matching_fingerprint():
    p = HoverPolicy()
    p.sensor_fingerprint = SENSOR.fingerprint()
    Runtime(ScriptedPoseSource([pose(0, 1)]), FakeRenderer(SENSOR, ROOM), p)


def test_none_fingerprint_is_allowed():
    Runtime(ScriptedPoseSource([pose(0, 1)]), FakeRenderer(SENSOR, ROOM), HoverPolicy())


def test_enforcement_can_be_disabled():
    p = HoverPolicy(); p.sensor_fingerprint = "nope"
    Runtime(ScriptedPoseSource([pose(0, 1)]), FakeRenderer(SENSOR, ROOM), p,
            config=RuntimeConfig(enforce_sensor_match=False))


# ------------------------------------------------------------- normal ticks
def test_no_pose_yet_finishes_cleanly():
    rt, src = build()
    r = rt.step(0.0)                       # never advanced -> latest() is None
    assert r.finished and r.reason == "no_pose"


def test_a_normal_tick_produces_a_command():
    rt, src = build(policy=ForwardPolicy(0.5))
    src.advance()
    r = rt.step(0.0)
    assert r.state == RUNNING and r.command is not None
    assert math.isclose(r.command.vx, 0.5, abs_tol=1e-9)
    assert math.isclose(r.command.z_distance, rt.cfg.hold_altitude_m)


def test_finished_is_sticky():
    rt, src = build()
    rt.step(0.0)                           # no_pose -> FINISHED
    for _ in range(3):
        assert rt.step(1.0).finished


def test_max_duration_ends_the_run():
    rt, src = build(cfg=RuntimeConfig(max_duration_s=1.0))
    src.advance()
    rt.step(0.0)
    assert rt.step(2.0).reason == "max_duration"


# ------------------------------------------------------------------ policy
class Exploding(Policy):
    name = "exploding"
    def act(self, obs, state):
        raise RuntimeError("student bug")


class NaNPolicy(Policy):
    name = "nan"
    def act(self, obs, state):
        return Action("velocity", "body_flu", [float("nan"), 0, 0])


class AccelPolicy(Policy):
    name = "accel"
    def act(self, obs, state):
        return Action("acceleration", "body_flu", [1.0, 0, 0])


def test_a_policy_that_raises_ends_the_run():
    rt, src = build(policy=Exploding())
    src.advance()
    r = rt.step(0.0)
    assert r.finished and r.reason == "policy_error"
    assert any("student bug" in e for e in r.events)


def test_a_non_finite_action_ends_the_run():
    """Action itself rejects NaN, so this surfaces as a policy error."""
    rt, src = build(policy=NaNPolicy())
    src.advance()
    assert rt.step(0.0).reason == "policy_error"


def test_an_unmappable_action_ends_the_run_with_its_own_reason():
    rt, src = build(policy=AccelPolicy())
    src.advance()
    r = rt.step(0.0)
    assert r.reason == "action_error"
    assert any("could not be mapped" in e for e in r.events)


def test_a_renderer_that_throws_ends_the_run():
    class Broken(FakeRenderer):
        def render(self, p, rpy):
            raise IOError("worker died")
    rt, src = build(renderer=Broken(SENSOR, ROOM))
    src.advance()
    r = rt.step(0.0)
    assert r.reason == "render_error" and any("worker died" in e for e in r.events)


# ----------------------------------------------------------------- staleness
def test_a_stale_pose_holds_rather_than_flying_blind():
    rt, src = build(policy=ForwardPolicy(0.8),
                    samples=[pose(0.0, 1.0)],
                    cfg=RuntimeConfig(pose_stale_s=0.15))
    src.advance()
    r = rt.step(0.5)                       # pose is 500 ms old
    assert r.state == HOLDING
    assert r.command.vx == 0.0 and r.command.vy == 0.0
    assert any("pose stale" in e for e in r.events)


def test_holding_recovers_when_the_pose_comes_back():
    rt, src = build(samples=[pose(0.0, 1.0), pose(1.0, 1.0)],
                    cfg=RuntimeConfig(pose_stale_s=0.15))
    src.advance()
    assert rt.step(0.5).state == HOLDING
    src.advance()                          # fresh sample at t=1.0
    r = rt.step(1.0)
    assert r.state == RUNNING and "recovered" in r.events


def test_holding_too_long_becomes_landing_then_lands():
    cfg = RuntimeConfig(pose_stale_s=0.15, hold_before_land_s=0.5,
                        land_complete_m=0.12, land_speed_ms=2.0)
    rt, src = build(samples=[pose(0.0, 1.0, z=0.6)], cfg=cfg)
    src.advance()
    assert rt.step(1.0).state == HOLDING
    r = rt.step(2.0)                       # held > 0.5 s
    assert r.state == LANDING
    assert any("LANDING" in e for e in r.events)
    assert r.command.z_distance < 0.6      # commanding a descent


def test_landing_completes_near_the_ground():
    cfg = RuntimeConfig(pose_stale_s=0.15, hold_before_land_s=0.1)
    rt, src = build(samples=[pose(0.0, 1.0, z=0.05)], cfg=cfg)
    src.advance()
    rt.step(1.0); rt.step(2.0)             # HOLDING then LANDING
    assert rt.step(3.0).reason == "landed"


def test_landing_bypasses_the_minimum_altitude_clamp():
    """Clamping to min_altitude_m while landing would stop it in mid-air."""
    cfg = RuntimeConfig(pose_stale_s=0.15, hold_before_land_s=0.1,
                        land_complete_m=0.01, land_speed_ms=2.0)
    rt, src = build(samples=[pose(0.0, 1.0, z=0.20)], cfg=cfg)
    src.advance()
    rt.step(1.0); r = rt.step(2.0)
    assert r.state == LANDING
    assert r.command.z_distance < rt.cfg.limits.min_altitude_m


# --------------------------------------------------------------- late ticks
def test_repeated_slow_ticks_degrade_to_holding():
    class Slow(FakeRenderer):
        def render(self, p, rpy):
            import time as _t
            _t.sleep(0.02)
            return super().render(p, rpy)
    cfg = RuntimeConfig(tick_budget_s=0.001, late_ticks_to_hold=2)
    rt, src = build(renderer=Slow(SENSOR, ROOM), cfg=cfg,
                    samples=[pose(float(i) * 0.1, 1.0) for i in range(8)])
    states = []
    for i in range(6):
        src.advance()
        states.append(rt.step(float(i) * 0.1).state)
    assert HOLDING in states
    assert any("over budget" in e for r in rt.history for e in r.events)


# ------------------------------------------------------------------ scoring
def straight_run(n=40, x0=0.5, dx=0.1, z=1.25):
    return [pose(i * 0.1, x0 + i * dx, 1.5, z) for i in range(n)]


def test_a_course_is_scored_from_the_real_pose():
    course = GateCourse([Gate("a", np.array([1.5, 1.5, 1.25]),
                              np.array([1.0, 0, 0]), 1.0, 1.0)])
    rt, src = build(samples=straight_run(), course=course)
    for _ in range(30):
        src.advance()
        r = rt.step(src.samples[src.i].t_s)
        if r.finished:
            break
    assert r.reason == "course_complete"
    assert course.passed == 1


def test_a_virtual_collision_ends_the_run():
    esdf = synthetic_room(size_m=ROOM, voxel_m=0.05)
    mon = CollisionMonitor(esdf, 0.10)
    rt, src = build(samples=straight_run(n=45), collision=mon)
    r = None
    for _ in range(45):
        src.advance()
        r = rt.step(src.samples[src.i].t_s)
        if r.finished:
            break
    assert r.reason == "virtual_collision"
    assert mon.failed


def test_scoring_still_runs_while_holding():
    """A stale pose must not stop the drone from being scored on where it is."""
    esdf = synthetic_room(size_m=ROOM, voxel_m=0.05)
    mon = CollisionMonitor(esdf, 0.10)
    samples = [pose(0.0, 1.0, 1.5, 1.25), pose(0.0, 3.99, 1.5, 1.25)]
    rt, src = build(samples=samples, collision=mon,
                    cfg=RuntimeConfig(pose_stale_s=0.05))
    src.advance(); rt.step(1.0)            # stale -> HOLDING
    src.advance()
    r = rt.step(1.0)                       # still stale, but now next to a wall
    assert r.finished and r.reason == "virtual_collision"


# ------------------------------------------------------------- end-to-end
def test_gate_seek_flies_a_course_end_to_end():
    """All eight modules in one loop, no GPU, no ROS, no drone."""
    course = GateCourse([Gate("a", np.array([2.0, 1.5, 1.25]),
                              np.array([1.0, 0, 0]), 1.2, 1.2)])
    esdf = synthetic_room(size_m=ROOM, voxel_m=0.05)
    rt, src = build(policy=GateSeekPolicy(), course=course,
                    collision=CollisionMonitor(esdf, 0.10),
                    samples=straight_run(n=30, x0=1.0, dx=0.05))
    r = None
    for _ in range(30):
        src.advance()
        r = rt.step(src.samples[src.i].t_s)
        if r.finished:
            break
    assert r.reason == "course_complete"
    txt = rt.summary()
    assert "gate_seek" in txt and "COURSE COMPLETE" in txt


def test_reset_clears_everything():
    course = GateCourse([Gate("a", np.array([1.5, 1.5, 1.25]),
                              np.array([1.0, 0, 0]), 1.0, 1.0)])
    rt, src = build(samples=straight_run(), course=course)
    for _ in range(30):
        src.advance()
        if rt.step(src.samples[src.i].t_s).finished:
            break
    rt.reset()
    assert rt.state == RUNNING and rt.reason is None and rt.steps == 0
    assert course.passed == 0


# ------------------------------------------------- scene-frame pose source
def test_transformed_pose_source_converts_position_and_orientation():
    from splat_hitl.frames import SplatTransform, rpy_to_matrix, matrix_to_quat
    from splat_hitl.runtime import TransformedPoseSource
    R = rpy_to_matrix(0, 0, math.pi / 2)              # scene yawed 90 deg
    tf = SplatTransform(R, np.array([1.0, 0.0, 0.0]), 2.0)
    inner = ScriptedPoseSource([PoseSample(0.0, np.array([1.0, 0.0, 0.5]),
                                           (0.0, 0.0, 0.0, 1.0))])
    inner.advance()
    out = TransformedPoseSource(inner, tf).latest()
    # position: 2 * (R @ [1,0,0.5]) + [1,0,0] = 2*[0,1,0.5] + [1,0,0]
    assert np.allclose(out.position_m, [1.0, 2.0, 1.0], atol=1e-9)
    # a level drone in a 90-deg-yawed scene reads as 90 deg of scene yaw
    assert abs(out.yaw_rad - math.pi / 2) < 1e-9


def test_transformed_source_passes_none_through():
    from splat_hitl.frames import SplatTransform
    from splat_hitl.runtime import TransformedPoseSource
    src = TransformedPoseSource(ScriptedPoseSource([pose(0, 1)]),
                                SplatTransform.identity_metres(1.0))
    assert src.latest() is None                       # never advanced


def test_identity_transform_is_a_no_op_apart_from_scale():
    from splat_hitl.frames import SplatTransform
    from splat_hitl.runtime import TransformedPoseSource
    inner = ScriptedPoseSource([pose(0.0, 2.0, z=1.0)])
    inner.advance()
    out = TransformedPoseSource(inner, SplatTransform.identity_metres(0.5)).latest()
    assert np.allclose(out.position_m, [4.0, 3.0, 2.0])   # 1 unit = 0.5 m


def test_runtime_scores_in_the_scene_frame_through_the_wrapper():
    """A gate defined in scene metres must be hit when the drone crosses the
    corresponding place in Vicon metres."""
    from splat_hitl.frames import SplatTransform
    from splat_hitl.runtime import TransformedPoseSource
    tf = SplatTransform.identity_metres(0.5)          # scene unit = 0.5 m
    gate_scene = Gate("a", np.array([3.0, 3.0, 2.5]), np.array([1.0, 0, 0]),
                      2.0, 2.0)
    course = GateCourse([gate_scene])
    inner = ScriptedPoseSource([pose(i * 0.1, 1.0 + i * 0.1, 1.5, 1.25)
                                for i in range(20)])
    rt = Runtime(TransformedPoseSource(inner, tf),
                 FakeRenderer(SENSOR, ROOM), HoverPolicy(), course)
    r = None
    for _ in range(20):
        inner.advance()
        r = rt.step(inner.samples[inner.i].t_s)
        if r.finished:
            break
    assert r.reason == "course_complete"


def test_stop_makes_the_runtime_and_the_caller_agree():
    """A dry run used to end with the log saying operator_stop and the summary
    still saying running."""
    rt, src = build()
    src.advance()
    rt.step(0.0)
    assert rt.state == RUNNING
    rt.stop()
    assert rt.state == FINISHED and rt.reason == "operator_stop"
    assert "operator_stop" in rt.summary()
    assert rt.step(1.0).finished


def test_stop_does_not_overwrite_a_real_outcome():
    rt, src = build()
    rt.step(0.0)                                   # no_pose -> FINISHED
    rt.stop("operator_stop")
    assert rt.reason == "no_pose"
