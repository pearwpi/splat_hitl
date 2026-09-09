"""Colour observations, end to end.

Assignments 3 and 4 -- gate segmentation and optical flow through an unknown
gap -- cannot be expressed at all without these. A depth-only policy cannot see
a painted gate, and optical flow is a visual method.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splat_hitl.contract import (ActionSpec, ControlSpec, ObservationSpec,
                                 PolicyContract, metric_splat_depth_ppo_v1)
from splat_hitl.env import EnvConfig, SplatEnv
from splat_hitl.gates import Gate, GateCourse
from splat_hitl.observation import ObservationBuilder
from splat_hitl.renderer import FakeRenderer
from splat_hitl.sensor import DepthEncoding, SensorModel

H, W = 12, 16
ROOM = (4.0, 3.0, 2.5)


def spec(channels="rgb_depth", history=2, rgb_range_max=1.0):
    base = metric_splat_depth_ppo_v1(90.0).observation
    sensor = SensorModel(name="t", width=W, height=H, fov_x_deg=90.0,
                         depth=DepthEncoding(kind="metric", near_m=0.05,
                                             far_m=4.0, empty_depth_m=4.0),
                         rate_hz=15.0)
    return ObservationSpec(**{**base.__dict__, "sensor": sensor,
                              "channels": channels, "history": history,
                              "rgb_range_max": rgb_range_max})


def depth(v=2.0):
    return np.full((H, W), v)


def rgb(v=0.5):
    return np.full((H, W, 3), v)


# ------------------------------------------------------------------- shapes
@pytest.mark.parametrize("channels,per_frame", [
    ("depth", 1), ("rgb", 3), ("rgb_depth", 4)])
def test_shape_is_history_times_channels(channels, per_frame):
    s = spec(channels, history=3)
    assert s.channels_per_frame == per_frame
    assert s.shape == (3 * per_frame, H, W)


def test_depth_only_shape_is_unchanged():
    """The existing contract must not move. (history, H, W), as before."""
    c = metric_splat_depth_ppo_v1(90.0)
    assert c.observation.shape == (4, 64, 96)


# ------------------------------------------------------------ channel order
def test_channel_order_within_a_frame_is_rgbd():
    b = ObservationBuilder(spec("rgb_depth", history=1))
    b.reset()
    colour = np.zeros((H, W, 3))
    colour[..., 0], colour[..., 1], colour[..., 2] = 0.1, 0.2, 0.3
    o = b.push(depth(2.0), colour)
    assert o.shape == (4, H, W)
    assert o[0, 0, 0] == pytest.approx(0.1)      # R
    assert o[1, 0, 0] == pytest.approx(0.2)      # G
    assert o[2, 0, 0] == pytest.approx(0.3)      # B
    assert o[3, 0, 0] == pytest.approx(0.5)      # D: 2 m of a 4 m clip


def test_frames_are_contiguous_not_interleaved_by_channel():
    """One frame stays together, so a network that takes ONE frame can slice it."""
    b = ObservationBuilder(spec("rgb_depth", history=2))
    b.reset()
    b.push(depth(2.0), rgb(0.25))
    o = b.push(depth(1.0), rgb(0.75))
    assert np.allclose(o[0:3, 0, 0], 0.25)       # oldest frame's RGB
    assert o[3, 0, 0] == pytest.approx(0.5)      # oldest frame's depth
    assert np.allclose(o[4:7, 0, 0], 0.75)       # newest frame's RGB
    assert o[7, 0, 0] == pytest.approx(0.25)


# ------------------------------------------------------------ the encoding
def test_rgb_is_transposed_to_channels_first():
    b = ObservationBuilder(spec("rgb", history=1))
    b.reset()
    colour = np.zeros((H, W, 3))
    colour[3, 5, 1] = 1.0                        # one green pixel
    o = b.push(rgb=colour)
    assert o.shape == (3, H, W)
    assert o[1, 3, 5] == pytest.approx(1.0)
    assert o[0, 3, 5] == pytest.approx(0.0)


def test_byte_range_renderers_are_scaled():
    b = ObservationBuilder(spec("rgb", history=1, rgb_range_max=255.0))
    b.reset()
    o = b.push(rgb=np.full((H, W, 3), 255.0))
    assert np.allclose(o, 1.0)


def test_colour_is_clipped_and_nan_filled():
    b = ObservationBuilder(spec("rgb", history=1))
    b.reset()
    c = np.full((H, W, 3), 0.5)
    c[0, 0, 0], c[0, 1, 0], c[0, 2, 0] = np.nan, 5.0, -3.0
    o = b.push(rgb=c)
    assert o[0, 0, 0] == pytest.approx(0.0)      # nan -> black
    assert o[0, 0, 1] == pytest.approx(1.0)      # clipped
    assert o[0, 0, 2] == pytest.approx(0.0)      # clipped


# ------------------------------------------------------------- the refusals
def test_a_colour_contract_refuses_a_renderer_that_gives_none():
    b = ObservationBuilder(spec("rgb"))
    b.reset()
    with pytest.raises(ValueError, match="returned no"):
        b.push(depth(2.0), None)


def test_wrong_colour_shape_is_refused_not_reshaped():
    b = ObservationBuilder(spec("rgb"))
    b.reset()
    with pytest.raises(ValueError, match="contract says"):
        b.push(rgb=np.zeros((H, W, 4)))


def test_changing_channels_changes_the_fingerprint():
    """A depth policy and a colour policy are not interchangeable."""
    base = metric_splat_depth_ppo_v1(90.0)
    other = PolicyContract(name="x", action=base.action, control=base.control,
                           observation=ObservationSpec(
                               **{**base.observation.__dict__,
                                  "channels": "rgb_depth"}))
    assert other.fingerprint() != base.fingerprint()
    with pytest.raises(ValueError, match="observation.channels"):
        base.assert_compatible(other)


# --------------------------------------------------------- the fake renderer
def test_the_fake_renderer_shades_each_surface_differently():
    """Enough to develop a segmentation loop with no GPU and no scene."""
    r = FakeRenderer(SensorModel(name="t", width=W, height=H, fov_x_deg=90.0),
                     ROOM)
    o = r.render([1.0, 1.5, 0.6], (0.0, 0.0, 0.0))
    assert o.rgb.shape == (H, W, 3)
    assert len(np.unique(o.rgb.reshape(-1, 3), axis=0)) >= 3


def test_an_obstacle_can_be_given_its_own_colour():
    """So a coloured sphere can stand in for the landmark you will detect."""
    r = FakeRenderer(SensorModel(name="t", width=W, height=H, fov_x_deg=90.0),
                     ROOM, obstacles=[{"centre": (2.0, 1.5, 0.6),
                                       "radius": 0.4, "colour": (1.0, 0.0, 0.0)}])
    o = r.render([1.0, 1.5, 0.6], (0.0, 0.0, 0.0))
    assert np.allclose(o.rgb[H // 2, W // 2], [1.0, 0.0, 0.0])


def test_colour_and_depth_agree_about_where_the_obstacle_is():
    """The sphere must be nearer AND differently coloured in the same pixels."""
    sensor = SensorModel(name="t", width=W, height=H, fov_x_deg=90.0)
    plain = FakeRenderer(sensor, ROOM).render([1.0, 1.5, 0.6], (0, 0, 0))
    ball = FakeRenderer(sensor, ROOM,
                        obstacles=[{"centre": (2.0, 1.5, 0.6), "radius": 0.4,
                                    "colour": (1.0, 0.0, 0.0)}]
                        ).render([1.0, 1.5, 0.6], (0, 0, 0))
    nearer = ball.depth_m < plain.depth_m - 1e-6
    recoloured = np.any(np.abs(ball.rgb - plain.rgb) > 1e-6, axis=-1)
    assert nearer.any()
    assert np.array_equal(nearer, recoloured)


# ------------------------------------------------------------- through a run
def rgb_contract(channels="rgb_depth"):
    o = spec(channels, history=2)
    return PolicyContract(name="rgbd", observation=o,
                          action=metric_splat_depth_ppo_v1(90.0).action,
                          control=ControlSpec(rate_hz=15.0))


def test_the_training_env_hands_a_policy_colour():
    c = rgb_contract()
    course = GateCourse([Gate("g1", np.array([3.0, 1.5, 0.6]),
                              np.array([1.0, 0.0, 0.0]), 1.0, 1.0)])
    env = SplatEnv(c, FakeRenderer(c.observation.sensor, ROOM), course,
                   config=EnvConfig(start_position_m=(1.0, 1.5, 0.6)))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2 * 4, H, W)
    # the colour channels must not be constant: the policy is looking at a room
    assert float(obs[0:3].std()) > 0.0
    obs, _, _, _, _ = env.step([1.0, 0.0, 0.0])
    assert obs.shape == (2 * 4, H, W)
