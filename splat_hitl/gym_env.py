"""Optional Gymnasium adapter for `SplatEnv`.

Kept in its own module, imported lazily by `SplatEnv.as_gym()`, so that
gymnasium is a dependency of TRAINING and not of this package. The environment
itself, and every test of it, runs on numpy alone -- which matters because the
same package has to import on the flight machine, where nothing but the
standard scientific stack is installed and a missing RL dependency at 2 a.m.
in the lab is a wasted session.
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:                                   # pragma: no cover
    raise ImportError(
        "gymnasium is not installed. `SplatEnv` works without it -- "
        "reset()/step() already return the standard tuples -- so only "
        "`as_gym()` needs this. Install with: pip install gymnasium"
    ) from exc

__all__ = ["GymSplatEnv"]


class GymSplatEnv(gym.Env):
    """Spaces derived from the contract, so they cannot disagree with it."""

    metadata = {"render_modes": []}

    def __init__(self, env):
        self.env = env
        o = env.contract.observation
        low, high = (0.0, 1.0) if o.normalize == "clip_unit" else (0.0, float(o.sensor.depth.far_m))
        self.observation_space = spaces.Box(low=low, high=high,
                                            shape=env.observation_shape,
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=env.action_shape, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()
