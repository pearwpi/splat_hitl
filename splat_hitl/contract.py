"""The full contract between training and flight -- not just the camera.

WHY THIS EXISTS ON TOP OF sensor.py
-----------------------------------
`SensorModel` pins what the camera sees. That is necessary and it is not
sufficient. Reading the trainer that actually produced our policies
(`splat_rl_env.py` in the metric-splat pipeline) against this runtime turned up
four more ways the two sides can disagree while every component reports healthy:

  1. ACTION KIND.  The trainer emits a body acceleration fraction and
     integrates it itself (`v += a*dt`). `commands.to_hover` refuses an
     acceleration outright. A policy trained today cannot be flown at all
     until something owns that integrator -- see `commands.VelocityIntegrator`.

  2. CONTROL RATE.  The trainer runs at `control_dt_s = 1/15`. The first HITL
     dry run stepped at 30 Hz. A double integrator driven at twice its training
     rate produces twice the velocity per unit time, and nothing complains.

  3. OBSERVATION SHAPE.  The trainer clips depth at 4 m, divides by 4, and
     stacks the last FOUR frames. `SensorModel` has no field for a clip, a
     normalisation or a history length, so its fingerprint cannot see any of
     them differ.

  4. FIELD OF VIEW, and this one is the worst.  The trainer never sends
     `fov_x_half_tan` to the render worker, so the worker falls back to
     `fx = ref_camera.fx * scale_x` -- the intrinsics of whatever camera
     CAPTURED that scene. The policy's field of view is therefore a property of
     the recording, differs between scenes, and is written down nowhere. A
     policy trained on one scene and evaluated on another was looking through
     two different lenses with nothing to notice.

So this module states all of it in one hashable object. `fov_x_half_tan` is
made EXPLICIT here and sent on every render request, which converts an
accident of capture into a stated property.

The trainer's own guard is a pair of strings, `ACTION_SEMANTICS` and
`OBSERVATION_SEMANTICS`. Those are recorded here too, so a contract loaded on
either side can be compared against the strings the other side believes in.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Optional

from .sensor import SensorModel

__all__ = ["ObservationSpec", "ActionSpec", "ControlSpec", "PolicyContract",
           "NORMALIZATIONS", "YAW_MODES", "INTEGRATORS",
           "fov_x_deg_from_transforms", "metric_splat_depth_ppo_v1"]

NORMALIZATIONS = ("none", "clip_unit")
YAW_MODES = ("fixed", "free")
INTEGRATORS = ("open_loop", "measured")
_HISTORY_ORDERS = ("oldest_first", "newest_first")
_PRIME = ("repeat_first", "zeros")


@dataclass(frozen=True)
class ObservationSpec:
    """Metric depth in, policy input out. Every step of that stated.

    clip_far_m       metres at which depth saturates BEFORE normalisation.
                     Distinct from `sensor.depth.far_m`, which describes the
                     renderer. The trainer clips at 4 m while the renderer
                     reports out to 24 m; both numbers are real and they are
                     not the same number.
    normalize        "clip_unit" divides by clip_far_m to land in [0, 1].
    history          stacked frames. 1 means no history.
    history_order    "oldest_first" matches `np.stack(tuple(deque))` on a deque
                     appended newest-last, which is what the trainer does.
    prime            what fills the history on the first frame of a run.
                     "repeat_first" copies frame 0 into every slot -- again the
                     trainer's behaviour. "zeros" would hand the policy a
                     wall of zero depth, i.e. an obstacle in its face.
    nan_fill_m       NaN and +inf become this many metres. The trainer uses the
                     clip distance: an unknown pixel reads as "far".
    neginf_fill_m    -inf becomes this. The trainer uses 0.0, i.e. "touching".
    """
    sensor: SensorModel
    clip_far_m: Optional[float] = None
    normalize: str = "none"
    history: int = 1
    history_order: str = "oldest_first"
    prime: str = "repeat_first"
    nan_fill_m: Optional[float] = None
    neginf_fill_m: float = 0.0

    def __post_init__(self):
        if self.normalize not in NORMALIZATIONS:
            raise ValueError("normalize %r not in %s" % (self.normalize, NORMALIZATIONS))
        if self.history_order not in _HISTORY_ORDERS:
            raise ValueError("history_order %r not in %s"
                             % (self.history_order, _HISTORY_ORDERS))
        if self.prime not in _PRIME:
            raise ValueError("prime %r not in %s" % (self.prime, _PRIME))
        if int(self.history) < 1:
            raise ValueError("history must be >= 1, got %r" % (self.history,))
        if self.clip_far_m is not None and not self.clip_far_m > 0:
            raise ValueError("clip_far_m must be positive or None")
        if self.normalize == "clip_unit" and self.clip_far_m is None:
            raise ValueError("normalize='clip_unit' divides by clip_far_m, so "
                             "clip_far_m must be set")

    @property
    def shape(self) -> tuple:
        """(channels, height, width) as the policy receives it."""
        return (int(self.history), int(self.sensor.height), int(self.sensor.width))

    @property
    def fov_x_half_tan(self) -> float:
        """tan(fov_x / 2), which is what the render worker actually takes.

        Sending this makes the field of view a stated property instead of the
        capture camera's intrinsics leaking through by default.
        """
        return math.tan(math.radians(self.sensor.fov_x_deg) / 2.0)


@dataclass(frozen=True)
class ActionSpec:
    """What the policy's numbers mean.

    kind        "velocity" | "acceleration" | "position", as in commands.Action.
    frame       the frame `vector` is expressed in.
    scale       magnitude of a unit action. The trainer's
                `max_acceleration_m_s2 = 1.5`.
    clip_unit   clip the raw network output to [-1, 1] before scaling. The
                trainer does this; a policy whose head can exceed 1 behaves
                differently if we do not.
    limit_norm  after scaling, limit the 3-vector NORM to `scale` rather than
                each component. Also the trainer's behaviour, and it is not the
                same as per-axis clipping: (1,1,1) has norm 1.73.
    yaw_mode    "fixed" is the important one, and it is what the trainer does:
                `episode_basis` is captured at reset and never updated, and
                `yaw_policy_rad` is hard 0. The policy has therefore never seen
                the scene from any heading but the one it started at. Flying it
                with a free yaw shows it a world it was not trained in.
    integrator  only meaningful for kind="acceleration". See
                commands.VelocityIntegrator.
    """
    kind: str = "velocity"
    frame: str = "body_flu"
    scale: float = 1.0
    clip_unit: bool = True
    limit_norm: bool = True
    yaw_mode: str = "free"
    integrator: str = "open_loop"

    def __post_init__(self):
        from .commands import FRAMES, KINDS          # local: avoid a cycle
        if self.kind not in KINDS:
            raise ValueError("kind %r not in %s" % (self.kind, KINDS))
        if self.frame not in FRAMES:
            raise ValueError("frame %r not in %s" % (self.frame, FRAMES))
        if self.yaw_mode not in YAW_MODES:
            raise ValueError("yaw_mode %r not in %s" % (self.yaw_mode, YAW_MODES))
        if self.integrator not in INTEGRATORS:
            raise ValueError("integrator %r not in %s" % (self.integrator, INTEGRATORS))
        if not self.scale > 0:
            raise ValueError("scale must be positive, got %r" % (self.scale,))
        if self.kind == "position" and self.integrator != "open_loop":
            raise ValueError("integrator is meaningless for a position action")


@dataclass(frozen=True)
class ControlSpec:
    """How often the policy is asked, which is part of what it learned.

    A policy that integrates its own action carries its timestep inside its
    behaviour. Running it faster does not make it smoother; it makes it faster.
    """
    rate_hz: float = 30.0

    def __post_init__(self):
        if not self.rate_hz > 0:
            raise ValueError("rate_hz must be positive, got %r" % (self.rate_hz,))

    @property
    def dt_s(self) -> float:
        return 1.0 / float(self.rate_hz)


@dataclass(frozen=True)
class PolicyContract:
    """Everything a policy assumes, in one hashable object.

    `trainer_action_semantics` and `trainer_observation_semantics` carry the
    trainer's own guard strings verbatim, so the two independent schemes can be
    checked against each other instead of coexisting silently.
    """
    name: str
    observation: ObservationSpec
    action: ActionSpec
    control: ControlSpec
    trainer_action_semantics: str = ""
    trainer_observation_semantics: str = ""
    notes: str = ""

    # -- identity ---------------------------------------------------------
    def fingerprint(self) -> str:
        """12-char hash of everything that changes how the policy behaves.

        `name` and `notes` are excluded, as in SensorModel: renaming a config
        must not invalidate a trained policy. The trainer's semantics strings
        ARE included -- they are claims about behaviour, not labels.
        """
        d = asdict(self)
        d.pop("name", None)
        d.pop("notes", None)
        d["observation"]["sensor"].pop("name", None)
        d["observation"]["sensor"].pop("notes", None)
        blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def assert_compatible(self, other: "PolicyContract") -> None:
        """Raise, naming the first difference, unless the two agree.

        A bare hash mismatch tells you something is wrong and nothing about
        what, which in practice means the message gets ignored. So this reports
        the actual field.
        """
        if self.fingerprint() == other.fingerprint():
            return
        diffs = []
        a, b = asdict(self), asdict(other)
        for section in ("observation", "action", "control"):
            for key in sorted(set(a[section]) | set(b[section])):
                if key == "sensor":
                    for sk in sorted(set(a[section][key]) | set(b[section][key])):
                        if sk in ("name", "notes"):
                            continue
                        if a[section][key].get(sk) != b[section][key].get(sk):
                            diffs.append("observation.sensor.%s: %r != %r"
                                         % (sk, a[section][key].get(sk),
                                            b[section][key].get(sk)))
                    continue
                if a[section].get(key) != b[section].get(key):
                    diffs.append("%s.%s: %r != %r"
                                 % (section, key, a[section].get(key),
                                    b[section].get(key)))
        for key in ("trainer_action_semantics", "trainer_observation_semantics"):
            if a.get(key) != b.get(key):
                diffs.append("%s: %r != %r" % (key, a.get(key), b.get(key)))
        raise ValueError(
            "policy contract mismatch: %s (%s) vs %s (%s)\n  %s\n"
            "A policy trained under one of these has never run under the "
            "other. Fix the config, not the policy."
            % (self.name, self.fingerprint(), other.name, other.fingerprint(),
               "\n  ".join(diffs) or "(fields equal but hashes differ -- "
                                     "report this, it is a bug here)"))

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        d["derived"] = {
            "observation_shape": list(self.observation.shape),
            "fov_x_half_tan": self.observation.fov_x_half_tan,
            "fov_y_deg": self.observation.sensor.fov_y_deg,
            "control_dt_s": self.control.dt_s,
            "sensor_fingerprint": self.observation.sensor.fingerprint(),
        }
        return d

    def save(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyContract":
        skip = ("fingerprint", "derived")
        d = {k: v for k, v in d.items() if not k.startswith("_") and k not in skip}
        obs = {k: v for k, v in dict(d.pop("observation")).items()
               if not k.startswith("_")}
        sensor = SensorModel.from_dict(obs.pop("sensor"))
        act = {k: v for k, v in dict(d.pop("action")).items()
               if not k.startswith("_")}
        ctl = {k: v for k, v in dict(d.pop("control")).items()
               if not k.startswith("_")}
        return cls(observation=ObservationSpec(sensor=sensor, **obs),
                   action=ActionSpec(**act), control=ControlSpec(**ctl), **d)

    @classmethod
    def load(cls, path) -> "PolicyContract":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def __repr__(self) -> str:
        o, a, c = self.observation, self.action, self.control
        return ("PolicyContract(%s, obs %dx%dx%d %s clip %s, act %s %s x%.2f "
                "yaw=%s, %.0f Hz, %s)"
                % (self.name, o.shape[0], o.shape[1], o.shape[2], o.normalize,
                   o.clip_far_m, a.kind, a.frame, a.scale, a.yaw_mode,
                   c.rate_hz, self.fingerprint()))


def metric_splat_depth_ppo_v1(fov_x_deg: float, name: str = "metric_splat_depth_ppo_v1",
                              notes: str = "") -> PolicyContract:
    """The contract `splat_rl_env.py` actually implements today.

    Every number here was read out of that file rather than chosen:

        depth_width / depth_height     96 x 64      SplatEnvConfig
        POLICY_DEPTH_CLIP_RAW_M        4.0
        DEPTH_HISTORY_LENGTH           4            newest appended last
        nan / +inf -> 4.0, -inf -> 0.0              _observe()
        max_acceleration_m_s2          1.5          SplatEnvConfig
        control_dt_s                   1/15         SplatEnvConfig
        episode_basis fixed at reset, yaw_policy_rad = 0.0

    `fov_x_deg` has no default ON PURPOSE. The trainer does not send
    `fov_x_half_tan`, so the worker falls back to the capture camera's
    intrinsics and the true value is a property of the SCENE. It has to be
    measured per scene and written down here, which is the whole point.
    """
    from .sensor import DepthEncoding
    sensor = SensorModel(
        name=name, width=96, height=64, fov_x_deg=float(fov_x_deg),
        depth=DepthEncoding(kind="metric", near_m=0.05, far_m=4.0,
                            empty_depth_m=4.0),
        rate_hz=15.0,
        notes="empty_depth_m matches the worker's --empty-depth-raw-m 4.0")
    return PolicyContract(
        name=name,
        observation=ObservationSpec(
            sensor=sensor, clip_far_m=4.0, normalize="clip_unit", history=4,
            history_order="oldest_first", prime="repeat_first",
            nan_fill_m=4.0, neginf_fill_m=0.0),
        action=ActionSpec(kind="acceleration", frame="body_flu", scale=1.5,
                          clip_unit=True, limit_norm=True, yaw_mode="fixed",
                          integrator="open_loop"),
        control=ControlSpec(rate_hz=15.0),
        trainer_action_semantics="body_acceleration_fraction_double_integrator_v1",
        trainer_observation_semantics="metric_depth_clipped_4m_normalized_history4_v1",
        notes=notes)


def fov_x_deg_from_transforms(path) -> float:
    """Recover the field of view the render worker will actually use.

    The worker only takes an explicit `fov_x_half_tan` if you send one. The
    trainer does not, so it falls back to

        fx = ref_camera.fx * (image_width / ref_width)

    and the horizontal field of view works out to

        fov_x = 2 * atan(ref_width / (2 * fl_x))

    which is INDEPENDENT of the render resolution -- it is simply the field of
    view of the camera that captured the scene. That is the number to put in
    the contract, and this reads it out of Nerfstudio's `transforms.json` so
    nobody has to guess it.

        python3 -m splat_hitl.contract path/to/transforms.json
    """
    with open(path) as fh:
        d = json.load(fh)
    for k in ("fl_x", "fl_y", "w", "h"):
        if k not in d and k in ("fl_x", "w"):
            raise ValueError("%s has no %r -- is this a Nerfstudio "
                             "transforms.json?" % (path, k))
    fl_x, width = float(d["fl_x"]), float(d["w"])
    if not (fl_x > 0 and width > 0):
        raise ValueError("fl_x and w must be positive, got %r and %r" % (fl_x, width))
    return math.degrees(2.0 * math.atan(width / (2.0 * fl_x)))


if __name__ == "__main__":                                   # pragma: no cover
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 -m splat_hitl.contract <transforms.json>\n"
                         "Prints the scene's horizontal field of view in degrees,\n"
                         "which is what belongs in the contract's sensor.fov_x_deg.")
    fov = fov_x_deg_from_transforms(sys.argv[1])
    print("fov_x_deg = %.4f   (fov_x_half_tan = %.6f)"
          % (fov, math.tan(math.radians(fov) / 2.0)))
