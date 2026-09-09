# The assignment workflow

`splat_hitl` is the half of the system that a policy lives in. You train in a
Gaussian splat of a real room, then fly the trained policy on a real drone that
believes it is still in the splat.

The drone flies in an **empty net**. Its policy sees the splat, rendered from
wherever Vicon says the drone is. When the policy says "forward through the
gate", the real drone goes forward — at a real net. That is the whole trick,
and it is why §5 exists.

Hardware and lab setup are in `cf_vicon_stack/GUIDE.md`. Read that first.

---

## 1. The scene you are given

A **scene bundle** is a directory. Check it before you use it:

```bash
python3 -m splat_hitl.bundle scenes/<scene>
```

It verifies the parts against each other — that the gates lie inside the
mapped volume, that the ESDF and the splat were built at the same scale, that
the contract file matches what the manifest says it is. Exit code 0 means the
scene is coherent. It is not a formality: a scene whose ESDF came from a
different export runs perfectly and scores you against a differently sized copy
of the room.

```python
from splat_hitl.bundle import SceneBundle

bundle = SceneBundle.load("scenes/playTunnels")
report = bundle.check()
assert report.ok, str(report)

contract = bundle.contract()
course   = bundle.gates()
esdf     = bundle.esdf()
```

If `check()` reports *"cannot be FLOWN"*, the bundle has no Vicon
registration yet. You can still train; you cannot yet fly it.

---

## 2. The contract — read this once, properly

A policy learns the world it was shown. Change anything about that world
between training and flying and the policy is looking at something it has never
seen, while every component still reports healthy.

`PolicyContract` pins all of it in one file with one fingerprint:

- **observation** — resolution, field of view, mount angle, depth clip,
  normalisation, and **how many frames are stacked**
- **action** — kind (`acceleration`), frame, scale, and `yaw_mode`
- **control** — the rate, in hertz

```python
print(contract)
# PolicyContract(playTunnels, obs 4x64x96 clip_unit clip 4.0,
#                act acceleration body_flu x1.50 yaw=fixed, 15 Hz, 4b7684afaa70)
```

**Pick your channels first.** `observation.channels` is `depth`, `rgb` or
`rgb_depth`, and it is the single most consequential line in the file for an
assignment: a depth-only policy cannot see a painted gate, and a colour-only
one cannot tell how far away it is. Per frame the channels are ordered
`[R, G, B, D]` and frames stack along the channel axis, so a 4-frame
`rgb_depth` observation is `(16, H, W)` laid out `R0 G0 B0 D0 R1 G1 B1 D1 ...`
— one frame stays contiguous, so a network that also runs on a single frame
can slice it out.

Three parts of that are easy to miss and expensive to get wrong:

**The rate is part of the policy.** Your policy integrates its own action. Run
it at 30 Hz when it was trained at 15 and it produces twice the velocity per
unit time. The runtime measures the actual step rate against the clock and
stops the run if it disagrees — it does not trust the config.

**`yaw_mode="fixed"` means the drone does not turn.** The environment captures
the heading at reset and never changes it, so your policy has only ever seen
the course from its starting heading. In flight the runtime captures the
heading from the first pose and drops to HOLDING if the drone drifts more than
20° from it. If you want your policy to cope with different headings, that is
what `start_yaw_jitter_rad` in `EnvConfig` is for.

**The field of view belongs to the scene, not to you.** It is the camera that
captured the room. `python3 -m splat_hitl.contract <transforms.json>` recovers
it; the bundle should already have it right.

Declare the contract on your policy and the runtime will check it:

```python
class MyPolicy(Policy):
    name = "gate_racer_v3"
    contract_fingerprint = "4b7684afaa70"   # from contract.fingerprint()

    def act(self, obs, state):
        a = self.net(obs.policy_input)      # (C, H, W) float32 in [0, 1]
        return action_from_raw(CONTRACT.action, a)
```

`obs.policy_input` is the encoded, stacked observation. `obs.depth_m` is raw
metres and `obs.rgb` is the raw colour image — neither is what you trained on,
so read `policy_input` unless you are debugging.

Developing a colour policy does not need a GPU. `FakeRenderer` shades its six
walls and every obstacle differently, and an obstacle can carry its own colour,
so a coloured sphere stands in for the landmark you intend to detect:

```python
FakeRenderer(sensor, (4.0, 3.0, 2.5),
             obstacles=[{"centre": (2.0, 1.5, 0.6), "radius": 0.3,
                         "colour": (1.0, 0.0, 0.0)}])
```

That is enough to check your loop is wired up — the shapes move correctly and
the colours are consistent between frames. It is **not** enough to train a
perception network on: flat colours, exact labels, no texture, no lighting, no
noise. Train on real or Blender-rendered imagery and use this for the plumbing.

---

## 3. Training

```python
from splat_hitl.env import SplatEnv, EnvConfig
from splat_hitl.renderer import SplatWorkerClient

# The worker command is a list, and it runs in its own interpreter: nerfstudio
# and gsplat cannot share a dependency tree with your policy stack.
WORKER = ["python3", "splat_rendering.py", "--worker",
          "--backend", "cleaned-splat",
          "--splat", bundle.path("splat"),
          "--empty-depth-raw-m", "4.0"]
renderer = SplatWorkerClient(contract.observation.sensor, WORKER)
env = SplatEnv(contract, renderer, course, esdf,
               EnvConfig(start_position_m=(1.0, 2.5, 0.6),
                         start_yaw_jitter_rad=0.35,
                         max_steps=500))

obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(raw_action)
```

`raw_action` is your network's raw output — the `[-1, 1]` box. Clipping,
scaling and the norm limit happen inside, so you are training against exactly
the interpretation the drone will apply. (The norm limit is not per-axis
clipping: `(1, 1, 1)` is pulled back to the action scale, so the diagonal is
not faster than forward.)

With stable-baselines3:

```python
model = PPO("CnnPolicy", env.as_gym(),
            policy_kwargs=dict(normalize_images=False),  # already 0..1 float32
            verbose=1)
model.learn(2_000_000)
```

`normalize_images=False` matters — the observation is normalised float32, not
a uint8 image, and SB3 will otherwise divide it by 255 again.

Expect roughly **150–240 environment steps per second** on one GPU with the
subprocess renderer. Two million steps is under three hours. If you are far
below that, the renderer is the bottleneck, not your network.

Terminations use one vocabulary shared with flight: `course_complete`,
`virtual_collision`, `virtual_outside`, `max_duration`. Leaving the mapped
volume is *not* the same outcome as hitting something and is not scored as one.

---

## 4. Flying it

Dry run first — the whole loop, with the motors inhibited:

```bash
python3 -m splat_hitl.ros_node \
  --topic /vicon/<object>/<object> \
  --contract  scenes/playTunnels/policy_contract.json \
  --transform scenes/playTunnels/vicon_transform.json \
  --gates     scenes/playTunnels/gates.json \
  --esdf      scenes/playTunnels/scene_esdf.npy \
  --worker    "<render worker command>" \
  --policy    mypackage.policies:MyPolicy \
  --log /tmp/run.json \
  --dry-run
```

Then drop `--dry-run`. Nothing else changes.

Do not pass `--sensor` alongside `--contract`. The contract carries its own
sensor model, and loading a second one is exactly how the two drift apart.

Before the first flight of a session, and after any change: `preflight.py`,
exit 0. See `cf_vicon_stack/GUIDE.md` §4.

---

## 5. Why your policy will fail, and how to tell which reason

A policy that fails because it was trained through a different camera and a
policy that fails because it is bad both present as flying into things. These
are the ways to tell them apart, in the order they cost you time.

**Contract mismatch.** The runtime refuses to start: it names the field that
differs, not just a hash. Fix the config, not the policy.

**Rate mismatch.** The run ends with `rate_mismatch` after about thirty ticks.
The loop is not running at the rate your policy was trained at.

**Yaw drift.** The runtime drops to HOLDING and says the observations are no
longer valid. Your drone is not holding the heading the policy assumes. That
is a controller or a trim problem, not a policy problem.

**The divergence column.** Every run log has `divergence_ms`: the gap between
the velocity your policy *believes* it has — its own double integrator, the one
it was trained with — and the velocity the drone actually has. In simulation
that gap is zero by construction. In flight it grows with drag, thrust error
and everything else the simulator does not model.

That column is the sim-to-real gap, measured, per tick. If your policy works in
sim and fails in the air, plot it first. A divergence that grows steadily means
your policy learned to fly a frictionless drone.

**What is not simulated:** aerodynamics. There is no drag, no rotor dynamics,
no ground effect off a virtual floor, no wall wash from virtual geometry. HITL
gives you real latency, real tracking and real dynamics response — not real
air. Say so in your write-up; it is the honest boundary of the method and a
better discussion than pretending otherwise.

---

## 6. Reading a run

Each run writes a JSON summary and a CSV of every tick. The columns worth
knowing:

| column | what it tells you |
|---|---|
| `pose_age_ms` | how old the pose was when the policy acted |
| `latency_ms` | camera exposure to policy output. ~6 ms is healthy on this rig |
| `divergence_ms` | the sim-to-real gap, per tick — see §5 |
| `yaw_error_deg` | drift from the heading the policy assumes |
| `clamped` | the envelope had to change your command |
| `gates_passed` | progress |

A run that ends `operator_stop` is you pressing Ctrl-C. `goal`, `collision`,
`out_of_bounds` and `timeout` are the four real outcomes, and they match the
names your training logs use, so the two are directly comparable.

---

## 7. Where things live

| | |
|---|---|
| `contract.py` | the train/fly contract and its fingerprint |
| `observation.py` | metric depth → what the policy reads |
| `commands.py` | actions → Crazyflie commands; the integrator |
| `env.py` | the training environment |
| `runtime.py` | the flight loop and every degradation path |
| `bundle.py` | scene bundles and their validation |
| `gates.py`, `collision.py` | the task and the ESDF |
| `recorder.py` | run logs |

The package is numpy-only on purpose, so it imports on the flight machine
where nothing else is installed. `gymnasium` is needed only for `as_gym()`.
