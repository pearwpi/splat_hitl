# splat_hitl

The bridge between a Vicon-flown Crazyflie and a metric Gaussian-splat scene.

A real drone flies in an empty net. Its pose comes from Vicon. That pose is used
to render what a camera would see *if the drone were inside the splat scene*.
A policy consumes that image and emits commands, which fly the real drone.

**Nothing here imports torch, gsplat or nerfstudio.** The renderer is reached
through a client interface, so the entire runtime is testable — and teachable —
on a laptop with no GPU. That is deliberate: it keeps the flight loop free of
the heaviest dependency in the stack, and it means a student without a GPU can
still develop everything except the rendering itself.

## Status

Built and tested (240 tests, no hardware required):

| module | what it does |
|---|---|
| `frames.py` | quaternion/rotation/Euler conversions, ENU↔NED, and `SplatTransform` — the Vicon-metres → splat-normalised similarity transform |
| `registration.py` | solves that transform from measured correspondences, reports residuals in millimetres, and **refuses degenerate geometry** |
| `sensor.py` | the sensor-model contract: resolution, FOV, mount angle, depth encoding, with a fingerprint that answers "did this policy ever see this world?" |
| `commands.py` | policy action → Crazyflie command, with every unit and sign made explicit, envelope clamping, and saturation reporting |
| `gates.py` | racing gates: swept-segment pass detection with direction, misses reported as events, ordered course progress |
| `collision.py` | virtual collision against the scene ESDF, in metres, with a `synthetic_room()` you can develop against with no splat at all |
| `renderer.py` | `RendererClient` interface, a GPU-free `FakeRenderer` that ray-casts the same toy world, and the subprocess client for `metric-splat`'s worker |
| `policy.py` | the slot a student's work goes into: `Policy.act(obs, state) -> Action`, plus hover / forward / gate-seek references |
| `runtime.py` | the HITL loop, the pose-source interface, and the degradation state machine |
| `calibrate.py` | collect Vicon↔splat correspondences in the lab, pair them by label, solve and save the transform |
| `recorder.py` | run logs in `metric-splat`'s own vocabulary, so a real flight and a simulated rollout land in the same figure |
| `ros_node.py` | the ROS 2 shell: `ViconPoseSource`, a command publisher, and a CLI. **Untested — no live ROS graph here.** |

Everything is written. What remains is integration on `pear-2`: the first live
render, the first live flight, and the calibration data. See `PIPELINE_STATUS.md` for the full inventory and
the critical path.

## Why these two first

Both are on the critical path and both fail *silently*.

**Registration.** A 2° rotation or a 10 cm offset between Vicon and the splat
produces a system where every component reports healthy, the render looks
beautiful, and the policy is confidently wrong about where the walls are. There
is no runtime symptom — the drone just flies into things for reasons that look
like bad control. So `registration.solve()` never returns a bare answer: it
reports per-point and RMS residuals in metres, and it **refuses** point sets
that cannot constrain a rotation.

That refusal matters more than it sounds. Four corners of the floor is the
obvious way to calibrate and it is coplanar — the fit is unconstrained out of
plane and will be confidently wrong. It is the same failure the marker-layout
work hit from the other direction: a degenerate geometry produces a steady,
plausible, incorrect answer.

```
    ok        -> solve
    marginal  -> solve, but warn: extrapolates poorly outside the calibrated region
    coplanar  -> REFUSE: "add points at different HEIGHTS"
    colinear  -> REFUSE: rotation about that line is unconstrained
    too_few   -> REFUSE
```

Umeyama's determinant correction is applied and `SplatTransform` re-checks it,
because a reflected fit renders a mirrored world that looks entirely normal.

**Sensor model.** A policy learns the camera it was trained through. Change the
FOV, resolution, mount angle or depth encoding between training and flight and
the policy is looking at a world it has never seen — and that failure is
indistinguishable from "the policy is bad", because both present as flying into
things. The fingerprint collapses a week of debugging the wrong layer into
comparing two 12-character strings.

Depth encoding is part of the contract, not a preprocessing detail. Note that
`normalized_uint8` is supported but discouraged: per-frame min/max scaling means
a threshold has no fixed meaning in metres and moves with the farthest visible
surface.

## Why `commands.py` is its own module

The driver reproduces Crazyswarm2's conventions exactly and deliberately, and
those conventions disagree with each other:

| message | linear | yaw field | units |
|---|---|---|---|
| `cmd_position` | world m | `yaw` | **degrees** |
| `cmd_hover` | **body** m/s | `yaw_rate` | **rad/s, sign-flipped by the driver** |
| `cmd_velocity_world` | world m/s | `yaw_rate` | **deg/s** |
| `cmd_full_state` | world m, m/s | `twist.angular` | **deg/s** |

Meanwhile policies emit whatever they were trained in — body-frame velocity,
world acceleration, or NED with **+z down** while the Crazyflie's **+z is up**.
Every mismatch flies the drone confidently in the wrong direction, and half of
them are invisible while yaw is zero, which is exactly how every example in the
surrounding repos is written.

So `Action` states its frame rather than assuming one, conversions are named
functions, and anything needing state the caller has not supplied is **refused**
rather than guessed:

```
to_hover(Action("acceleration", ...))
ValueError: cannot build a Hover command from an acceleration action.
Turning acceleration into velocity needs the current velocity and a timestep --
state this module does not have. Integrate it at the call site, where dt is
known...
```

Clamping is reported, not silent. A policy that asks for 8 m/s and receives 1.5
is a finding.

`yaw_sign` exists for the same reason the teleop has `--yaw-sign`: the sign of
the CRTP yaw setpoint has changed across firmware releases. This module does not
pretend to know yours. Verify it **props off**, set it once, record it in the
run log.

## Why gates are planes, not waypoint spheres

A sphere cannot distinguish "went through" from "went past", and cannot tell
forwards from backwards. Both distinctions are the whole point of a race, so a
gate here is a rectangular aperture in a plane with a required direction.

Two consequences worth knowing:

**Crossings are detected on the segment, not the sample.** A policy at 30 Hz and
3 m/s moves 10 cm per step and a plane is infinitely thin, so testing whether
individual poses are "near" a gate misses every fast pass. Same reason the
collision checker in `metric-splat` sweeps segments. There is a test that fires
a 3 m step at a gate 1.2 m in and requires the pass to register.

**A miss is an event, not silence.** Crossing the plane outside the opening is
reported with the distance by which it was missed. "Flew past 8 cm to the left"
and "never went near the gate" are different failures and should not produce
identical logs.

Crossing is defined as a state transition on *at or past the plane*, rather than
as a sign product — `d0 * d1 < 0` silently drops the case where a sample lands
exactly on the plane, which is rare and maddening to debug.

Gates arm in order. Gate k+1 cannot be passed while gate k is pending, so a
drone flying the course out of sequence gets no credit and no event, and the
stalled count is the honest signal that it is off-course.

## Virtual collision, and what an ESDF will not tell you

The drone flies in an empty net, so there is nothing to crash into — which is
the problem. Without this, "the trajectory looked reasonable" is the only
available verdict.

Keep the two mechanisms apart:

| | protects | consequence |
|---|---|---|
| `cf_core` geofence | the real room | cuts motors |
| `CollisionMonitor` | the simulation's verdict | ends the run |

Four properties of the field that the API refuses to let you forget:

- **It is unsigned.** Inside an obstacle reads the same as just outside it.
- **It is truncated.** A value at the cap means *at least* this far.
  `Clearance.truncated` tells you when you are in that regime, and the monitor
  refuses a clearance threshold at or beyond the truncation — every free voxel
  would read as the cap and nothing could ever fail.
- **Outside the map there is no answer.** `Clearance.clear_of()` returns False
  when the point is outside, because unknown is not safe. Leaving the mapped
  region is a failure by default.
- **A voxel coarser than your threshold gives confident nonsense.** The
  constructor refuses that combination outright, and warns when the threshold
  is under two voxels.

Checks are swept, for the same reason gates are: there is a test where both
endpoints are comfortably clear and the segment between them passes straight
through a sphere.

`synthetic_room()` builds an analytically exact field for a box room with
optional spherical obstacles — no point cloud, no splat, no GPU. It is what the
tests run against, and it means a student can develop and debug the whole
collision and scoring path before a scene exists.

## Rendering behind an interface

The real renderer needs a GPU, a CUDA gsplat build, a trained splat and its own
conda environment. Putting that in the flight loop's import path would make the
entire runtime testable in exactly one place. So everything downstream talks to
`RendererClient`, and `FakeRenderer` — pure numpy, no assets — stands in.

`FakeRenderer` ray-casts **the same world description** that
`collision.synthetic_room()` takes, so the depth a policy sees and the field
that scores it describe one consistent toy scene. A student can fly the whole
loop, hit a virtual wall, and inspect the depth image that should have warned
them, before any splat exists. There is a test asserting the two agree.

**The interface takes and returns metres.** `metric-splat`'s worker wants
positions in *normalised scene units* and returns depth in *metres*; that
asymmetry is real, easy to get wrong, and absorbed by `SplatWorkerClient` so
nothing above this layer ever sees a normalised coordinate.

**The mount is applied here, once.** A camera pitched 10° down is part of the
sensor model, not the policy's business, so callers pass the *drone's* pose and
get the *camera's* view. `mount_pitch_deg` is positive-down, which is how people
describe a mount and the opposite of the right-handed pitch sign — there are
tests pinning both that and the equivalence of turning the drone versus turning
the mount.

`SplatWorkerClient`'s protocol is transcribed from `splat_rendering.py` rather
than assumed, but it **has not been run against a live worker** — that needs a
GPU and a scene. Treat the first run on pear-2 as an integration test.

## The runtime, and what happens when it slips

`Runtime.step()` takes a pose, renders, runs the policy, maps the action to a
command, scores the result, and **returns what should be published**. It does
not publish. That keeps ROS out of the logic and makes every failure path
testable on a laptop — including the ones you cannot safely produce in a lab.

The timing contract is the driver's, not ours. It transmits from its own 50 Hz
timer and stops the drone if no command arrives within `COMMAND_TIMEOUT_S`
(0.30 s); the firmware stops stabilising after 0.50 s. So this loop does not
have to hit 50 Hz — it has to notice trouble *before* those blunt timeouts do:

```
RUNNING   the policy is flying
HOLDING   pose stale, or ticks running over budget -- hover and wait
LANDING   it did not clear -- descend under our own control
FINISHED  terminal, with a reason
```

A drone landed deliberately is an experiment that ended. A drone stopped by a
watchdog is a drone that falls.

**Scoring runs on the real pose regardless of state.** A stale pose stops the
policy from flying; it does not stop the drone from being scored on where it
actually is. There is a test for exactly that.

**A policy that raises, or returns a non-finite action, ends the run.** For a
course that is the honest verdict — an exception in flight is a failure, not a
hiccup to smooth over. Catching and continuing would hide the bug and score the
student on a controller they did not write.

The end-to-end test flies `GateSeekPolicy` through a gate, past an ESDF, with
collision monitoring, using the fake renderer — all eight modules in one loop,
no GPU, no ROS, no drone.

## One frame, converted once

Everything downstream of the pose source is in the **scene frame** — the
renderer, the gates, and the ESDF all describe the splat, not the room. So
`TransformedPoseSource` wraps the Vicon source and converts there, once.

Putting the transform inside the renderer instead would have left the gates and
the collision field still talking about Vicon coordinates — a total mismatch
with no symptom except results that make no sense. There is a test that defines
a gate in scene metres and requires it to be hit when the drone crosses the
corresponding place in Vicon metres.

Commands still leave in the drone's **body** frame, which is physical and
therefore the same in either world.

## Run logs that match the simulator's

`recorder.py` maps terminations onto `metric-splat`'s vocabulary:

```
course_complete   -> goal
virtual_collision -> collision
virtual_outside   -> out_of_bounds
max_duration      -> timeout
everything else   -> other        (the exact reason is kept alongside)
```

So a HITL run drops into `scene_visualization.py` and `success_rates.txt`
unchanged, and "did the policy that worked in sim work on the drone" becomes a
diff rather than a project. Each log also carries the **sensor fingerprint** and
the **calibration residual**, because six months later the first question about
any surprising result is whether it was flown through the sensor model it was
trained on and how good the registration was — and if that is not in the file it
is gone.

## Usage

```python
from splat_hitl import registration
from splat_hitl.frames import SplatTransform
from splat_hitl.sensor import SensorModel

# --- calibration, once per scene -------------------------------------------
res = registration.solve(vicon_points_m, splat_points_normalised)
print(res.report())
if res.ok:
    res.transform.save("scene_a/vicon_to_splat.json")

# --- at run time ------------------------------------------------------------
tf = SplatTransform.load("scene_a/vicon_to_splat.json")
pos, rpy = tf.pose_to_splat(vicon_position_m, vicon_quaternion_xyzw)
# -> hand to the renderer

sensor = SensorModel.load("config/sensor_model.example.json")
sensor.assert_compatible(SensorModel.load(policy_dir / "sensor.json"))
```

## Calibration, in two sittings

You cannot hold a drone and click a point cloud at the same time, so it is two
commands run at two different times.

**In the lab, carrying the drone:**

```bash
python3 -m splat_hitl.calibrate collect \
    --topic /vicon/crazyflie2/crazyflie2 --out vicon_points.json
```

Rest the drone *on* a recognisable feature — a table corner, a floor marking, a
taped cross — type a label, press enter. It averages a two-second burst and
**rejects** it if the drone wandered more than 5 mm, if any sample was the
occluded-segment sentinel, or if too few arrived. A correspondence taken while
the body was drifting or half-tracked is a wrong number that looks exactly like
a right one, and it poisons the transform everywhere.

**At a desk, in the point cloud:** click the same features with `scene_tools.py`,
save them with matching labels, then

```bash
python3 -m splat_hitl.calibrate solve \
    --vicon vicon_points.json --splat splat_points.json \
    --out scene_a/vicon_to_splat.json
```

Labels are matched by name, and unmatched ones are **reported rather than
dropped** — a typo that halves your correspondences is otherwise invisible until
the residual looks vaguely odd. The report also names the single worst
correspondence, because the error is almost always one bad point rather than a
diffuse fog, and knowing which one saves re-taking all of them.

`rclpy` is imported inside the capture function, so this module and its tests
run on a laptop with no ROS.

## Collecting registration data

You need **at least 4 correspondences, not coplanar**, whose position you know
in both frames:

- *Vicon side*: carry the drone to the point, read the mocap position.
- *Splat side*: click the same physical feature in the cleaned point cloud —
  `scene_tools.py` already has pickers that return normalised coordinates.

Choose points that fill the volume in all three axes, including height. Then
read the residual: tens of millimetres is workable, hundreds means something is
mismatched — most likely a point pairing, or a splat that was transformed after
the ESDF was built.

## Tests

```bash
python3 -m pytest tests/ -q
```

No GPU, no ROS, no drone, no splat. If these pass on your laptop they pass in
the lab, which is the point.
