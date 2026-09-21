# Pipeline status

Updated **2026-09-17**. The critical path in §4 is closed: Vicon → policy →
command → radio → drone executed end to end, 5.12 m past four physical
obstacles, tracked to 32 mm rms. Supersedes the 2026-09-08 version.

**Scope, fixed:**

- Policies run **offboard** — lab PC or student workstation. Nothing on the drone.
- Students receive **cleaned metric splats** as assets. Splat creation is our job.
- Students receive **`cf_vicon_stack` and `splat_hitl` only.** Not the
  `MihirBhat` research tree. This is what forces the training environment into
  `splat_hitl` (see §4).
- The same runtime serves every assignment. An RL gate racer and a learned
  optical-flow module differ only in which object fills the policy slot.

---

## 1. Measured, not assumed

Everything here was verified on `pear-2` on 2026-09-07/08 and is written up in
`cf_vicon_stack/FINDINGS_2026-09-07.md`.

| | value | how |
|---|---|---|
| Tracker frame rate | **240.0 Hz** | log-interval arithmetic, matched to the ms across two loss regimes |
| Vicon link latency | **4.56 ms** | agreed by three independent measurements to within 0.03 ms |
| End to end, camera to policy | **5.9 ms** median, 9.8 p95, 18.5 max | 2243 steps |
| Frame loss | **0.22%**, holes of 17 ms | was 1.85% in 200 ms holes before the RTO fix |
| Marker visibility | **7/7**, 0 partial frames over 200 s | `/quality` telemetry, live since 2026-09-08 |

---

## 2. The rigid body — diagnosis changed

The old §1.7 mechanism (attitude-dependent partial occlusion) is **wrong**.
Measured on 2026-09-08: single-frame attitude jumps of 86–95° and 158–160°,
out and back within one 240 Hz frame, **with all seven markers visible**, no
data gap, and a parameterisation-free metric (`d_att`).

`marker_geom.py` predicts both families — 167° about a near-horizontal axis
and 100.7° about the vertical — and scored the layout 25.2 mm, which under its
original thresholds read PASS. The analysis was right; the thresholds were
guesses. They are now set from that measurement.

The mechanism is **planarity**: 12.7 mm of depth over a 133 mm body, 0.096 of
the spread. A flat plate maps onto itself under a half turn about any in-plane
axis.

**The planned fix was also wrong.** Adding an eighth marker on a centre post
was simulated and makes it *worse* — 25.2 mm down to 13.3 mm at 30 mm height —
because a point on the axis of the competing rotations is nearly invariant
under them. The fix is to **raise one existing marker**:

| marker | radius | +20 mm | +30 mm | +35 mm |
|---|---|---|---|---|
| crazyflie22 | 28.6 mm | 36.3 | 38.9 | **41.2** |
| crazyflie27 | 78.8 mm | 17.3 | 45.3 | 45.5 |
| crazyflie23 | 9.0 mm | 17.3 | 39.2 | 38.8 |

`crazyflie22` is the recommendation: it improves monotonically, so a few
millimetres of build error stays safe, and at 28.6 mm radius it costs little
inertia and sits away from the props. `crazyflie27` scores higher but collapses
at +20 mm and is far out on an arm.

These are model numbers. The acceptance test is empirical: build it, re-teach
the Tracker object, then fly and watch `track_monitor.py`.

### 2026-09-17, `crazyflie_a2` — the evidence now contradicts itself

The rebuilt 8-marker body was bench-tested with `spin_test.py`, motors audibly
running, all 8 markers visible at every level:

| throttle | flips | peak deg/s | yaw sd |
|---|---|---|---|
| 0.0 % | **2** | 21 647 | 3.14 |
| 7.5 % | 0 | 661 | 1.07 |
| 15.0 % | 0 | 746 | 1.06 |
| 22.5 % | 0 | 702 | 1.05 |
| 30.0 % | **2** | 23 025 | 4.29 |
| 37.5 % | **4** | 27 286 | 4.28 |
| 45.0 % | 0 | 748 | 1.08 |

FAIL, with no monotonic relationship to throttle — it flips with the motors
stopped, so the ambiguity is present **at rest**, not merely under vibration.
The driver independently logged a 93° single-frame jump the same minute.

Yet three flights that evening — a 15.4 s teleop hover, a 40.6 s HITL hover and
a 26.6 s A2 run — logged **zero** yaw rejects between them. The 148 rejects in
the crashed run all fall in the last 0.56 s, after ground contact, with the
airframe lying at an angle.

Those two observations do not reconcile, and until they do the body is
mitigated rather than fixed: `cf_core.flip_check` keeps a flipped quaternion
out of the EKF and `splat_hitl.ros_node.is_solver_flip` keeps it out of the
policy. Next step is the `crazyflie_a2` `.vsk` through `marker_geom.py`, which
names the symmetry family and its margin from marker positions alone and needs
no lab time.

One asymmetry worth noting: `spin_test.py`'s own docstring says props-off
vibration is weaker than props-on, so a PASS there does not clear a body. The
2026-09-17 data is the inverse case, which the docstring does not cover — a
bench FAIL that flight does not reproduce.

---

## 3. Built and tested

**`cf_vicon_stack`** — 260-test safety core; Docker image that runs its own
suite at build time; bridge with capture-time stamps, marker telemetry and
frame-drop counting; `preflight.py`; `marker_geom.py`; `track_monitor.py`.

**`splat_hitl`** — 301 tests, no GPU, no ROS, no drone required. Frames,
registration, sensor model, commands, gates, collision, renderer clients,
policy interface, runtime, recorder, calibration tools, **and as of 2026-09-08
the policy contract**: `contract.py`, `observation.py`, `VelocityIntegrator`.

The contract closes four ways training and flight could disagree silently:
action kind (the trainer emits accelerations, which `to_hover` refuses),
control rate (15 Hz vs 30), observation encoding (4 m clip, ÷4, 4-frame
history), and field of view — which leaks in from the capture camera because
the trainer never sends `fov_x_half_tan`, making it a property of the recording
that differs between scenes and is written down nowhere.

---

## 4. What is left

### Critical path to a first HITL flight

No trained policy needed. This proves Vicon → transform → render → policy →
command → radio → drone, which has never once executed end to end.

1. Marker standoff on `crazyflie22`, delete and re-teach the Tracker object,
   clear `marker_geom.py --sweep`, then confirm empirically with
   `track_monitor.py`. *(lab, hardware)*
2. ~~**Vicon↔splat calibration.**~~ Done for `net_2026-09-16`: 23 mm RMS over
   218k box-surface points, fitted against four tape-measured boxes rather than
   through `calibrate collect`, which still has no run against real data. That
   scene is a complete bundle and `bundle.check()` exits 0 on it. No other scene
   has a registration.
3. ~~Render worker up; `SplatWorkerClient` against it.~~ Done 2026-09-17 on
   `net_2026-09-16`: `render_check` renders a tape-measured box 1.000 m away
   and reads 1.002-1.026 m, agreeing with an independent ESDF ray to 21 mm.
   2-7 ms per frame after warm-up, against a 66 ms budget at 15 Hz. Not
   mirrored, by 22.5x. Getting there cost three real bugs -- normalised depth,
   camera basis, mount sign -- none of which raised an error.
4. ~~`--yaw-sign` props off.~~ Not a blocker and not a props-off test.
   `yaw_sign` is applied only in `_send_position`, so it touches takeoff, land
   and hold; the cruise goes through `cmd_hover`, where `yaw_rate` is
   sign-flipped separately. Verifying it needs the drone able to rotate, so it
   needs props on. Started at yaw −3.4°, a wrong sign would have commanded a
   6.8° rotation during the climb.
5. ~~`HoverPolicy` HITL flight.~~ Done 2026-09-17. 610 steps over 40.6 s at
   15.0 Hz, zero yaw rejects. Held 0.804 m ± 22 mm and −5.6° ± 1.6°, and
   drifted 0.9 m in x — which is correct: `HoverPolicy` commands zero *velocity*
   and closes no position loop, so a couple of cm/s of EKF velocity bias
   integrates. A velocity contract is not a position contract.
6. ~~A2 reference policy, full trajectory.~~ Done 2026-09-17. 400 steps, 5.12 m
   at a 0.5 m/s cruise, 32 mm rms / 90 mm peak planar error, 0.809 m ± 12 mm
   altitude, stopping 20 mm from the goal. No command clamped, no mocap sample
   rejected. Logs in `a2_reference/a2_hitl_run1.{json,csv}`.

Flight-planning note: `yaw_mode="fixed"`, so the flight commands zero yaw and
the drone must hold its starting heading. The runtime drops to HOLDING past 20°
of drift — that is the guard working, and it was observed working.

**What the bring-up cost, and where.** Eight defects, none of them in tested
code and all of them in the wiring between repositories: the driver never
selected the Kalman estimator (the firmware sat on the complementary filter and
discarded every injected pose, publishing barometric altitude above sea level);
`bounds` declared with an empty-list default, which rclpy types as BYTE_ARRAY,
killing the node *because* its geofence was configured; no `setup.cfg` in
`crazyflie_ros`, so its console scripts never reached libexec and the ROS driver
had never been launched anywhere; log blocks not stopped on shutdown, so each
Ctrl-C poisoned the next connection; takeoff latching a possibly-rejected yaw;
no solver-flip filter upstream of the policy; a config naming a Tracker object
that does not exist; and a missing `import math`.

**Still unproven on hardware:** the live splat worker *in the loop* — the flight
used `FakeRenderer`, and `render_check` verified the worker separately — plus
the ESDF collision monitor, gate scoring, and `TransformedPoseSource`.

### Then, for a student-ready assignment

| | | size |
|---|---|---|
| ~~`SplatEnv` in `splat_hitl`~~ | **Built.** `env.py` + `gym_env.py`: reset/step, both action kinds, gate rewards with progress shaping, the collision monitor, timeout, and a lazily-imported Gymnasium adapter. It imports the same `ObservationBuilder`, action stage and contract as the runtime, so parity is structural rather than policed. What it does NOT have is attitude — see below. | done |
| Render worker container image | from `splat_rendering.py`. Keeps torch/gsplat/CUDA out of `splat_hitl`, which is numpy-only and should stay that way | packaging |
| Student guide | drone care, marker placement and verification, Tracker object, radio checks, `ROS_DOMAIN_ID` per team, preflight, splat_hitl usage | the big writing job |
| Operations doc | scheduling, supervision, LiPo, spares | small, not optional |

**What the sim does not model, and what that costs per assignment.** The
dynamics are a velocity envelope, integrated — no attitude loop, no drag, no
rotor dynamics, no ground effect, and a commanded velocity is reached instantly.
For A2 that is honest and was validated on 2026-09-17: the real drone tracked
the cruise BETTER than the sim (6–24 mm against a 33 mm steady lag), so the sim
is not flattering the controller. Its optimism is on the profile's ramps, 34 mm
simulated against 90 mm flown.

For A3 and A5 two of those omissions start to matter:

- **The camera never tilts.** `env._observe()` poses the render with yaw only.
  A real drone pitches to accelerate, so a vision policy trains on level images
  and flies on tilted ones. A point mass's tilt is *determined* by its
  commanded acceleration, so this is recoverable exactly, without any rotor
  model.
- **There is no dead time.** Mocap latency, the control period, radio transit
  and the firmware's own response are all zero in sim. A reactive gate policy
  is exactly the kind that oscillates when that is wrong.

Training throughput is **not** a blocker: measured 150–240 env-steps/s on the
single subprocess renderer, 2 M steps in 2.8 h. The unmerged batched-rendering
branch is off the critical path.

### Small and scoped

`track_monitor` `dt` from the header stamp and time-since-last-marker-loss
instead of the 200 ms boolean; the rclpy Ctrl-C catch; per-window drop
reporting in the bridge.

---

## 5. Risks that have not changed

**The drone flies in an empty net while its policy believes it is in a room.**
The geofence protects the physical room; the virtual-collision monitor scores
the simulation. Different mechanisms, both required.

**Registration error is invisible.** A 10 cm offset or a 2° rotation produces a
system that runs perfectly and flies wrong. Render the splat from the drone's
live pose next to a real camera view and eyeball it before trusting a result.

**HITL captures real dynamics and real latency, not real aerodynamics.** No
ground effect off a virtual floor. Say so to students; it is the honest
boundary of the method.

**Everything silent is worse than everything loud.** Every failure found on
2026-09-07 — a two-day-dead RTO route, a six-day-broken Docker build, a
telemetry patch that had never executed, latency reading as NaN in exactly the
real configuration — was reported healthy by its own status command. That is
what `preflight.py` and the contract fingerprint exist to prevent.
