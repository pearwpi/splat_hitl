# Pipeline status

Updated **2026-09-08**. Supersedes the 2026-09-03 version, which listed as
missing several things that turned out to exist and several that turned out to
be built wrong.

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
2. **Vicon↔splat calibration.** Solved for `net_2026-09-16` — 23 mm RMS over
   218k box-surface points, fitted against four tape-measured boxes rather than
   through `calibrate collect`, which still has no run against real data. No
   other scene has one, and that scene's bundle is not yet assembled (no ESDF,
   gates or manifest). *(done for one scene)*
3. Render worker up; `SplatWorkerClient` against it. Never tested. *(lab, GPU)*
4. `--yaw-sign` props off. *(lab, 10 min)*
5. `HoverPolicy` HITL flight. *(lab)*

Flight-planning note: `yaw_mode="fixed"`, so the first flight commands zero yaw
and the drone must hold its starting heading. The runtime drops to HOLDING past
20° of drift — that is the guard working.

### Then, for a student-ready assignment

| | | size |
|---|---|---|
| `SplatEnv` in `splat_hitl` | the training environment. Students never see `MihirBhat`, so it has to live here. Reuses the contract, `ObservationBuilder`, `VelocityIntegrator`, `gates.py`, `collision.py`. New: dynamics, start sampling, reward, the Gym wrapper. | the biggest remaining build |
| Render worker container image | from `splat_rendering.py`. Keeps torch/gsplat/CUDA out of `splat_hitl`, which is numpy-only and should stay that way | packaging |
| Student guide | drone care, marker placement and verification, Tracker object, radio checks, `ROS_DOMAIN_ID` per team, preflight, splat_hitl usage | the big writing job |
| Operations doc | scheduling, supervision, LiPo, spares | small, not optional |

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
