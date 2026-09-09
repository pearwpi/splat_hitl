# splat_hitl

Train a policy in a metric Gaussian splat of a real room, then fly it
**hardware-in-the-loop**: a real Crazyflie in an empty net, its pose from
Vicon, its view rendered from the splat.

```
Vicon ──► registration ──► splat renderer ──► policy ──► Crazyflie
   pose         (metres)         image        action      command
```

The drone's dynamics, latency and tracking are real. Its *world* is not. That
is the point, and its limits are set out in `WORKFLOW.md` §5.

---

## Install

```bash
pip install -e .            # numpy only
pip install -e ".[train]"   # + gymnasium, for SplatEnv.as_gym()
python3 -m pytest -q        # 345 tests, no GPU, no ROS, no drone
```

numpy is the only hard dependency on purpose. This package has to import on the
flight machine, where a missing RL library at 2 a.m. in the lab is a wasted
session — `rclpy`, `gymnasium` and `crazyflie_interfaces` are imported lazily
inside the functions that need them.

---

## Where to start

| you want to | read |
|---|---|
| do the assignment | **`WORKFLOW.md`** |
| set up the drone and the lab | `../cf_vicon_stack/GUIDE.md` |
| know why a module is shaped as it is | its docstring — that is where the reasoning lives |

---

## The modules

**The contract** — everything a policy assumes, in one hashable object. This is
the spine: training and flight load the same file, and a mismatch is refused by
name rather than discovered on a drone.

| | |
|---|---|
| `contract.py` | observation + action + control, and one fingerprint over all of it |
| `sensor.py` | the camera: resolution, field of view, mount, depth encoding |
| `observation.py` | metric depth and colour → the array the policy reads |
| `commands.py` | actions → Crazyflie commands; the integrator and the passthrough |

**The scene** — what a student is handed.

| | |
|---|---|
| `bundle.py` | a scene as a checkable set of files, verified against each other |
| `frames.py`, `registration.py` | the Vicon ↔ splat transform, and solving for it |
| `calibrate.py` | collecting the correspondences that transform is solved from |
| `collision.py` | the ESDF, and the swept-segment monitor over a flight |
| `gates.py` | gates as planes with an opening; one course scores sim *and* flight |

**Running it.**

| | |
|---|---|
| `env.py` | the training environment. Same builder, same action stage as flight |
| `gym_env.py` | optional Gymnasium adapter |
| `renderer.py` | `SplatWorkerClient` for a real scene, `FakeRenderer` for a toy one |
| `policy.py` | the slot your work goes in |
| `runtime.py` | the flight loop, and every way it degrades |
| `recorder.py` | run logs, in the same vocabulary as the training logs |
| `ros_node.py` | the ROS wiring. Deliberately thin |

---

## The three ideas worth knowing before you read code

**A wrong pose is more dangerous than no pose.** A dropout is detected and
handled; a confidently wrong pose is obeyed. Every guard in the runtime follows
from that.

**Parity is structural, not policed.** `SplatEnv` and `Runtime` import the same
`ObservationBuilder` and the same `VelocityIntegrator`, driven by the same
contract. There is no second implementation to drift, and
`test_env_and_runtime_agree` asserts it step for step.

**Measure the thing, not a proxy for it.** The runtime checks its rate against
the clock rather than against config; the bundle checks gate corners against the
ESDF rather than gate centres; `divergence_ms` measures the sim-to-real gap per
tick instead of assuming it away. Each of those exists because the proxy version
once reported health over a broken state.

---

## Development

```bash
python3 -m pytest -q          # everything
python3 -m pytest tests/test_contract.py -q
ruff check .                  # config in pyproject.toml
```

Tests run from a fresh clone without installing — `conftest.py` supplies the
path. No test needs a GPU, ROS, a network or a drone; `FakeRenderer` and
`ScriptedPoseSource` stand in for all of it.

Internal engineering notes are under `docs/`. They are not part of the
assignment and students do not need them.
