# splat_hitl

Train a policy in a metric Gaussian splat of a real room, then fly it
hardware-in-the-loop: a real Crazyflie in the net, its pose from Vicon, its
camera view rendered from the splat.

```
Vicon ──► registration ──► splat renderer ──► policy ──► Crazyflie
   pose         (metres)         image        action      command
```

The drone, its latency and its tracking are real. The world it sees is not.

**Course documentation:** <https://pear-wiki.wpi.edu/rbe595/>. It covers
installing this package, the scenes, the contract, the simulator and flying.

## Install

```bash
pip install -e .            # numpy only
pip install -e ".[train]"   # + gymnasium, for SplatEnv.as_gym()
pip install -e ".[dev]"     # + pytest
```

The renderer that draws camera images from a real scene runs in its own Python
environment, with PyTorch and gsplat on an NVIDIA GPU: see `worker/README.md`.

## Modules

**The contract:** everything a policy assumes, in one file with one fingerprint.

| | |
|---|---|
| `contract.py` | observation, action and control rate, and the fingerprint over all of them |
| `sensor.py` | the camera: resolution, field of view, mount, depth |
| `observation.py` | depth and colour → the array the policy reads |
| `commands.py` | actions → Crazyflie commands |

**The scene**

| | |
|---|---|
| `bundle.py` | a scene's files, checked against each other |
| `frames.py`, `registration.py` | the Vicon ↔ splat transform, and fitting it |
| `calibrate.py` | collecting the points that transform is fitted to |
| `anchors.py` | room anchors: has the Vicon frame moved since the scene was registered? |
| `collision.py` | the collision map (ESDF) and the collision monitor |
| `gates.py` | gates, scored the same way in simulation and in flight |
| `blockmap.py` | the A2 map: a boundary and boxes |
| `pointcloud.py` | reading a scene's point cloud |

**Running it**

| | |
|---|---|
| `env.py`, `gym_env.py` | the simulator, and its optional Gymnasium adapter |
| `renderer.py` | `SplatWorkerClient` for a real scene, `FakeRenderer` for a plain room |
| `render_check.py` | checks rendered depth against tape-measured objects |
| `policy.py` | the `Policy` class your work goes in |
| `runtime.py` | the flight loop, and what it does when something goes wrong |
| `recorder.py` | run logs |
| `ros_node.py` | the ROS 2 node that flies a policy |
| `worker_server.py` | serves the GPU renderer to a flight in the Docker container |

The simulator and the flight loop share the same observation builder, action
stage and contract; `tests/test_env.py` checks that they agree step for step.

## Tests

```bash
python3 -m pytest -q        # no GPU, ROS, network or drone needed
ruff check .
```
