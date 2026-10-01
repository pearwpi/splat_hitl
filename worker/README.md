# The render worker

`splat_rendering.py` draws what the drone's camera sees. Given a pose, it
renders the scene's Gaussian splat and returns a depth image and a colour
image.

You do not run it by hand. `SplatWorkerClient` starts it as a separate process
and asks it for one frame at a time.

## What it needs

An NVIDIA GPU with CUDA, and a Python environment with:

- PyTorch, built for that CUDA version
- gsplat
- numpy

Keep these in their own environment. `splat_hitl` itself needs only numpy and
never imports any of them, which is why the worker is a separate process.

## Starting it

```python
from splat_hitl.bundle import SceneBundle
from splat_hitl.renderer import SplatWorkerClient, worker_command

bundle = SceneBundle.load("scenes/a3_train")
cmd = worker_command(bundle, python="/path/to/gpu-env/bin/python")
renderer = SplatWorkerClient(bundle.contract().observation.sensor, cmd)
```

`worker_command` fills in the scene's files, its measured scale and the
contract's empty depth. `python` is the environment above; leave it out to use
the Python you are running.

## In flight

The flight runs in the Docker container, which has no GPU. So the worker runs
on the lab PC itself, served by `worker_server`, and the flight connects to it:

```bash
# on the lab PC, outside Docker
python3 -m splat_hitl.worker_server --bundle scenes/a3_train \
    --python /path/to/gpu-env/bin/python

# in the container
python3 -m splat_hitl.ros_node ... --worker-port 7790
```

Each flight that connects gets its own worker, which stops when the flight
does.

## Where it came from

A copy of `scripts/splat_rendering.py` from
`pearwpi/metric-splat-policy-pipeline` (commit `e1d1650`), kept byte for byte.
If you change it, run `python3 -m splat_hitl.render_check` on a GPU machine
before trusting it again: it renders views of tape-measured objects and
compares the depth it gets with the tape.

The same file also holds an interactive viewer (`visualize-cleaned`) that
needs opencv-python and open3d as well. The course does not use it.
