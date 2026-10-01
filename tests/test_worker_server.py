"""Rendering across the container boundary, with a fake worker and no GPU.

The flight runs in Docker, which has no GPU, so the worker runs on the lab PC
and `worker_server` passes its traffic over 127.0.0.1. These tests run the real
server and the real client against a stand-in worker that speaks the same
protocol, and check the two things that matter in flight: the frames come back
the same as from a local worker, and a worker never outlives its flight.
"""
import argparse
import json
import socket
import sys
import threading

import numpy as np
import pytest

from splat_hitl.frames import SplatTransform
from splat_hitl.renderer import FakeRenderer, SplatWorkerClient
from splat_hitl.ros_node import make_renderer
from splat_hitl.sensor import SensorModel
from splat_hitl.worker_server import check_worker, listen, serve

#: Speaks splat_rendering.py's protocol and returns a constant depth in
#: NORMALISED units. Appends one line per process to `starts`, and writes the
#: requests it saw to `log` when it exits -- which it does on {"cmd": "close"}
#: or when its input closes, like the real worker.
FAKE_WORKER = """
import sys, json, base64
import numpy as np
starts, log = sys.argv[1], sys.argv[2]
scale, value = float(sys.argv[3]), float(sys.argv[4])
open(starts, "a").write("started\\n")
print(json.dumps({"ready": True, "scale_to_metres": scale, "backend": "fake"}),
      flush=True)
seen = []
for line in sys.stdin:
    r = json.loads(line)
    if r.get("cmd") == "close":
        break
    seen.append(r)
    h, w = int(r["image_height"]), int(r["image_width"])
    d = np.full((h, w), value, dtype=np.float32)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    print(json.dumps({"ok": True, "shape": [h, w],
                      "depth_b64": base64.b64encode(d.tobytes()).decode(),
                      "rgb_shape": [h, w, 3],
                      "rgb_b64": base64.b64encode(rgb.tobytes()).decode()}),
          flush=True)
open(log, "w").write(json.dumps(seen))
"""

SCALE = 3.371039
SENSOR = SensorModel(name="t", width=8, height=6, fov_x_deg=60.0)


@pytest.fixture
def worker(tmp_path):
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    starts, log = tmp_path / "starts.txt", tmp_path / "seen.json"
    cmd = [sys.executable, str(script), str(starts), str(log), str(SCALE), "0.25"]
    return cmd, starts, log


def _server(cmd, once=True, scene="a3_train"):
    srv = listen(0)
    port = srv.getsockname()[1]

    def run():
        try:
            serve(srv, cmd, scene=scene, once=once, log=lambda m: None)
        except OSError:
            pass                                  # the test closed the socket

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return srv, port, t


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# --------------------------------------------------------------- the server
def test_a_served_worker_renders_like_a_local_one(worker):
    cmd, _, log = worker
    srv, port, t = _server(cmd)
    client = SplatWorkerClient(SENSOR, port=port)
    try:
        obs = client.render([1.0, 2.0, 1.0], [0.0, 0.0, 0.0])
    finally:
        client.close()
    t.join(5)
    srv.close()
    assert not t.is_alive()
    assert client.scale_to_metres == pytest.approx(SCALE)
    assert client.scene == "a3_train"
    assert obs.depth_m.shape == (6, 8)
    assert np.allclose(obs.depth_m, 0.25 * SCALE)          # metres, as locally
    seen = json.loads(log.read_text())
    assert seen[0]["position"] == pytest.approx(np.array([1.0, 2.0, 1.0]) / SCALE)


def test_the_worker_stops_when_the_flight_vanishes(worker):
    """ros_node killed mid-flight never sends close. The worker must still go,
    or the GPU fills with orphans, one per crashed run."""
    cmd, _, log = worker
    srv, port, t = _server(cmd)
    s = socket.create_connection(("127.0.0.1", port))
    assert json.loads(s.makefile("r").readline())["ready"]
    s.close()                                              # no goodbye
    t.join(5)
    srv.close()
    assert not t.is_alive()
    assert log.exists()                                    # the worker exited


def test_every_flight_gets_a_fresh_worker(worker):
    cmd, starts, _ = worker
    srv, port, t = _server(cmd, once=False)
    for _ in range(2):
        c = SplatWorkerClient(SENSOR, port=port)
        c.render([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
        c.close()
    srv.close()
    assert starts.read_text().count("started") == 2


def test_the_start_up_check_reads_the_handshake(worker):
    cmd, starts, log = worker
    hello = check_worker(cmd)
    assert hello["ready"] and hello["scale_to_metres"] == pytest.approx(SCALE)
    assert log.exists()                                    # and it was closed


def test_a_worker_that_dies_at_start_up_is_reported():
    with pytest.raises(RuntimeError, match="exit code 3"):
        check_worker([sys.executable, "-c", "import sys; sys.exit(3)"])


# --------------------------------------------------------------- the client
def test_no_server_is_a_clear_error():
    with pytest.raises(RuntimeError, match="no render server"):
        SplatWorkerClient(SENSOR, port=_free_port())


def test_a_command_or_a_port_but_not_both(worker):
    cmd, _, _ = worker
    with pytest.raises(ValueError):
        SplatWorkerClient(SENSOR, cmd, port=7790)
    with pytest.raises(ValueError):
        SplatWorkerClient(SENSOR)


# ------------------------------------------------------- ros_node's choice
def _flags(**kw):
    a = dict(worker=None, worker_port=None, fake_room=None)
    a.update(kw)
    return argparse.Namespace(**a)


def _tf(mpu=SCALE):
    return SplatTransform.identity_metres(mpu)


def test_exactly_one_renderer_is_asked_for(worker):
    with pytest.raises(SystemExit, match="exactly one"):
        make_renderer(_flags(), SENSOR, _tf())
    with pytest.raises(SystemExit, match="--worker-port and --fake-room"):
        make_renderer(_flags(worker_port=7790, fake_room=[8, 4, 3]), SENSOR, _tf())


def test_a_fake_room_needs_no_transform():
    assert isinstance(make_renderer(_flags(fake_room=[8, 4, 3]), SENSOR, None),
                      FakeRenderer)


def test_a_real_scene_without_a_transform_is_refused():
    with pytest.raises(SystemExit, match="without --transform"):
        make_renderer(_flags(worker_port=7790), SENSOR, None)


def test_the_served_scene_is_flown(worker):
    cmd, _, _ = worker
    srv, port, t = _server(cmd)
    r = make_renderer(_flags(worker_port=port), SENSOR, _tf())
    r.close()
    t.join(5)
    srv.close()
    assert r.scene == "a3_train"


def test_a_server_rendering_another_scene_is_refused(worker):
    """Serving a5_test while flying a3_train is the mistake this catches: the
    two scenes are drawn at different scales."""
    cmd, _, _ = worker
    srv, port, t = _server(cmd, scene="a5_test")
    with pytest.raises(SystemExit, match="different scene.*a5_test"):
        make_renderer(_flags(worker_port=port), SENSOR, _tf(3.910744))
    t.join(5)
    srv.close()
    assert not t.is_alive()


def test_a_local_worker_command_still_works(worker):
    cmd, _, _ = worker
    r = make_renderer(_flags(worker=" ".join(cmd)), SENSOR, _tf())
    r.close()
    assert r.scene is None
