"""The render worker that ships in worker/, checked without a GPU.

The worker needs PyTorch and gsplat, which these tests do not have, so nothing
here imports it. What can be checked is that it is there, that it accepts every
flag `worker_command` passes, and that it speaks the keys `SplatWorkerClient`
reads and writes. Rendering itself is `render_check`'s job, on a GPU machine.
"""
import ast
import json
import os
import sys

import pytest

from splat_hitl.bundle import SceneBundle
from splat_hitl.contract import metric_splat_depth_ppo_v1
from splat_hitl.frames import SplatTransform
from splat_hitl.renderer import WORKER_SCRIPT, worker_command


def _source():
    with open(WORKER_SCRIPT) as fh:
        return fh.read()


def _worker_flags():
    """Every --flag the worker's own parser declares, read from its source."""
    tree = ast.parse(_source())
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    flags = set()
    for node in ast.walk(main):
        if isinstance(node, ast.Call) \
                and getattr(node.func, "attr", "") == "add_argument":
            first = node.args[0]
            if isinstance(first, ast.Constant) and str(first.value).startswith("--"):
                flags.add(first.value)
    return flags


def _bundle(tmp_path, mpu=0.297, with_transform=True):
    root = tmp_path / "scene"
    root.mkdir()
    (root / "scene.splat").write_bytes(b"not a real splat")
    metric_splat_depth_ppo_v1(60.7).save(str(root / "policy_contract.json"))
    files = {"splat": "scene.splat", "contract": "policy_contract.json"}
    if with_transform:
        SplatTransform.identity_metres(mpu).save(str(root / "vicon_transform.json"))
        files["vicon_transform"] = "vicon_transform.json"
    (root / "manifest.json").write_text(json.dumps({"name": "t", "files": files}))
    return SceneBundle.load(str(root))


def _value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


# ------------------------------------------------------------- the file itself
def test_the_worker_ships_with_the_repository():
    assert os.path.isfile(WORKER_SCRIPT)
    assert os.path.basename(os.path.dirname(WORKER_SCRIPT)) == "worker"
    ast.parse(_source())


def test_the_worker_accepts_every_flag_worker_command_passes(tmp_path):
    cmd = worker_command(_bundle(tmp_path))
    passed = {a for a in cmd if a.startswith("--")}
    assert passed <= _worker_flags(), passed - _worker_flags()


def test_the_worker_speaks_the_clients_protocol():
    """SplatWorkerClient sends these keys and reads these back. A rename on
    either side would otherwise only show up on a GPU machine."""
    src = _source()
    for key in ("ready", "scale_to_metres", "position", "orientation_rpy",
                "image_width", "image_height", "fov_x_half_tan",
                "ok", "shape", "depth_b64", "rgb_shape", "rgb_b64",
                "cmd", "close"):
        assert '"%s"' % key in src, key


# -------------------------------------------------------- the command it gets
def test_the_scale_and_the_empty_depth_come_from_the_bundle(tmp_path):
    b = _bundle(tmp_path, mpu=0.297)
    cmd = worker_command(b)
    assert float(_value(cmd, "--scale-to-metres")) == pytest.approx(0.297)
    want = b.contract().observation.sensor.depth.empty_depth_m
    assert float(_value(cmd, "--empty-depth-raw-m")) == pytest.approx(want)
    assert _value(cmd, "--backend") == "cleaned-splat"
    assert _value(cmd, "--splat") == os.path.join(b.root, "scene.splat")


def test_the_paths_are_absolute_so_the_command_runs_from_anywhere(tmp_path,
                                                                  monkeypatch):
    _bundle(tmp_path)
    monkeypatch.chdir(tmp_path)
    cmd = worker_command(SceneBundle.load("scene"))
    for flag in ("--splat", "--splat-config", "--transforms-json"):
        assert os.path.isabs(_value(cmd, flag)), flag
    assert os.path.isabs(cmd[1])


def test_the_interpreter_defaults_to_this_one_and_can_be_named(tmp_path):
    b = _bundle(tmp_path)
    assert worker_command(b)[:2] == [sys.executable, WORKER_SCRIPT]
    assert worker_command(b, python="/opt/gpu/bin/python")[0] == "/opt/gpu/bin/python"


def test_a_scene_without_a_registration_is_refused(tmp_path):
    """Without the measured scale the worker falls back on the capture's own,
    about a percent out, and renders from somewhere the drone is not."""
    with pytest.raises(ValueError, match="no vicon_transform"):
        worker_command(_bundle(tmp_path, with_transform=False))


def test_a_missing_worker_file_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="no render worker"):
        worker_command(_bundle(tmp_path), script=str(tmp_path / "nope.py"))
