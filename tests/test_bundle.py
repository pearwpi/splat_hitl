"""The scene bundle: does it check the parts against EACH OTHER.

The interesting tests here are not "a file is missing" -- that is loud on its
own. They are the quiet ones: an ESDF built from a different export than the
splat, a gate outside the mapped volume, a manifest edited without the file it
describes.
"""

import hashlib
import json
import numpy as np
import os
import pytest

from splat_hitl.bundle import MANIFEST_NAME, SceneBundle
from splat_hitl.contract import metric_splat_depth_ppo_v1
from splat_hitl.frames import SplatTransform
from splat_hitl.gates import Gate, GateCourse

SPLAT_BYTES = b"not a real splat, and this module never parses one"


def write_bundle(root, *, gate_centre=(2.0, 2.0, 0.6), scale=1.0,
                 esdf_mpu=1.0, voxel=0.05, truncation=1.0,
                 checksum=True, fingerprint=True, with_transform=False,
                 drop=(), dataparser=True, splat_frame=None, map_lines=None,
                 task=None, splat_frame_key="metres_per_splat_unit"):
    root = str(root)
    os.makedirs(root, exist_ok=True)
    files = {}

    with open(os.path.join(root, "scene.splat"), "wb") as fh:
        fh.write(SPLAT_BYTES)
    files["splat"] = "scene.splat"

    # 4 x 4 x 2 m of free space, everything a metre from anything
    grid = np.full((80, 80, 40), truncation, dtype=float)
    np.save(os.path.join(root, "scene_esdf.npy"),
            {"esdf": grid, "voxel_size": voxel, "origin": [0.0, 0.0, 0.0],
             "truncation": truncation, "scale_to_metres": esdf_mpu},
            allow_pickle=True)
    files["esdf"] = "scene_esdf.npy"

    c = metric_splat_depth_ppo_v1(90.0)
    c.save(os.path.join(root, "policy_contract.json"))
    files["contract"] = "policy_contract.json"

    GateCourse([Gate("g1", np.array(gate_centre), np.array([1.0, 0.0, 0.0]),
                     1.0, 1.0)]).save(os.path.join(root, "gates.json"))
    files["gates"] = "gates.json"

    if dataparser:
        with open(os.path.join(root, "dataparser_transforms.json"), "w") as fh:
            json.dump({"scale": scale}, fh)
        files["dataparser_transforms"] = "dataparser_transforms.json"

    if splat_frame is not None:
        with open(os.path.join(root, "splat_frame.json"), "w") as fh:
            json.dump({"scale": {splat_frame_key: float(splat_frame)},
                       "gravity": {"nerfstudio_up_error_deg": 63.1}}, fh)
        files["splat_frame"] = "splat_frame.json"

    if with_transform:
        SplatTransform.identity_metres(esdf_mpu).save(
            os.path.join(root, "vicon_transform.json"))
        files["vicon_transform"] = "vicon_transform.json"

    if map_lines is not None:
        with open(os.path.join(root, "map.txt"), "w") as fh:
            fh.write(map_lines)
        files["map"] = "map.txt"

    if task is not None:
        with open(os.path.join(root, "task.json"), "w") as fh:
            json.dump(task, fh)
        files["task"] = "task.json"

    for k in drop:
        files.pop(k, None)

    manifest = {"_comment": "one scene, as handed to a student",
                "name": "test_scene", "files": files}
    if checksum:
        manifest["checksums"] = {
            "splat": hashlib.sha256(SPLAT_BYTES).hexdigest()}
    if fingerprint:
        manifest["contract_fingerprint"] = c.fingerprint()
    with open(os.path.join(root, MANIFEST_NAME), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return root


# ------------------------------------------------------------------ loading
def test_a_directory_without_a_manifest_is_not_a_bundle(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a bundle"):
        SceneBundle.load(tmp_path)


def test_a_well_formed_bundle_passes(tmp_path):
    b = SceneBundle.load(write_bundle(tmp_path))
    rep = b.check()
    assert rep.ok, str(rep)
    assert b.contract().observation.shape == (4, 64, 96)
    assert len(b.gates().gates) == 1


def test_a_declared_but_missing_file_is_an_error(tmp_path):
    root = write_bundle(tmp_path)
    os.remove(os.path.join(root, "gates.json"))
    rep = SceneBundle.load(root).check()
    assert not rep.ok
    assert any("gates.json" in e for e in rep.errors)


def test_an_undeclared_required_file_is_an_error(tmp_path):
    rep = SceneBundle.load(write_bundle(tmp_path, drop=("esdf",))).check()
    assert not rep.ok and any("no 'esdf'" in e for e in rep.errors)


# ------------------------------------------------------------- the quiet ones
def test_a_truncated_splat_is_caught_by_the_checksum(tmp_path):
    root = write_bundle(tmp_path)
    with open(os.path.join(root, "scene.splat"), "wb") as fh:
        fh.write(SPLAT_BYTES[:10])              # half-copied
    rep = SceneBundle.load(root).check()
    assert not rep.ok and any("checksum mismatch" in e for e in rep.errors)


def test_no_checksum_is_only_a_warning(tmp_path):
    rep = SceneBundle.load(write_bundle(tmp_path, checksum=False)).check()
    assert rep.ok and any("no splat checksum" in w for w in rep.warnings)


def test_a_manifest_edited_without_its_contract_is_caught(tmp_path):
    root = write_bundle(tmp_path)
    p = os.path.join(root, MANIFEST_NAME)
    m = json.load(open(p))
    m["contract_fingerprint"] = "deadbeefcafe"
    json.dump(m, open(p, "w"))
    rep = SceneBundle.load(root).check()
    assert not rep.ok and any("edited one without the other" in e
                              for e in rep.errors)


def test_a_gate_outside_the_mapped_volume_is_an_error(tmp_path):
    rep = SceneBundle.load(write_bundle(tmp_path,
                                        gate_centre=(9.0, 2.0, 0.6))).check()
    assert not rep.ok and any("outside the ESDF volume" in e for e in rep.errors)


def test_a_gate_only_partly_outside_is_still_caught(tmp_path):
    """The centre is inside and a corner is not.

    A gate 0.2 m off the floor with a 1 m opening has its lower edge 300 mm
    BELOW the mapped volume. Checking the centre alone would pass it, and then
    half the opening is unscoreable -- a drone through the bottom of that gate
    is somewhere the collision field cannot see.
    """
    rep = SceneBundle.load(write_bundle(tmp_path,
                                        gate_centre=(2.0, 2.0, 0.2))).check()
    assert not rep.ok and any("outside the ESDF volume" in e for e in rep.errors)


def test_scale_disagreement_between_esdf_and_dataparser(tmp_path):
    """The ESDF built from a different export than the splat."""
    rep = SceneBundle.load(write_bundle(tmp_path, scale=2.0,
                                        esdf_mpu=1.0)).check()
    assert not rep.ok and any("scale disagreement" in e for e in rep.errors)


def test_a_percent_of_capture_scale_error_is_a_warning(tmp_path):
    """A metric capture lands within about a percent of a tape measure.

    That gap is the capture's scale error, measured -- not a mismatched export.
    Failing the bundle for it would fail every honestly registered scene, so it
    is reported with both numbers and the measured value is the one kept.
    """
    rep = SceneBundle.load(write_bundle(tmp_path, scale=1.0 / 1.014,
                                        esdf_mpu=1.0)).check()
    assert rep.ok, str(rep)
    assert any("capture's scale error" in w and "1.4%" in w
               for w in rep.warnings), str(rep)


def test_a_registration_at_a_different_scale_is_an_error(tmp_path):
    root = write_bundle(tmp_path, with_transform=True)
    SplatTransform.identity_metres(3.0).save(
        os.path.join(root, "vicon_transform.json"))
    rep = SceneBundle.load(root).check()
    assert not rep.ok and any("differently sized copy" in e for e in rep.errors)


# ------------------------------------------------------------------ clearance
def test_clearance_finer_than_a_voxel_is_refused(tmp_path):
    rep = SceneBundle.load(write_bundle(tmp_path)).check(clearance_m=0.01)
    assert not rep.ok and any("finer than the ESDF voxel" in e
                              for e in rep.errors)


def test_clearance_past_the_truncation_is_refused(tmp_path):
    rep = SceneBundle.load(write_bundle(tmp_path)).check(clearance_m=1.5)
    assert not rep.ok and any("nothing would ever fail" in e
                              for e in rep.errors)


# --------------------------------------------------------------- the lab gap
def test_a_bundle_without_a_registration_is_trainable_but_not_flyable(tmp_path):
    rep = SceneBundle.load(write_bundle(tmp_path)).check()
    assert rep.ok
    assert any("cannot be FLOWN" in n for n in rep.notes)
    assert SceneBundle.load(write_bundle(tmp_path)).transform() is None


def test_a_bundle_with_a_registration_says_nothing_about_it(tmp_path):
    b = SceneBundle.load(write_bundle(tmp_path, with_transform=True))
    rep = b.check()
    assert rep.ok and not any("cannot be FLOWN" in n for n in rep.notes)
    assert b.transform() is not None


# ------------------------------------------------- a scene that is not a race
#: 4 x 4 x 2 m, matching write_bundle's all-free ESDF. No blocks, so nothing in
#: it can disagree with a distance field that says everything is clear.
EMPTY_MAP = "# a room with nothing in it\nboundary 0 0 0 4 4 2\n"


def test_a_bundle_without_gates_is_valid_and_says_so(tmp_path):
    """A planning scene poses a start and a goal, not a course. Requiring gates
    forced every such scene to invent some, which is how this bundle got a
    four-gate slalom it had no use for."""
    rep = SceneBundle.load(write_bundle(tmp_path, drop=("gates",))).check()
    assert rep.ok, str(rep)
    assert any("not a race course" in n for n in rep.notes), str(rep)


def test_a_map_of_empty_space_agrees_with_an_empty_esdf(tmp_path):
    rep = SceneBundle.load(write_bundle(tmp_path, drop=("gates",),
                                        with_transform=True,
                                        map_lines=EMPTY_MAP)).check()
    assert rep.ok, str(rep)
    assert any("BlockMap" in n for n in rep.notes), str(rep)


def test_a_block_that_is_open_space_in_the_esdf_is_an_error(tmp_path):
    """Someone moved a box and edited the map without recapturing, or the other
    way round. Both files still load; every verified plan flies into nothing,
    or through something."""
    rep = SceneBundle.load(write_bundle(
        tmp_path, drop=("gates",), with_transform=True,
        map_lines=EMPTY_MAP + "block 1 1 0 2 2 1 255 0 0\n")).check()
    assert not rep.ok
    assert any("open space in the ESDF" in e for e in rep.errors), str(rep)


def test_a_map_that_cannot_be_compared_says_so_rather_than_passing(tmp_path):
    """No registration means the map and the ESDF are in different frames.
    Silently skipping the comparison would read as a clean bill of health."""
    rep = SceneBundle.load(write_bundle(tmp_path, drop=("gates",),
                                        map_lines=EMPTY_MAP)).check()
    assert rep.ok, str(rep)
    assert any("different frames" in n for n in rep.notes), str(rep)


def test_a_goal_outside_the_map_boundary_is_an_error(tmp_path):
    rep = SceneBundle.load(write_bundle(
        tmp_path, drop=("gates",), with_transform=True, map_lines=EMPTY_MAP,
        task={"pairs": [{"name": "p", "start": [1, 1, 1],
                         "goal": [9, 1, 1]}]})).check()
    assert not rep.ok
    assert any("outside the map boundary" in e for e in rep.errors), str(rep)


def test_a_start_inside_a_block_is_an_error(tmp_path):
    rep = SceneBundle.load(write_bundle(
        tmp_path, drop=("gates",), with_transform=True,
        map_lines=EMPTY_MAP + "block 1 1 0 2 2 1 0 255 0\n",
        task={"pairs": [{"name": "p", "start": [1.5, 1.5, 0.5],
                         "goal": [3, 3, 1]}]})).check()
    assert not rep.ok
    assert any("inside a block" in e for e in rep.errors), str(rep)


def test_a_task_with_no_pairs_poses_no_problem(tmp_path):
    rep = SceneBundle.load(write_bundle(
        tmp_path, drop=("gates",), with_transform=True, map_lines=EMPTY_MAP,
        task={"pairs": []})).check()
    assert any("poses no problem" in w for w in rep.warnings), str(rep)


def test_endpoints_in_open_space_pass(tmp_path):
    rep = SceneBundle.load(write_bundle(
        tmp_path, drop=("gates",), with_transform=True, map_lines=EMPTY_MAP,
        task={"pairs": [{"name": "p", "start": [0.5, 0.5, 0.5],
                         "goal": [3.5, 3.5, 1.5]}]})).check()
    assert rep.ok, str(rep)



# -- a COLMAP scene's metric scale has a different, better provenance ---------

def test_no_dataparser_and_no_splat_frame_is_a_warning(tmp_path):
    write_bundle(tmp_path, dataparser=False)
    rep = SceneBundle.load(str(tmp_path)).check()
    assert rep.ok
    assert any("no provenance" in w for w in rep.warnings)


def test_splat_frame_answers_for_the_scale_instead(tmp_path):
    """A COLMAP capture ships splat_frame.json, not a dataparser scale.

    Its dataparser scale is in COLMAP units and claims nothing about the room,
    so omitting it is correct and the scene is not short of provenance -- it has
    better provenance. That has to read as a note, not as a warning, or every
    COLMAP scene carries a permanent complaint that says the opposite of what is
    true.
    """
    write_bundle(tmp_path, dataparser=False, splat_frame=1.0, esdf_mpu=1.0)
    rep = SceneBundle.load(str(tmp_path)).check()
    assert rep.ok
    assert not any("no provenance" in w for w in rep.warnings)
    note = " ".join(rep.notes)
    assert "COLMAP" in note and "63.1 deg from gravity" in note


def test_splat_frame_reports_how_far_arkit_was_from_the_measurement(tmp_path):
    write_bundle(tmp_path, dataparser=False, splat_frame=1.05, esdf_mpu=1.0)
    rep = SceneBundle.load(str(tmp_path)).check()
    assert rep.ok
    assert "5.0%" in " ".join(rep.notes)


def test_a_splat_frame_written_before_the_fold_still_reads(tmp_path):
    """`metres_per_splat_unit_arkit` was the raw ARKit or dataparser claim, with the
    pole-top fit's correction reported beside it and NOT applied. `splat_frame.py`
    now folds that correction in and writes the corrected figure as
    `metres_per_splat_unit`. Dropping the old key rather than keeping it as an
    alias is deliberate -- a reader that has not been updated should fail loudly
    instead of quietly using an uncorrected scale -- but frames already on disk,
    a5_test's shipped one among them, still carry only the old name and must keep
    working.
    """
    write_bundle(tmp_path, dataparser=False, splat_frame=1.05, esdf_mpu=1.0,
                 splat_frame_key="metres_per_splat_unit_arkit")
    rep = SceneBundle.load(str(tmp_path)).check()
    assert rep.ok
    assert "5.0%" in " ".join(rep.notes)


def test_an_unreadable_splat_frame_is_a_warning(tmp_path):
    write_bundle(tmp_path, dataparser=False, splat_frame=1.0)
    with open(os.path.join(str(tmp_path), "splat_frame.json"), "w") as fh:
        fh.write("{ this is not json")
    rep = SceneBundle.load(str(tmp_path)).check()
    assert any("splat_frame unreadable" in w for w in rep.warnings)


# ------------------------------------------------------------------- level
def _turned(roll_deg, yaw_deg=0.0):
    from splat_hitl.frames import rpy_to_matrix
    R = rpy_to_matrix(np.radians(roll_deg), 0.0, np.radians(yaw_deg))
    return SplatTransform(R, np.zeros(3), 1.0)


def test_a_scene_stored_tilted_from_level_is_an_error(tmp_path):
    """A COLMAP-solved splat keeps COLMAP's frame, tens of degrees from level,
    and SplatEnv would render and fly it that way."""
    root = write_bundle(tmp_path, with_transform=True)
    _turned(63.0).save(os.path.join(root, "vicon_transform.json"))
    rep = SceneBundle.load(root).check()
    assert not rep.ok and any("from level" in e for e in rep.errors)


def test_a_scene_turned_in_heading_and_off_by_a_little_is_level(tmp_path):
    root = write_bundle(tmp_path, with_transform=True)
    _turned(0.6, yaw_deg=136.6).save(os.path.join(root, "vicon_transform.json"))
    rep = SceneBundle.load(root).check()
    assert rep.ok, rep.errors
