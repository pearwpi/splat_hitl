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
                 drop=(), dataparser=True):
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

    if with_transform:
        SplatTransform.identity_metres(esdf_mpu).save(
            os.path.join(root, "vicon_transform.json"))
        files["vicon_transform"] = "vicon_transform.json"

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
