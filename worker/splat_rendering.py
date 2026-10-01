#!/usr/bin/env python3
"""RGB/depth rendering backends and JSON worker for metric splat scenes."""

# PEP 604 annotations (`Path | None`) are evaluated at def time on
# Python 3.9 and older, where they raise TypeError at IMPORT. The render
# machine's nerfstudio env is 3.8, so this module could not be imported
# there at all. Deferring annotations makes them strings and costs
# nothing; do not remove it without checking the oldest interpreter this
# has to run on.
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch


def relocated_transforms_near_config(splat_config):
    """Find dataset intrinsics after a Nerfstudio output tree is relocated."""
    config_path = Path(splat_config).resolve()
    for parent in config_path.parents:
        candidate = parent / "transforms.json"
        if candidate.is_file():
            return candidate
    return None


def rotation_matrix_from_rpy(rpy):
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(roll), -np.sin(roll)], [0.0, np.sin(roll), np.cos(roll)]],
        dtype=np.float64,
    )
    ry = np.array(
        [[np.cos(pitch), 0.0, np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0.0, np.cos(pitch)]],
        dtype=np.float64,
    )
    rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def _camera_intrinsics(data, path):
    """Intrinsics from a Nerfstudio transforms.json: top level, or per frame.

    ns-process-data writes them at the TOP LEVEL when one camera describes every
    frame, and PER FRAME when it will not promise that -- which is what the
    polycam path does unconditionally, even where every frame in fact agrees.
    Reading only the top level raises KeyError on every Polycam-captured scene,
    so the worker dies before its first frame.

    Frames that disagree are refused rather than averaged: one pinhole model
    cannot describe two cameras, and a quietly averaged focal length renders a
    plausible picture of the wrong room.
    """
    need = ("fl_x", "fl_y", "cx", "cy", "w", "h")
    if all(k in data for k in need):
        src = data
    else:
        seen = {tuple(f[k] for k in need) for f in (data.get("frames") or [])
                if all(k in f for k in need)}
        if not seen:
            raise RuntimeError(
                "%s carries no camera intrinsics, neither at the top level nor "
                "on any frame -- is this a Nerfstudio transforms.json?" % (path,))
        if len(seen) > 1:
            raise RuntimeError(
                "%s has %d different intrinsic sets across its frames, so one "
                "camera model cannot describe it. Re-process the capture."
                % (path, len(seen)))
        src = dict(zip(need, seen.pop()))
    return {
        "path": str(path),
        "width": int(src["w"]),
        "height": int(src["h"]),
        "fx": float(src["fl_x"]),
        "fy": float(src["fl_y"]),
        "cx": float(src["cx"]),
        "cy": float(src["cy"]),
    }


def load_scale_to_metres_from_config(config_path):
    dataparser_path = Path(config_path).resolve().parent / "dataparser_transforms.json"
    if dataparser_path.exists():
        with open(dataparser_path) as f:
            data = json.load(f)
        return 1.0 / float(data.get("scale", 1.0))
    return 1.0


class NerfstudioRGBDRenderer:
    """Render RGB/depth from a trained Nerfstudio splatfacto checkpoint."""

    def __init__(self, config_path: str):
        from nerfstudio.cameras.cameras import Cameras
        from nerfstudio.utils.eval_utils import eval_setup

        self.Cameras = Cameras
        self.config_path = Path(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        original_torch_load = torch.load

        def torch_load_patched(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        torch.load = torch_load_patched
        try:
            result = eval_setup(config_path=self.config_path, test_mode="inference")
        finally:
            torch.load = original_torch_load

        if len(result) == 4:
            _, pipeline, _, _ = result
        else:
            _, pipeline = result

        self.pipeline = pipeline
        self.pipeline.eval()
        self.model = self.pipeline.model
        self.scale_to_metres = load_scale_to_metres_from_config(self.config_path)

        train_cameras = self.pipeline.datamanager.train_dataset.cameras
        if len(train_cameras) == 0:
            raise ValueError("No training cameras found in pipeline")

        ref_camera = train_cameras[0]
        self.camera_height = int(ref_camera.height[0].item())
        self.camera_width = int(ref_camera.width[0].item())
        self.fx = float(ref_camera.fx[0].item())
        self.fy = float(ref_camera.fy[0].item())
        self.cx = float(ref_camera.cx[0].item())
        self.cy = float(ref_camera.cy[0].item())

        if hasattr(ref_camera, "distortion_params") and ref_camera.distortion_params is not None:
            self.distortion_params = ref_camera.distortion_params[0].clone()
        else:
            self.distortion_params = torch.zeros(6, device=self.device)

    def render(
        self,
        position: np.ndarray,
        orientation_rpy: np.ndarray,
        image_height=None,
        image_width=None,
        fov_x_half_tan=None,
    ) -> dict:
        if image_height is None:
            image_height = self.camera_height
        if image_width is None:
            image_width = self.camera_width

        rotation = rotation_matrix_from_rpy(orientation_rpy)
        c2w = torch.eye(4, dtype=torch.float32, device=self.device)
        c2w[:3, :3] = torch.as_tensor(rotation, dtype=torch.float32, device=self.device)
        c2w[:3, 3] = torch.as_tensor(position, dtype=torch.float32, device=self.device)

        scale_x = image_width / self.camera_width
        scale_y = image_height / self.camera_height
        if fov_x_half_tan is None:
            fx = self.fx * scale_x
            fy = self.fy * scale_y
            cx = self.cx * scale_x
            cy = self.cy * scale_y
            distortion_params = self.distortion_params.unsqueeze(0)
        else:
            fx = float(image_width) / (2.0 * float(fov_x_half_tan))
            fy = fx
            cx = (float(image_width) - 1.0) * 0.5
            cy = (float(image_height) - 1.0) * 0.5
            distortion_params = torch.zeros_like(self.distortion_params).unsqueeze(0)
        camera = self.Cameras(
            camera_to_worlds=c2w.unsqueeze(0),
            fx=torch.tensor([fx], device=self.device),
            fy=torch.tensor([fy], device=self.device),
            cx=torch.tensor([cx], device=self.device),
            cy=torch.tensor([cy], device=self.device),
            height=image_height,
            width=image_width,
            distortion_params=distortion_params,
        )

        with torch.no_grad():
            outputs = self.model.get_outputs_for_camera(camera)

        depth = outputs["depth"].squeeze().cpu().numpy()
        rgb = np.clip(outputs["rgb"].squeeze().cpu().numpy(), 0.0, 1.0)

        # Convert the renderer's bottom-up raster to the shared top-down image
        # convention. Horizontal image coordinates already match the camera
        # frame and must not be mirrored.
        depth = np.flipud(depth).copy()
        rgb = np.flipud(rgb).copy()
        return {"depth": depth, "rgb": rgb}


def dataset_transforms_from_config(splat_config):
    text = Path(splat_config).read_text()
    match = re.search(r"^data:.*?\n((?:- .*\n)+)", text, flags=re.MULTILINE)
    if not match:
        return relocated_transforms_near_config(splat_config)
    parts = []
    for line in match.group(1).splitlines():
        value = line.strip()[2:]
        if value:
            parts.append(value)
    if not parts:
        return relocated_transforms_near_config(splat_config)
    root = Path("/" + "/".join(part for part in parts if part != "/")).resolve()
    transforms = root / "transforms.json"
    if transforms.is_file():
        return transforms
    return relocated_transforms_near_config(splat_config)


def load_intrinsics(transforms_json, splat_config):
    path = Path(transforms_json).resolve() if transforms_json else None
    if path is None and splat_config:
        path = dataset_transforms_from_config(splat_config)
    if path is None or not path.exists():
        raise RuntimeError("Could not find transforms.json. Pass --transforms-json explicitly.")
    with open(path) as f:
        data = json.load(f)
    return _camera_intrinsics(data, path)


def scaled_intrinsics_matrix(intrinsics, image_width, image_height):
    """Scale native camera intrinsics to the requested raster dimensions."""
    image_width = int(image_width)
    image_height = int(image_height)
    native_width = int(intrinsics["width"])
    native_height = int(intrinsics["height"])
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Rendered image dimensions must be positive")
    if native_width <= 0 or native_height <= 0:
        raise ValueError("Native camera dimensions must be positive")
    scale_x = float(image_width) / float(native_width)
    scale_y = float(image_height) / float(native_height)
    return np.array(
        [
            [float(intrinsics["fx"]) * scale_x, 0.0, float(intrinsics["cx"]) * scale_x],
            [0.0, float(intrinsics["fy"]) * scale_y, float(intrinsics["cy"]) * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def load_supersplat(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size % 32 != 0:
        raise RuntimeError(f"{path} size is not divisible by 32 bytes; not the expected .splat layout")
    records = data.reshape(-1, 32)
    floats = records[:, :24].copy().view("<f4").reshape(-1, 6)
    positions = floats[:, :3].astype(np.float32)
    scales = floats[:, 3:6].astype(np.float32)
    colors = records[:, 24:28].astype(np.float32) / 255.0
    rotations = records[:, 28:32].astype(np.float32)
    return positions, scales, colors, rotations


def decode_supersplat_quats(rotation_bytes):
    quats = (rotation_bytes.astype(np.float32) - 128.0) / 128.0
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8)
    return quats.astype(np.float32)


def viewmat_for_gsplat(camera_position, orientation_rpy):
    rotation = rotation_matrix_from_rpy(orientation_rpy)
    to_cv = np.diag([1.0, 1.0, -1.0])
    world_to_camera = to_cv @ rotation.T
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = world_to_camera.astype(np.float32)
    viewmat[:3, 3] = (-world_to_camera @ camera_position.reshape(3)).astype(np.float32)
    return viewmat


class CleanedSplatRGBDRenderer:
    """Render RGB/depth from a cleaned SuperSplat .splat export using gsplat."""

    def __init__(
        self,
        splat_path,
        splat_config,
        transforms_json=None,
        max_points=None,
        empty_depth_raw_m=12.0,
        flip_ud=False,
        flip_lr=False,
        scale_to_metres=None,
    ):
        from gsplat.rendering import rasterization

        if not torch.cuda.is_available():
            raise RuntimeError("CleanedSplatRGBDRenderer requires CUDA because gsplat rasterization is CUDA-backed.")

        self.rasterization = rasterization
        self.splat_path = Path(splat_path)
        self.splat_config = Path(splat_config)
        # The capture's dataparser scale is what the CAPTURE believes; a scene
        # registered against a tape measure knows better, and on a LiDAR/VIO
        # capture the two differ by around a percent. A percent of camera
        # placement across a 6 m room is ~7 cm, which is a rendered view of
        # somewhere the drone is not -- and every component still reports
        # healthy. So an explicit value always wins, and the source is printed.
        if scale_to_metres is not None:
            self.scale_to_metres = float(scale_to_metres)
            self.scale_source = "explicit"
        else:
            self.scale_to_metres = load_scale_to_metres_from_config(self.splat_config)
            self.scale_source = "dataparser_transforms.json (the capture's own claim)"
        if not self.scale_to_metres > 0:
            raise ValueError("scale_to_metres must be positive, got %r"
                             % (self.scale_to_metres,))
        self.intrinsics = load_intrinsics(transforms_json, self.splat_config)
        self.camera_width = self.intrinsics["width"]
        self.camera_height = self.intrinsics["height"]
        self.empty_depth = float(empty_depth_raw_m) / self.scale_to_metres
        self.flip_ud = bool(flip_ud)
        self.flip_lr = bool(flip_lr)
        self.device = torch.device("cuda")

        positions, scales, colors, rotations = load_supersplat(self.splat_path)
        if max_points is not None and len(positions) > max_points:
            rng = np.random.default_rng(0)
            indices = rng.choice(len(positions), size=int(max_points), replace=False)
            positions = positions[indices]
            scales = scales[indices]
            colors = colors[indices]
            rotations = rotations[indices]

        self.means = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        self.quats = torch.as_tensor(decode_supersplat_quats(rotations), dtype=torch.float32, device=self.device)
        self.scales = torch.as_tensor(np.maximum(scales, 1e-6), dtype=torch.float32, device=self.device)
        self.colors = torch.as_tensor(colors[:, :3], dtype=torch.float32, device=self.device)
        self.opacities = torch.as_tensor(np.clip(colors[:, 3], 0.0, 1.0), dtype=torch.float32, device=self.device)
        self.K = torch.as_tensor(
            scaled_intrinsics_matrix(self.intrinsics, self.camera_width, self.camera_height),
            dtype=torch.float32,
            device=self.device,
        )[None]

    def render(self, position, orientation_rpy, image_height=None, image_width=None, fov_x_half_tan=None):
        image_width = int(image_width) if image_width is not None else self.camera_width
        image_height = int(image_height) if image_height is not None else self.camera_height
        if fov_x_half_tan is None:
            K = torch.as_tensor(
                scaled_intrinsics_matrix(self.intrinsics, image_width, image_height),
                dtype=torch.float32,
                device=self.device,
            )[None]
        else:
            fx = float(image_width) / (2.0 * float(fov_x_half_tan))
            fy = fx
            cx = (float(image_width) - 1.0) * 0.5
            cy = (float(image_height) - 1.0) * 0.5
            K = torch.as_tensor(
                np.array(
                    [
                        [fx, 0.0, cx],
                        [0.0, fy, cy],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                dtype=torch.float32,
                device=self.device,
            )[None]
        viewmat = torch.as_tensor(
            viewmat_for_gsplat(np.asarray(position, dtype=np.float64), np.asarray(orientation_rpy, dtype=np.float64)),
            dtype=torch.float32,
            device=self.device,
        )[None]
        with torch.no_grad():
            rendered, alphas, _ = self.rasterization(
                means=self.means,
                quats=self.quats,
                scales=self.scales,
                opacities=self.opacities,
                colors=self.colors,
                viewmats=viewmat,
                Ks=K,
                width=image_width,
                height=image_height,
                near_plane=0.01,
                far_plane=1e6,
                packed=True,
                backgrounds=None,
                render_mode="RGB+ED",
                rasterize_mode="classic",
            )

        rendered_np = rendered[0].detach().cpu().numpy()
        alpha_np = np.squeeze(alphas[0].detach().cpu().numpy())
        rgb = np.clip(rendered_np[..., :3], 0.0, 1.0).astype(np.float32)
        depth = rendered_np[..., 3].astype(np.float32)
        depth = np.where(alpha_np > 1e-4, depth, self.empty_depth).astype(np.float32)

        # Match the canonical image convention used by NerfstudioRGBDRenderer.
        # gsplat's raster is bottom-up here, but its horizontal coordinates
        # already match the camera frame and must not be mirrored.
        # Legacy flip_lr/flip_ud arguments remain accepted for old commands and
        # legacy scene files but are intentionally ignored to prevent double flips.
        rgb = np.flipud(rgb).copy()
        depth = np.flipud(depth).copy()

        return {"rgb": rgb, "depth": depth}

    def render_viewmat(self, viewmat, K, image_width, image_height, canonicalize=True):
        """Render from an explicit OpenCV world-to-camera matrix and intrinsics."""
        image_width = int(image_width)
        image_height = int(image_height)
        viewmat = torch.as_tensor(
            np.asarray(viewmat, dtype=np.float32), dtype=torch.float32, device=self.device
        )[None]
        K = torch.as_tensor(
            np.asarray(K, dtype=np.float32), dtype=torch.float32, device=self.device
        )[None]
        with torch.no_grad():
            rendered, alphas, _ = self.rasterization(
                means=self.means,
                quats=self.quats,
                scales=self.scales,
                opacities=self.opacities,
                colors=self.colors,
                viewmats=viewmat,
                Ks=K,
                width=image_width,
                height=image_height,
                near_plane=0.01,
                far_plane=1e6,
                packed=True,
                backgrounds=None,
                render_mode="RGB+ED",
                rasterize_mode="classic",
            )

        rendered_np = rendered[0].detach().cpu().numpy()
        alpha_np = np.squeeze(alphas[0].detach().cpu().numpy())
        rgb = np.clip(rendered_np[..., :3], 0.0, 1.0).astype(np.float32)
        depth = rendered_np[..., 3].astype(np.float32)
        depth = np.where(alpha_np > 1e-4, depth, self.empty_depth).astype(np.float32)
        if canonicalize:
            rgb = np.flipud(rgb).copy()
            depth = np.flipud(depth).copy()
        return {"rgb": rgb, "depth": depth}


def run_rgbd_worker(args):
    if args.backend == "cleaned-splat":
        renderer = CleanedSplatRGBDRenderer(
            splat_path=args.splat,
            splat_config=args.splat_config,
            transforms_json=args.transforms_json,
            max_points=args.max_points,
            empty_depth_raw_m=args.empty_depth_raw_m,
            flip_ud=args.flip_ud,
            flip_lr=args.flip_lr,
            scale_to_metres=args.scale_to_metres,
        )
    else:
        renderer = NerfstudioRGBDRenderer(config_path=args.splat_config)

    print(
        json.dumps(
            {
                "ready": True,
                "scale_to_metres": renderer.scale_to_metres,
                "scale_source": getattr(renderer, "scale_source", "config"),
                "width": getattr(renderer, "camera_width", None),
                "height": getattr(renderer, "camera_height", None),
                "backend": args.backend,
            }
        ),
        flush=True,
    )

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("cmd") == "close":
                break

            position = np.asarray(request["position"], dtype=np.float32)
            orientation = np.asarray(request["orientation_rpy"], dtype=np.float32)
            result = renderer.render(
                position,
                orientation,
                image_height=request.get("image_height"),
                image_width=request.get("image_width"),
                fov_x_half_tan=request.get("fov_x_half_tan"),
            )
            depth = np.asarray(result["depth"], dtype=np.float32)
            rgb = np.asarray(result["rgb"], dtype=np.float32)

            response = {
                "ok": True,
                "shape": list(depth.shape),
                "dtype": "float32",
                "depth_b64": base64.b64encode(depth.tobytes()).decode("ascii"),
                "rgb_shape": list(rgb.shape),
                "rgb_dtype": "float32",
                "rgb_b64": base64.b64encode(rgb.tobytes()).decode("ascii"),
                "min": float(np.min(depth)),
                "max": float(np.max(depth)),
                "mean": float(np.mean(depth)),
            }
        except Exception as exc:
            response = {"ok": False, "error": repr(exc)}

        print(json.dumps(response), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "visualize-cleaned":
        visualize_cleaned_splat_cli(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description="Splat RGB/depth rendering worker")
    parser.add_argument("--backend", choices=("nerfstudio", "cleaned-splat"), default="nerfstudio")
    parser.add_argument("--splat-config", required=True, help="Nerfstudio config used for renderer or scale/intrinsics lookup")
    parser.add_argument("--splat", default=None, help="Cleaned SuperSplat .splat file for --backend cleaned-splat")
    parser.add_argument("--transforms-json", default=None, help="Optional explicit transforms.json for cleaned splat intrinsics")
    parser.add_argument("--max-points", type=int, default=None, help="Optional random subset for faster cleaned-splat rendering")
    parser.add_argument("--empty-depth-raw-m", type=float, default=12.0, help="Depth assigned to pixels with no splat hit")
    parser.add_argument("--scale-to-metres", type=float, default=None,
                        help="Metres per normalised unit, MEASURED. Overrides the "
                             "capture's own dataparser scale, which is a claim and "
                             "is typically about a percent out. A registered scene "
                             "carries the right number in its bundle.")
    parser.add_argument("--flip-ud", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--flip-lr", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.backend == "cleaned-splat" and not args.splat:
        parser.error("--splat is required with --backend cleaned-splat")
    run_rgbd_worker(args)


# ---- Cleaned SuperSplat visualizer subcommand ----

"""Visualize and sanity-render a SuperSplat .splat export.

This is intentionally standalone. It does not alter the policy adapters. It
loads the cleaned .splat binary, lets the user pick a camera position on a
point cloud, then renders an approximate RGB/depth view from the cleaned splat
centers using the dataset intrinsics.
"""

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.fonts=false")

import numpy as np
import torch

# cv2, open3d and gsplat belong to the INTERACTIVE VISUALISER below, not to the
# render worker. Imported at module scope they were a hard dependency of the
# whole file, so the worker -- the one thing that has to come up on the flight
# GPU machine -- could not even be imported without a desktop GUI stack. Loaded
# on demand instead, the same way scene_tools.py does it.
cv2 = None
o3d = None
rasterization = None


def require_visualiser_deps():
    """Import what only the interactive viewer needs, and say so if it is absent."""
    global cv2, o3d, rasterization
    if o3d is not None:
        return
    try:
        import cv2 as _cv2
        import open3d as _o3d
        from gsplat.rendering import rasterization as _raster
    except ImportError as exc:
        raise RuntimeError(
            "the cleaned-splat visualiser needs opencv-python, open3d and "
            "gsplat: %s. The render worker does not -- run that instead if you "
            "are on a headless machine." % (exc,)) from exc
    _o3d.utility.set_verbosity_level(_o3d.utility.VerbosityLevel.Error)
    cv2, o3d, rasterization = _cv2, _o3d, _raster

BASE_RENDER_ROLL_RADIANS = np.pi / 2.0


def load_supersplat(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size % 32 != 0:
        raise RuntimeError(f"{path} size is not divisible by 32 bytes; not the expected .splat layout")
    records = data.reshape(-1, 32)
    floats = records[:, :24].copy().view("<f4").reshape(-1, 6)
    positions = floats[:, :3].astype(np.float32)
    scales = floats[:, 3:6].astype(np.float32)
    colors = records[:, 24:28].astype(np.float32) / 255.0
    rotations = records[:, 28:32].astype(np.float32)
    return positions, scales, colors, rotations


def decode_supersplat_quats(rotation_bytes):
    # SuperSplat/antimatter .splat stores normalized quaternions as 4 uint8s.
    quats = (rotation_bytes.astype(np.float32) - 128.0) / 128.0
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8)
    return quats.astype(np.float32)


def dataparser_from_splat_config(splat_config: str) -> Path:
    return Path(splat_config).resolve().parent / "dataparser_transforms.json"


def load_scale_to_metres(splat_config: str) -> float:
    dataparser_path = dataparser_from_splat_config(splat_config)
    with open(dataparser_path) as f:
        data = json.load(f)
    return 1.0 / float(data.get("scale", 1.0))


def dataset_transforms_from_config(splat_config: str) -> Path | None:
    text = Path(splat_config).read_text()
    match = re.search(r"^data:.*?\n((?:- .*\n)+)", text, flags=re.MULTILINE)
    if not match:
        return relocated_transforms_near_config(splat_config)
    parts = []
    for line in match.group(1).splitlines():
        value = line.strip()[2:]
        if value:
            parts.append(value)
    if not parts:
        return relocated_transforms_near_config(splat_config)
    root = Path("/" + "/".join(part for part in parts if part != "/")).resolve()
    transforms = root / "transforms.json"
    if transforms.is_file():
        return transforms
    return relocated_transforms_near_config(splat_config)


def load_intrinsics(transforms_json: str | None, splat_config: str | None):
    path = Path(transforms_json).resolve() if transforms_json else None
    if path is None and splat_config:
        path = dataset_transforms_from_config(splat_config)
    if path is None or not path.exists():
        raise RuntimeError("Could not find transforms.json. Pass --transforms-json explicitly.")
    with open(path) as f:
        data = json.load(f)
    return _camera_intrinsics(data, path)


def rotation_matrix_from_rpy(rpy):
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(roll), -np.sin(roll)], [0.0, np.sin(roll), np.cos(roll)]],
        dtype=np.float64,
    )
    ry = np.array(
        [[np.cos(pitch), 0.0, np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0.0, np.cos(pitch)]],
        dtype=np.float64,
    )
    rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def axis_angle_rotation(axis_name, degrees):
    axis_map = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    axis = axis_map[axis_name]
    angle = np.deg2rad(float(degrees))
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def rpy_from_rotation(rotation):
    sy = -float(rotation[2, 0])
    cy = float(np.sqrt(max(0.0, 1.0 - sy * sy)))
    if cy > 1e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arcsin(sy)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        pitch = np.arcsin(sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float32)


def corrected_orientation(yaw, render_axis, render_degrees):
    base = rotation_matrix_from_rpy(np.array([BASE_RENDER_ROLL_RADIANS, 0.0, float(yaw)], dtype=np.float64))
    correction = axis_angle_rotation(render_axis, render_degrees)
    return rpy_from_rotation(correction @ base)


def orientation_forward(rpy):
    return rotation_matrix_from_rpy(rpy) @ np.array([0.0, 0.0, -1.0], dtype=np.float64)


def axis_index(axis_name):
    return {"x": 0, "y": 1, "z": 2}[axis_name]


def pick_camera_position(pointcloud_path, scale_to_metres, height_axis, prompt_height):
    pcd = o3d.io.read_point_cloud(pointcloud_path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"Empty point cloud: {pointcloud_path}")
    print("\nCAMERA: SHIFT+CLICK a camera position, press Q, then enter height if needed.", flush=True)
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window("Pick camera position - SHIFT+CLICK then Q", width=1200, height=800)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 3.0
    vis.run()
    vis.destroy_window()
    picks = [p.index for p in vis.get_picked_points()]
    if not picks:
        raise RuntimeError("No camera position selected")
    points = np.asarray(pcd.points)
    position = points[picks[-1]].astype(np.float64)
    if prompt_height:
        idx = axis_index(height_axis)
        raw = position * scale_to_metres
        default_height = raw[idx]
        height = input(
            f"Enter {height_axis.upper()} camera height in RAW metres, or Enter to keep {default_height:.3f}: "
        ).strip()
        if height:
            raw[idx] = float(height)
            position = raw / scale_to_metres
    return position.astype(np.float32), pcd


def show_pose(pointcloud, position, forward, arrow_length):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=arrow_length * 0.12)
    sphere.translate(position)
    sphere.paint_uniform_color([1.0, 0.7, 0.0])
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=arrow_length * 0.04,
        cone_radius=arrow_length * 0.09,
        cylinder_height=arrow_length * 0.75,
        cone_height=arrow_length * 0.25,
    )
    arrow.paint_uniform_color([1.0, 0.45, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    direction = forward / max(np.linalg.norm(forward), 1e-8)
    dot = np.clip(float(z_axis @ direction), -1.0, 1.0)
    if dot > 1.0 - 1e-8:
        rotation = np.eye(3)
    elif dot < -1.0 + 1e-8:
        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([np.pi, 0.0, 0.0]))
    else:
        axis = np.cross(z_axis, direction)
        axis = axis / np.linalg.norm(axis)
        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.arccos(dot))
    arrow.rotate(rotation, center=np.zeros(3))
    arrow.translate(position)
    vis = o3d.visualization.Visualizer()
    vis.create_window("SuperSplat camera pose", width=1200, height=800)
    vis.add_geometry(pointcloud)
    vis.add_geometry(sphere)
    vis.add_geometry(arrow)
    vis.run()
    vis.destroy_window()


def render_point_splat(positions, scales, colors, camera_position, orientation_rpy, intrinsics, max_points=None):
    if max_points is not None and len(positions) > max_points:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(positions), size=max_points, replace=False)
        positions = positions[indices]
        scales = scales[indices]
        colors = colors[indices]

    height = intrinsics["height"]
    width = intrinsics["width"]
    rotation = rotation_matrix_from_rpy(orientation_rpy)
    camera_points = (positions.astype(np.float64) - camera_position.reshape(1, 3)) @ rotation
    depth = -camera_points[:, 2]
    valid = depth > 1e-4
    x = camera_points[:, 0]
    y = camera_points[:, 1]
    u = np.round(intrinsics["fx"] * x / depth + intrinsics["cx"]).astype(np.int32)
    v = np.round(intrinsics["fy"] * y / depth + intrinsics["cy"]).astype(np.int32)
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)

    if not np.any(valid):
        return np.zeros((height, width, 3), dtype=np.float32), np.full((height, width), np.inf, dtype=np.float32)

    u = u[valid]
    v = v[valid]
    depth = depth[valid]
    color = colors[valid, :3]
    pix = v * width + u
    order = np.lexsort((depth, pix))
    pix_sorted = pix[order]
    unique_pix, first = np.unique(pix_sorted, return_index=True)
    chosen = order[first]

    rgb = np.zeros((height * width, 3), dtype=np.float32)
    depth_map = np.full(height * width, np.inf, dtype=np.float32)
    rgb[unique_pix] = color[chosen]
    depth_map[unique_pix] = depth[chosen].astype(np.float32)
    rgb = rgb.reshape(height, width, 3)
    depth_map = depth_map.reshape(height, width)

    hit = np.isfinite(depth_map).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    rgb = cv2.dilate(rgb, kernel, iterations=1)
    finite_depth = np.where(np.isfinite(depth_map), depth_map, 0.0).astype(np.float32)
    finite_depth = cv2.dilate(finite_depth, kernel, iterations=1)
    hit = cv2.dilate(hit, kernel, iterations=1).astype(bool)
    depth_map = np.where(hit, finite_depth, np.inf).astype(np.float32)
    return rgb, depth_map


def viewmat_for_gsplat(camera_position, orientation_rpy):
    rotation = rotation_matrix_from_rpy(orientation_rpy)
    # The local convention used by the adapters renders forward along camera -Z.
    # gsplat expects OpenCV-style +Z depth, so flip the local Z axis.
    to_cv = np.diag([1.0, 1.0, -1.0])
    world_to_camera = to_cv @ rotation.T
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = world_to_camera.astype(np.float32)
    viewmat[:3, 3] = (-world_to_camera @ camera_position.reshape(3)).astype(np.float32)
    return viewmat


def render_gsplat(positions, scales, colors, rotations, camera_position, orientation_rpy, intrinsics, max_points=None):
    if not torch.cuda.is_available():
        raise RuntimeError("gsplat rasterization needs CUDA for this script; use --renderer point for the rough fallback")

    if max_points is not None and len(positions) > max_points:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(positions), size=max_points, replace=False)
        positions = positions[indices]
        scales = scales[indices]
        colors = colors[indices]
        rotations = rotations[indices]

    device = torch.device("cuda")
    means = torch.as_tensor(positions, dtype=torch.float32, device=device)
    quats = torch.as_tensor(decode_supersplat_quats(rotations), dtype=torch.float32, device=device)
    gaussian_scales = torch.as_tensor(np.maximum(scales, 1e-6), dtype=torch.float32, device=device)
    rgb = torch.as_tensor(colors[:, :3], dtype=torch.float32, device=device)
    opacities = torch.as_tensor(np.clip(colors[:, 3], 0.0, 1.0), dtype=torch.float32, device=device)

    viewmat = torch.as_tensor(viewmat_for_gsplat(camera_position, orientation_rpy), dtype=torch.float32, device=device)[None]
    k = np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    ks = torch.as_tensor(k, dtype=torch.float32, device=device)[None]

    rendered, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=gaussian_scales,
        opacities=opacities,
        colors=rgb,
        viewmats=viewmat,
        Ks=ks,
        width=intrinsics["width"],
        height=intrinsics["height"],
        near_plane=0.01,
        far_plane=1e6,
        packed=True,
        backgrounds=None,
        render_mode="RGB+ED",
        rasterize_mode="classic",
    )
    rendered = rendered[0].detach().cpu().numpy()
    return np.clip(rendered[..., :3], 0.0, 1.0).astype(np.float32), rendered[..., 3].astype(np.float32)


def show_rgb_depth(rgb, depth_norm, scale_to_metres, flip_lr=False, flip_ud=False, title_suffix=""):
    if flip_lr:
        rgb = np.flip(rgb, axis=1).copy()
        depth_norm = np.flip(depth_norm, axis=1).copy()
    if flip_ud:
        rgb = np.flip(rgb, axis=0).copy()
        depth_norm = np.flip(depth_norm, axis=0).copy()

    finite = np.isfinite(depth_norm)
    if np.any(finite):
        depth_raw = depth_norm * scale_to_metres
        dmin = float(np.nanmin(depth_raw[finite]))
        dmax = float(np.nanmax(depth_raw[finite]))
        depth_vis = np.zeros(depth_norm.shape, dtype=np.uint8)
        depth_vis[finite] = np.clip((depth_raw[finite] - dmin) / max(dmax - dmin, 1e-6) * 255, 0, 255).astype(np.uint8)
    else:
        dmin = dmax = float("nan")
        depth_vis = np.zeros(depth_norm.shape, dtype=np.uint8)
        print("WARNING: no finite depth hits; camera is probably pointed away from the splat.")
    rgb_bgr = cv2.cvtColor((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    combined = np.hstack([rgb_bgr, cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)])
    print(f"Rendered cleaned .splat{title_suffix}: depth_raw_range=[{dmin:.3f}, {dmax:.3f}]m")
    print("Showing cleaned .splat RGB/depth. Press Q/Esc to close.")
    title = "Cleaned SuperSplat RGB (left) / depth (right)"
    if title_suffix:
        title = f"{title} {title_suffix}"
    while True:
        cv2.imshow(title, combined)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def render_and_show(
    args,
    positions,
    scales,
    colors,
    rotations,
    camera_position,
    intrinsics,
    scale_to_metres,
    render_degrees,
):
    orientation = corrected_orientation(float(args.yaw), args.render_rotate_axis, render_degrees)
    forward = orientation_forward(orientation)
    print(f"Render rotate: axis={args.render_rotate_axis}, degrees={render_degrees:.3f}")
    print(f"Render orientation RPY: {orientation}")
    print(f"Forward norm: {forward / max(np.linalg.norm(forward), 1e-8)}")

    if args.renderer == "gsplat":
        rgb, depth = render_gsplat(
            positions,
            scales,
            colors,
            rotations,
            camera_position.astype(np.float64),
            orientation,
            intrinsics,
            args.max_points,
        )
    else:
        rgb, depth = render_point_splat(
            positions,
            scales,
            colors,
            camera_position.astype(np.float64),
            orientation,
            intrinsics,
            args.max_points,
        )
    show_rgb_depth(
        rgb,
        depth,
        scale_to_metres,
        flip_lr=args.cleaned_splat_flip_lr,
        flip_ud=args.cleaned_splat_flip_ud,
        title_suffix=f"(rot={render_degrees:.1f})",
    )
    return orientation, forward


def visualize_cleaned_splat_cli(argv=None):
    require_visualiser_deps()
    parser = argparse.ArgumentParser(description="Visualize a cleaned SuperSplat .splat export")
    parser.add_argument("--splat", required=True, help="Cleaned SuperSplat .splat file")
    parser.add_argument("--pointcloud", required=True, help="Cleaned point cloud for pose picking")
    parser.add_argument("--splat-config", required=True, help="Nerfstudio config, used only for scale/dataparser paths")
    parser.add_argument("--transforms-json", default=None, help="Dataset transforms.json for camera intrinsics")
    parser.add_argument("--height-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--prompt-height", action="store_true")
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--render-rotate-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--render-rotate-degrees", type=float, default=180.0)
    parser.add_argument("--cleaned-splat-flip-lr", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--cleaned-splat-flip-ud", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--show-pose", action="store_true")
    parser.add_argument("--renderer", choices=("gsplat", "point"), default="gsplat")
    parser.add_argument("--interactive-rotation", action="store_true",
                        help="Pick camera once, then repeatedly type initial render rotation values")
    args = parser.parse_args(argv)

    positions, scales, colors, rotations = load_supersplat(args.splat)
    scale_to_metres = load_scale_to_metres(args.splat_config)
    intrinsics = load_intrinsics(args.transforms_json, args.splat_config)

    print("Loaded cleaned .splat:")
    print(f"  file: {args.splat}")
    print(f"  gaussians: {len(positions)}")
    print(f"  scale_to_metres: {scale_to_metres:.6f}")
    print(f"  intrinsics: {intrinsics['width']}x{intrinsics['height']} fx={intrinsics['fx']:.2f} fy={intrinsics['fy']:.2f}")

    camera_position, pointcloud = pick_camera_position(
        args.pointcloud,
        scale_to_metres,
        args.height_axis,
        args.prompt_height,
    )
    print(f"Camera position norm: {camera_position}")
    print(f"Camera position RAW m: {camera_position * scale_to_metres}")

    orientation, forward = render_and_show(
        args,
        positions,
        scales,
        colors,
        rotations,
        camera_position,
        intrinsics,
        scale_to_metres,
        float(args.render_rotate_degrees),
    )

    if args.show_pose:
        extent = np.linalg.norm(np.asarray(pointcloud.points).max(axis=0) - np.asarray(pointcloud.points).min(axis=0))
        show_pose(pointcloud, camera_position, forward, max(0.25, min(0.6, extent * 0.08)))

    if args.interactive_rotation:
        render_degrees = float(args.render_rotate_degrees)
        while True:
            value = input("Enter 'rot <deg>' (or bare deg), 'pose', or 'q': ").strip()
            if value.lower() in ("q", "quit", "exit"):
                break
            if not value:
                continue
            if value.lower() == "pose":
                extent = np.linalg.norm(
                    np.asarray(pointcloud.points).max(axis=0) - np.asarray(pointcloud.points).min(axis=0)
                )
                show_pose(pointcloud, camera_position, forward, max(0.25, min(0.6, extent * 0.08)))
                continue
            parts = value.split()
            try:
                if len(parts) == 2 and parts[0].lower() in ("rot", "rotate", "render"):
                    render_degrees = float(parts[1])
                else:
                    render_degrees = float(value)
            except ValueError:
                print("Could not parse input. Examples: rot 170, 170, pose, q")
                continue
            orientation, forward = render_and_show(
                args,
                positions,
                scales,
                colors,
                rotations,
                camera_position,
                intrinsics,
                scale_to_metres,
                render_degrees,
            )



if __name__ == "__main__":
    main()
