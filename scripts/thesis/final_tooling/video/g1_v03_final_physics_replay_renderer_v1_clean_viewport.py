#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any

parser = argparse.ArgumentParser(
    description=(
        "V03 final deterministic physics-replay renderer. Re-simulates the exact recorded 900-step action/command "
        "sequence through PhysX and captures one 2560x1440 viewport PNG after every post-physics control step. "
        "Wall-clock render speed is irrelevant; 900 frames are intended for 50 fps encoding = 18.000 s."
    )
)
parser.add_argument("--trajectory", type=Path, required=True)
parser.add_argument("--frames-dir", type=Path, required=True)
parser.add_argument("--manifest-path", type=Path, required=True)
parser.add_argument("--video-path", type=Path, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--video-width", type=int, default=2560)
parser.add_argument("--video-height", type=int, default=1440)
parser.add_argument("--capture-max-updates", type=int, default=60)
parser.add_argument("--initial-settle-renders", type=int, default=8)
parser.add_argument("--encode-mp4", action="store_true")
parser.add_argument("--ffmpeg", type=Path, default=None)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# FINAL VIDEO: restore the exact clean-viewport bootstrap used by the frozen v5.42 runner.
# This must happen BEFORE AppLauncher so Kit creates the native-resolution render target/window.
if args_cli.video_width <= 0 or args_cli.video_height <= 0:
    parser.error("--video-width and --video-height must be positive")
args_cli.width = int(args_cli.video_width)
args_cli.height = int(args_cli.video_height)
args_cli.window_width = int(args_cli.video_width)
args_cli.window_height = int(args_cli.video_height)
_clean_args = " ".join([
    "--/app/window/hideUi=true",
    "--/app/docks/disabled=true",
    "--/app/window/fullscreen=true",
    "--/app/window/noDecorations=true",
    "--/app/window/hideMouseInFullscreen=true",
    f"--/app/window/width={int(args_cli.video_width)}",
    f"--/app/window/height={int(args_cli.video_height)}",
    "--/app/viewport/grid/enabled=false",
    "--/metricsAssembler/showUnitsOverlay=false",
])
_existing_kit_args = str(getattr(args_cli, "kit_args", "") or "").strip()
args_cli.kit_args = (_existing_kit_args + " " + _clean_args).strip()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb.settings
import gymnasium as gym
import torch
import humanoid_rl_lab.tasks  # noqa: F401
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.upright_yaw_direct_wide_yaw_flat_orientation_ppo_cfg import (
    UprightYawDirectWideYawFlatOrientationPPORunnerCfg,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg

TASK_ID = "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-v0"
EXPECTED_TRAJECTORY_SHA256 = "3E9645259AB65F4AE6E1FD46254F17AD657EED64E56568A7C787B9120B09442D"
EXPECTED_FORMAT = "G1_DUAL_POLICY_VISUAL_TRAJECTORY_V3"
EXPECTED_FRAMES = 900
EXPECTED_DT = 0.02
EXPECTED_FPS = 50
EXPECTED_JOINTS = 37
EXPECTED_V2G_SHA = "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3"
EXPECTED_V4N_SHA = "367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342"

# Frozen final visual preset from v5.42.
SHOWCASE_SHIFT_X = -60.0
CAMERA_FORWARD_OFFSET = -3.10
CAMERA_LEFT_OFFSET = -2.90
CAMERA_EYE_Z = 1.58
CAMERA_TARGET_Z = 0.92
CAMERA_TARGET_LOOKAHEAD = 0.00
CAMERA_POSITION_SMOOTHING_S = 0.85
CAMERA_VELOCITY_SMOOTHING_S = 0.65
CAMERA_HEADING_MODE = "fixed"
PRESENTATION_LIGHTING = "softbox"
GROUND_LOOK = "demo"
SHOWCASE_SCENERY = "rough-dark"
RENDER_QUALITY = "video-native"
FPS_MASTER = 50

PASS_MARKER = "G1_V03_FINAL_PHYSICS_REPLAY_RENDER_V1_PASS"
FAIL_MARKER = "G1_V03_FINAL_PHYSICS_REPLAY_RENDER_V1_FAIL"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def fail(message: str) -> None:
    raise RuntimeError(message)


def as_tensor(value: Any) -> torch.Tensor:
    if hasattr(value, "torch"):
        value = value.torch
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value


def cpu_row(value: Any) -> torch.Tensor:
    return as_tensor(value)[0].detach().to(device="cpu").clone()


def write_command(base_env: Any, cmd: torch.Tensor) -> None:
    buf = base_env.command_manager.get_command("base_velocity")
    with torch.no_grad():
        buf[:, 0] = float(cmd[0].item())
        buf[:, 1] = float(cmd[1].item())
        buf[:, 2] = float(cmd[2].item())

def quat_wxyz_to_yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class FrozenCameraTracker:
    """v5.42 final fixed-heading camera, evaluated using trajectory dt rather than wall-clock time."""

    def __init__(self) -> None:
        self.center_xy = None
        self.prev_root_xy = None
        self.yaw = None
        self.filtered_vel_xy = [0.0, 0.0]

    def update(self, root_pos, root_quat, dt: float):
        root_x = float(root_pos[0])
        root_y = float(root_pos[1])
        root_yaw = quat_wxyz_to_yaw(root_quat)

        if self.center_xy is None:
            self.center_xy = [root_x, root_y]
            self.prev_root_xy = [root_x, root_y]
            self.yaw = root_yaw
            self.filtered_vel_xy = [0.0, 0.0]
        else:
            alpha_pos = 1.0 - math.exp(-dt / CAMERA_POSITION_SMOOTHING_S)
            self.center_xy[0] += alpha_pos * (root_x - self.center_xy[0])
            self.center_xy[1] += alpha_pos * (root_y - self.center_xy[1])

            raw_vx = (root_x - self.prev_root_xy[0]) / dt
            raw_vy = (root_y - self.prev_root_xy[1]) / dt
            self.prev_root_xy[0] = root_x
            self.prev_root_xy[1] = root_y

            alpha_vel = 1.0 - math.exp(-dt / CAMERA_VELOCITY_SMOOTHING_S)
            self.filtered_vel_xy[0] += alpha_vel * (raw_vx - self.filtered_vel_xy[0])
            self.filtered_vel_xy[1] += alpha_vel * (raw_vy - self.filtered_vel_xy[1])
            # Final v5.42 mode is fixed: yaw intentionally does not follow travel/pelvis after initialization.

        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        eye = (
            self.center_xy[0] + CAMERA_FORWARD_OFFSET * cy + CAMERA_LEFT_OFFSET * (-sy),
            self.center_xy[1] + CAMERA_FORWARD_OFFSET * sy + CAMERA_LEFT_OFFSET * cy,
            CAMERA_EYE_Z,
        )
        target = (
            self.center_xy[0] + CAMERA_TARGET_LOOKAHEAD * cy,
            self.center_xy[1] + CAMERA_TARGET_LOOKAHEAD * sy,
            CAMERA_TARGET_Z,
        )
        return eye, target


def resolve_approved_viewport_camera_optics():
    """Read the active Kit perspective-camera optics when a viewport exists.

    v5.42 moved the active viewport camera with sim.set_camera_view but did not override its lens.
    Reading that prim here preserves the user's actually approved viewport optics.
    """
    fallback = {
        "source": "fallback_kit_default",
        "focal_length": 18.147562,
        "horizontal_aperture": 20.955,
        "vertical_aperture": 20.955 * float(args_cli.video_height) / float(args_cli.video_width),
        "clipping_range": [1.0, 10000000.0],
    }
    try:
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport is None:
            return fallback
        stage = omni.usd.get_context().get_stage()
        camera_path = viewport.get_active_camera()
        prim = stage.GetPrimAtPath(camera_path)
        if not prim.IsValid():
            return fallback
        focal = prim.GetAttribute("focalLength").Get()
        ha = prim.GetAttribute("horizontalAperture").Get()
        va = prim.GetAttribute("verticalAperture").Get()
        clipping = prim.GetAttribute("clippingRange").Get()
        if focal is None or ha is None or va is None or clipping is None:
            return fallback
        return {
            "source": "active_v542_style_viewport_camera",
            "camera_path": str(camera_path),
            "focal_length": float(focal),
            "horizontal_aperture": float(ha),
            "vertical_aperture": float(va),
            "clipping_range": [float(clipping[0]), float(clipping[1])],
        }
    except Exception as exc:
        fallback["probe_error"] = repr(exc)
        return fallback



_SHOWCASE_TERRAIN_IMPORTERS = []


def configure_presentation_lighting(mode):
    # Visual-only USD lights under /World/VideoPresentation.
    # softbox deliberately avoids DomeLight and DistantLight so the real ground
    # is not globally lifted. No physics/collision/material/contact property is authored.
    if mode == "off":
        return {"mode": "off", "authored": False}

    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    stage = omni.usd.get_context().get_stage()
    root_path = Sdf.Path("/World/VideoPresentation")
    stage.DefinePrim(root_path, "Xform")

    if mode == "softbox":
        light_path = root_path.AppendChild("Softbox")
        if stage.GetPrimAtPath(light_path).IsValid():
            raise RuntimeError(f"Presentation light path already exists: {light_path}")

        light = UsdLux.RectLight.Define(stage, light_path)
        light.CreateIntensityAttr(3200.0)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.965, 0.92))
        light.CreateWidthAttr(5.0)
        light.CreateHeightAttr(3.0)

        xform = UsdGeom.Xformable(light.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(1.5, -1.8, 4.8))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(-28.0, 18.0, 8.0))

        return {
            "mode": mode,
            "authored": True,
            "root_prim": str(root_path),
            "light_prim": str(light_path),
            "type": "RectLight",
            "intensity": 3200.0,
            "width_m": 5.0,
            "height_m": 3.0,
            "dome_light": False,
            "distant_light": False,
            "physics_modified": False,
        }

    key_path = root_path.AppendChild("Key")
    fill_path = root_path.AppendChild("Fill")

    for p in (key_path, fill_path):
        existing = stage.GetPrimAtPath(p)
        if existing.IsValid():
            raise RuntimeError(f"Presentation light path already exists: {p}")

    key = UsdLux.DistantLight.Define(stage, key_path)
    fill = UsdLux.DomeLight.Define(stage, fill_path)

    if mode == "science":
        key_intensity = 650.0
        fill_intensity = 120.0
        key_color = Gf.Vec3f(1.0, 0.98, 0.95)
        fill_color = Gf.Vec3f(0.92, 0.95, 1.0)
        key_rotation = Gf.Vec3f(-52.0, 20.0, -28.0)
    else:
        key_intensity = 680.0
        fill_intensity = 18.0
        key_color = Gf.Vec3f(1.0, 0.97, 0.93)
        fill_color = Gf.Vec3f(0.62, 0.72, 0.88)
        key_rotation = Gf.Vec3f(-64.0, 18.0, -24.0)

    key.CreateIntensityAttr(key_intensity)
    key.CreateColorAttr(key_color)
    key.CreateAngleAttr(4.0 if mode == "demo" else 1.5)

    fill.CreateIntensityAttr(fill_intensity)
    fill.CreateColorAttr(fill_color)

    xform = UsdGeom.Xformable(key.GetPrim())
    xform.AddRotateXYZOp().Set(key_rotation)

    return {
        "mode": mode,
        "authored": True,
        "root_prim": str(root_path),
        "key_intensity": key_intensity,
        "fill_intensity": fill_intensity,
        "physics_modified": False,
    }

def configure_real_ground_look(mode):
    # Probe-derived stage structure:
    #   visual mesh    = /World/ground/terrain/Environment/Geometry
    #   original mat   = /World/ground/terrain/Looks/theGrid
    #   collision plane= /World/ground/terrain/GroundPlane/CollisionPlane
    #
    # This function changes ONLY material binding on the visual mesh.
    # It does not edit points/transforms, collision schemas, friction, rigid bodies, or physics material.
    if mode == "original":
        return {"mode": "original", "authored": False, "physics_modified": False}

    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    visual_path = Sdf.Path("/World/ground/terrain/Environment/Geometry")
    collision_path = Sdf.Path("/World/ground/terrain/GroundPlane/CollisionPlane")

    visual_prim = stage.GetPrimAtPath(visual_path)
    collision_prim = stage.GetPrimAtPath(collision_path)

    if not visual_prim.IsValid() or not visual_prim.IsA(UsdGeom.Gprim):
        raise RuntimeError(f"Expected visual ground Gprim missing: {visual_path}")
    if not collision_prim.IsValid():
        raise RuntimeError(f"Expected physical collision ground missing: {collision_path}")

    visual_schemas_before = [str(x) for x in visual_prim.GetAppliedSchemas()]
    collision_schemas_before = [str(x) for x in collision_prim.GetAppliedSchemas()]

    if any("CollisionAPI" in s or "RigidBodyAPI" in s or "MassAPI" in s for s in visual_schemas_before):
        raise RuntimeError(
            f"Refusing ground-look edit: visual ground unexpectedly has physics schemas: {visual_schemas_before}"
        )
    if not any("CollisionAPI" in s for s in collision_schemas_before):
        raise RuntimeError(
            f"Refusing ground-look edit: expected collision plane lacks CollisionAPI: {collision_schemas_before}"
        )

    original_mat = None
    try:
        mat, _ = UsdShade.MaterialBindingAPI(visual_prim).ComputeBoundMaterial()
        if mat and mat.GetPrim().IsValid():
            original_mat = str(mat.GetPath())
    except Exception:
        original_mat = None

    root_path = Sdf.Path("/World/VideoPresentation")
    stage.DefinePrim(root_path, "Xform")
    material_path = root_path.AppendChild("RealGroundMaterial")
    shader_path = material_path.AppendChild("PreviewSurface")

    if stage.GetPrimAtPath(material_path).IsValid():
        raise RuntimeError(f"Real-ground presentation material already exists: {material_path}")

    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")

    if mode == "science":
        diffuse = Gf.Vec3f(0.16, 0.17, 0.19)
        roughness = 0.82
        metallic = 0.0
    else:
        # Technical-studio ground: dark enough to separate the white G1,
        # but not black, so contact shadows and foot placement remain readable.
        # Near-black graphite-blue demo floor. The previous low RGB values were still
        # lifted strongly by the global fill/tonemapping, so this deliberately goes darker.
        diffuse = Gf.Vec3f(0.006, 0.009, 0.014)
        roughness = 0.92
        metallic = 0.0

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diffuse)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    UsdShade.MaterialBindingAPI.Apply(visual_prim).Bind(material)

    # Post-edit safety verification: only the material binding may have changed.
    visual_schemas_after = [str(x) for x in visual_prim.GetAppliedSchemas()]
    collision_schemas_after = [str(x) for x in collision_prim.GetAppliedSchemas()]

    if visual_schemas_after != visual_schemas_before:
        raise RuntimeError(
            f"Visual ground applied schemas changed unexpectedly: before={visual_schemas_before}, "
            f"after={visual_schemas_after}"
        )
    if collision_schemas_after != collision_schemas_before:
        raise RuntimeError(
            f"Collision ground schemas changed unexpectedly: before={collision_schemas_before}, "
            f"after={collision_schemas_after}"
        )

    rebound_mat, _ = UsdShade.MaterialBindingAPI(visual_prim).ComputeBoundMaterial()
    rebound_path = (
        str(rebound_mat.GetPath())
        if rebound_mat and rebound_mat.GetPrim().IsValid()
        else None
    )
    if rebound_path != str(material_path):
        raise RuntimeError(
            f"Real-ground material binding failed: expected {material_path}, got {rebound_path}"
        )

    return {
        "mode": mode,
        "authored": True,
        "visual_ground_prim": str(visual_path),
        "collision_ground_prim": str(collision_path),
        "original_visual_material": original_mat,
        "new_visual_material": rebound_path,
        "visual_schemas_unchanged": True,
        "collision_schemas_unchanged": True,
        "geometry_modified": False,
        "physics_modified": False,
        "collision_modified": False,
        "friction_modified": False,
    }

def configure_showcase_scenery(mode, shift_x_m=0.0):
    if mode == "off":
        return {
            "mode": "off",
            "authored": False,
            "physics_modified": False,
            "main_ground_modified": False,
        }

    import omni.usd
    import isaaclab.sim as sim_utils
    from isaaclab.terrains import TerrainGenerator, TerrainImporter, TerrainImporterCfg
    from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    showcase_root = Sdf.Path("/World/VideoPresentation/IsaacLabShowcase")
    if stage.GetPrimAtPath(showcase_root).IsValid():
        raise RuntimeError(f"Showcase scenery already exists: {showcase_root}")
    showcase_root_prim = stage.DefinePrim(showcase_root, "Xform")
    showcase_shift_x = float(shift_x_m)
    UsdGeom.Xformable(showcase_root_prim).AddTranslateOp().Set(
        Gf.Vec3d(showcase_shift_x, 0.0, 0.0)
    )

    from pxr import UsdLux, UsdShade

    # ------------------------------------------------------------------
    # Visual/physics separation for the flat lane.
    # The task's physical flat collision plane remains authoritative and unchanged.
    # Only the old infinite VISUAL mesh is hidden, avoiding coplanar overlap/z-fighting
    # with the generated rough scenery outside the corridor.
    # ------------------------------------------------------------------
    ground_visual_path = Sdf.Path("/World/ground/terrain/Environment/Geometry")
    ground_collision_path = Sdf.Path("/World/ground/terrain/GroundPlane/CollisionPlane")

    ground_visual_prim = stage.GetPrimAtPath(ground_visual_path)
    ground_collision_prim = stage.GetPrimAtPath(ground_collision_path)

    if not ground_visual_prim.IsValid():
        raise RuntimeError(f"Main ground visual mesh missing: {ground_visual_path}")
    if not ground_collision_prim.IsValid():
        raise RuntimeError(f"Main ground collision prim missing: {ground_collision_path}")
    if not ground_collision_prim.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(
            f"Main ground collision prim lost CollisionAPI before showcase setup: {ground_collision_path}"
        )

    ground_collision_api = UsdPhysics.CollisionAPI(ground_collision_prim)
    collision_enabled_before = ground_collision_api.GetCollisionEnabledAttr().Get()
    if collision_enabled_before is False:
        raise RuntimeError("Main flat collision plane is unexpectedly disabled before showcase setup.")

    UsdGeom.Imageable(ground_visual_prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    # Visual-only flat corridor: 10 m wide, 130 m long.
    corridor_path = showcase_root.AppendChild("FlatCorridorVisual")
    corridor_mat_path = showcase_root.AppendChild("FlatCorridorMaterial")
    corridor_shader_path = corridor_mat_path.AppendChild("PreviewSurface")

    corridor = UsdGeom.Mesh.Define(stage, corridor_path)
    corridor.CreatePointsAttr([
        Gf.Vec3f(-30.0, -5.0, 0.003),
        Gf.Vec3f(116.0, -5.0, 0.003),
        Gf.Vec3f(116.0,  5.0, 0.003),
        Gf.Vec3f(-30.0,  5.0, 0.003),
    ])
    corridor.CreateFaceVertexCountsAttr([4])
    corridor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    corridor.CreateDoubleSidedAttr(True)
    corridor.CreateSubdivisionSchemeAttr("none")

    corridor_mat = UsdShade.Material.Define(stage, corridor_mat_path)
    corridor_shader = UsdShade.Shader.Define(stage, corridor_shader_path)
    corridor_shader.CreateIdAttr("UsdPreviewSurface")
    corridor_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.006, 0.009, 0.014)
    )
    corridor_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.92)
    corridor_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    corridor_mat.CreateSurfaceOutput().ConnectToSource(
        corridor_shader.ConnectableAPI(), "surface"
    )
    UsdShade.MaterialBindingAPI.Apply(corridor.GetPrim()).Bind(corridor_mat)

    # Corridor is intentionally visual-only.
    corridor_schemas = [str(x) for x in corridor.GetPrim().GetAppliedSchemas()]
    if any(
        "CollisionAPI" in schema or "RigidBodyAPI" in schema or "MassAPI" in schema
        for schema in corridor_schemas
    ):
        raise RuntimeError(f"Visual corridor unexpectedly has physics schemas: {corridor_schemas}")

    collision_enabled_after = ground_collision_api.GetCollisionEnabledAttr().Get()
    if collision_enabled_after is False:
        raise RuntimeError("Main flat collision plane was disabled during visual corridor setup.")

    from pxr import UsdLux

    ambient_path = showcase_root.AppendChild("AmbientDome")
    ambient = UsdLux.DomeLight.Define(stage, ambient_path)
    ambient.CreateIntensityAttr(300.0)
    ambient.CreateColorAttr(Gf.Vec3f(0.75, 0.78, 0.82))

    # Keep the original G1 flat ground untouched; only scenery is added around it.
    # Every rough group is offset from the robot; it is also made collision-disabled.
    import numpy as np
    import trimesh

    class _ShowcaseSwapTerrainGenerator(TerrainGenerator):
        # Swap exactly two deterministic cells in NearSideBand (seed 43).

        def __init__(self, cfg, device="cpu"):
            super().__init__(cfg=cfg, device=device)

            if int(cfg.seed) != 43:
                self._showcase_cell_swap = None
                return

            blue = (8, 3)
            red = (9, 3)
            green = (10, 3)

            for row, col in (blue, red, green):
                if not (0 <= row < int(cfg.num_rows) and 0 <= col < int(cfg.num_cols)):
                    raise RuntimeError(
                        f"Showcase swap cell outside grid: row={row}, col={col}, "
                        f"grid={cfg.num_rows}x{cfg.num_cols}"
                    )

            blue_idx = blue[0] * int(cfg.num_cols) + blue[1]
            green_idx = green[0] * int(cfg.num_cols) + green[1]

            blue_mesh = self.terrain_meshes[blue_idx].copy()
            green_mesh = self.terrain_meshes[green_idx].copy()

            # The cells differ by two 8 m rows = 16 m along local X.
            delta_x = float(green[0] - blue[0]) * float(cfg.size[0])

            move_blue_to_green = np.eye(4)
            move_blue_to_green[0, -1] = +delta_x
            blue_mesh.apply_transform(move_blue_to_green)

            move_green_to_blue = np.eye(4)
            move_green_to_blue[0, -1] = -delta_x
            green_mesh.apply_transform(move_green_to_blue)

            # Swap only BLUE and GREEN. RED (row 9, col 3) is untouched.
            self.terrain_meshes[blue_idx] = green_mesh
            self.terrain_meshes[green_idx] = blue_mesh

            # Additional visual correction from the annotated free-camera view:
            # duplicate the desired GREEN pyramid into the cell directly behind RED.
            # GREEN=(10,3), RED=(9,3), behind-RED=(9,2).
            behind_red = (9, 2)
            behind_red_idx = behind_red[0] * int(cfg.num_cols) + behind_red[1]

            green_pyramid_copy = self.terrain_meshes[green_idx].copy()
            copy_to_behind_red = np.eye(4)
            copy_to_behind_red[0, -1] = (
                float(behind_red[0] - green[0]) * float(cfg.size[0])
            )
            copy_to_behind_red[1, -1] = (
                float(behind_red[1] - green[1]) * float(cfg.size[1])
            )
            green_pyramid_copy.apply_transform(copy_to_behind_red)
            self.terrain_meshes[behind_red_idx] = green_pyramid_copy

            # Final orientation correction from the user's visual checks:
            # TRUE RIGHT of row9/col2 in the rendered view is row8/col2.
            # The previous wrong guesses row9/col1 and row10/col2 are left untouched.
            true_right_of_v539 = (8, 2)
            true_right_of_v539_idx = (
                true_right_of_v539[0] * int(cfg.num_cols) + true_right_of_v539[1]
            )

            true_right_pyramid_copy = self.terrain_meshes[behind_red_idx].copy()
            move_true_right = np.eye(4)
            move_true_right[0, -1] = (
                float(true_right_of_v539[0] - behind_red[0]) * float(cfg.size[0])
            )
            move_true_right[1, -1] = (
                float(true_right_of_v539[1] - behind_red[1]) * float(cfg.size[1])
            )
            true_right_pyramid_copy.apply_transform(move_true_right)
            self.terrain_meshes[true_right_of_v539_idx] = true_right_pyramid_copy

            # Keep destination XY origins; exchange only terrain-origin Z values.
            blue_z = float(self.terrain_origins[blue[0], blue[1], 2])
            green_z = float(self.terrain_origins[green[0], green[1], 2])
            self.terrain_origins[blue[0], blue[1], 2] = green_z
            self.terrain_origins[green[0], green[1], 2] = blue_z
            self.terrain_origins[behind_red[0], behind_red[1], 2] = self.terrain_origins[
                green[0], green[1], 2
            ]
            self.terrain_origins[
                true_right_of_v539[0], true_right_of_v539[1], 2
            ] = self.terrain_origins[behind_red[0], behind_red[1], 2]

            # Rebuild the combined mesh and apply the same centering transform used by
            # the official TerrainGenerator.
            self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)
            center_transform = np.eye(4)
            center_transform[:2, -1] = (
                -float(cfg.size[0]) * int(cfg.num_rows) * 0.5,
                -float(cfg.size[1]) * int(cfg.num_cols) * 0.5,
            )
            self.terrain_mesh.apply_transform(center_transform)

            self._showcase_cell_swap = {
                "band": "NearSideBand",
                "seed": 43,
                "blue_cell": [8, 3],
                "red_cell_restored_untouched": [9, 3],
                "green_cell": [10, 3],
                "behind_red_cell": [9, 2],
                "true_right_of_v539_cell": [8, 2],
                "wrong_v540_cell_preserved_original": [9, 1],
                "wrong_v541_cell_preserved_original": [10, 2],
                "operation": (
                    "preserve v5.39 exactly; additionally copy its pyramid from row9/col2 "
                    "into the true right-hand cell row8/col2"
                ),
                "local_row_delta_m": float(delta_x),
            }

    groups = [
        # label, translation xyz, rows, cols, deterministic seed
        #
        # TWO PARALLEL ROUGH BANDS ONLY.
        # 14x3 official 8 m cells -> ~112 m along X and ~24 m across Y per side.
        # Approximate x-range: -12..100 m.
        # Approximate y-ranges: +5..+29 m and -29..-5 m.
        # This keeps the 10 m central corridor completely flat while the rough scenery
        # continues far behind the fixed-camera background on BOTH sides.
        ("FarSideBand",  (52.0,  21.0, 0.000), 16, 4, 42),
        ("NearSideBand", (52.0, -21.0, 0.000), 16, 4, 43),
    ]

    terrain_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.006, 0.009, 0.014),
        roughness=0.92,
        metallic=0.0,
    )

    reports = []
    global _SHOWCASE_TERRAIN_IMPORTERS
    _SHOWCASE_TERRAIN_IMPORTERS = []

    for label, translation, rows, cols, seed in groups:
        parent_path = showcase_root.AppendChild(f"{label}Root")
        terrain_path = parent_path.AppendChild("Terrain")

        parent = stage.DefinePrim(parent_path, "Xform")
        UsdGeom.Xformable(parent).AddTranslateOp().Set(Gf.Vec3d(*translation))

        import copy

        terrain_gen_cfg = ROUGH_TERRAINS_CFG.replace(
            seed=int(seed),
            curriculum=False,
            num_rows=int(rows),
            num_cols=int(cols),
            border_width=0.25,
            color_scheme="none",
            use_cache=False,
        )

        # Full official Isaac Lab rough-terrain family mix.
        # Keep fixed seeds: visually random, scientifically reproducible.
        _stairs = copy.deepcopy(ROUGH_TERRAINS_CFG.sub_terrains["pyramid_stairs"])
        _stairs_inv = copy.deepcopy(ROUGH_TERRAINS_CFG.sub_terrains["pyramid_stairs_inv"])
        _boxes = copy.deepcopy(ROUGH_TERRAINS_CFG.sub_terrains["boxes"])
        _random_rough = copy.deepcopy(ROUGH_TERRAINS_CFG.sub_terrains["random_rough"])
        _slope = copy.deepcopy(ROUGH_TERRAINS_CFG.sub_terrains["hf_pyramid_slope"])
        _slope_inv = copy.deepcopy(ROUGH_TERRAINS_CFG.sub_terrains["hf_pyramid_slope_inv"])

        _stairs.proportion = 0.20
        _stairs_inv.proportion = 0.20
        _boxes.proportion = 0.20
        _random_rough.proportion = 0.20
        _slope.proportion = 0.10
        _slope_inv.proportion = 0.10

        terrain_gen_cfg.sub_terrains = {
            "pyramid_stairs": _stairs,
            "pyramid_stairs_inv": _stairs_inv,
            "boxes": _boxes,
            "random_rough": _random_rough,
            "hf_pyramid_slope": _slope,
            "hf_pyramid_slope_inv": _slope_inv,
        }
        terrain_gen_cfg.class_type = _ShowcaseSwapTerrainGenerator

        importer_cfg = TerrainImporterCfg(
            num_envs=1,
            env_spacing=3.0,
            prim_path=str(terrain_path),
            terrain_type="generator",
            terrain_generator=terrain_gen_cfg,
            visual_material=terrain_material,
            debug_vis=False,
        )
        importer = TerrainImporter(importer_cfg)
        _SHOWCASE_TERRAIN_IMPORTERS.append(importer)

        terrain_prim = stage.GetPrimAtPath(terrain_path)
        if not terrain_prim.IsValid():
            raise RuntimeError(f"Generated showcase terrain root missing: {terrain_path}")

        collision_prims = 0
        collision_disabled = 0
        for prim in Usd.PrimRange(terrain_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_prims += 1
                api = UsdPhysics.CollisionAPI(prim)
                api.CreateCollisionEnabledAttr(False)
                enabled = api.GetCollisionEnabledAttr().Get()
                if enabled is False:
                    collision_disabled += 1

        if collision_prims > 0 and collision_disabled != collision_prims:
            raise RuntimeError(
                f"Could not disable every showcase collision for {label}: "
                f"{collision_disabled}/{collision_prims}"
            )

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        world_range = bbox_cache.ComputeWorldBound(terrain_prim).ComputeAlignedRange()
        world_min = world_range.GetMin()
        world_max = world_range.GetMax()

        expected_half_x = 0.5 * rows * 8.0
        expected_half_y = 0.5 * cols * 8.0

        reports.append({
            "label": label,
            "parent_prim": str(parent_path),
            "terrain_prim": str(terrain_path),
            "translation_m": list(translation),
            "rows": int(rows),
            "cols": int(cols),
            "subterrain_size_m": [8.0, 8.0],
            "seed": int(seed),
            "expected_xy_bounds_m": {
                "x": [float(translation[0] - expected_half_x), float(translation[0] + expected_half_x)],
                "y": [float(translation[1] - expected_half_y), float(translation[1] + expected_half_y)],
            },
            "runtime_world_aabb_min": [float(world_min[0]), float(world_min[1]), float(world_min[2])],
            "runtime_world_aabb_max": [float(world_max[0]), float(world_max[1]), float(world_max[2])],
            "collision_prims_found": int(collision_prims),
            "collision_prims_disabled": int(collision_disabled),
        })

    return {
        "mode": mode,
        "authored": True,
        "whole_arena_shift_x_m": float(showcase_shift_x),
        "translation_scope": "visual showcase root only; robot and physical flat collision plane unchanged",
        "source_config": "isaaclab.terrains.config.rough.ROUGH_TERRAINS_CFG",
        "source_subterrains": [
            "MeshPyramidStairsTerrainCfg",
            "MeshInvertedPyramidStairsTerrainCfg",
            "MeshRandomGridTerrainCfg",
            "HfRandomUniformTerrainCfg",
            "HfPyramidSlopedTerrainCfg",
            "HfInvertedPyramidSlopedTerrainCfg",
        ],
        "targeted_cell_swap": {
            "band": "NearSideBand",
            "seed": 43,
            "blue_cell": [8, 3],
            "red_cell_restored_untouched": [9, 3],
            "green_cell": [10, 3],
            "behind_red_cell": [9, 2],
            "true_right_of_v539_cell": [8, 2],
            "wrong_v540_cell_preserved_original": [9, 1],
            "wrong_v541_cell_preserved_original": [10, 2],
            "operation": (
                "preserve v5.39; copy the same pyramid into the true right-hand "
                "neighbor row8/col2"
            ),
            "reason": "user-annotated free-camera identification",
        },
        "showcase_subterrain_mix": {
            "pyramid_stairs": 0.20,
            "pyramid_stairs_inv": 0.20,
            "boxes": 0.20,
            "random_rough": 0.20,
            "hf_pyramid_slope": 0.10,
            "hf_pyramid_slope_inv": 0.10,
            "selection": "seeded random; deterministic per band",
        },
        "safe_flat_lane_note": (
            "The original infinite flat COLLISION plane remains enabled and authoritative. "
            "Its infinite visual mesh is hidden and replaced by one continuous 10 m wide visual-only corridor. "
            "The two parallel rough bands remain scenery-only and collision-disabled."
        ),
        "layout_note": (
            "The arena is intentionally only two parallel 16x4 rough bands flanking one continuous "
            "10 m flat visual corridor. There is no transverse front cap. The rough bands run from "
            "approximately x=-12 to x=116 m, adding exactly two far-end rows beyond v5.30, "
            "while their reduced 24 m lateral depth removes the oversized outer spill."
        ),
        "flat_corridor_visual": {
            "prim": str(corridor_path),
            "width_m": 10.0,
            "x_range_m": [-30.0, 116.0],
            "z_m": 0.003,
            "physics_schemas": corridor_schemas,
        },
        "main_ground_visual": {
            "prim": str(ground_visual_path),
            "visibility_runtime": str(
                UsdGeom.Imageable(ground_visual_prim).GetVisibilityAttr().Get()
            ),
        },
        "main_ground_collision": {
            "prim": str(ground_collision_path),
            "collision_api_present": True,
            "collision_enabled_before": bool(collision_enabled_before),
            "collision_enabled_after": bool(collision_enabled_after),
        },
        "scenery_material": {
            "diffuse_color": [0.006, 0.009, 0.014],
            "roughness": 0.92,
            "metallic": 0.0,
        },
        "ambient_dome": {
            "prim": str(ambient_path),
            "intensity": 300.0,
            "color": [0.75, 0.78, 0.82],
            "reference_demo_intensity": 2000.0,
        },
        "groups": reports,
        "physics_modified": False,
        "main_ground_modified": False,
        "policy_modified": False,
    }





def quat_sign_invariant_max_abs(q: torch.Tensor, ref: torch.Tensor) -> float:
    return min(
        float(torch.max(torch.abs(q - ref)).item()),
        float(torch.max(torch.abs(q + ref)).item()),
    )


def find_ffmpeg() -> str | None:
    if args_cli.ffmpeg is not None:
        p = Path(args_cli.ffmpeg).resolve()
        if not p.is_file():
            fail(f"explicit ffmpeg missing: {p}")
        return str(p)
    return shutil.which("ffmpeg")


def main() -> None:
    trajectory_path = Path(args_cli.trajectory).resolve()
    frames_dir = Path(args_cli.frames_dir).resolve()
    manifest_path = Path(args_cli.manifest_path).resolve()
    video_path = (
        Path(args_cli.video_path).resolve()
        if args_cli.video_path is not None
        else manifest_path.parent.parent / "video" / "V03_curve_neg_50fps.mp4"
    )
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    if not trajectory_path.is_file():
        fail(f"trajectory missing: {trajectory_path}")
    trajectory_sha = sha256_file(trajectory_path)
    if trajectory_sha != EXPECTED_TRAJECTORY_SHA256:
        fail(f"trajectory SHA mismatch: {trajectory_sha}")

    trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=False)
    if trajectory.get("format") != EXPECTED_FORMAT:
        fail(f"unexpected trajectory format: {trajectory.get('format')}")
    if int(trajectory.get("num_frames", -1)) != EXPECTED_FRAMES:
        fail("trajectory frame-count mismatch")
    if not math.isclose(float(trajectory.get("dt_s", -1.0)), EXPECTED_DT, rel_tol=0.0, abs_tol=1e-12):
        fail("trajectory dt mismatch")
    controller_meta = trajectory.get("controller", {})
    if str(controller_meta.get("v2g_sha256", "")).upper() != EXPECTED_V2G_SHA:
        fail("trajectory v2-G provenance mismatch")
    if str(controller_meta.get("v4n_sha256", "")).upper() != EXPECTED_V4N_SHA:
        fail("trajectory v4-N provenance mismatch")

    joint_names_saved = list(trajectory["joint_names"])
    if len(joint_names_saved) != EXPECTED_JOINTS:
        fail("trajectory joint count mismatch")
    actions_all = trajectory["actions"].float().contiguous()
    policy_cmd_all = trajectory["policy_cmd"].float().contiguous()
    root_pos_ref = trajectory["root_pos_w"].float().contiguous()
    root_quat_ref = trajectory["root_quat_w_wxyz"].float().contiguous()
    joint_pos_ref = trajectory["joint_pos"].float().contiguous()
    router_state_all = trajectory["router_state_code"]
    phase_all = list(trajectory["phase"])
    if tuple(actions_all.shape) != (EXPECTED_FRAMES, EXPECTED_JOINTS):
        fail(f"actions shape mismatch: {tuple(actions_all.shape)}")
    if tuple(policy_cmd_all.shape) != (EXPECTED_FRAMES, 3):
        fail(f"policy_cmd shape mismatch: {tuple(policy_cmd_all.shape)}")

    # V03 semantic guard: exact 3 s ZERO + 12 s engineering curve-negative + 3 s ZERO schedule.
    expected_phases = (["ZERO_BEFORE"] * 150) + (["CURVE_NEG"] * 600) + (["ZERO_AFTER"] * 150)
    if phase_all != expected_phases:
        fail("V03 phase schedule mismatch")
    requested_cmd_all = trajectory.get("requested_cmd")
    if requested_cmd_all is None:
        fail("trajectory missing requested_cmd for V03 semantic guard")
    requested_cmd_all = requested_cmd_all.float().contiguous()
    if tuple(requested_cmd_all.shape) != (EXPECTED_FRAMES, 3):
        fail(f"requested_cmd shape mismatch: {tuple(requested_cmd_all.shape)}")
    zero_cmd = torch.tensor([0.0, 0.0, 0.0], dtype=requested_cmd_all.dtype)
    curve_neg_cmd = torch.tensor([0.40, 0.0, -0.25], dtype=requested_cmd_all.dtype)
    if not torch.allclose(requested_cmd_all[:150], zero_cmd.expand(150, -1), atol=1e-7, rtol=0.0):
        fail("V03 ZERO_BEFORE requested command mismatch")
    if not torch.allclose(requested_cmd_all[150:750], curve_neg_cmd.expand(600, -1), atol=1e-7, rtol=0.0):
        fail("V03 CURVE_NEG requested command mismatch")
    if not torch.allclose(requested_cmd_all[750:], zero_cmd.expand(150, -1), atol=1e-7, rtol=0.0):
        fail("V03 ZERO_AFTER requested command mismatch")

    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args_cli.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "base_external_force_torque"):
        env_cfg.events.base_external_force_torque = None
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    reset_base = getattr(env_cfg.events, "reset_base", None)
    if reset_base is not None:
        for key in list(reset_base.params.get("pose_range", {}).keys()):
            reset_base.params["pose_range"][key] = (0.0, 0.0)
        for key in list(reset_base.params.get("velocity_range", {}).keys()):
            reset_base.params["velocity_range"][key] = (0.0, 0.0)

    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.ranges.lin_vel_x = (0.0, 0.0)
    cmd_cfg.ranges.lin_vel_y = (0.0, 0.0)
    cmd_cfg.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(cmd_cfg, "heading_command"):
        cmd_cfg.heading_command = False
    if hasattr(cmd_cfg, "debug_vis"):
        cmd_cfg.debug_vis = False

    # Critical final-render contract: physics steps must not trigger their own Kit renders.
    # Every PNG is rendered exactly once after the corresponding control step below.
    env_cfg.sim.render_interval = 2**31 - 1

    installed_rsl = metadata.version("rsl-rl-lib")
    replay_agent_cfg = UprightYawDirectWideYawFlatOrientationPPORunnerCfg()
    if args_cli.device is not None:
        replay_agent_cfg.device = args_cli.device
    replay_agent_cfg = handle_deprecated_rsl_rl_cfg(replay_agent_cfg, installed_rsl)

    env = None
    rendered = []
    try:
        print("[RENDER] Creating deterministic physics-replay environment...", flush=True)
        raw_env = gym.make(TASK_ID, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=replay_agent_cfg.clip_actions)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        if list(robot.joint_names) != joint_names_saved:
            fail("joint-name/order mismatch")
        if not math.isclose(float(base_env.step_dt), EXPECTED_DT, rel_tol=0.0, abs_tol=1e-12):
            fail(f"unexpected replay dt={base_env.step_dt}")

        env_ids = torch.tensor([0], dtype=torch.int32, device=base_env.device)
        base_env.command_manager.reset(env_ids=env_ids)
        base_env.command_manager.get_command("base_velocity").zero_()

        reward_manager = base_env.reward_manager
        reward_buf = getattr(reward_manager, "_reward_buf", None)
        if reward_buf is None or not torch.is_tensor(reward_buf):
            fail("reward bypass unavailable")
        def reward_bypass(*_args, **_kwargs):
            reward_buf.zero_()
            return reward_buf
        reward_manager.compute = reward_bypass

        print("[RENDER] Applying frozen v5.42 softbox lighting...", flush=True)
        lighting_report = configure_presentation_lighting(PRESENTATION_LIGHTING)
        print("[RENDER] Applying frozen v5.42 ground material...", flush=True)
        ground_report = configure_real_ground_look(GROUND_LOOK)
        print("[RENDER] Building frozen v5.42 rough-dark showcase scenery...", flush=True)
        scenery_report = configure_showcase_scenery(SHOWCASE_SCENERY, shift_x_m=SHOWCASE_SHIFT_X)

        import omni.usd
        from pxr import UsdGeom
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        stage = omni.usd.get_context().get_stage()
        cmd_vis = stage.GetPrimAtPath("/Visuals/Command")
        if cmd_vis.IsValid() and cmd_vis.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(cmd_vis).MakeInvisible()

        render_settings = carb.settings.get_settings()
        render_settings.set("/rtx-transient/dlssg/enabled", False)
        render_settings.set("/rtx/post/motionblur/enabled", False)
        render_settings.set("/rtx/post/aa/op", 4)

        viewport = get_active_viewport()
        if viewport is None:
            fail("active Kit viewport unavailable; run with --viz kit")
        viewport.set_texture_resolution((int(args_cli.video_width), int(args_cli.video_height)))
        viewport.set_texture_resolution_scale(1.0)
        for _ in range(max(1, int(args_cli.initial_settle_renders))):
            base_env.sim.render()
        actual_resolution = tuple(int(round(v)) for v in viewport.get_texture_resolution())
        viewport_scale = float(viewport.get_texture_resolution_scale())
        if actual_resolution != (int(args_cli.video_width), int(args_cli.video_height)):
            fail(f"strict final viewport resolution mismatch: {actual_resolution}")
        if abs(viewport_scale - 1.0) > 1.0e-6:
            fail(f"viewport scale mismatch: {viewport_scale}")

        camera_tracker = FrozenCameraTracker()
        max_root_pos_err = 0.0
        max_root_quat_err = 0.0
        max_joint_pos_err = 0.0
        divergence_frame = None

        print("=" * 124)
        print("G1 V03 FINAL DETERMINISTIC PHYSICS-REPLAY RENDER")
        print("=" * 124)
        print(f"Master trajectory              : {trajectory_path}")
        print(f"Master SHA256                  : {trajectory_sha}")
        print("Physics replay                 : recorded command + recorded action; one real PhysX control step per master frame")
        print("Frame semantics                : post-physics state after each 0.02 s control step")
        print("Wall-clock pacing              : NONE")
        print("Final timing                   : 900 PNG / 50 fps = 18.000 s")
        print(f"Resolution                     : {actual_resolution[0]}x{actual_resolution[1]} / scale {viewport_scale:.3f}")
        print("Render                         : DLAA / motion blur OFF / frame generation OFF")
        print("Camera                         : v5.42 smoothed / fixed heading / -3.10,-2.90 / z 1.58->0.92")
        print("Command debug visualizer       : DISABLED")
        print(f"Frames directory               : {frames_dir}")
        print("=" * 124, flush=True)

        # Remove stale output frames so the directory cannot mix runs.
        for old in frames_dir.glob("frame_*.png"):
            old.unlink()

        for idx in range(EXPECTED_FRAMES):
            write_command(base_env, policy_cmd_all[idx])
            actions = actions_all[idx].to(base_env.device).unsqueeze(0)
            with torch.inference_mode():
                _, _, dones, _ = env.step(actions)
            if bool(torch.any(dones).item()):
                fail(f"environment termination/reset during replay at frame={idx}")

            # Compare the actual re-simulated post-physics state against the frozen master before rendering.
            root_pos_now = cpu_row(robot.data.root_pos_w)
            root_quat_now = cpu_row(robot.data.root_quat_w)
            joint_pos_now = cpu_row(robot.data.joint_pos)
            root_pos_err = float(torch.max(torch.abs(root_pos_now - root_pos_ref[idx])).item())
            root_quat_err = quat_sign_invariant_max_abs(root_quat_now, root_quat_ref[idx])
            joint_pos_err = float(torch.max(torch.abs(joint_pos_now - joint_pos_ref[idx])).item())
            max_root_pos_err = max(max_root_pos_err, root_pos_err)
            max_root_quat_err = max(max_root_quat_err, root_quat_err)
            max_joint_pos_err = max(max_joint_pos_err, joint_pos_err)
            if divergence_frame is None and (root_pos_err > 5e-4 or root_quat_err > 5e-4 or joint_pos_err > 5e-4):
                divergence_frame = idx
                fail(
                    f"physics replay diverged from master at frame={idx}: "
                    f"root_pos={root_pos_err:.3e}, root_quat={root_quat_err:.3e}, joint_pos={joint_pos_err:.3e}"
                )

            eye, target = camera_tracker.update(root_pos_now.tolist(), root_quat_now.tolist(), EXPECTED_DT)
            base_env.sim.set_camera_view(eye, target)
            # Physics has just stepped, so Fabric/Hydra receives a genuine updated articulation state.
            base_env.sim.forward()
            base_env.sim.render()

            png_path = frames_dir / f"frame_{idx:06d}.png"
            if png_path.exists():
                png_path.unlink()
            capture_viewport_to_file(viewport, file_path=str(png_path), is_hdr=False)
            completed_updates = None
            for update_index in range(1, int(args_cli.capture_max_updates) + 1):
                base_env.sim.render()
                if png_path.is_file() and png_path.stat().st_size > 1024:
                    completed_updates = update_index
                    break
            if completed_updates is None:
                fail(f"PNG capture timeout at frame={idx}")

            rendered.append({
                "frame": idx,
                "sim_time_s": float(trajectory["sim_time_s"][idx].item()),
                "phase": phase_all[idx],
                "router_state_code": int(router_state_all[idx].item()),
                "png": str(png_path),
                "png_bytes": int(png_path.stat().st_size),
                "root_pos_err": root_pos_err,
                "root_quat_err": root_quat_err,
                "joint_pos_err": joint_pos_err,
                "eye": [float(v) for v in eye],
                "target": [float(v) for v in target],
                "capture_updates": int(completed_updates),
            })

            if idx in (0, 149, 150, 299, 449, 599, 749, 750, 767, 768, 899) or idx % 100 == 0:
                state_name = {0: "STAND", 1: "LOCO", 2: "BRAKE"}[int(router_state_all[idx].item())]
                print(
                    f"[RENDER] frame={idx:03d}/899 t={(idx+1)*EXPECTED_DT:6.2f}s "
                    f"phase={phase_all[idx]:11s} router={state_name:5s} "
                    f"err(root/joint)={root_pos_err:.2e}/{joint_pos_err:.2e} "
                    f"png={png_path.stat().st_size} B",
                    flush=True,
                )

        if len(rendered) != EXPECTED_FRAMES:
            fail(f"rendered frame count mismatch: {len(rendered)}")

        manifest = {
            "status": "PASS",
            "classification": "final deterministic V03 engineering curve-negative video render; physics re-simulation from frozen master actions/commands; not ROS deployment evidence",
            "trajectory": str(trajectory_path),
            "trajectory_sha256": trajectory_sha,
            "renderer_sha256": sha256_file(Path(__file__).resolve()),
            "frames": EXPECTED_FRAMES,
            "fps": EXPECTED_FPS,
            "duration_s": EXPECTED_FRAMES / EXPECTED_FPS,
            "resolution": list(actual_resolution),
            "viewport_scale": viewport_scale,
            "physics_replay": {
                "real_physx_step_per_frame": True,
                "wallclock_pacing": False,
                "recorded_actions_used": True,
                "recorded_policy_commands_used": True,
                "master_state_guard_tolerance": 5e-4,
                "max_root_pos_abs_error": max_root_pos_err,
                "max_root_quat_sign_invariant_abs_error": max_root_quat_err,
                "max_joint_pos_abs_error": max_joint_pos_err,
            },
            "visual_preset": {
                "showcase_shift_x": SHOWCASE_SHIFT_X,
                "camera_forward": CAMERA_FORWARD_OFFSET,
                "camera_left": CAMERA_LEFT_OFFSET,
                "camera_eye_z": CAMERA_EYE_Z,
                "camera_target_z": CAMERA_TARGET_Z,
                "camera_lookahead": CAMERA_TARGET_LOOKAHEAD,
                "camera_position_smoothing_s": CAMERA_POSITION_SMOOTHING_S,
                "camera_heading_mode": CAMERA_HEADING_MODE,
                "lighting": PRESENTATION_LIGHTING,
                "ground": GROUND_LOOK,
                "scenery": SHOWCASE_SCENERY,
                "aa_op": render_settings.get("/rtx/post/aa/op"),
                "motion_blur": render_settings.get("/rtx/post/motionblur/enabled"),
                "frame_generation": render_settings.get("/rtx-transient/dlssg/enabled"),
            },
            "lighting_report": lighting_report,
            "ground_report": ground_report,
            "scenery_report": scenery_report,
            "frames_dir": str(frames_dir),
            "video_path": str(video_path),
        }

        encode_status = "NOT_REQUESTED"
        ffmpeg_path = None
        if args_cli.encode_mp4:
            ffmpeg_path = find_ffmpeg()
            if ffmpeg_path is None:
                encode_status = "FFMPEG_NOT_FOUND"
                print("[RENDER] ffmpeg not found; PNG sequence is complete and canonical.", flush=True)
            else:
                if video_path.exists():
                    video_path.unlink()
                cmd = [
                    ffmpeg_path, "-y",
                    "-framerate", str(EXPECTED_FPS),
                    "-i", str(frames_dir / "frame_%06d.png"),
                    "-c:v", "libx264",
                    "-preset", "slow",
                    "-crf", "16",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(video_path),
                ]
                print("[RENDER] Encoding MP4 with ffmpeg...", flush=True)
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    fail(f"ffmpeg failed rc={proc.returncode}: {proc.stderr[-2000:]}")
                encode_status = "PASS"
                manifest["video_sha256"] = sha256_file(video_path)
                manifest["video_bytes"] = int(video_path.stat().st_size)

        manifest["encode_status"] = encode_status
        manifest["ffmpeg"] = ffmpeg_path
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print("=" * 124)
        print("V03 FINAL PHYSICS-REPLAY RENDER SUMMARY")
        print("=" * 124)
        print("Status                              : PASS")
        print(f"Frames / FPS / duration             : {EXPECTED_FRAMES} / {EXPECTED_FPS} / {EXPECTED_FRAMES/EXPECTED_FPS:.3f} s")
        print(f"Resolution                          : {actual_resolution[0]} x {actual_resolution[1]}")
        print(f"Max master root-pos error           : {max_root_pos_err:.9e}")
        print(f"Max master root-quat error          : {max_root_quat_err:.9e}")
        print(f"Max master joint-pos error          : {max_joint_pos_err:.9e}")
        print(f"Frames directory                    : {frames_dir}")
        print(f"Manifest                            : {manifest_path}")
        print(f"MP4 encode status                   : {encode_status}")
        if video_path.is_file():
            print(f"Video                               : {video_path}")
            print(f"Video SHA256                        : {sha256_file(video_path)}")
        print(PASS_MARKER, flush=True)

    finally:
        if env is not None:
            env.close()


try:
    main()
except BaseException as exc:
    print(FAIL_MARKER, flush=True)
    print(f"EXCEPTION_TYPE={type(exc).__name__}", flush=True)
    print(f"EXCEPTION_MESSAGE={exc}", flush=True)
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
