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
        "FINAL 2560x1440/50fps renderer for the frozen v4-N graded rough-terrain LIMIT demonstration. "
        "Scientific and visual behavior are frozen from camera preview V9. One native-resolution PNG "
        "is captured after every 0.02 s physics step and the completed sequence is encoded to MP4."
    )
)
parser.add_argument("--command-vx", type=float, default=0.40)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--terrain-start-x-m", type=float, default=0.70)
parser.add_argument("--connector-length-m", type=float, default=0.18)

# Frozen final camera: approved V9 stabilized follow, then delayed radius-locked orbit.
parser.add_argument("--camera-orbit-start-s", type=float, default=15.0)
parser.add_argument("--camera-orbit-rate-deg-s", type=float, default=9.0)
parser.add_argument("--camera-position-smoothing-s", type=float, default=0.65)
parser.add_argument("--camera-forward-offset-m", type=float, default=-3.10)
parser.add_argument("--camera-left-offset-m", type=float, default=-2.90)
parser.add_argument("--camera-eye-z-m", type=float, default=1.58)
parser.add_argument("--camera-target-z-m", type=float, default=0.92)
parser.add_argument("--camera-target-lookahead-m", type=float, default=0.00)
parser.add_argument("--softbox-intensity", type=float, default=3200.0)

# Frozen final-capture semantic stop logic. The final video uses these
# simulation-time criteria and keep a short post-event tail before encoding closes.
parser.add_argument("--fall-height-drop-m", type=float, default=0.28)
parser.add_argument("--fall-up-z-threshold", type=float, default=0.45)
parser.add_argument("--fall-confirm-s", type=float, default=0.30)
parser.add_argument("--stall-window-s", type=float, default=6.0)
parser.add_argument("--stall-min-progress-m", type=float, default=0.12)
parser.add_argument("--finish-clearance-m", type=float, default=0.25)
parser.add_argument("--post-event-hold-s", type=float, default=2.0)
parser.add_argument("--safety-timeout-s", type=float, default=120.0)

parser.add_argument(
    "--output-dir", type=Path,
    default=Path(r"C:\humanoid_rl_lab\logs\thesis\video_final\V07_v4n_graded_rough_limit"),
)
parser.add_argument("--video-width", type=int, default=2560)
parser.add_argument("--video-height", type=int, default=1440)
parser.add_argument("--capture-max-updates", type=int, default=60)
parser.add_argument("--initial-settle-renders", type=int, default=8)
parser.add_argument("--ffmpeg", type=Path, default=None)
parser.add_argument("--overwrite", action="store_true")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not (0.0 < args_cli.command_vx <= 1.0):
    parser.error("--command-vx must be in (0, 1.0]")
if args_cli.connector_length_m < 0.05:
    parser.error("--connector-length-m must be >= 0.05")
if args_cli.camera_orbit_start_s < 0.0:
    parser.error("--camera-orbit-start-s must be >= 0")
if not (-30.0 <= args_cli.camera_orbit_rate_deg_s <= 30.0):
    parser.error("--camera-orbit-rate-deg-s must be within [-30, 30]")
if not (0.05 <= args_cli.camera_position_smoothing_s <= 5.0):
    parser.error("--camera-position-smoothing-s must be within [0.05, 5.0]")
if not (200.0 <= args_cli.softbox_intensity <= 6000.0):
    parser.error("--softbox-intensity must be within [200, 6000]")
if not (0.10 <= args_cli.fall_height_drop_m <= 0.60):
    parser.error("--fall-height-drop-m must be within [0.10, 0.60]")
if not (-1.0 <= args_cli.fall_up_z_threshold <= 1.0):
    parser.error("--fall-up-z-threshold must be within [-1, 1]")
if not (0.05 <= args_cli.fall_confirm_s <= 2.0):
    parser.error("--fall-confirm-s must be within [0.05, 2.0]")
if not (1.0 <= args_cli.stall_window_s <= 30.0):
    parser.error("--stall-window-s must be within [1, 30]")
if not (0.01 <= args_cli.stall_min_progress_m <= 1.0):
    parser.error("--stall-min-progress-m must be within [0.01, 1.0]")
if not (0.0 <= args_cli.finish_clearance_m <= 1.0):
    parser.error("--finish-clearance-m must be within [0, 1.0]")
if not (0.0 <= args_cli.post_event_hold_s <= 10.0):
    parser.error("--post-event-hold-s must be within [0, 10]")
if args_cli.safety_timeout_s <= 0.0:
    parser.error("--safety-timeout-s must be > 0")
if args_cli.video_width <= 0 or args_cli.video_height <= 0:
    parser.error("--video-width and --video-height must be positive")
if args_cli.capture_max_updates < 1 or args_cli.initial_settle_renders < 1:
    parser.error("capture update/settle counts must be >= 1")

# FINAL clean fullscreen bootstrap, before AppLauncher.
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
args_cli.kit_args = (str(getattr(args_cli, "kit_args", "") or "").strip() + " " + _clean_args).strip()

# Launch with --viz kit: final capture requires a Kit viewport.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb.settings
import gymnasium as gym
import torch
import humanoid_rl_lab.tasks  # noqa: F401
import omni.usd

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.v4n_ppo_cfg import (
    V4NSymmetryAugmentationPPORunnerCfg,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner


TASK_ID = "Humanoid-RL-Lab-G1-Flat-V4N-Symmetry-Augmentation-v0"
CHECKPOINT = Path(
    r"C:\humanoid_rl_lab\logs\rsl_rl"
    r"\humanoid_rl_lab_g1_v4n_symmetry_augmentation_scratch"
    r"\2026-09-04_13-58-13_v4n_genuine_scratch_seed42_official1500"
    r"\model_1499.pt"
)
EXPECTED_SHA256 = (
    "367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342"
)
EXPECTED_DT = 0.02
EXPECTED_ACTION_DIM = 37
EXPECTED_FPS = 50
FROZEN_PREVIEW_V9_SHA256 = "3D39FAC169D0914F28CEAC411DF4510D10046FA1BC2EB0A5407F24C7B8C8FEE6"
FINAL_VIDEO_NAME = "V07_v4n_graded_rough_limit_50fps.mp4"
FINAL_MANIFEST_NAME = "V07_v4n_graded_rough_limit_manifest.json"

# V4 FLUSH-FLOOR:
# A continuous physical floor remains at nominal z=0 everywhere beneath/around
# the rough course. Roughness is authored upward from that floor, so there is no
# artificial cliff into a deep support trench.
FLOOR_TOP_Z = 0.0
FLOOR_THICKNESS_M = 0.20

# Primary lane is centered exactly on the robot spawn (y=0).
# Secondary lane is directly adjacent on +Y, not mirrored around the robot.
TERRAIN_LANE_WIDTH_M = 4.20
PRIMARY_LANE_CENTER_OFFSET_Y_M = 0.0
SECONDARY_LANE_CENTER_OFFSET_Y_M = TERRAIN_LANE_WIDTH_M
COURSE_MIN_Y_OFFSET_M = -0.5 * TERRAIN_LANE_WIDTH_M
COURSE_MAX_Y_OFFSET_M = 1.5 * TERRAIN_LANE_WIDTH_M
COURSE_WIDTH_M = COURSE_MAX_Y_OFFSET_M - COURSE_MIN_Y_OFFSET_M

# Flat safety floor extends beyond both rough lanes.
SAFETY_FLOOR_MARGIN_M = 2.10
SAFETY_SUPPORT_WIDTH_M = COURSE_WIDTH_M + 2.0 * SAFETY_FLOOR_MARGIN_M
START_PLATFORM_BACK_M = 2.20

PASS_MARKER = "G1_V07_V4N_GRADED_ROUGH_LIMIT_FINAL_RENDER_PASS"
FAIL_MARKER = "G1_V07_V4N_GRADED_ROUGH_LIMIT_FINAL_RENDER_FAIL"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def as_tensor(value: Any) -> torch.Tensor:
    if hasattr(value, "torch"):
        value = value.torch
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value


def configure_policy_environment(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args_cli.seed)
    agent_cfg.seed = int(args_cli.seed)

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "base_external_force_torque"):
        env_cfg.events.base_external_force_torque = None

    reset_base = getattr(env_cfg.events, "reset_base", None)
    if reset_base is None:
        raise RuntimeError("Expected reset_base event")
    for key in list(reset_base.params.get("pose_range", {}).keys()):
        reset_base.params["pose_range"][key] = (0.0, 0.0)
    for key in list(reset_base.params.get("velocity_range", {}).keys()):
        reset_base.params["velocity_range"][key] = (0.0, 0.0)

    # Keep failures visible instead of immediately resetting.
    for term_name in ("time_out", "base_contact", "bad_orientation"):
        if hasattr(env_cfg.terminations, term_name):
            setattr(env_cfg.terminations, term_name, None)

    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.heading_command = False
    if hasattr(cmd_cfg, "rel_heading_envs"):
        cmd_cfg.rel_heading_envs = 0.0
    cmd_cfg.ranges.lin_vel_x = (0.0, 0.0)
    cmd_cfg.ranges.lin_vel_y = (0.0, 0.0)
    cmd_cfg.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(cmd_cfg, "debug_vis"):
        cmd_cfg.debug_vis = False

    # Normal Kit rendering cadence remains active.


APPROVED_GRAPHITE_RGB = (0.006, 0.009, 0.014)
APPROVED_GRAPHITE_ROUGHNESS = 0.92


def make_material(root: Sdf.Path, name: str, rgb=APPROVED_GRAPHITE_RGB,
                  roughness: float = APPROVED_GRAPHITE_ROUGHNESS) -> UsdShade.Material:
    """Visual-only UsdPreviewSurface matching the approved dark graphite video look."""
    stage = omni.usd.get_context().get_stage()
    mat_path = root.AppendChild(name + "Material")
    shader_path = mat_path.AppendChild("PreviewSurface")
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*[float(v) for v in rgb])
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


class TrackingSoftboxLight:
    """Approved local RectLight look, translated with the robot so exposure stays consistent."""

    def __init__(self, stage, root_pos) -> None:
        root_path = Sdf.Path("/World/VideoPresentation")
        stage.DefinePrim(root_path, "Xform")

        light_path = root_path.AppendChild("TrackingSoftbox")
        if stage.GetPrimAtPath(light_path).IsValid():
            raise RuntimeError(f"Softbox path already exists: {light_path}")

        self.light = UsdLux.RectLight.Define(stage, light_path)
        self.light.CreateIntensityAttr(float(args_cli.softbox_intensity))
        self.light.CreateColorAttr(Gf.Vec3f(1.0, 0.965, 0.92))
        self.light.CreateWidthAttr(5.0)
        self.light.CreateHeightAttr(3.0)

        xform = UsdGeom.Xformable(self.light.GetPrim())
        self.translate_op = xform.AddTranslateOp()
        xform.AddRotateXYZOp().Set(Gf.Vec3f(-28.0, 18.0, 8.0))

        self.update(root_pos)

    def update(self, root_pos) -> None:
        self.translate_op.Set(
            Gf.Vec3d(
                float(root_pos[0]) + 1.5,
                float(root_pos[1]) - 1.8,
                4.8,
            )
        )


def author_solid_column(stage, path: Sdf.Path, x: float, y: float, top_z: float,
                        sx: float, sy: float, material) -> None:
    """
    Collision column whose bottom is exactly the nominal floor z=0.

    `top_z` is therefore a non-negative roughness height above the floor.
    The continuous flat collision floor remains below/around the columns,
    eliminating holes without changing peak-to-valley roughness.
    """
    top_z = max(float(FLOOR_TOP_Z), float(top_z))
    height = max(0.002, top_z - FLOOR_TOP_Z)
    center_z = FLOOR_TOP_Z + 0.5 * height

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(2.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), center_z))
    xf.AddScaleOp().Set(Gf.Vec3f(0.5 * sx, 0.5 * sy, 0.5 * height))

    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    collision = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    collision.CreateCollisionEnabledAttr(True)



def author_flat_connector(stage, root: Sdf.Path, x0: float, length: float,
                          y0: float, material, label: str) -> float:
    # Visual/collision connector flush with the z=0 floor. A very thin top slab
    # avoids any vertical discontinuity while remaining physically explicit.
    cube = UsdGeom.Cube.Define(stage, root.AppendChild(label))
    cube.CreateSizeAttr(2.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(
        Gf.Vec3d(
            x0 + 0.5 * length,
            y0 + 0.5 * (COURSE_MIN_Y_OFFSET_M + COURSE_MAX_Y_OFFSET_M),
            -0.001,
        )
    )
    xf.AddScaleOp().Set(
        Gf.Vec3f(
            0.5 * length,
            0.5 * COURSE_WIDTH_M,
            0.001,
        )
    )
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
    return x0 + length



def paver_top(row: int, col: int) -> float:
    """Strong but still progressive staggered pavers, 8--50 mm above nominal zero."""
    levels = (0.18, 0.48, 0.78, 0.34)
    alpha = levels[(row + 2 * col) % len(levels)]
    alpha += 0.10 * math.sin(0.91 * row - 1.13 * col + 0.35)
    alpha = max(0.0, min(1.0, alpha))

    h_min, h_max = 0.008, 0.050
    h = h_min + (h_max - h_min) * alpha
    entry_caps = (0.014, 0.025, 0.038)
    if row < len(entry_caps):
        cap = entry_caps[row]
        h = h_min + (cap - h_min) * alpha
    return h


def smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def rocky_top(local_x: float, local_y: float, entry_length: float) -> float:
    """
    Continuous rocky surface with both shallow depressions and protrusions.
    Approximate intended range after entry: -25 mm .. +90 mm.
    """
    entry = smoothstep01(local_x / max(1.0e-9, entry_length))

    raw = (
        0.05
        + 0.45 * math.sin(2.40 * local_x + 1.10 * local_y + 0.2)
        + 0.24 * math.sin(5.30 * local_x - 2.70 * local_y + 1.1)
        + 0.15 * math.sin(9.60 * local_x + 5.80 * local_y + 0.8)
        + 0.10 * math.sin(15.0 * local_x - 9.0 * local_y + 2.2)
    )

    rocks = 0.0
    for cx, cy, amp, sigma in (
        (1.10, -0.55, 0.42, 0.25),
        (1.75,  0.45, 0.48, 0.22),
        (2.35, -0.05, 0.50, 0.20),
    ):
        dx = local_x - cx
        dy = local_y - cy
        rocks += amp * math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))

    raw += rocks
    # Convert bounded relief into a signed height.
    raw = max(-1.0, min(1.0, raw))
    signed = -0.025 + 0.115 * (0.5 * (raw + 1.0))

    # Preserve the original ~11.5 cm peak-to-valley relief but lift the
    # complete surface so its minimum is the flush floor at z=0.
    lifted = signed + 0.025
    return entry * max(0.0, lifted)


def deterministic_signed_cell(row: int, col: int, amplitude: float, phase: float) -> float:
    """
    Deterministic signed pseudo-random field in [-amplitude,+amplitude].
    First 3 rows ramp difficulty so entry does not dominate the test.
    """
    value = (
        0.62 * math.sin(1.79 * row + 2.31 * col + phase)
        + 0.38 * math.sin(3.17 * row - 1.27 * col + 0.73 + phase)
    )
    value = max(-1.0, min(1.0, value))

    entry_scale = (0.35, 0.62, 0.82)
    scale = entry_scale[row] if row < len(entry_scale) else 1.0

    # V4 floor-flush transform:
    # signed [-A,+A] becomes [0,2A]. Peak-to-valley remains exactly 2A,
    # but no cell top lies below the continuous z=0 floor.
    signed = float(amplitude * scale * value)
    local_amplitude = float(amplitude * scale)
    return signed + local_amplitude


def build_pavers(stage, root, x0, y0, material) -> dict:
    rows, cols_per_lane = 7, 8
    lane_count = 2
    px, py = 0.40, 0.50

    # No seams: each paver reaches exactly to the next pitch boundary.
    sx, sy = px, py
    tops = []

    occupied_lane_width = cols_per_lane * py
    lane_centers = (
        y0 + PRIMARY_LANE_CENTER_OFFSET_Y_M,
        y0 + occupied_lane_width,
    )

    for row in range(rows):
        x = x0 + row * px
        y_shift = 0.25 * py if row % 2 == 0 else -0.25 * py

        # Primary lane is centered on the robot. Secondary lane is attached on +Y.
        for lane in range(lane_count):
            lane_center = lane_centers[lane]
            for col in range(cols_per_lane):
                y = (
                    lane_center
                    + (col - (cols_per_lane - 1) / 2.0) * py
                    + y_shift
                )
                top = paver_top(row, col)
                tops.append(top)
                author_solid_column(
                    stage,
                    root.AppendChild(
                        f"paver_lane{lane}_r{row:02d}_c{col:02d}"
                    ),
                    x, y, top, sx, sy, material
                )

    length = (rows - 1) * px + sx
    return {
        "name": "Z1_STRONG_STAGGERED_PAVERS",
        "family": "strong staggered pavers",
        "start_x": x0,
        "end_x": x0 + length,
        "min_top_m": min(tops),
        "max_top_m": max(tops),
        "expected": "strong entry / likely pass",
    }


def build_rocky(stage, root, x0, y0, material) -> dict:
    length, width = 3.10, COURSE_WIDTH_M
    grid = 0.085
    entry_length = 0.65

    nx = int(math.ceil(length / grid)) + 1
    ny = int(math.ceil(width / grid)) + 1
    dx = length / (nx - 1)
    dy = width / (ny - 1)

    points, tops = [], []
    for ix in range(nx):
        lx = ix * dx
        for iy in range(ny):
            # Course spans one centered primary lane [-2.1,+2.1] and one
            # immediately adjacent secondary lane [+2.1,+6.3].
            ly = COURSE_MIN_Y_OFFSET_M + iy * dy

            # Repeat the exact primary-lane rocky pattern in the secondary lane.
            lane_y = (
                (ly + 0.5 * TERRAIN_LANE_WIDTH_M)
                % TERRAIN_LANE_WIDTH_M
            ) - 0.5 * TERRAIN_LANE_WIDTH_M

            top = rocky_top(lx, lane_y, entry_length)
            tops.append(top)
            points.append(Gf.Vec3f(x0 + lx, y0 + ly, top))

    counts, indices = [], []
    for ix in range(nx - 1):
        for iy in range(ny - 1):
            i00 = ix * ny + iy
            i10 = (ix + 1) * ny + iy
            i01 = ix * ny + iy + 1
            i11 = (ix + 1) * ny + iy + 1
            counts.extend([3, 3])
            indices.extend([i00, i10, i11, i00, i11, i01])

    mesh = UsdGeom.Mesh.Define(stage, root.AppendChild("RockySurface"))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision.CreateCollisionEnabledAttr(True)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr("none")

    return {
        "name": "Z2_ROCKY_SIGNED_HEIGHTFIELD",
        "family": "dense signed rocky heightfield",
        "start_x": x0,
        "end_x": x0 + length,
        "min_top_m": min(tops),
        "max_top_m": max(tops),
        "expected": "strong continuous OOD surface",
    }


def build_signed_grid(stage, root, x0, y0, material,
                      amplitude: float, cell: float, rows: int,
                      phase: float, name: str, expected: str) -> dict:
    cols_per_lane = int(math.floor(TERRAIN_LANE_WIDTH_M / cell))
    if cols_per_lane % 2 == 1:
        cols_per_lane -= 1
    lane_count = 2
    tops = []

    occupied_lane_width = cols_per_lane * cell
    lane_centers = (
        y0 + PRIMARY_LANE_CENTER_OFFSET_Y_M,
        y0 + occupied_lane_width,
    )

    for row in range(rows):
        x = x0 + (row + 0.5) * cell
        for lane in range(lane_count):
            lane_center = lane_centers[lane]
            for col in range(cols_per_lane):
                y = (
                    lane_center
                    + (col - (cols_per_lane - 1) / 2.0) * cell
                )

                # Same deterministic cell pattern in both lanes.
                top = deterministic_signed_cell(
                    row, col, amplitude, phase
                )
                tops.append(top)
                author_solid_column(
                    stage,
                    root.AppendChild(
                        f"cell_lane{lane}_r{row:02d}_c{col:02d}"
                    ),
                    x, y, top,
                    cell, cell,
                    material,
                )

    length = rows * cell
    return {
        "name": name,
        "family": "signed boxes-style random grid",
        "start_x": x0,
        "end_x": x0 + length,
        "cell_m": cell,
        "amplitude_m": amplitude,
        "min_top_m": min(tops),
        "max_top_m": max(tops),
        "max_theoretical_adjacent_jump_m": 2.0 * amplitude,
        "expected": expected,
    }


def build_course(env_origin) -> list[dict]:
    stage = omni.usd.get_context().get_stage()
    root = Sdf.Path("/World/ProfessorOODGradedLimitCourseV4FlushFloor")
    if stage.GetPrimAtPath(root).IsValid():
        raise RuntimeError(f"Course root already exists: {root}")
    stage.DefinePrim(root, "Xform")

    # Keep the approved near-black graphite only for the continuous floor/connectors.
    # Restore distinct rough-zone colors so the terrain progression remains visually obvious.
    mats = {
        "flat": make_material(root, "Connector", (0.006, 0.009, 0.014), 0.92),
        "z1": make_material(root, "Pavers", (0.080, 0.060, 0.038), 0.97),
        "z2": make_material(root, "Rocky", (0.085, 0.075, 0.062), 0.97),
        "z3": make_material(root, "Grid30", (0.080, 0.065, 0.050), 0.97),
        "z4": make_material(root, "Grid45", (0.095, 0.055, 0.040), 0.97),
    }

    env_x, env_y, env_z = [float(v) for v in env_origin]
    if abs(env_z) > 1.0e-6:
        raise RuntimeError(f"Unexpected environment origin z={env_z}")

    # Disable/hide the original task plane only to avoid duplicate coplanar physics.
    # Replace it immediately with our own continuous z=0 physical floor covering the
    # full rough-course corridor plus generous lateral safety margins.
    flat_visual_path = Sdf.Path("/World/ground/terrain/Environment/Geometry")
    flat_collision_path = Sdf.Path("/World/ground/terrain/GroundPlane/CollisionPlane")
    flat_visual = stage.GetPrimAtPath(flat_visual_path)
    flat_collision = stage.GetPrimAtPath(flat_collision_path)

    if not flat_visual.IsValid() or not flat_collision.IsValid():
        raise RuntimeError("Original flat terrain prims are missing")
    if not flat_collision.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("Original flat collision lost CollisionAPI")

    UsdGeom.Imageable(flat_visual).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    flat_api = UsdPhysics.CollisionAPI(flat_collision)
    flat_api.CreateCollisionEnabledAttr(False)
    if flat_api.GetCollisionEnabledAttr().Get() is not False:
        raise RuntimeError("Could not disable original flat collision")

    start_x = env_x + float(args_cli.terrain_start_x_m)
    # Z5 was deliberately removed: observed failure already occurs near the end
    # of Z4, so extending the course beyond that adds no useful evidence.
    estimated_end = (
        start_x
        + 2.80                         # Z1 pavers
        + float(args_cli.connector_length_m)
        + 3.10                         # Z2 rocky
        + float(args_cli.connector_length_m)
        + 2.56                         # Z3: 8 * 0.32
        + float(args_cli.connector_length_m)
        + 2.70                         # Z4: 9 * 0.30
    )

    floor_x0 = env_x - START_PLATFORM_BACK_M
    floor_x1 = estimated_end + 2.0
    floor_length = floor_x1 - floor_x0
    floor_center_x = 0.5 * (floor_x0 + floor_x1)

    # Center the safety floor under the asymmetric two-lane course.
    course_center_y_offset = 0.5 * (
        COURSE_MIN_Y_OFFSET_M + COURSE_MAX_Y_OFFSET_M
    )
    floor_center_y = env_y + course_center_y_offset

    floor = UsdGeom.Cube.Define(stage, root.AppendChild("FlushSafetyFloor"))
    floor.CreateSizeAttr(2.0)
    xf = UsdGeom.Xformable(floor.GetPrim())
    xf.AddTranslateOp().Set(
        Gf.Vec3d(
            floor_center_x,
            floor_center_y,
            FLOOR_TOP_Z - 0.5 * FLOOR_THICKNESS_M,
        )
    )
    xf.AddScaleOp().Set(
        Gf.Vec3f(
            0.5 * floor_length,
            0.5 * SAFETY_SUPPORT_WIDTH_M,
            0.5 * FLOOR_THICKNESS_M,
        )
    )
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(mats["flat"])
    floor_collision = UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    floor_collision.CreateCollisionEnabledAttr(True)

    # Spawn region already lies on this continuous floor, so no raised or lowered
    # start platform is needed. Rough zone 1 begins after the normal flat entry.

    zones = []
    x = start_x

    z1_root = root.AppendChild("Zone1_StrongPavers")
    stage.DefinePrim(z1_root, "Xform")
    z1 = build_pavers(stage, z1_root, x, env_y, mats["z1"])
    zones.append(z1)
    x = z1["end_x"]
    x = author_flat_connector(
        stage, root, x, args_cli.connector_length_m, env_y, mats["flat"], "Connector_1"
    )

    z2_root = root.AppendChild("Zone2_RockySigned")
    stage.DefinePrim(z2_root, "Xform")
    z2 = build_rocky(stage, z2_root, x, env_y, mats["z2"])
    zones.append(z2)
    x = z2["end_x"]
    x = author_flat_connector(
        stage, root, x, args_cli.connector_length_m, env_y, mats["flat"], "Connector_2"
    )

    # Relative-height grid ladder: preserves the same peak-to-valley ladder above a flush z=0 floor.
    grid_specs = [
        ("Z3_SIGNED_GRID_PM30MM", 0.030, 0.32, 8, 0.2, mats["z3"],
         "first genuine depression/protrusion grid"),
        ("Z4_SIGNED_GRID_PM45MM", 0.045, 0.30, 9, 1.1, mats["z4"],
         "known qualitative limit region; final rough zone"),
    ]

    for idx, (name, amp, cell, rows, phase, mat, expected) in enumerate(grid_specs, start=3):
        zr = root.AppendChild(f"Zone{idx}_{name}")
        stage.DefinePrim(zr, "Xform")
        z = build_signed_grid(
            stage, zr, x, env_y, mat,
            amplitude=amp, cell=cell, rows=rows,
            phase=phase, name=name, expected=expected,
        )
        zones.append(z)
        x = z["end_x"]
        if idx < 4:
            x = author_flat_connector(
                stage, root, x, args_cli.connector_length_m, env_y,
                mats["flat"], f"Connector_{idx}"
            )

    # Small flat catch platform after the final zone.
    author_flat_connector(stage, root, x, 1.20, env_y, mats["flat"], "FinishPlatform")

    return zones


def zone_for_x(x: float, zones: list[dict]) -> str:
    for zone in zones:
        if zone["start_x"] <= x <= zone["end_x"]:
            return zone["name"]
    if x < zones[0]["start_x"]:
        return "FLAT_START"
    if x > zones[-1]["end_x"]:
        return "COURSE_COMPLETE"
    return "CONNECTOR"



def quat_wxyz_to_yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_wxyz_to_body_up_z(q) -> float:
    """World-Z component of the robot body +Z axis; +1 upright, 0 sideways, -1 inverted."""
    w, x, y, z = (float(v) for v in q)
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm <= 1.0e-9:
        return -1.0
    x /= norm
    y /= norm
    return 1.0 - 2.0 * (x*x + y*y)


class StabilizedDelayedOrbitCamera:
    """
    v5.42-style stabilized camera with a delayed, radius-locked orbit.

    The low-pass center is retained to suppress noisy horizontal follow motion, but
    the camera eye is clamped to the INITIAL horizontal radius around the CURRENT
    robot XY. Therefore filtering can make the shot slightly closer, never farther
    away than the approved starting distance while the orbit turns.
    """

    def __init__(self) -> None:
        self.center_xy = None
        self.initial_yaw = None
        self.initial_horizontal_radius = None
        self.last_eye = None
        self.last_target = None

    def update(self, root_pos, root_quat, sim_time_s: float, dt: float):
        root_x = float(root_pos[0])
        root_y = float(root_pos[1])

        forward = float(args_cli.camera_forward_offset_m)
        left = float(args_cli.camera_left_offset_m)
        nominal_radius = math.hypot(forward, left)

        if self.center_xy is None:
            self.center_xy = [root_x, root_y]
            self.initial_yaw = quat_wxyz_to_yaw(root_quat)
            self.initial_horizontal_radius = nominal_radius
        else:
            tau = max(1.0e-6, float(args_cli.camera_position_smoothing_s))
            alpha = 1.0 - math.exp(-float(dt) / tau)
            self.center_xy[0] += alpha * (root_x - self.center_xy[0])
            self.center_xy[1] += alpha * (root_y - self.center_xy[1])

        orbit_elapsed = max(
            0.0,
            float(sim_time_s) - float(args_cli.camera_orbit_start_s),
        )
        orbit_angle = math.radians(
            float(args_cli.camera_orbit_rate_deg_s) * orbit_elapsed
        )
        camera_yaw = float(self.initial_yaw) + orbit_angle

        cy = math.cos(camera_yaw)
        sy = math.sin(camera_yaw)

        desired_eye_x = self.center_xy[0] + forward * cy + left * (-sy)
        desired_eye_y = self.center_xy[1] + forward * sy + left * cy

        # Lock maximum distance to the exact initial horizontal radius.
        # If low-pass lag would push the eye farther away, project it back onto
        # the circle around the CURRENT robot. If it is already closer, keep it.
        vx = desired_eye_x - root_x
        vy = desired_eye_y - root_y
        actual_radius = math.hypot(vx, vy)
        max_radius = float(self.initial_horizontal_radius)

        if actual_radius > max_radius and actual_radius > 1.0e-9:
            scale = max_radius / actual_radius
            eye_x = root_x + vx * scale
            eye_y = root_y + vy * scale
        else:
            eye_x = desired_eye_x
            eye_y = desired_eye_y

        eye = (
            eye_x,
            eye_y,
            float(args_cli.camera_eye_z_m),
        )

        # Target the current robot XY directly so the G1 remains centered while
        # the eye retains the smoother, radius-limited motion.
        target = (
            root_x + float(args_cli.camera_target_lookahead_m) * cy,
            root_y + float(args_cli.camera_target_lookahead_m) * sy,
            float(args_cli.camera_target_z_m),
        )

        self.last_eye = eye
        self.last_target = target
        return eye, target, orbit_angle



def resolve_ffmpeg() -> str:
    if args_cli.ffmpeg is not None:
        p = Path(args_cli.ffmpeg).resolve()
        if not p.is_file():
            raise RuntimeError(f"Explicit ffmpeg executable not found: {p}")
        return str(p)
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError("ffmpeg is not available on PATH; aborting before expensive final capture")
    return found


def prepare_final_output():
    root = Path(args_cli.output_dir).resolve()
    frames_dir = root / "frames"
    video_dir = root / "video"
    replay_dir = root / "replay"
    video_path = video_dir / FINAL_VIDEO_NAME
    manifest_path = replay_dir / FINAL_MANIFEST_NAME
    old_frames = list(frames_dir.glob("frame_*.png")) if frames_dir.is_dir() else []
    if (old_frames or video_path.exists() or manifest_path.exists()) and not args_cli.overwrite:
        raise RuntimeError("Existing V07 artifacts found; use --overwrite only to intentionally replace this render")
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)
    if args_cli.overwrite:
        for p in frames_dir.glob("frame_*.png"):
            p.unlink()
        for p in (video_path, manifest_path):
            if p.exists(): p.unlink()
    return root, frames_dir, video_path, manifest_path


def capture_png_strict(sim, viewport, png_path: Path) -> int:
    if png_path.exists(): png_path.unlink()
    capture_viewport_to_file(viewport, file_path=str(png_path), is_hdr=False)
    for update_index in range(1, int(args_cli.capture_max_updates) + 1):
        sim.render()
        if png_path.is_file() and png_path.stat().st_size > 1024:
            return update_index
    raise RuntimeError(f"PNG capture timeout: {png_path}")


def encode_final_mp4(ffmpeg_path: str, frames_dir: Path, video_path: Path) -> None:
    if video_path.exists(): video_path.unlink()
    cmd = [
        ffmpeg_path, "-y", "-framerate", str(EXPECTED_FPS),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264", "-preset", "slow", "-crf", "16",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path),
    ]
    print("[ENCODE] H.264 / libx264 / slow / CRF 16 / yuv420p / faststart", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed rc={proc.returncode}:\n{proc.stderr[-4000:]}")
    if not video_path.is_file() or video_path.stat().st_size <= 1024:
        raise RuntimeError("ffmpeg returned success but final MP4 is missing/empty")


def main() -> None:
    output_root, frames_dir, video_path, manifest_path = prepare_final_output()
    ffmpeg_path = resolve_ffmpeg()
    checkpoint_sha = sha256_file(CHECKPOINT)
    if checkpoint_sha != EXPECTED_SHA256:
        raise RuntimeError(
            f"v4-N checkpoint SHA mismatch: expected={EXPECTED_SHA256}, actual={checkpoint_sha}"
        )

    installed_rsl = metadata.version("rsl-rl-lib")
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1)
    agent_cfg = V4NSymmetryAugmentationPPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl)
    configure_policy_environment(env_cfg, agent_cfg)

    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    env = None

    try:
        raw_uenv = raw_env.unwrapped
        env_origin = (
            as_tensor(raw_uenv.scene.env_origins)[0]
            .detach().cpu().tolist()
        )

        stage = omni.usd.get_context().get_stage()
        # IMPORTANT: keep every pre-existing Isaac/task light unchanged.
        # The approved video softbox is an ADDITIONAL local RectLight; it must not
        # zero/disable the environment's own DomeLight or DistantLight.
        zones = build_course(env_origin)
        raw_uenv.sim.forward()

        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        uenv = env.unwrapped
        robot = uenv.scene["robot"]

        if not math.isclose(float(uenv.step_dt), EXPECTED_DT, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(f"Unexpected step_dt={uenv.step_dt}")
        if int(uenv.action_manager.total_action_dim) != EXPECTED_ACTION_DIM:
            raise RuntimeError(f"Unexpected action dim={uenv.action_manager.total_action_dim}")

        reward_manager = uenv.reward_manager
        reward_buf = getattr(reward_manager, "_reward_buf", None)
        if reward_buf is None or not torch.is_tensor(reward_buf):
            raise RuntimeError("Reward bypass buffer unavailable")

        def reward_bypass(*_args, **_kwargs):
            reward_buf.zero_()
            return reward_buf

        reward_manager.compute = reward_bypass

        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        runner.load(str(CHECKPOINT))
        policy = runner.get_inference_policy(device=uenv.device)

        env_ids = torch.tensor([0], dtype=torch.int32, device=uenv.device)
        uenv.command_manager.reset(env_ids=env_ids)
        cmd_tensor = uenv.command_manager.get_command("base_velocity")
        cmd_tensor.zero_()

        # Stabilized follow camera: same visual principle as the frozen v5.42
        # final camera, with a 9 deg/s radius-locked orbit enabled only after the requested delay.
        camera_tracker = StabilizedDelayedOrbitCamera()

        root_pos_initial = as_tensor(
            robot.data.root_pos_w
        )[0].detach().cpu().tolist()
        root_quat_initial = as_tensor(
            robot.data.root_quat_w
        )[0].detach().cpu().tolist()

        softbox = TrackingSoftboxLight(stage, root_pos_initial)
        raw_uenv.sim.forward()

        initial_eye, initial_target, _ = camera_tracker.update(
            root_pos_initial,
            root_quat_initial,
            sim_time_s=0.0,
            dt=EXPECTED_DT,
        )
        uenv.sim.set_camera_view(
            eye=list(initial_eye),
            target=list(initial_target),
        )
        uenv.sim.forward()
        uenv.sim.render()

        render_settings = carb.settings.get_settings()
        render_settings.set("/rtx-transient/dlssg/enabled", False)
        render_settings.set("/rtx/post/motionblur/enabled", False)
        render_settings.set("/rtx/post/aa/op", 4)
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("Active Kit viewport unavailable; launch with --viz kit")
        viewport.set_texture_resolution((int(args_cli.video_width), int(args_cli.video_height)))
        viewport.set_texture_resolution_scale(1.0)
        for _ in range(max(1, int(args_cli.initial_settle_renders))):
            uenv.sim.render()
        actual_viewport_resolution = tuple(int(round(v)) for v in viewport.get_texture_resolution())
        full_viewport_resolution = tuple(int(round(v)) for v in viewport.get_full_texture_resolution())
        viewport_scale = float(viewport.get_texture_resolution_scale())
        expected_resolution = (int(args_cli.video_width), int(args_cli.video_height))
        if actual_viewport_resolution != expected_resolution:
            raise RuntimeError(f"Viewport mismatch: expected={expected_resolution}, actual={actual_viewport_resolution}, full={full_viewport_resolution}")
        if abs(viewport_scale - 1.0) > 1.0e-6:
            raise RuntimeError(f"Viewport scale mismatch: {viewport_scale}")

        print("=" * 132)
        print("G1 V07 FINAL — v4-N GRADED ROUGH LIMIT / FROZEN V9 CONTRACT")
        print("=" * 132)
        print("Policy                         : frozen v4-N, trained on FLAT only")
        print("Command                        : [0.40, 0.00, 0.00]")
        print("Physics/control dt             : 0.020 s")
        print("Original task floor            : replaced by equivalent custom z=0 floor")
        print("Custom physical floor top      : z=0.000 m / flush with lowest rough level")
        print(f"Primary rough lane             : centered exactly on robot y={PRIMARY_LANE_CENTER_OFFSET_Y_M:+.2f} m")
        print("Secondary rough lane           : +Y side, offset by exact occupied width of primary")
        print(f"Nominal continuous-lane width  : {TERRAIN_LANE_WIDTH_M:.2f} m")
        print(f"Maximum rough corridor width   : {COURSE_WIDTH_M:.2f} m")
        print(f"Safety floor width             : {SAFETY_SUPPORT_WIDTH_M:.2f} m")
        print("Discrete-cell seams            : ZERO by construction (size == pitch)")
        print("Capture / encoding             : 1 PNG PER PHYSICS STEP / H.264 MP4")
        print("Wall-clock pacing              : NONE")
        print(f"Final resolution                : {actual_viewport_resolution[0]} x {actual_viewport_resolution[1]} / scale {viewport_scale:.3f}")
        print("Render quality                 : DLAA / motion blur OFF / frame generation OFF")
        print(f"Final FPS                      : {EXPECTED_FPS}")
        print(f"Frames directory               : {frames_dir}")
        print(f"Final MP4                      : {video_path}")
        print("Camera mode                    : stabilized v5.42-style follow")
        print(f"Camera position smoothing      : {args_cli.camera_position_smoothing_s:.2f} s")
        print("Camera pelvis yaw/roll/pitch   : IGNORED")
        print("Camera eye/target Z            : FIXED")
        print(f"Camera orbit start             : {args_cli.camera_orbit_start_s:.2f} s SIM-TIME")
        print(f"Camera orbit rate              : {args_cli.camera_orbit_rate_deg_s:.2f} deg/s")
        print(f"Camera max horizontal radius   : {math.hypot(args_cli.camera_forward_offset_m, args_cli.camera_left_offset_m):.3f} m (initial radius lock)")
        print("Floor shader                   : approved near-black graphite RGB=(0.006,0.009,0.014), roughness=0.92")
        print("Rough-zone shaders             : original distinct Z1/Z2/Z3/Z4 colors restored")
        print(f"Tracking softbox               : RectLight {args_cli.softbox_intensity:.0f}, 5.0 x 3.0 m, warm-neutral")
        print("Existing Isaac/task lights     : PRESERVED UNCHANGED")
        print(f"Fall confirm                   : drop>={args_cli.fall_height_drop_m:.2f} m OR up_z<{args_cli.fall_up_z_threshold:.2f}, held {args_cli.fall_confirm_s:.2f} s")
        print(f"Stall detection                : <{args_cli.stall_min_progress_m:.2f} m new forward progress for {args_cli.stall_window_s:.1f} s")
        print(f"Finish clearance               : {args_cli.finish_clearance_m:.2f} m beyond final rough zone")
        print(f"Post-event hold                : {args_cli.post_event_hold_s:.2f} s SIM-TIME")
        print(f"Safety timeout                 : {args_cli.safety_timeout_s:.1f} s SIM-TIME")
        print("Scientific classification      : QUALITATIVE OOD LIMIT SEARCH ONLY")
        print("-" * 132)

        for i, z in enumerate(zones, 1):
            extra = ""
            if "amplitude_m" in z:
                extra = (
                    f" amp=+/-{z['amplitude_m']*100:.1f}cm"
                    f" theoretical_jump<={z['max_theoretical_adjacent_jump_m']*100:.1f}cm"
                )
            print(
                f"Z{i}: {z['family']:<36} "
                f"x={z['start_x']:.2f}..{z['end_x']:.2f}m "
                f"top={z['min_top_m']*100:+.1f}..{z['max_top_m']*100:+.1f}cm"
                f"{extra}"
            )
            print(f"    role: {z['expected']}")

        print("-" * 132)
        print("Difficulty ladder:")
        print("  strong pavers -> signed rocky -> signed grid +/-3.0cm -> +/-4.5cm [FINAL ROUGH ZONE]")
        print("Lateral layout:")
        print("  primary lane centered on robot; secondary +Y lane uses exact occupied-width offset, so borders touch")
        print("  continuous floor at z=0 removes artificial cliffs/holes outside tile footprints")
        print("  signed grid peak-to-valley severity is preserved by shifting [-A,+A] -> [0,2A]")
        print("Observe where locomotion first stalls or a real fall occurs.")
        print("Camera stays fixed-angle/smoothed for 15 s, then begins the 9 deg/s radius-locked orbit.")
        print("Z5 +/-6.0 cm grid is intentionally absent: observed failure already occurs in late Z4.")
        print("Visual preview uses graphite floor + original colored rough zones + local softbox, preserving Isaac/task lighting.")
        print("Final capture auto-stops on confirmed fall, prolonged stall, course completion, or safety timeout.")
        print("=" * 132, flush=True)

        step = 0
        last_zone = None

        initial_root_z = float(root_pos_initial[2])
        fall_candidate_since_s = None
        progress_anchor_x = None
        last_progress_time_s = None
        stop_reason = None
        stop_trigger_time_s = None
        stop_trigger_x = None
        rendered_frames = 0
        max_capture_updates_seen = 0

        finish_trigger_x = float(zones[-1]["end_x"]) + float(args_cli.finish_clearance_m)

        while simulation_app.is_running():
            # Preserve the exact commanded locomotion until a semantic stop is confirmed.
            # During the short post-event tail, command zero so the final shot settles.
            if stop_reason is None:
                cmd_tensor[:, 0] = float(args_cli.command_vx)
            else:
                cmd_tensor[:, 0] = 0.0
            cmd_tensor[:, 1] = 0.0
            cmd_tensor[:, 2] = 0.0

            obs = env.get_observations()
            with torch.inference_mode():
                actions = policy(obs)

            if not torch.isfinite(actions).all():
                raise RuntimeError(f"Non-finite action at step={step}")

            env.step(actions)
            step += 1

            root_pos = as_tensor(robot.data.root_pos_w)[0].detach().cpu()
            root_quat = as_tensor(robot.data.root_quat_w)[0].detach().cpu()
            current_zone = zone_for_x(float(root_pos[0]), zones)

            # Visual-only local softbox follows the robot; no global light washes the course.
            softbox.update(root_pos.tolist())

            # Camera timing is based only on deterministic simulation time.
            sim_time_s = step * EXPECTED_DT
            eye, target, orbit_angle = camera_tracker.update(
                root_pos.tolist(),
                root_quat.tolist(),
                sim_time_s=sim_time_s,
                dt=EXPECTED_DT,
            )
            uenv.sim.set_camera_view(
                eye=list(eye),
                target=list(target),
            )

            root_x = float(root_pos[0])
            root_z = float(root_pos[2])
            body_up_z = quat_wxyz_to_body_up_z(root_quat.tolist())

            if stop_reason is None:
                # 1) Genuine course completion: robot has cleared the final rough zone
                # and advanced onto the flat finish platform.
                if root_x >= finish_trigger_x:
                    stop_reason = "COURSE_COMPLETE"
                    stop_trigger_time_s = sim_time_s
                    stop_trigger_x = root_x

                # 2) Confirmed fall: large pelvis-height collapse OR body almost sideways.
                # The condition must persist, avoiding one-frame rough-terrain transients.
                if stop_reason is None:
                    height_drop = initial_root_z - root_z
                    fall_candidate = (
                        height_drop >= float(args_cli.fall_height_drop_m)
                        or body_up_z <= float(args_cli.fall_up_z_threshold)
                    )
                    if fall_candidate:
                        if fall_candidate_since_s is None:
                            fall_candidate_since_s = sim_time_s
                        elif sim_time_s - fall_candidate_since_s >= float(args_cli.fall_confirm_s):
                            stop_reason = "FALL_CONFIRMED"
                            stop_trigger_time_s = sim_time_s
                            stop_trigger_x = root_x
                    else:
                        fall_candidate_since_s = None

                # 3) Prolonged loss of forward progress. This catches a robot that
                # remains upright but becomes physically stuck on the rough course.
                if stop_reason is None and root_x >= float(zones[0]["start_x"]):
                    if progress_anchor_x is None:
                        progress_anchor_x = root_x
                        last_progress_time_s = sim_time_s
                    elif root_x >= progress_anchor_x + float(args_cli.stall_min_progress_m):
                        progress_anchor_x = root_x
                        last_progress_time_s = sim_time_s
                    elif (
                        last_progress_time_s is not None
                        and sim_time_s - last_progress_time_s >= float(args_cli.stall_window_s)
                    ):
                        stop_reason = "STALL_CONFIRMED"
                        stop_trigger_time_s = sim_time_s
                        stop_trigger_x = root_x

                # 4) Last-resort guard only. This is not the normal end condition.
                if stop_reason is None and sim_time_s >= float(args_cli.safety_timeout_s):
                    stop_reason = "SAFETY_TIMEOUT"
                    stop_trigger_time_s = sim_time_s
                    stop_trigger_x = root_x

                if stop_reason is not None:
                    print(
                        f"[AUTO-STOP-TRIGGER] reason={stop_reason} "
                        f"sim_t={sim_time_s:.2f}s x={root_x:+.3f} z={root_z:+.3f} "
                        f"up_z={body_up_z:+.3f}; holding {args_cli.post_event_hold_s:.2f}s",
                        flush=True,
                    )

            if current_zone != last_zone:
                print(
                    f"[ZONE] sim_t={step*EXPECTED_DT:7.2f}s "
                    f"-> {current_zone}  root_x={float(root_pos[0]):+.3f}",
                    flush=True,
                )
                last_zone = current_zone

            if step % 100 == 0:
                root_lin = as_tensor(robot.data.root_lin_vel_b)[0].detach().cpu()
                print(
                    f"[preview] t={step*EXPECTED_DT:7.2f}s "
                    f"zone={current_zone:<28} "
                    f"x={root_pos[0]:+.2f} y={root_pos[1]:+.2f} z={root_pos[2]:+.2f} "
                    f"vx_body={root_lin[0]:+.3f} "
                    f"up_z={body_up_z:+.3f} "
                    f"camera_orbit={math.degrees(orbit_angle):+.1f}deg",
                    flush=True,
                )

            # Capture exactly one output frame for each completed 0.02 s physics step.
            uenv.sim.forward()
            uenv.sim.render()
            png_path = frames_dir / f"frame_{rendered_frames:06d}.png"
            capture_updates = capture_png_strict(uenv.sim, viewport, png_path)
            max_capture_updates_seen = max(max_capture_updates_seen, capture_updates)
            rendered_frames += 1
            if rendered_frames == 1 or rendered_frames % 100 == 0:
                print(f"[CAPTURE] frame={rendered_frames-1:06d} sim_t={sim_time_s:7.2f}s zone={current_zone:<28} png={png_path.stat().st_size} B", flush=True)

            if (
                stop_reason is not None
                and stop_trigger_time_s is not None
                and sim_time_s - stop_trigger_time_s >= float(args_cli.post_event_hold_s)
            ):
                print(
                    f"[AUTO-STOP] reason={stop_reason} trigger_t={stop_trigger_time_s:.2f}s "
                    f"end_t={sim_time_s:.2f}s trigger_x={stop_trigger_x:+.3f}", flush=True,
                )
                break

        if stop_reason is None:
            raise RuntimeError("Kit closed before semantic auto-stop; partial PNGs retained and no final MP4 certified")
        if rendered_frames <= 0:
            raise RuntimeError("No frames captured")
        duration_s = rendered_frames / EXPECTED_FPS
        sim_duration_s = rendered_frames * EXPECTED_DT
        if abs(duration_s - sim_duration_s) > 1.0e-9:
            raise RuntimeError(f"Timing mismatch: video={duration_s}, sim={sim_duration_s}")

        encode_final_mp4(ffmpeg_path, frames_dir, video_path)
        manifest = {
            "status": "PASS",
            "classification": "final qualitative OOD graded-rough-limit demonstration; frozen flat-trained v4-N; not terrain-trained benchmark; not ROS deployment evidence",
            "frozen_preview_v9_sha256": FROZEN_PREVIEW_V9_SHA256,
            "final_renderer_sha256": sha256_file(Path(__file__).resolve()),
            "task_id": TASK_ID, "checkpoint": str(CHECKPOINT), "checkpoint_sha256": checkpoint_sha,
            "seed": int(args_cli.seed), "command": [float(args_cli.command_vx), 0.0, 0.0],
            "physics_dt_s": EXPECTED_DT, "fps": EXPECTED_FPS, "frames": rendered_frames,
            "duration_s": duration_s, "resolution": list(actual_viewport_resolution),
            "full_viewport_resolution": list(full_viewport_resolution), "viewport_scale": viewport_scale,
            "render": {
                "aa_op": render_settings.get("/rtx/post/aa/op"),
                "motion_blur": render_settings.get("/rtx/post/motionblur/enabled"),
                "dlss_frame_generation": render_settings.get("/rtx-transient/dlssg/enabled"),
                "clean_fullscreen": True, "png_per_physics_step": True,
                "max_capture_updates_seen": max_capture_updates_seen,
            },
            "course": {"z5_removed": True, "zones": zones, "final_zone": zones[-1]["name"], "finish_trigger_x": finish_trigger_x},
            "camera": {
                "orbit_start_s": float(args_cli.camera_orbit_start_s),
                "orbit_rate_deg_s": float(args_cli.camera_orbit_rate_deg_s),
                "position_smoothing_s": float(args_cli.camera_position_smoothing_s),
                "forward_offset_m": float(args_cli.camera_forward_offset_m),
                "left_offset_m": float(args_cli.camera_left_offset_m),
                "eye_z_m": float(args_cli.camera_eye_z_m), "target_z_m": float(args_cli.camera_target_z_m),
                "target_lookahead_m": float(args_cli.camera_target_lookahead_m), "radius_lock": True,
            },
            "visual": {
                "floor_rgb": list(APPROVED_GRAPHITE_RGB), "floor_roughness": APPROVED_GRAPHITE_ROUGHNESS,
                "rough_zone_colors_restored": True, "softbox_intensity": float(args_cli.softbox_intensity),
                "existing_isaac_task_lights_preserved": True,
            },
            "auto_stop": {
                "reason": stop_reason, "trigger_time_s": stop_trigger_time_s, "trigger_x": stop_trigger_x,
                "post_event_hold_s": float(args_cli.post_event_hold_s), "final_sim_time_s": sim_duration_s,
                "fall_height_drop_m": float(args_cli.fall_height_drop_m),
                "fall_up_z_threshold": float(args_cli.fall_up_z_threshold), "fall_confirm_s": float(args_cli.fall_confirm_s),
                "stall_window_s": float(args_cli.stall_window_s), "stall_min_progress_m": float(args_cli.stall_min_progress_m),
                "safety_timeout_s": float(args_cli.safety_timeout_s),
            },
            "encoding": {"ffmpeg": ffmpeg_path, "codec": "libx264", "preset": "slow", "crf": 16, "pixel_format": "yuv420p", "faststart": True},
            "paths": {"output_root": str(output_root), "frames_dir": str(frames_dir), "video_path": str(video_path), "manifest_path": str(manifest_path)},
        }
        manifest["video_sha256"] = sha256_file(video_path)
        manifest["video_bytes"] = int(video_path.stat().st_size)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("=" * 132)
        print("G1 V07 FINAL ROUGH-LIMIT VIDEO — PASS")
        print(f"Stop reason                     : {stop_reason}")
        print(f"Frames / duration               : {rendered_frames} / {duration_s:.3f} s @ {EXPECTED_FPS} fps")
        print(f"Video                           : {video_path}")
        print(f"Video SHA256                    : {manifest['video_sha256']}")
        print(f"Manifest                        : {manifest_path}")
        print("=" * 132, flush=True)
        print(PASS_MARKER, flush=True)

    finally:
        try:
            if env is not None:
                env.close()
            else:
                raw_env.close()
        except Exception:
            pass


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
