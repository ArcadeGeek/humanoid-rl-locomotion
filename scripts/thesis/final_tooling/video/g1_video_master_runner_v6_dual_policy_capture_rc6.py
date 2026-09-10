#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

parser = argparse.ArgumentParser(
    description=(
        "Final G1 ROS-to-Isaac dual-policy deployment/video runner. "
        "Frozen v2-G handles STAND, frozen v4-N handles LOCOMOTION and the 0.36 s BRAKE dwell; "
        "the validated v5.42 ROS/network/presentation layer is preserved."
    )
)
parser.add_argument(
    "--task-id",
    type=str,
    default="Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-v0",
    help="Shared physical runtime task. The final controller requires the frozen v2-G task.",
)
parser.add_argument(
    "--stand-checkpoint",
    type=Path,
    default=Path(r"C:\humanoid_rl_lab\checkpoints\local_g1_yaw_direct_wide_yaw_flat_orientation_v2g\model_600.pt"),
    help="Frozen v2-G STAND checkpoint.",
)
parser.add_argument(
    "--stand-expected-sha256",
    type=str,
    default="58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3",
    help="Expected frozen v2-G STAND checkpoint SHA256.",
)
parser.add_argument(
    "--loco-checkpoint",
    type=Path,
    default=Path(
        r"C:\humanoid_rl_lab\logs\rsl_rl\humanoid_rl_lab_g1_v4n_symmetry_augmentation_scratch"
        r"\2026-09-04_13-58-13_v4n_genuine_scratch_seed42_official1500\model_1499.pt"
    ),
    help="Frozen v4-N LOCOMOTION checkpoint.",
)
parser.add_argument(
    "--loco-expected-sha256",
    type=str,
    default="367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342",
    help="Expected frozen v4-N LOCOMOTION checkpoint SHA256.",
)
parser.add_argument("--output-dir", type=Path, required=True, help="Dedicated deployment/video output directory.")
parser.add_argument(
    "--policy-label",
    type=str,
    default="frozen_v2g_stand_v4n_locomotion_final",
    help="Human-readable integrated-controller label for provenance.",
)
parser.add_argument(
    "--validation-profile",
    choices=("generic", "straight", "curve-pos", "curve-neg", "full-soak"),
    default="generic",
    help=(
        "Functional command profile expected in this run. "
        "Use straight for V01, curve-pos for V02, curve-neg for V03, full-soak for the integrated sequence."
    ),
)
parser.add_argument(
    "--capture-trajectory",
    action="store_true",
    help=(
        "Record the post-physics G1 state at every 50 Hz control step after the first fresh canonical command. "
        "The capture is GPU-buffered during runtime and transferred/written only after the run."
    ),
)
parser.add_argument(
    "--trajectory-capture-path",
    type=Path,
    default=None,
    help="Optional .pt output path. Default: <output-dir>/g1_trajectory_<policy-label>.pt",
)
parser.add_argument("--max-runtime-s", type=float, default=180.0)
parser.add_argument("--watchdog-s", type=float, default=0.50)
parser.add_argument("--status-hz", type=float, default=5.0)
parser.add_argument("--telemetry-hz", type=float, default=20.0)
parser.add_argument("--real-time", action="store_true", default=True)
parser.add_argument("--spin-margin-ms", type=float, default=1.0)
parser.add_argument(
    "--camera-preset",
    choices=("three-quarter", "side", "front-quarter"),
    default="three-quarter",
    help="Automatic stabilized viewport framing relative to robot yaw.",
)
parser.add_argument("--camera-update-hz", type=float, default=50.0)
parser.add_argument(
    "--camera-position-smoothing-s",
    type=float,
    default=0.80,
    help="Low-pass time constant for camera translation. Larger = steadier/slower follow.",
)
parser.add_argument(
    "--camera-heading-smoothing-s",
    type=float,
    default=1.20,
    help="Low-pass time constant for camera heading. Heading is estimated from travel direction, not pelvis yaw.",
)
parser.add_argument(
    "--camera-velocity-smoothing-s",
    type=float,
    default=0.60,
    help="Low-pass time constant for horizontal robot velocity used to estimate travel heading.",
)
parser.add_argument(
    "--camera-heading-speed-threshold",
    type=float,
    default=0.06,
    help="Below this horizontal speed [m/s], hold the current camera heading.",
)
parser.add_argument(
    "--camera-heading-mode",
    choices=("fixed", "travel"),
    default="fixed",
    help=(
        "fixed: keep the initial world camera angle while translating with the robot; "
        "travel: slowly orbit to remain 3/4 relative to travel direction."
    ),
)
parser.add_argument("--camera-eye-z", type=float, default=2.00)
parser.add_argument("--camera-target-z", type=float, default=0.70)
parser.add_argument(
    "--video-preset",
    choices=("legacy", "demo", "science-side", "science-high"),
    default="legacy",
    help=(
        "Visual presentation preset only. legacy preserves v4 camera settings; "
        "demo uses a polished 3/4 travel-follow view; science-side uses a fixed side view; "
        "science-high uses a fixed elevated 3/4 view."
    ),
)
parser.add_argument(
    "--clean-viewport",
    action="store_true",
    help="Start Kit as a borderless fullscreen viewport with UI/docks hidden for native-resolution OBS capture.",
)
parser.add_argument("--video-width", type=int, default=1920)
parser.add_argument("--video-height", type=int, default=1080)
parser.add_argument(
    "--presentation-lighting",
    choices=("off", "science", "demo", "softbox"),
    default="off",
    help=(
        "Visual-only USD lighting. softbox adds one local RectLight over/front of the robot "
        "without DomeLight or DistantLight."
    ),
)
parser.add_argument(
    "--render-quality",
    choices=("legacy", "video-dlss", "video-native"),
    default="legacy",
    help=(
        "video-dlss = DLSS Quality; video-native = DLAA at the actual viewport resolution. "
        "Both disable motion blur and DLSS Frame Generation."
    ),
)
parser.add_argument(
    "--presentation-floor",
    choices=("off", "science", "demo"),
    default="off",
    help=(
        "Add a visual-only matte overlay floor above the original terrain to hide the training grid. "
        "No collision or physics API is authored."
    ),
)
parser.add_argument(
    "--ground-look",
    choices=("original", "science", "demo"),
    default="original",
    help=(
        "Rebind only the material of the REAL visual ground mesh "
        "/World/ground/terrain/Environment/Geometry. Collision geometry and physics material are untouched."
    ),
)
parser.add_argument(
    "--studio-backdrop",
    choices=("off", "demo"),
    default="off",
    help=(
        "Add a distant visual-only circular cyclorama beginning 14 m from the origin. "
        "It never sits under the robot and has no collision/physics APIs."
    ),
)
parser.add_argument(
    "--background-wall",
    choices=("off", "clean"),
    default="off",
    help=(
        "Add one distant visual-only matte background wall at x=45 m. "
        "It is flat, far outside the locomotion/contact region, and has no collision/physics APIs."
    ),
)
parser.add_argument(
    "--showcase-scenery",
    choices=("off", "rough-dark"),
    default="off",
    help=(
        "Add deterministic Isaac Lab ROUGH_TERRAINS_CFG scenery outside a wide flat safe lane. "
        "Scenery collisions are explicitly disabled; the G1 still walks only on the original flat ground."
    ),
)
parser.add_argument(
    "--showcase-shift-x",
    type=float,
    default=0.0,
    help=(
        "Translate the ENTIRE visual showcase arena along world X without moving the G1 or the "
        "authoritative flat collision plane. Positive values move the arena farther into the "
        "current fixed-camera background."
    ),
)
parser.add_argument(
    "--demo-camera-distance-scale",
    type=float,
    default=1.08,
    help="Horizontal distance multiplier for the demo 3/4 camera only; <1 brings the camera closer.",
)
parser.add_argument(
    "--demo-camera-forward-offset",
    type=float,
    default=-3.95,
    help="DEMO camera horizontal forward offset [m] in stabilized robot-heading frame.",
)
parser.add_argument(
    "--demo-camera-left-offset",
    type=float,
    default=-1.55,
    help="DEMO camera horizontal left offset [m] in stabilized robot-heading frame.",
)
parser.add_argument(
    "--demo-camera-eye-z",
    type=float,
    default=1.58,
    help="DEMO camera eye height [m].",
)
parser.add_argument(
    "--demo-camera-target-z",
    type=float,
    default=0.92,
    help="DEMO camera target height [m].",
)
parser.add_argument(
    "--local-preview-sequence",
    action="store_true",
    help=(
        "Quick Windows-only visual preview: bypass external ROS commands and drive the existing "
        "base_velocity command tensor directly with zero -> vx=0.40 -> zero. "
        "This is NOT deployment evidence and is disabled by default."
    ),
)
parser.add_argument("--local-preview-zero-before-s", type=float, default=3.0)
parser.add_argument("--local-preview-walk-s", type=float, default=12.0)
parser.add_argument("--local-preview-zero-after-s", type=float, default=5.0)

from isaaclab.app import AppLauncher
parser.add_argument(
    "--free-camera",
    action="store_true",
    help=(
        "Environment-inspection mode: set the normal camera pose only once, then stop automatic "
        "follow-camera updates so the Kit viewport camera can be moved manually."
    ),
)

parser.add_argument(
    "--demo-camera-target-lookahead",
    type=float,
    default=0.20,
    help=(
        "Forward offset [m] of the camera target relative to the smoothed robot center. "
        "Use 0.0 for true robot centering in the final demo composition."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# VIDEO-MASTER VISUAL PRESETS.
# Camera/presentation only: policy, physics, command semantics and ROS transport remain unchanged.
if args_cli.video_width <= 0 or args_cli.video_height <= 0:
    parser.error("--video-width and --video-height must be positive")

for _preview_name in (
    "local_preview_zero_before_s",
    "local_preview_walk_s",
    "local_preview_zero_after_s",
):
    if float(getattr(args_cli, _preview_name)) < 0.0:
        parser.error(f"--{_preview_name.replace('_', '-')} must be >= 0")

if args_cli.video_preset == "demo":
    args_cli.camera_preset = "three-quarter"
    args_cli.camera_heading_mode = "fixed"
    args_cli.camera_update_hz = 50.0
    args_cli.camera_position_smoothing_s = 0.85
    args_cli.camera_heading_smoothing_s = 1.50
    args_cli.camera_velocity_smoothing_s = 0.65
    args_cli.camera_heading_speed_threshold = 0.06
    args_cli.camera_eye_z = float(args_cli.demo_camera_eye_z)
    args_cli.camera_target_z = float(args_cli.demo_camera_target_z)
elif args_cli.video_preset == "science-side":
    args_cli.camera_preset = "side"
    args_cli.camera_heading_mode = "fixed"
    args_cli.camera_update_hz = 50.0
    args_cli.camera_position_smoothing_s = 0.65
    args_cli.camera_heading_smoothing_s = 1.20
    args_cli.camera_velocity_smoothing_s = 0.60
    args_cli.camera_heading_speed_threshold = 0.06
    args_cli.camera_eye_z = 1.65
    args_cli.camera_target_z = 0.78
elif args_cli.video_preset == "science-high":
    args_cli.camera_preset = "three-quarter"
    args_cli.camera_heading_mode = "fixed"
    args_cli.camera_update_hz = 50.0
    args_cli.camera_position_smoothing_s = 0.70
    args_cli.camera_heading_smoothing_s = 1.20
    args_cli.camera_velocity_smoothing_s = 0.60
    args_cli.camera_heading_speed_threshold = 0.06
    args_cli.camera_eye_z = 4.20
    args_cli.camera_target_z = 0.55

if args_cli.clean_viewport:
    # SimulationApp width/height control viewport/generated-image resolution.
    # window_width/window_height control the application window independently.
    args_cli.width = int(args_cli.video_width)
    args_cli.height = int(args_cli.video_height)
    args_cli.window_width = int(args_cli.video_width)
    args_cli.window_height = int(args_cli.video_height)

    clean_args = " ".join([
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
    existing_kit_args = str(getattr(args_cli, "kit_args", "") or "").strip()
    args_cli.kit_args = (existing_kit_args + " " + clean_args).strip()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb.settings

# Headless timing/profiling runs do not load the Kit viewport extension.
# Keep the visual path byte-for-byte equivalent when that extension exists, but
# allow the same controller/runtime to execute without a viewport when Isaac is headless.
try:
    from omni.kit.viewport.utility import get_active_viewport
except ModuleNotFoundError:
    get_active_viewport = None

_render_settings = carb.settings.get_settings()

if get_active_viewport is None:
    if args_cli.clean_viewport:
        raise RuntimeError("--clean-viewport requires the Kit viewport extension; unavailable in headless mode")
    if args_cli.render_quality != "legacy":
        raise RuntimeError("non-legacy --render-quality requires the Kit viewport extension; unavailable in headless mode")
    _active_viewport = None
    _actual_viewport_resolution = (0, 0)
    _full_viewport_resolution = (0, 0)
    _viewport_scale = 1.0
    _expected_viewport_resolution = (0, 0)
    _viewport_resolution_pass = True
    _viewport_available = False
else:
    # Give Kit a few frames to create the active viewport.
    _active_viewport = None
    for _ in range(12):
        simulation_app.update()
        _active_viewport = get_active_viewport()
        if _active_viewport is not None:
            break

    if _active_viewport is None:
        raise RuntimeError("Could not obtain active Kit viewport after startup")

    if args_cli.clean_viewport:
        _active_viewport.set_texture_resolution(
            (int(args_cli.video_width), int(args_cli.video_height))
        )
        _active_viewport.set_texture_resolution_scale(1.0)
        for _ in range(4):
            simulation_app.update()

    _actual_viewport_resolution = tuple(
        int(round(v)) for v in _active_viewport.get_texture_resolution()
    )
    _full_viewport_resolution = tuple(
        int(round(v)) for v in _active_viewport.get_full_texture_resolution()
    )
    _viewport_scale = float(_active_viewport.get_texture_resolution_scale())

    _expected_viewport_resolution = (
        (int(args_cli.video_width), int(args_cli.video_height))
        if args_cli.clean_viewport
        else _actual_viewport_resolution
    )

    _viewport_resolution_pass = (
        _actual_viewport_resolution == _expected_viewport_resolution
        and abs(_viewport_scale - 1.0) <= 1.0e-6
    )

    if args_cli.clean_viewport and not _viewport_resolution_pass:
        raise RuntimeError(
            "Active viewport render-target mismatch: "
            f"expected={_expected_viewport_resolution}, "
            f"actual={_actual_viewport_resolution}, "
            f"full={_full_viewport_resolution}, scale={_viewport_scale}"
        )
    _viewport_available = True

render_quality_report = {
    "mode": args_cli.render_quality,
    "viewport_available": bool(_viewport_available),
    "viewport_resolution_expected": list(_expected_viewport_resolution),
    "viewport_resolution_runtime": list(_actual_viewport_resolution),
    "viewport_full_resolution_runtime": list(_full_viewport_resolution),
    "viewport_resolution_scale_runtime": _viewport_scale,
    "viewport_resolution_pass": bool(_viewport_resolution_pass),
}

if args_cli.render_quality in ("video-dlss", "video-native"):
    # Keep captured motion physical: no generated display frames or cinematic motion blur.
    _render_settings.set("/rtx-transient/dlssg/enabled", False)
    _render_settings.set("/rtx/post/motionblur/enabled", False)

    if args_cli.render_quality == "video-native":
        # Native-resolution NVIDIA DLAA.
        _render_settings.set("/rtx/post/aa/op", 4)
    else:
        # DLSS Super Resolution, Quality mode.
        _render_settings.set("/rtx/post/aa/op", 3)
        _render_settings.set("/rtx/post/dlss/execMode", 2)

    for _ in range(3):
        simulation_app.update()

render_quality_report.update({
    "aa_op_runtime": _render_settings.get("/rtx/post/aa/op"),
    "dlss_exec_mode_runtime": _render_settings.get("/rtx/post/dlss/execMode"),
    "motion_blur_runtime": _render_settings.get("/rtx/post/motionblur/enabled"),
    "dlss_frame_generation_runtime": _render_settings.get("/rtx-transient/dlssg/enabled"),
})

TASK_ID = args_cli.task_id
EXPECTED_STAND_TASK = "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-v0"
LOCO_TASK_LABEL = "Humanoid-RL-Lab-G1-Flat-V4N-Symmetry-Augmentation-v0"

STAND_CHECKPOINT = Path(args_cli.stand_checkpoint)
STAND_EXPECTED_SHA256 = str(args_cli.stand_expected_sha256).upper()
LOCO_CHECKPOINT = Path(args_cli.loco_checkpoint)
LOCO_EXPECTED_SHA256 = str(args_cli.loco_expected_sha256).upper()

OUT_DIR = Path(args_cli.output_dir)
POLICY_LABEL = args_cli.policy_label
OUT_DIR.mkdir(parents=True, exist_ok=True)

CMD_TOPIC = "/g1/deployment/cmd_vel"
ECHO_TOPIC = "/g1/deployment/cmd_echo"
STATUS_TOPIC = "/g1/deployment/status"
BASE_TWIST_BODY_TOPIC = "/g1/deployment/base_twist_body"
BASE_POSE_TOPIC = "/g1/deployment/base_pose"
STOP_TOPIC = "/g1/deployment/stop"

SCI_CMD = (0.40, 0.0, 0.0)
CURVE_POS_CMD = (0.40, 0.0, 0.25)
CURVE_NEG_CMD = (0.40, 0.0, -0.25)
ZERO_CMD = (0.0, 0.0, 0.0)
CMD_TOL = 1.0e-4
EXPECTED_SOURCE_FRAME_ID = "g1_cmd_source_linux_phase10b_canonical"

EXPECTED_OBS_DIM = 123
EXPECTED_ACTION_DIM = 37
EXPECTED_STEP_DT = 0.02

# Frozen integrated architecture.
ENTER_LOCO_SPEED = 0.20
EXIT_LOCO_SPEED = 0.10
BRAKE_S = 0.36
BRAKE_STEPS = 18

report = {
    "phase": "FINAL-DUAL-POLICY-DEPLOYMENT-VIDEO",
    "purpose": "canonical ROS2-to-Isaac deployment/video runner for the frozen v2-G + v4-N integrated controller",
    "policy_label": POLICY_LABEL,
    "validation_profile": args_cli.validation_profile,
    "scientific_command": list(SCI_CMD),
    "task_id": TASK_ID,
    "stand_task": EXPECTED_STAND_TASK,
    "loco_task_label": LOCO_TASK_LABEL,
    "stand_checkpoint": str(STAND_CHECKPOINT),
    "stand_expected_sha256": STAND_EXPECTED_SHA256,
    "loco_checkpoint": str(LOCO_CHECKPOINT),
    "loco_expected_sha256": LOCO_EXPECTED_SHA256,
    "training_run": False,
    "repository_source_modified": False,
    "checkpoint_modified": False,
    "trajectory_capture_requested": bool(args_cli.capture_trajectory),
    "trajectory_capture_path_requested": (
        str(args_cli.trajectory_capture_path) if args_cli.trajectory_capture_path is not None else None
    ),
    "trajectory_capture_contract": (
        "post-physics state at every controller step; starts on first fresh canonical ROS command; "
        "preallocated GPU buffer during runtime; one host transfer/write after rollout"
    ),
    "controller_architecture": {
        "stand_policy": "v2-G",
        "locomotion_policy": "v4-N",
        "enter_loco_speed_mps": ENTER_LOCO_SPEED,
        "exit_loco_speed_mps": EXIT_LOCO_SPEED,
        "brake_s": BRAKE_S,
        "brake_steps": BRAKE_STEPS,
        "brake_policy": "v4-N",
        "brake_policy_command": list(ZERO_CMD),
        "blending": False,
        "startup_contract": "nominal/quasi-static; reset pose and all six root reset velocities forced to zero",
    },
    "accepted_commands_only": [list(ZERO_CMD), list(SCI_CMD), list(CURVE_POS_CMD), list(CURVE_NEG_CMD)],
    "scientific_vs_engineering": {
        "straight_040": "scientific",
        "curve_pos_040_025": "engineering",
        "curve_neg_040_m025": "engineering",
    },
    "classification": "hardware-free end-to-end deployment rehearsal / final video runner",
    "reward_bypass_deployment_only": True,
    "timing_clock": "time.perf_counter / QueryPerformanceCounter",
    "pacing": "absolute 20ms deadlines + sleep + final spin",
    "network_path": "direct Ethernet 10.42.0.0/24",
    "expected_windows_ip": "10.42.0.1",
    "expected_linux_ip": "10.42.0.2",
    "physics_config_modified": False,
    "policy_modified": False,
    "errors": [],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def as_tensor(x):
    return x.torch if hasattr(x, "torch") else x


def robot_vec(robot, name):
    return as_tensor(getattr(robot.data, name))[0].detach().cpu()


def near_cmd(vals, target):
    return max(abs(a - b) for a, b in zip(vals, target)) <= CMD_TOL


class RouterState(Enum):
    STAND = auto()
    LOCO = auto()
    BRAKE = auto()


@dataclass(frozen=True)
class RouterOutput:
    state: RouterState
    active_policy: str
    brake_index: int


class Router:
    """Frozen router logic validated by the development + untouched-holdout campaign."""

    def __init__(self, enter_loco_speed: float, exit_loco_speed: float, brake_steps: int) -> None:
        if not (0.0 <= exit_loco_speed < enter_loco_speed):
            raise ValueError("require 0 <= EXIT_LOCO_SPEED < ENTER_LOCO_SPEED")
        if brake_steps <= 0:
            raise ValueError("brake_steps must be positive")
        self.enter = float(enter_loco_speed)
        self.exit = float(exit_loco_speed)
        self.brake_steps = int(brake_steps)
        self.state = RouterState.STAND
        self._brake_followup_count = 0

    def update(self, requested_vx: float) -> RouterOutput:
        # Thresholds were calibrated and validated on |vx|. Do not silently redefine
        # the frozen router as a planar-speed or 3-D command-norm router.
        speed = abs(float(requested_vx))

        if self.state is RouterState.STAND:
            if speed >= self.enter:
                self.state = RouterState.LOCO

        elif self.state is RouterState.LOCO:
            if speed <= self.exit:
                self.state = RouterState.BRAKE
                self._brake_followup_count = 0

        elif self.state is RouterState.BRAKE:
            if speed >= self.enter:
                self.state = RouterState.LOCO
                self._brake_followup_count = 0
            else:
                self._brake_followup_count += 1
                if self._brake_followup_count >= self.brake_steps:
                    self.state = RouterState.STAND
                    self._brake_followup_count = 0
        else:
            raise RuntimeError(f"unknown router state {self.state}")

        if self.state is RouterState.STAND:
            return RouterOutput(self.state, "v2-G", 0)
        if self.state is RouterState.LOCO:
            return RouterOutput(self.state, "v4-N", 0)

        # BRAKE: v4-N stays active at ZERO command for exactly 18 control steps.
        return RouterOutput(
            self.state,
            "v4-N",
            self._brake_followup_count + 1,
        )


def extract_policy_observation_tensor(obs):
    # RSL-RL 5.x may return a TensorDict whose batch shape is (1,).
    # Resolve the named policy field before consulting generic .shape; this is
    # the same contract used by the validated v2-G/v4-N runtime harness.
    candidate = None
    if torch.is_tensor(obs) or hasattr(obs, "torch"):
        candidate = obs
    else:
        try:
            candidate = obs["policy"]
        except Exception:
            get_fn = getattr(obs, "get", None)
            if callable(get_fn):
                try:
                    candidate = get_fn("policy")
                except Exception:
                    candidate = None

    if candidate is None:
        raise RuntimeError(f"Cannot resolve policy observation from {type(obs).__name__}")

    candidate = as_tensor(candidate)
    if candidate.ndim != 2 or candidate.shape[0] != 1:
        raise RuntimeError(f"Unexpected policy observation shape: {tuple(candidate.shape)}")
    return candidate


def quat_wxyz_to_yaw(q):
    """Return world yaw [rad] from Isaac quaternion ordered w,x,y,z."""
    w, x, y, z = (float(v) for v in q)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def camera_body_offsets(preset):
    # (forward, left) offsets in the robot-yaw frame, metres.
    # Camera altitude/target altitude are controlled separately.
    if preset == "side":
        return (0.0, -4.30)
    if preset == "front-quarter":
        return (3.55, -2.35)
    # Default: stable rear-right three-quarter view.
    return (-3.55, -2.35)


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



def configure_presentation_floor(mode):
    # Visual-only overlay. It deliberately has no collision/rigid-body/physics API.
    if mode == "off":
        return {"mode": "off", "authored": False, "physics_modified": False}

    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    root_path = Sdf.Path("/World/VideoPresentation")
    stage.DefinePrim(root_path, "Xform")

    floor_path = root_path.AppendChild("Floor")
    material_path = root_path.AppendChild("FloorMaterial")
    shader_path = material_path.AppendChild("PreviewSurface")

    for p in (floor_path, material_path):
        if stage.GetPrimAtPath(p).IsValid():
            raise RuntimeError(f"Presentation floor path already exists: {p}")

    # 160 x 160 m is enough for every planned visual run.
    half = 80.0
    z = 0.0025
    mesh = UsdGeom.Mesh.Define(stage, floor_path)
    mesh.CreatePointsAttr([
        Gf.Vec3f(-half, -half, z),
        Gf.Vec3f( half, -half, z),
        Gf.Vec3f( half,  half, z),
        Gf.Vec3f(-half,  half, z),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)])
    mesh.SetNormalsInterpolation("constant")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")

    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")

    if mode == "science":
        diffuse = Gf.Vec3f(0.19, 0.20, 0.22)
        roughness = 0.82
        metallic = 0.0
    else:
        # Dark neutral graphite: presentation-friendly but still technically clean.
        diffuse = Gf.Vec3f(0.030, 0.040, 0.055)
        roughness = 0.82
        metallic = 0.00

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diffuse)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    # Explicitly verify that the visual floor has no collision/rigid-body schemas.
    applied = [str(x) for x in mesh.GetPrim().GetAppliedSchemas()]
    forbidden = [
        s for s in applied
        if "CollisionAPI" in s or "RigidBodyAPI" in s or "MassAPI" in s
    ]
    if forbidden:
        raise RuntimeError(f"Visual presentation floor unexpectedly has physics schemas: {forbidden}")

    return {
        "mode": mode,
        "authored": True,
        "floor_prim": str(floor_path),
        "material_prim": str(material_path),
        "z_m": z,
        "size_m": 2.0 * half,
        "roughness": roughness,
        "metallic": metallic,
        "applied_schemas": applied,
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


def configure_studio_backdrop(mode):
    if mode == "off":
        return {
            "mode": "off",
            "authored": False,
            "physics_modified": False,
        }

    import math as _math
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    root_path = Sdf.Path("/World/VideoPresentation")
    stage.DefinePrim(root_path, "Xform")

    mesh_path = root_path.AppendChild("StudioCyclorama")
    material_path = root_path.AppendChild("StudioCycloramaMaterial")
    shader_path = material_path.AppendChild("PreviewSurface")

    for p in (mesh_path, material_path):
        if stage.GetPrimAtPath(p).IsValid():
            raise RuntimeError(f"Studio backdrop path already exists: {p}")

    # Circular cyclorama:
    # - starts at radius 14 m at ground level;
    # - curves smoothly upward with an 8 m quarter-circle;
    # - then continues vertically to 18 m.
    #
    # The G1 travels only a few metres in our preview/final sequences, so this surface
    # stays far away and never replaces the real ground/contact region.
    angular_segments = 72
    curve_segments = 14
    r0 = 14.0
    curve_radius = 8.0
    top_z = 18.0

    profile = []
    for j in range(curve_segments + 1):
        phi = 0.5 * _math.pi * (j / curve_segments)
        r = r0 + curve_radius * _math.sin(phi)
        z = curve_radius * (1.0 - _math.cos(phi)) + 0.002
        profile.append((r, z))

    # Vertical continuation.
    profile.append((r0 + curve_radius, top_z))

    points = []
    for i in range(angular_segments):
        theta = 2.0 * _math.pi * i / angular_segments
        c = _math.cos(theta)
        s = _math.sin(theta)
        for r, z in profile:
            points.append(Gf.Vec3f(r * c, r * s, z))

    pcount = len(profile)
    face_counts = []
    face_indices = []
    for i in range(angular_segments):
        ni = (i + 1) % angular_segments
        for j in range(pcount - 1):
            a = i * pcount + j
            b = ni * pcount + j
            cidx = ni * pcount + (j + 1)
            d = i * pcount + (j + 1)
            face_counts.append(4)
            face_indices.extend([a, b, cidx, d])

    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")

    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")

    # Match the real demo ground closely so the transition reads as a studio sweep,
    # while curvature + lighting create the tonal gradient naturally.
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.075, 0.090, 0.115)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.84)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    applied = [str(x) for x in mesh.GetPrim().GetAppliedSchemas()]
    forbidden = [
        s for s in applied
        if "CollisionAPI" in s or "RigidBodyAPI" in s or "MassAPI" in s
    ]
    if forbidden:
        raise RuntimeError(f"Studio backdrop unexpectedly has physics schemas: {forbidden}")

    return {
        "mode": mode,
        "authored": True,
        "mesh_prim": str(mesh_path),
        "material_prim": str(material_path),
        "inner_radius_m": r0,
        "outer_radius_m": r0 + curve_radius,
        "top_z_m": top_z,
        "angular_segments": angular_segments,
        "curve_segments": curve_segments,
        "physics_modified": False,
        "collision_modified": False,
        "ground_contact_region_modified": False,
        "applied_schemas": applied,
    }


def configure_clean_background_wall(mode):
    if mode == "off":
        return {
            "mode": "off",
            "authored": False,
            "physics_modified": False,
        }

    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    root_path = Sdf.Path("/World/VideoPresentation")
    stage.DefinePrim(root_path, "Xform")

    wall_path = root_path.AppendChild("CleanBackgroundWall")
    material_path = root_path.AppendChild("CleanBackgroundWallMaterial")
    shader_path = material_path.AppendChild("PreviewSurface")

    for p in (wall_path, material_path):
        if stage.GetPrimAtPath(p).IsValid():
            raise RuntimeError(f"Background wall path already exists: {p}")

    # Fixed demo camera looks predominantly toward +X. Put the wall very far ahead,
    # so a 20–40 s locomotion clip cannot reach it and the robot cannot cast a shadow onto it.
    x = 45.0
    half_y = 60.0
    top_z = 28.0

    mesh = UsdGeom.Mesh.Define(stage, wall_path)
    mesh.CreatePointsAttr([
        Gf.Vec3f(x, -half_y, 0.0),
        Gf.Vec3f(x,  half_y, 0.0),
        Gf.Vec3f(x,  half_y, top_z),
        Gf.Vec3f(x, -half_y, top_z),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")

    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")

    # Neutral blue-gray technical wall. A small emissive lift prevents it from
    # collapsing into black without turning it into a self-lit backdrop.
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.115, 0.135, 0.175)
    )
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.018, 0.024, 0.036)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.94)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    applied = [str(x) for x in mesh.GetPrim().GetAppliedSchemas()]
    forbidden = [
        s for s in applied
        if "CollisionAPI" in s or "RigidBodyAPI" in s or "MassAPI" in s
    ]
    if forbidden:
        raise RuntimeError(f"Background wall unexpectedly has physics schemas: {forbidden}")

    return {
        "mode": mode,
        "authored": True,
        "wall_prim": str(wall_path),
        "material_prim": str(material_path),
        "wall_x_m": x,
        "wall_half_width_m": half_y,
        "wall_height_m": top_z,
        "physics_modified": False,
        "collision_modified": False,
        "ground_contact_region_modified": False,
        "applied_schemas": applied,
    }


_SHOWCASE_TERRAIN_IMPORTERS = []


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



def summarize_ms(values):
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return None

    def pct(p):
        if len(vals) == 1:
            return vals[0]
        x = (len(vals) - 1) * p
        lo = int(math.floor(x))
        hi = int(math.ceil(x))
        if lo == hi:
            return vals[lo]
        f = x - lo
        return vals[lo] * (1.0 - f) + vals[hi] * f

    return {
        "n": len(vals),
        "mean_ms": statistics.fmean(vals),
        "median_ms": statistics.median(vals),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "min_ms": vals[0],
        "max_ms": vals[-1],
    }


class RosState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_cmd = ZERO_CMD
        self.last_valid_rx = None
        self.valid_rx_count = 0
        self.valid_zero_count = 0
        self.valid_040_count = 0
        self.valid_curve_pos_count = 0
        self.valid_curve_neg_count = 0
        self.rejected_command_count = 0
        self.invalid_count = 0
        self.source_mismatch_count = 0
        self.max_cmd_publishers_seen = 0
        self.stop_requested = False
        self.watchdog_after_fresh = False
        self.have_ever_been_fresh = False

    def snapshot(self):
        with self.lock:
            return {
                "latest_cmd": tuple(self.latest_cmd),
                "last_valid_rx": self.last_valid_rx,
                "valid_rx_count": self.valid_rx_count,
                "valid_zero_count": self.valid_zero_count,
                "valid_040_count": self.valid_040_count,
                "valid_curve_pos_count": self.valid_curve_pos_count,
                "valid_curve_neg_count": self.valid_curve_neg_count,
                "rejected_command_count": self.rejected_command_count,
                "invalid_count": self.invalid_count,
                "source_mismatch_count": self.source_mismatch_count,
                "max_cmd_publishers_seen": self.max_cmd_publishers_seen,
                "stop_requested": self.stop_requested,
                "watchdog_after_fresh": self.watchdog_after_fresh,
                "have_ever_been_fresh": self.have_ever_been_fresh,
            }


env = None
ros_node = None
ros_executor = None
ros_thread = None
rclpy = None

try:
    if TASK_ID != EXPECTED_STAND_TASK:
        raise RuntimeError(
            f"Final dual-policy runtime requires task {EXPECTED_STAND_TASK!r}, got {TASK_ID!r}"
        )
    if BRAKE_STEPS != int(round(BRAKE_S / EXPECTED_STEP_DT)):
        raise RuntimeError("Frozen BRAKE guardrail failed: 0.36 s must equal 18 control steps")

    for label, path in (("v2-G STAND", STAND_CHECKPOINT), ("v4-N LOCOMOTION", LOCO_CHECKPOINT)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} checkpoint missing: {path}")

    stand_actual_sha = sha256_file(STAND_CHECKPOINT)
    loco_actual_sha = sha256_file(LOCO_CHECKPOINT)
    stand_sha_pass = stand_actual_sha == STAND_EXPECTED_SHA256
    loco_sha_pass = loco_actual_sha == LOCO_EXPECTED_SHA256
    report["stand_checkpoint_sha256"] = stand_actual_sha
    report["loco_checkpoint_sha256"] = loco_actual_sha
    report["stand_checkpoint_sha_pass"] = stand_sha_pass
    report["loco_checkpoint_sha_pass"] = loco_sha_pass
    # Backward-compatible legacy aggregate consumed by the existing Linux V01/V02 sources.
    report["checkpoint_sha_pass"] = bool(stand_sha_pass and loco_sha_pass)
    if not stand_sha_pass:
        raise RuntimeError(
            f"v2-G SHA mismatch: expected {STAND_EXPECTED_SHA256}, got {stand_actual_sha}"
        )
    if not loco_sha_pass:
        raise RuntimeError(
            f"v4-N SHA mismatch: expected {LOCO_EXPECTED_SHA256}, got {loco_actual_sha}"
        )

    fastdds_profile = os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE")
    if not fastdds_profile:
        raise RuntimeError("FASTRTPS_DEFAULT_PROFILES_FILE is not set")
    fastdds_path = Path(fastdds_profile)
    if not fastdds_path.is_file():
        raise RuntimeError(f"FastDDS profile missing: {fastdds_path}")
    fastdds_text = fastdds_path.read_text(encoding="utf-8", errors="ignore")
    if "10.42.0.1" not in fastdds_text:
        raise RuntimeError("FastDDS profile does not whitelist expected Windows Ethernet IP 10.42.0.1")
    report["fastdds_profile"] = str(fastdds_path)
    report["ethernet_profile_ip_pass"] = True

    report["clock_info"] = {
        "perf_counter": {
            "implementation": time.get_clock_info("perf_counter").implementation,
            "resolution_s": time.get_clock_info("perf_counter").resolution,
            "monotonic": time.get_clock_info("perf_counter").monotonic,
            "adjustable": time.get_clock_info("perf_counter").adjustable,
        },
        "monotonic": {
            "implementation": time.get_clock_info("monotonic").implementation,
            "resolution_s": time.get_clock_info("monotonic").resolution,
            "monotonic": time.get_clock_info("monotonic").monotonic,
            "adjustable": time.get_clock_info("monotonic").adjustable,
        },
    }

    import gymnasium as gym
    import torch
    import humanoid_rl_lab.tasks  # noqa: F401
    from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.upright_yaw_direct_wide_yaw_flat_orientation_ppo_cfg import (
        UprightYawDirectWideYawFlatOrientationPPORunnerCfg,
    )
    from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.v4n_ppo_cfg import (
        V4NSymmetryAugmentationPPORunnerCfg,
    )

    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import parse_env_cfg

    # Direct bundled Jazzy rclpy; no ROS OmniGraph bridge extension.
    ros_prefix = Path(r"C:\IsaacLab\_isaac_sim\exts\isaacsim.ros2.core\jazzy")
    ros_python = ros_prefix / "rclpy"
    ros_lib = ros_prefix / "lib"

    for p in (ros_prefix, ros_python, ros_lib):
        if not p.exists():
            raise RuntimeError(f"Missing Isaac Sim bundled ROS path: {p}")

    if str(ros_python) not in sys.path:
        sys.path.insert(0, str(ros_python))
    os.environ["PATH"] = str(ros_lib) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("AMENT_PREFIX_PATH", str(ros_prefix))
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(ros_lib))

    import rclpy as _rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from std_msgs.msg import Bool, String

    rclpy = _rclpy

    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1)
    stand_agent_cfg = UprightYawDirectWideYawFlatOrientationPPORunnerCfg()
    loco_agent_cfg = V4NSymmetryAugmentationPPORunnerCfg()

    installed_rsl = metadata.version("rsl-rl-lib")
    stand_agent_cfg = handle_deprecated_rsl_rl_cfg(stand_agent_cfg, installed_rsl)
    loco_agent_cfg = handle_deprecated_rsl_rl_cfg(loco_agent_cfg, installed_rsl)

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        stand_agent_cfg.device = args_cli.device
        loco_agent_cfg.device = args_cli.device

    if hasattr(env_cfg, "seed"):
        env_cfg.seed = 42
    if hasattr(stand_agent_cfg, "seed"):
        stand_agent_cfg.seed = 42
    if hasattr(loco_agent_cfg, "seed"):
        loco_agent_cfg.seed = 42

    if stand_agent_cfg.class_name != "OnPolicyRunner":
        raise RuntimeError(f"Unexpected v2-G runner class: {stand_agent_cfg.class_name!r}")
    if loco_agent_cfg.class_name != "OnPolicyRunner":
        raise RuntimeError(f"Unexpected v4-N runner class: {loco_agent_cfg.class_name!r}")
    if stand_agent_cfg.clip_actions != loco_agent_cfg.clip_actions:
        raise RuntimeError(
            f"v2-G/v4-N clip_actions mismatch: {stand_agent_cfg.clip_actions!r} != {loco_agent_cfg.clip_actions!r}"
        )

    env_cfg.scene.num_envs = 1
    env_cfg.log_dir = str(OUT_DIR)

    # Frozen deployment startup contract: nominal/quasi-static initialization.
    env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    reset_base = getattr(env_cfg.events, "reset_base", None)
    if reset_base is None:
        raise RuntimeError("Expected reset_base event for final startup contract")
    pose_range = reset_base.params.get("pose_range", {})
    velocity_range = reset_base.params.get("velocity_range", {})
    for key in list(pose_range.keys()):
        pose_range[key] = (0.0, 0.0)
    for key in list(velocity_range.keys()):
        velocity_range[key] = (0.0, 0.0)

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

    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=stand_agent_cfg.clip_actions)
    uenv = env.unwrapped

    presentation_lighting_report = configure_presentation_lighting(args_cli.presentation_lighting)
    report["presentation_lighting_report"] = presentation_lighting_report

    presentation_floor_report = configure_presentation_floor(args_cli.presentation_floor)
    report["presentation_floor_report"] = presentation_floor_report

    real_ground_report = configure_real_ground_look(args_cli.ground_look)
    report["real_ground_look_report"] = real_ground_report

    studio_backdrop_report = configure_studio_backdrop(args_cli.studio_backdrop)
    report["studio_backdrop_report"] = studio_backdrop_report

    background_wall_report = configure_clean_background_wall(args_cli.background_wall)
    report["background_wall_report"] = background_wall_report

    showcase_scenery_report = configure_showcase_scenery(
        args_cli.showcase_scenery,
        shift_x_m=args_cli.showcase_shift_x,
    )
    report["showcase_scenery_report"] = showcase_scenery_report

    env_ids = torch.tensor([0], dtype=torch.int32, device=uenv.device)
    uenv.command_manager.reset(env_ids=env_ids)
    cmd_tensor = uenv.command_manager.get_command("base_velocity")
    cmd_tensor.zero_()

    stand_runner = OnPolicyRunner(
        env,
        stand_agent_cfg.to_dict(),
        log_dir=None,
        device=stand_agent_cfg.device,
    )
    stand_runner.load(str(STAND_CHECKPOINT))
    stand_policy = stand_runner.get_inference_policy(device=uenv.device)

    loco_runner = OnPolicyRunner(
        env,
        loco_agent_cfg.to_dict(),
        log_dir=None,
        device=loco_agent_cfg.device,
    )
    loco_runner.load(str(LOCO_CHECKPOINT))
    loco_policy = loco_runner.get_inference_policy(device=uenv.device)

    initial_obs = env.get_observations()
    policy_obs = extract_policy_observation_tensor(initial_obs)
    if int(policy_obs.shape[1]) != EXPECTED_OBS_DIM:
        raise RuntimeError(
            f"Unexpected policy observation dim: {policy_obs.shape[1]} != {EXPECTED_OBS_DIM}"
        )

    startup_robot = uenv.scene["robot"]
    startup_root_lin_w = as_tensor(startup_robot.data.root_lin_vel_w)[0]
    startup_root_ang_w = as_tensor(startup_robot.data.root_ang_vel_w)[0]
    startup_root_velocity_max_abs = float(
        torch.max(
            torch.cat(
                (
                    torch.abs(startup_root_lin_w),
                    torch.abs(startup_root_ang_w),
                )
            )
        ).detach().cpu().item()
    )
    report["startup_root_velocity_max_abs"] = startup_root_velocity_max_abs
    report["startup_root_velocity_zero_pass"] = startup_root_velocity_max_abs <= 1.0e-6
    if not report["startup_root_velocity_zero_pass"]:
        raise RuntimeError(
            "Nominal startup contract not realized: "
            f"max |root velocity|={startup_root_velocity_max_abs}"
        )

    if abs(float(uenv.step_dt) - EXPECTED_STEP_DT) > 1.0e-9:
        raise RuntimeError(f"Unexpected step_dt: {uenv.step_dt}")
    if int(uenv.action_manager.total_action_dim) != EXPECTED_ACTION_DIM:
        raise RuntimeError(f"Unexpected action dim: {uenv.action_manager.total_action_dim}")

    report["stand_policy_loaded"] = True
    report["loco_policy_loaded"] = True
    # Backward-compatible legacy aggregate consumed by Linux V01/V02.
    report["policy_loaded"] = True

    # Phase 8C-v2 validated deployment-only optimization:
    # reward computation is not consumed by the policy or ROS deployment path.
    # Preserve the RewardManager return buffer identity/shape/device while
    # skipping reward-term computation. Physics, observations, terminations,
    # commands, actions, and the frozen policy remain unchanged.
    reward_manager = uenv.reward_manager
    original_reward_compute = reward_manager.compute
    reward_buf = getattr(reward_manager, "_reward_buf", None)
    if reward_buf is None or not torch.is_tensor(reward_buf):
        raise RuntimeError("Reward bypass unavailable: reward_manager._reward_buf missing/not tensor")

    def deployment_reward_compute_bypass(*_args, **_kwargs):
        reward_buf.zero_()
        return reward_buf

    reward_manager.compute = deployment_reward_compute_bypass
    report["reward_bypass_active"] = True
    report["reward_buffer_shape"] = list(reward_buf.shape)
    report["reward_buffer_device"] = str(reward_buf.device)
    report["reward_buffer_dtype"] = str(reward_buf.dtype)

    report["policy_loaded"] = True
    report["rsl_rl_version"] = installed_rsl
    report["step_dt"] = float(uenv.step_dt)
    report["action_dim"] = int(uenv.action_manager.total_action_dim)
    report["visual_only"] = False
    report["demo_camera_target_lookahead_m"] = float(args_cli.demo_camera_target_lookahead)
    report["free_viewport_camera"] = bool(args_cli.free_camera)
    report["video_presentation"] = {
        "runner_generation": "v6-dual-policy",
        "video_preset": args_cli.video_preset,
        "clean_viewport": bool(args_cli.clean_viewport),
        "requested_resolution": [int(args_cli.video_width), int(args_cli.video_height)],
        "actual_viewport_resolution": list(_actual_viewport_resolution),
        "full_viewport_resolution": list(_full_viewport_resolution),
        "viewport_resolution_scale": _viewport_scale,
        "viewport_resolution_verified": bool(_viewport_resolution_pass),
        "presentation_lighting": args_cli.presentation_lighting,
        "render_quality": render_quality_report,
        "presentation_floor": args_cli.presentation_floor,
        "ground_look": args_cli.ground_look,
        "studio_backdrop": args_cli.studio_backdrop,
        "background_wall": args_cli.background_wall,
        "showcase_scenery": args_cli.showcase_scenery,
        "showcase_shift_x_m": float(args_cli.showcase_shift_x),
        "demo_camera_distance_scale_legacy": float(args_cli.demo_camera_distance_scale),
        "demo_camera_direct_offsets_m": {
            "forward": float(args_cli.demo_camera_forward_offset),
            "left": float(args_cli.demo_camera_left_offset),
            "eye_z": float(args_cli.demo_camera_eye_z),
            "target_z": float(args_cli.demo_camera_target_z),
        },
        "demo_look_revision": "isaaclab_showcase_true_right_pyramid_v542",
        "local_preview_sequence": bool(args_cli.local_preview_sequence),
        "local_preview_timing_s": {
            "zero_before": float(args_cli.local_preview_zero_before_s),
            "walk_040": float(args_cli.local_preview_walk_s),
            "zero_after": float(args_cli.local_preview_zero_after_s),
        },
        "command_source": (
            "INTERNAL_WINDOWS_PREVIEW"
            if args_cli.local_preview_sequence
            else "ROS2_EXTERNAL"
        ),
        "canonical_deployment_evidence": False if args_cli.local_preview_sequence else True,
        "visual_only": True,
        "physics_modified": False,
        "policy_modified": False,
    }
    report["camera"] = {
        "preset": args_cli.camera_preset,
        "update_hz": float(args_cli.camera_update_hz),
        "position_smoothing_s": float(args_cli.camera_position_smoothing_s),
        "heading_smoothing_s": float(args_cli.camera_heading_smoothing_s),
        "velocity_smoothing_s": float(args_cli.camera_velocity_smoothing_s),
        "heading_speed_threshold_mps": float(args_cli.camera_heading_speed_threshold),
        "heading_mode": args_cli.camera_heading_mode,
        "eye_z_m": float(args_cli.camera_eye_z),
        "target_z_m": float(args_cli.camera_target_z),
        "stabilization": (
            "camera translation low-pass filtered; pelvis roll/pitch and gait-cycle yaw ignored; "
            "heading is fixed in world by default, with optional low-pass travel-follow mode; eye/target z fixed"
        ),
    }

    if abs(float(uenv.step_dt) - EXPECTED_STEP_DT) > 1.0e-9:
        raise RuntimeError(f"Unexpected step_dt: {uenv.step_dt}")
    if int(uenv.action_manager.total_action_dim) != EXPECTED_ACTION_DIM:
        raise RuntimeError(f"Unexpected action dim: {uenv.action_manager.total_action_dim}")

    ros_state = RosState()

    qos_cmd = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    qos_status = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )

    class DeploymentNode(Node):
        def __init__(self):
            super().__init__("g1_visual_follow_camera_windows_runner_v3")
            self.cmd_sub = self.create_subscription(TwistStamped, CMD_TOPIC, self._on_cmd, qos_cmd)
            self.echo_pub = self.create_publisher(TwistStamped, ECHO_TOPIC, qos_cmd)
            self.status_pub = self.create_publisher(String, STATUS_TOPIC, qos_status)
            self.twist_pub = self.create_publisher(TwistStamped, BASE_TWIST_BODY_TOPIC, qos_cmd)
            self.pose_pub = self.create_publisher(PoseStamped, BASE_POSE_TOPIC, qos_cmd)
            self.stop_sub = self.create_subscription(Bool, STOP_TOPIC, self._on_stop, qos_status)

        def _on_cmd(self, msg):
            if msg.header.frame_id != EXPECTED_SOURCE_FRAME_ID:
                with ros_state.lock:
                    ros_state.source_mismatch_count += 1
                return

            vals = (
                float(msg.twist.linear.x),
                float(msg.twist.linear.y),
                float(msg.twist.angular.z),
            )
            if not all(math.isfinite(v) for v in vals):
                with ros_state.lock:
                    ros_state.invalid_count += 1
                return

            if near_cmd(vals, ZERO_CMD):
                accepted = ZERO_CMD
                kind = "zero"
            elif near_cmd(vals, SCI_CMD):
                accepted = SCI_CMD
                kind = "040"
            elif near_cmd(vals, CURVE_POS_CMD):
                accepted = CURVE_POS_CMD
                kind = "curve_pos"
            elif near_cmd(vals, CURVE_NEG_CMD):
                accepted = CURVE_NEG_CMD
                kind = "curve_neg"
            else:
                with ros_state.lock:
                    ros_state.rejected_command_count += 1
                return

            now = time.perf_counter()
            with ros_state.lock:
                ros_state.latest_cmd = accepted
                ros_state.last_valid_rx = now
                ros_state.valid_rx_count += 1
                if kind == "zero":
                    ros_state.valid_zero_count += 1
                elif kind == "040":
                    ros_state.valid_040_count += 1
                elif kind == "curve_pos":
                    ros_state.valid_curve_pos_count += 1
                elif kind == "curve_neg":
                    ros_state.valid_curve_neg_count += 1

            echo = TwistStamped()
            echo.header = msg.header
            echo.header.frame_id = "g1_accepted_command"
            echo.twist.linear.x = accepted[0]
            echo.twist.linear.y = accepted[1]
            echo.twist.angular.z = accepted[2]
            self.echo_pub.publish(echo)

        def _on_stop(self, msg):
            if bool(msg.data):
                with ros_state.lock:
                    ros_state.stop_requested = True

    rclpy.init()
    ros_node = DeploymentNode()
    ros_executor = SingleThreadedExecutor()
    ros_executor.add_node(ros_node)

    def ros_spin():
        ros_executor.spin()

    ros_thread = threading.Thread(target=ros_spin, daemon=True)
    ros_thread.start()

    robot = uenv.scene["robot"]
    dt = float(uenv.step_dt)

    # VISUAL-ONLY stabilized viewport follow camera.
    # Uses robot world x/y + yaw only; it deliberately ignores robot roll/pitch
    # and fixes eye/target height so gait bobbing is not inherited by the camera.
    sim = uenv.sim
    camera_period = 1.0 / max(1.0, float(args_cli.camera_update_hz))
    camera_pos_tau = max(1.0e-6, float(args_cli.camera_position_smoothing_s))
    camera_heading_tau = max(1.0e-6, float(args_cli.camera_heading_smoothing_s))
    camera_vel_tau = max(1.0e-6, float(args_cli.camera_velocity_smoothing_s))
    camera_heading_speed_threshold = max(0.0, float(args_cli.camera_heading_speed_threshold))

    camera_last_update = None
    camera_center_xy = None
    camera_yaw = None
    camera_prev_root_xy = None
    camera_filtered_vel_xy = [0.0, 0.0]
    camera_forward_offset, camera_left_offset = camera_body_offsets(args_cli.camera_preset)
    if args_cli.video_preset == "demo":
        # Direct low/front 3/4 composition. Independent forward/left controls let us
        # tune angle and distance without rebuilding the runner.
        camera_forward_offset = float(args_cli.demo_camera_forward_offset)
        camera_left_offset = float(args_cli.demo_camera_left_offset)

        for _name, _value in (
            ("demo-camera-forward-offset", camera_forward_offset),
            ("demo-camera-left-offset", camera_left_offset),
            ("demo-camera-eye-z", float(args_cli.camera_eye_z)),
            ("demo-camera-target-z", float(args_cli.camera_target_z)),
        ):
            if not math.isfinite(_value):
                raise RuntimeError(f"Invalid --{_name}: {_value}")

        _demo_horizontal_distance = math.hypot(camera_forward_offset, camera_left_offset)
        if _demo_horizontal_distance < 2.5 or _demo_horizontal_distance > 8.0:
            raise RuntimeError(
                "Invalid DEMO horizontal camera distance "
                f"{_demo_horizontal_distance:.3f} m; expected [2.5, 8.0] m"
            )

    router = Router(
        enter_loco_speed=ENTER_LOCO_SPEED,
        exit_loco_speed=EXIT_LOCO_SPEED,
        brake_steps=BRAKE_STEPS,
    )
    router_transitions = []
    router_state_steps = {"STAND": 0, "LOCO": 0, "BRAKE": 0}
    active_policy_steps = {"v2-G": 0, "v4-N": 0}
    completed_brake_lengths = []
    canceled_brake_lengths = []
    active_brake_len = 0
    current_route = RouterOutput(RouterState.STAND, "v2-G", 0)
    current_policy_cmd = ZERO_CMD

    step_count = 0
    reset_count = 0
    nonfinite_action_count = 0
    watchdog_steps = 0
    fresh_steps = 0
    loop_overruns = 0
    missed_deadlines = 0
    compute_ms_samples = []
    loop_period_ms_samples = []
    previous_loop_start = None
    max_action_abs = 0.0
    max_root_lin_speed_body = 0.0
    max_root_ang_speed_body = 0.0
    min_root_z = float("inf")
    max_root_z = float("-inf")
    accepted_040_effective_steps = 0
    accepted_curve_pos_effective_steps = 0
    accepted_curve_neg_effective_steps = 0
    accepted_zero_effective_steps = 0

    # 50 Hz VISUAL trajectory capture for deterministic offline video replay.
    # RC6 stores only the state required to reproduce the rendered robot exactly:
    # root position + root quaternion + joint positions. Runtime velocities/actions
    # remain available in the deployment telemetry/report and are intentionally not
    # copied on every control step. One torch.cat(..., out=preallocated_row) operation
    # writes the full visual state on-device per captured step.
    trajectory_capture_started = False
    trajectory_capture_start_control_step = None
    trajectory_frame_count = 0
    trajectory_capture_capacity = max(8, int(math.ceil(float(args_cli.max_runtime_s) / dt)) + 8)

    _cap_root_pos_sample = as_tensor(robot.data.root_pos_w)[0].detach()
    _cap_root_quat_sample = as_tensor(robot.data.root_quat_w)[0].detach()
    _cap_joint_pos_sample = as_tensor(robot.data.joint_pos)[0].detach()
    _cap_num_joints = int(_cap_joint_pos_sample.numel())
    _cap_visual_dim = 3 + 4 + _cap_num_joints
    trajectory_visual_state = torch.empty(
        (trajectory_capture_capacity, _cap_visual_dim),
        device=_cap_joint_pos_sample.device,
        dtype=_cap_joint_pos_sample.dtype,
    )

    trajectory_effective_cmd = []
    trajectory_policy_cmd = []
    trajectory_router_state = []
    trajectory_active_policy = []
    trajectory_brake_index = []
    trajectory_watchdog = []
    trajectory_control_step = []

    status_period = 1.0 / args_cli.status_hz
    telemetry_period = 1.0 / args_cli.telemetry_hz
    last_status_pub = 0.0
    last_telemetry_pub = 0.0

    start_wall = time.perf_counter()
    next_deadline = start_wall
    spin_margin_s = max(0.0, args_cli.spin_margin_ms / 1000.0)
    obs = env.get_observations()

    print("=" * 112)
    print("G1 ROS ↔ ISAAC — VIDEO MASTER RUNNER V6 DUAL-POLICY")
    print("=" * 112)
    print(f"v2-G STAND checkpoint SHA           : PASS ({stand_actual_sha})")
    print(f"v4-N LOCOMOTION checkpoint SHA      : PASS ({loco_actual_sha})")
    print(f"Integrated policy label             : {POLICY_LABEL}")
    print(f"Validation profile                  : {args_cli.validation_profile}")
    print(f"Router EXIT / ENTER                 : {EXIT_LOCO_SPEED:.2f} / {ENTER_LOCO_SPEED:.2f} m/s")
    print(f"BRAKE dwell                         : {BRAKE_S:.2f} s = {BRAKE_STEPS} steps, v4-N @ ZERO")
    print("Blending                            : ABSENT")
    print("Startup contract                    : nominal/quasi-static; reset pose/root velocities ZERO")
    print(f"Task                                : {TASK_ID}")
    print(f"RSL-RL                              : {installed_rsl}")
    print(f"step_dt                             : {dt}")
    print("ROS transport                       : direct bundled rclpy (no OmniGraph bridge extension)")
    print("ROS cmd QoS                         : BEST_EFFORT / KEEP_LAST(1)")
    print(f"Watchdog                            : {args_cli.watchdog_s:.3f} s")
    print("Allowed commands                    : zero; scientific [0.40,0,0]; engineering [0.40,0,±0.25]")
    print(f"Listening                           : {CMD_TOPIC}")
    print(f"Publishing                          : {ECHO_TOPIC}, {STATUS_TOPIC}, {BASE_TWIST_BODY_TOPIC}, {BASE_POSE_TOPIC}")
    print(f"Stop topic                          : {STOP_TOPIC}")
    print("Network path                        : direct Ethernet 10.42.0.1 <-> 10.42.0.2")
    print(f"FASTRTPS profile                    : {os.environ.get('FASTRTPS_DEFAULT_PROFILES_FILE')}")
    print(f"Local timing clock                  : {time.get_clock_info('perf_counter').implementation}")
    print(f"Pacing                              : absolute {dt*1000.0:.3f} ms deadlines; spin margin {args_cli.spin_margin_ms:.3f} ms")
    print(f"Camera preset                       : {args_cli.camera_preset}")
    print(f"Camera update rate                  : {args_cli.camera_update_hz:.1f} Hz")
    print(f"Camera position smoothing           : {args_cli.camera_position_smoothing_s:.3f} s")
    print(f"Camera heading smoothing            : {args_cli.camera_heading_smoothing_s:.3f} s")
    print(f"Camera velocity smoothing           : {args_cli.camera_velocity_smoothing_s:.3f} s")
    print(f"Camera heading speed threshold      : {args_cli.camera_heading_speed_threshold:.3f} m/s")
    print(f"Camera heading mode                 : {args_cli.camera_heading_mode}")
    print(f"Demo camera target lookahead        : {args_cli.demo_camera_target_lookahead:+.3f} m")
    print(f"Free viewport camera                : {bool(args_cli.free_camera)}")
    print(f"Camera eye / target z               : {args_cli.camera_eye_z:.2f} / {args_cli.camera_target_z:.2f} m")
    print(f"Video preset                        : {args_cli.video_preset}")
    print(f"Clean fullscreen viewport           : {bool(args_cli.clean_viewport)}")
    print(f"Requested video resolution          : {args_cli.video_width} x {args_cli.video_height}")
    print(f"ACTUAL viewport render resolution   : {_actual_viewport_resolution[0]} x {_actual_viewport_resolution[1]}")
    print(f"Full viewport texture resolution    : {_full_viewport_resolution[0]} x {_full_viewport_resolution[1]}")
    print(f"Viewport render scale               : {_viewport_scale:.3f}")
    print(f"Viewport resolution verification    : {'PASS' if _viewport_resolution_pass else 'FAIL'}")
    print(f"Presentation lighting               : {args_cli.presentation_lighting}")
    print(f"Presentation floor                  : {args_cli.presentation_floor}")
    print(f"REAL ground look                    : {args_cli.ground_look}")
    print(f"Studio backdrop                     : {args_cli.studio_backdrop}")
    print(f"Background wall                     : {args_cli.background_wall}")
    print(f"Isaac Lab showcase scenery          : {args_cli.showcase_scenery}")
    print(f"Whole showcase arena shift X        : {args_cli.showcase_shift_x:+.3f} m")
    if args_cli.showcase_scenery == "rough-dark":
        print("Showcase layout                      : 2 PARALLEL 16x4 ROUGH BANDS + FULL FLAT CORRIDOR")
        print("Showcase terrain cells               : 128 official rough cells total")
        print("Showcase terrain mix                 : 20/20 stairs/inv/boxes/random + 10/10 slopes")
        print("Showcase material                    : MATCHED to flat corridor graphite")
        print("Targeted terrain correction          : RED restored; BLUE row8/col3 <-> GREEN row10/col3")
        print("Additional pyramid copy              : GREEN row10/col3 -> behind RED row9/col2")
        print("True-right pyramid copy              : v5.39 row9/col2 -> row8/col2")
        print("Wrong prior guesses                  : row9/col1 and row10/col2 left ORIGINAL")
        print("Targeted band                        : NearSideBand seed=43")
        print("Rough band approximate X bounds      : x=-12..116 m")
        print("Visual flat corridor X range         : x=-30..116 m")
        print("Original infinite ground visual      : HIDDEN (visual only)")
        print("Flat collision plane                 : PRESERVED + ENABLED")
        print("Visual corridor                      : 10 m wide, x=-30..116 m, NO collision")
    if args_cli.showcase_scenery == "rough-dark":
        for _group in showcase_scenery_report.get("groups", []):
            print(
                "  scenery %-12s world AABB          : min=%s max=%s"
                % (
                    _group.get("label", "?"),
                    _group.get("runtime_world_aabb_min"),
                    _group.get("runtime_world_aabb_max"),
                )
            )
    if args_cli.video_preset == "demo":
        print(
            "Demo camera offsets [fwd,left]      : "
            f"{args_cli.demo_camera_forward_offset:.3f}, {args_cli.demo_camera_left_offset:.3f} m"
        )
        print(
            "Demo camera eye/target z            : "
            f"{args_cli.demo_camera_eye_z:.3f}, {args_cli.demo_camera_target_z:.3f} m"
        )
    print(f"Local Windows preview sequence      : {bool(args_cli.local_preview_sequence)}")
    if args_cli.local_preview_sequence:
        print(
            "Local preview sequence              : "
            f"zero {args_cli.local_preview_zero_before_s:.1f}s -> "
            f"[0.40,0,0] {args_cli.local_preview_walk_s:.1f}s -> "
            f"zero {args_cli.local_preview_zero_after_s:.1f}s"
        )
        print("Command-source classification       : INTERNAL VISUAL PREVIEW — NOT ROS DEPLOYMENT EVIDENCE")
    print(f"Trajectory capture                  : {bool(args_cli.capture_trajectory)}")
    if args_cli.capture_trajectory:
        _capture_path_preview = (
            Path(args_cli.trajectory_capture_path)
            if args_cli.trajectory_capture_path is not None
            else OUT_DIR / f"g1_trajectory_{''.join(c if c.isalnum() or c in '._-' else '_' for c in POLICY_LABEL).strip('._') or 'policy'}.pt"
        )
        print(f"Trajectory capture path             : {_capture_path_preview}")
        print("Trajectory capture cadence          : EVERY 50 Hz post-physics control step")
        print("Trajectory capture start            : FIRST FRESH canonical command")
        print("Trajectory runtime storage          : PREALLOCATED GPU BUFFER -> ONE host write after rollout")
    print(f"Render quality                      : {args_cli.render_quality}")
    print(f"RTX AA op runtime                   : {render_quality_report.get('aa_op_runtime')} (4=DLAA, 3=DLSS)")
    print(f"Motion blur runtime                 : {render_quality_report.get('motion_blur_runtime')}")
    print(f"DLSS frame generation runtime       : {render_quality_report.get('dlss_frame_generation_runtime')}")
    print("Presentation changes                : VISUAL ONLY — no physics/policy/contact changes")
    print("Evidence classification             : VISUAL ONLY — timing from this run is not canonical")
    print("Repository source modified          : NO")
    print("Checkpoint modified                 : NO")
    print()

    while simulation_app.is_running():
        loop_start = time.perf_counter()
        if previous_loop_start is not None:
            loop_period_ms_samples.append((loop_start - previous_loop_start) * 1000.0)
        previous_loop_start = loop_start
        elapsed = loop_start - start_wall
        cmd_publishers_now = int(ros_node.count_publishers(CMD_TOPIC))
        with ros_state.lock:
            ros_state.max_cmd_publishers_seen = max(
                ros_state.max_cmd_publishers_seen, cmd_publishers_now
            )
        snap = ros_state.snapshot()

        if snap["stop_requested"]:
            report["stop_reason"] = "ROS stop request"
            break
        if elapsed >= args_cli.max_runtime_s:
            report["stop_reason"] = "max runtime"
            break

        if args_cli.local_preview_sequence:
            _preview_total_s = (
                float(args_cli.local_preview_zero_before_s)
                + float(args_cli.local_preview_walk_s)
                + float(args_cli.local_preview_zero_after_s)
            )
            if elapsed >= _preview_total_s:
                report["stop_reason"] = "local preview sequence complete"
                break

        age = None if snap["last_valid_rx"] is None else loop_start - snap["last_valid_rx"]
        watchdog = age is None or age > args_cli.watchdog_s

        with ros_state.lock:
            if not watchdog:
                ros_state.have_ever_been_fresh = True
            elif ros_state.have_ever_been_fresh:
                ros_state.watchdog_after_fresh = True

        if args_cli.local_preview_sequence:
            _t0 = float(args_cli.local_preview_zero_before_s)
            _t1 = _t0 + float(args_cli.local_preview_walk_s)
            if elapsed < _t0:
                effective = ZERO_CMD
            elif elapsed < _t1:
                effective = SCI_CMD
            else:
                effective = ZERO_CMD
            watchdog = False
        else:
            # Safety contract: a stale/missing ROS command becomes ZERO, then follows
            # the same validated router path (LOCO -> BRAKE -> STAND) rather than
            # bypassing the braking dwell with a direct policy switch.
            effective = ZERO_CMD if watchdog else snap["latest_cmd"]

        if args_cli.capture_trajectory and not trajectory_capture_started:
            if args_cli.local_preview_sequence:
                trajectory_capture_started = True
                trajectory_capture_start_control_step = int(step_count)
            elif snap["last_valid_rx"] is not None and not watchdog:
                trajectory_capture_started = True
                trajectory_capture_start_control_step = int(step_count)
                print(
                    f"[TRAJECTORY] capture START control_step={step_count} "
                    f"first_effective={tuple(round(v, 3) for v in effective)}",
                    flush=True,
                )

        if watchdog:
            watchdog_steps += 1
        else:
            fresh_steps += 1
            if near_cmd(effective, SCI_CMD):
                accepted_040_effective_steps += 1
            elif near_cmd(effective, CURVE_POS_CMD):
                accepted_curve_pos_effective_steps += 1
            elif near_cmd(effective, CURVE_NEG_CMD):
                accepted_curve_neg_effective_steps += 1
            elif near_cmd(effective, ZERO_CMD):
                accepted_zero_effective_steps += 1

        old_router_state = router.state
        route = router.update(effective[0])
        new_router_state = route.state

        if new_router_state is RouterState.BRAKE:
            active_brake_len += 1
        elif old_router_state is RouterState.BRAKE and new_router_state is not RouterState.BRAKE:
            if new_router_state is RouterState.LOCO:
                canceled_brake_lengths.append(active_brake_len)
            elif new_router_state is RouterState.STAND:
                completed_brake_lengths.append(active_brake_len)
            active_brake_len = 0

        if new_router_state is not old_router_state:
            transition = {
                "step": int(step_count),
                "runtime_s": float(elapsed),
                "from": old_router_state.name,
                "to": new_router_state.name,
                "requested_cmd": list(effective),
                "active_policy": route.active_policy,
                "brake_index": int(route.brake_index),
            }
            router_transitions.append(transition)
            print(
                f"[ROUTER] step={step_count:05d} t={elapsed:8.3f}s "
                f"{old_router_state.name}->{new_router_state.name} "
                f"request={tuple(round(v, 3) for v in effective)} "
                f"policy={route.active_policy} brake_index={route.brake_index}",
                flush=True,
            )

        if new_router_state is RouterState.LOCO:
            # Forward the complete validated ROS command, including yaw-rate.
            policy_cmd = tuple(float(v) for v in effective)
        else:
            # STAND uses v2-G at ZERO. BRAKE keeps v4-N at ZERO.
            policy_cmd = ZERO_CMD

        cmd_tensor[0, 0] = policy_cmd[0]
        cmd_tensor[0, 1] = policy_cmd[1]
        cmd_tensor[0, 2] = policy_cmd[2]

        if route.active_policy == "v2-G":
            active_policy = stand_policy
        elif route.active_policy == "v4-N":
            active_policy = loco_policy
        else:
            raise RuntimeError(f"Unknown active policy from router: {route.active_policy!r}")

        current_route = route
        current_policy_cmd = policy_cmd
        router_state_steps[new_router_state.name] += 1
        active_policy_steps[route.active_policy] += 1

        obs = env.get_observations()

        with torch.inference_mode():
            actions = active_policy(obs)

        if not torch.isfinite(actions).all():
            nonfinite_action_count += 1
            report["stop_reason"] = "non-finite policy action"
            break

        max_action_abs = max(max_action_abs, float(torch.max(torch.abs(actions)).item()))

        obs, _, dones, _ = env.step(actions)
        step_count += 1

        if int(torch.count_nonzero(dones).item()) > 0:
            reset_count += int(torch.count_nonzero(dones).item())
            cmd_tensor.zero_()
            report["stop_reason"] = "environment termination/base-contact"
            break

        root_pos = robot_vec(robot, "root_pos_w")
        root_quat = robot_vec(robot, "root_quat_w")
        root_lin_b = robot_vec(robot, "root_lin_vel_b")
        root_ang_b = robot_vec(robot, "root_ang_vel_b")

        root_z = float(root_pos[2].item())
        lin_speed_b = float(torch.linalg.norm(root_lin_b[:3]).item())
        ang_speed_b = float(torch.linalg.norm(root_ang_b[:3]).item())

        min_root_z = min(min_root_z, root_z)
        max_root_z = max(max_root_z, root_z)
        max_root_lin_speed_body = max(max_root_lin_speed_body, lin_speed_b)
        max_root_ang_speed_body = max(max_root_ang_speed_body, ang_speed_b)

        if args_cli.capture_trajectory and trajectory_capture_started:
            # Device-to-device writes into preallocated storage only. No per-step host
            # transfer and no Tensor.clone() allocation on the timed controller path.
            if trajectory_frame_count >= trajectory_capture_capacity:
                raise RuntimeError(
                    f"Trajectory capture capacity exceeded: {trajectory_frame_count} >= "
                    f"{trajectory_capture_capacity}"
                )
            _ci = trajectory_frame_count
            torch.cat(
                (
                    as_tensor(robot.data.root_pos_w)[0].detach(),
                    as_tensor(robot.data.root_quat_w)[0].detach(),
                    as_tensor(robot.data.joint_pos)[0].detach(),
                ),
                dim=0,
                out=trajectory_visual_state[_ci],
            )
            trajectory_effective_cmd.append(effective)
            trajectory_policy_cmd.append(current_policy_cmd)
            trajectory_router_state.append(int(current_route.state.value))
            trajectory_active_policy.append(0 if current_route.active_policy == "v2-G" else 1)
            trajectory_brake_index.append(int(current_route.brake_index))
            trajectory_watchdog.append(bool(watchdog))
            trajectory_control_step.append(int(step_count))
            trajectory_frame_count += 1

        now = time.perf_counter()

        # VISUAL-ONLY stabilized follow camera.
        #
        # V1 used pelvis yaw directly. That is numerically correct but visually poor:
        # small gait-cycle yaw/x-y oscillations become camera shake. V2 therefore:
        #   1) low-pass filters robot x/y before moving the camera;
        #   2) estimates heading from low-pass WORLD travel velocity;
        #   3) holds heading while nearly stationary;
        #   4) ignores pelvis roll/pitch and gait-cycle yaw completely;
        #   5) updates at 50 Hz by default to avoid visible 25 Hz stepping.
        if camera_last_update is None or ((not args_cli.free_camera) and (now - camera_last_update) >= camera_period):
            root_x = float(root_pos[0].item())
            root_y = float(root_pos[1].item())
            root_yaw = quat_wxyz_to_yaw(root_quat.tolist())

            if camera_center_xy is None:
                camera_center_xy = [root_x, root_y]
                camera_prev_root_xy = [root_x, root_y]
                camera_yaw = root_yaw
                camera_filtered_vel_xy = [0.0, 0.0]
                cam_dt = camera_period
            else:
                cam_dt = max(1.0e-6, now - camera_last_update)

                # Smooth translation strongly enough to reject gait-cycle fore/aft and lateral bob.
                alpha_pos = 1.0 - math.exp(-cam_dt / camera_pos_tau)
                camera_center_xy[0] += alpha_pos * (root_x - camera_center_xy[0])
                camera_center_xy[1] += alpha_pos * (root_y - camera_center_xy[1])

                # Estimate horizontal travel velocity from world displacement.
                raw_vx = (root_x - camera_prev_root_xy[0]) / cam_dt
                raw_vy = (root_y - camera_prev_root_xy[1]) / cam_dt
                camera_prev_root_xy[0] = root_x
                camera_prev_root_xy[1] = root_y

                alpha_vel = 1.0 - math.exp(-cam_dt / camera_vel_tau)
                camera_filtered_vel_xy[0] += alpha_vel * (raw_vx - camera_filtered_vel_xy[0])
                camera_filtered_vel_xy[1] += alpha_vel * (raw_vy - camera_filtered_vel_xy[1])

                travel_speed = math.hypot(
                    camera_filtered_vel_xy[0],
                    camera_filtered_vel_xy[1],
                )

                # During locomotion, orient by path tangent rather than pelvis yaw.
                # While stopped, hold the previous camera heading instead of chasing posture noise.
                if (
                    args_cli.camera_heading_mode == "travel"
                    and travel_speed >= camera_heading_speed_threshold
                ):
                    travel_heading = math.atan2(
                        camera_filtered_vel_xy[1],
                        camera_filtered_vel_xy[0],
                    )
                    alpha_heading = 1.0 - math.exp(-cam_dt / camera_heading_tau)
                    yaw_err = wrap_pi(travel_heading - camera_yaw)
                    camera_yaw = wrap_pi(camera_yaw + alpha_heading * yaw_err)

            cy = math.cos(camera_yaw)
            sy = math.sin(camera_yaw)

            eye_x = (
                camera_center_xy[0]
                + camera_forward_offset * cy
                + camera_left_offset * (-sy)
            )
            eye_y = (
                camera_center_xy[1]
                + camera_forward_offset * sy
                + camera_left_offset * cy
            )

            # Keep a small look-ahead, but based on the stabilized heading.
            target_x = camera_center_xy[0] + float(args_cli.demo_camera_target_lookahead) * cy
            target_y = camera_center_xy[1] + float(args_cli.demo_camera_target_lookahead) * sy

            sim.set_camera_view(
                eye=(eye_x, eye_y, float(args_cli.camera_eye_z)),
                target=(target_x, target_y, float(args_cli.camera_target_z)),
            )
            camera_last_update = now

        if now - last_telemetry_pub >= telemetry_period:
            tmsg = TwistStamped()
            tmsg.header.stamp = ros_node.get_clock().now().to_msg()
            tmsg.header.frame_id = "base"
            tmsg.twist.linear.x = float(root_lin_b[0].item())
            tmsg.twist.linear.y = float(root_lin_b[1].item())
            tmsg.twist.linear.z = float(root_lin_b[2].item())
            tmsg.twist.angular.x = float(root_ang_b[0].item())
            tmsg.twist.angular.y = float(root_ang_b[1].item())
            tmsg.twist.angular.z = float(root_ang_b[2].item())
            ros_node.twist_pub.publish(tmsg)

            pmsg = PoseStamped()
            pmsg.header.stamp = tmsg.header.stamp
            pmsg.header.frame_id = "world"
            pmsg.pose.position.x = float(root_pos[0].item())
            pmsg.pose.position.y = float(root_pos[1].item())
            pmsg.pose.position.z = float(root_pos[2].item())
            # Isaac quaternion is wxyz.
            pmsg.pose.orientation.w = float(root_quat[0].item())
            pmsg.pose.orientation.x = float(root_quat[1].item())
            pmsg.pose.orientation.y = float(root_quat[2].item())
            pmsg.pose.orientation.z = float(root_quat[3].item())
            ros_node.pose_pub.publish(pmsg)

            last_telemetry_pub = now

        if now - last_status_pub >= status_period:
            snap2 = ros_state.snapshot()
            age_ms = None if snap2["last_valid_rx"] is None else max(
                0.0, (now - snap2["last_valid_rx"]) * 1000.0
            )
            payload = {
                "phase": "FINAL-DUAL-POLICY-DEPLOYMENT-VIDEO",
                # Legacy aggregate fields retained for the already-validated Linux V01/V02 clients.
                "policy_loaded": bool(report.get("policy_loaded", False)),
                "checkpoint_sha_pass": bool(report.get("checkpoint_sha_pass", False)),
                # New explicit dual-policy provenance.
                "stand_policy_loaded": bool(report.get("stand_policy_loaded", False)),
                "loco_policy_loaded": bool(report.get("loco_policy_loaded", False)),
                "stand_checkpoint_sha_pass": bool(report.get("stand_checkpoint_sha_pass", False)),
                "loco_checkpoint_sha_pass": bool(report.get("loco_checkpoint_sha_pass", False)),
                "scientific_command": list(SCI_CMD),
                "engineering_curve_pos_command": list(CURVE_POS_CMD),
                "engineering_curve_neg_command": list(CURVE_NEG_CMD),
                "step_count": step_count,
                "runtime_s": now - start_wall,
                "watchdog_active": (
                    False
                    if args_cli.local_preview_sequence
                    else bool(age_ms is None or age_ms > args_cli.watchdog_s * 1000.0)
                ),
                "cmd_age_ms": age_ms,
                "effective_cmd": list(effective),
                "policy_cmd": list(current_policy_cmd),
                "router_state": current_route.state.name,
                "active_policy": current_route.active_policy,
                "brake_index": int(current_route.brake_index),
                "router_exit_loco_speed_mps": EXIT_LOCO_SPEED,
                "router_enter_loco_speed_mps": ENTER_LOCO_SPEED,
                "router_brake_steps": BRAKE_STEPS,
                "valid_rx_count": snap2["valid_rx_count"],
                "valid_zero_count": snap2["valid_zero_count"],
                "valid_040_count": snap2["valid_040_count"],
                "valid_curve_pos_count": snap2["valid_curve_pos_count"],
                "valid_curve_neg_count": snap2["valid_curve_neg_count"],
                "rejected_command_count": snap2["rejected_command_count"],
                "invalid_count": snap2["invalid_count"],
                "source_mismatch_count": snap2["source_mismatch_count"],
                "max_cmd_publishers_seen": snap2["max_cmd_publishers_seen"],
                "expected_source_frame_id": EXPECTED_SOURCE_FRAME_ID,
                "reset_count": reset_count,
                "root_z": root_z,
                "body_vx": float(root_lin_b[0].item()),
                "body_vy": float(root_lin_b[1].item()),
                "body_wz": float(root_ang_b[2].item()),
            }
            smsg = String()
            smsg.data = json.dumps(payload, separators=(",", ":"), sort_keys=True)
            ros_node.status_pub.publish(smsg)
            last_status_pub = now

        spent = time.perf_counter() - loop_start
        compute_ms_samples.append(spent * 1000.0)

        if spent > dt:
            loop_overruns += 1

        if args_cli.real_time:
            next_deadline += dt
            now_sched = time.perf_counter()

            if now_sched > next_deadline:
                missed_deadlines += 1
                # Re-anchor after a true missed deadline instead of accumulating lag.
                next_deadline = now_sched
            else:
                sleep_until = next_deadline - spin_margin_s
                now_sched = time.perf_counter()
                if sleep_until > now_sched:
                    time.sleep(sleep_until - now_sched)
                while time.perf_counter() < next_deadline:
                    pass

    cmd_tensor.zero_()
    final_snap = ros_state.snapshot()

    compute_stats = summarize_ms(compute_ms_samples)
    period_stats = summarize_ms(loop_period_ms_samples)
    overrun_pct = (100.0 * loop_overruns / step_count) if step_count > 0 else None
    achieved_loop_hz = (
        1000.0 / period_stats["mean_ms"]
        if period_stats is not None and period_stats["mean_ms"] > 0.0
        else None
    )

    trajectory_path = None
    trajectory_sha256 = None
    trajectory_duration_s = trajectory_frame_count * dt
    trajectory_shape_pass = False
    if args_cli.capture_trajectory:
        if trajectory_frame_count <= 0:
            raise RuntimeError("Trajectory capture requested but no frames were recorded")
        if not (
            len(trajectory_effective_cmd)
            == len(trajectory_policy_cmd)
            == len(trajectory_router_state)
            == len(trajectory_active_policy)
            == len(trajectory_brake_index)
            == len(trajectory_watchdog)
            == len(trajectory_control_step)
            == trajectory_frame_count
        ):
            raise RuntimeError("Trajectory capture metadata length mismatch")

        safe_capture_label = (
            "".join(c if c.isalnum() or c in "._-" else "_" for c in POLICY_LABEL).strip("._") or "policy"
        )
        trajectory_path = (
            Path(args_cli.trajectory_capture_path)
            if args_cli.trajectory_capture_path is not None
            else OUT_DIR / f"g1_trajectory_{safe_capture_label}.pt"
        )
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)

        # Single post-rollout device -> host transfer of the populated visual-state prefix.
        _visual_state = trajectory_visual_state[:trajectory_frame_count].cpu()
        _root_pos = _visual_state[:, 0:3]
        _root_quat = _visual_state[:, 3:7]
        _joint_pos = _visual_state[:, 7:]
        trajectory_shape_pass = bool(
            _visual_state.shape == (trajectory_frame_count, _cap_visual_dim)
            and _root_pos.shape == (trajectory_frame_count, 3)
            and _root_quat.shape == (trajectory_frame_count, 4)
            and _joint_pos.shape == (trajectory_frame_count, _cap_num_joints)
        )
        if not trajectory_shape_pass:
            raise RuntimeError(
                "Visual trajectory tensor shape guard failed: "
                f"visual_state={tuple(_visual_state.shape)} root_pos={tuple(_root_pos.shape)} "
                f"root_quat={tuple(_root_quat.shape)} joint_pos={tuple(_joint_pos.shape)}"
            )

        trajectory_payload = {
            "format": "G1_DUAL_POLICY_VISUAL_TRAJECTORY_V2",
            "format_version": 2,
            "capture_semantics": "post-physics visual state at every 50 Hz control step",
            "payload_scope": "root pose + joint positions + command/router provenance; runtime dynamics remain in deployment telemetry",
            "task_id": TASK_ID,
            "policy_label": POLICY_LABEL,
            "validation_profile": args_cli.validation_profile,
            "step_dt_s": float(dt),
            "fps": float(1.0 / dt),
            "frame_count": int(trajectory_frame_count),
            "duration_s": float(trajectory_duration_s),
            "capture_start_control_step": trajectory_capture_start_control_step,
            "capture_runner_sha256": sha256_file(Path(__file__)),
            "stand_checkpoint_sha256": stand_actual_sha,
            "loco_checkpoint_sha256": loco_actual_sha,
            "joint_names": list(robot.joint_names),
            "visual_state": _visual_state,
            "visual_state_layout": {
                "root_pos_w": [0, 3],
                "root_quat_w_wxyz": [3, 7],
                "joint_pos": [7, 7 + _cap_num_joints],
            },
            "root_pos_w": _root_pos,
            "root_quat_w_wxyz": _root_quat,
            "joint_pos": _joint_pos,
            "effective_cmd": torch.tensor(trajectory_effective_cmd, dtype=torch.float32),
            "policy_cmd": torch.tensor(trajectory_policy_cmd, dtype=torch.float32),
            "router_state_code": torch.tensor(trajectory_router_state, dtype=torch.int16),
            "router_state_codebook": {
                int(RouterState.STAND.value): "STAND",
                int(RouterState.LOCO.value): "LOCO",
                int(RouterState.BRAKE.value): "BRAKE",
            },
            "active_policy_code": torch.tensor(trajectory_active_policy, dtype=torch.int8),
            "active_policy_codebook": {0: "v2-G", 1: "v4-N"},
            "brake_index": torch.tensor(trajectory_brake_index, dtype=torch.int16),
            "watchdog": torch.tensor(trajectory_watchdog, dtype=torch.bool),
            "control_step": torch.tensor(trajectory_control_step, dtype=torch.int64),
            "capture_time_s": torch.arange(1, trajectory_frame_count + 1, dtype=torch.float64) * float(dt),
        }
        torch.save(trajectory_payload, trajectory_path)
        trajectory_sha256 = sha256_file(trajectory_path)
        print(
            f"[TRAJECTORY] saved frames={trajectory_frame_count} duration={trajectory_duration_s:.3f}s "
            f"path={trajectory_path}",
            flush=True,
        )
        print(f"[TRAJECTORY] SHA256={trajectory_sha256}", flush=True)

    report["trajectory_capture_started"] = bool(trajectory_capture_started)
    report["trajectory_capture_start_control_step"] = trajectory_capture_start_control_step
    report["trajectory_frame_count"] = int(trajectory_frame_count)
    report["trajectory_duration_s"] = float(trajectory_duration_s)
    report["trajectory_path"] = str(trajectory_path) if trajectory_path is not None else None
    report["trajectory_sha256"] = trajectory_sha256
    report["trajectory_shape_pass"] = bool(trajectory_shape_pass) if args_cli.capture_trajectory else None

    transition_pairs = [(t["from"], t["to"]) for t in router_transitions]
    router_completed_brakes_ok = bool(completed_brake_lengths) and all(
        n == BRAKE_STEPS for n in completed_brake_lengths
    )

    report.update(
        {
            "runtime_s": time.perf_counter() - start_wall,
            "step_count": step_count,
            "reset_count": reset_count,
            "nonfinite_action_count": nonfinite_action_count,
            "loop_overruns": loop_overruns,
            "loop_overrun_pct": overrun_pct,
            "missed_deadlines": missed_deadlines,
            "missed_deadline_pct": (100.0 * missed_deadlines / step_count) if step_count > 0 else None,
            "spin_margin_ms": args_cli.spin_margin_ms,
            "compute_cycle_ms": compute_stats,
            "loop_period_ms": period_stats,
            "achieved_loop_hz": achieved_loop_hz,
            "valid_ros_cmd_count": final_snap["valid_rx_count"],
            "valid_zero_count": final_snap["valid_zero_count"],
            "valid_040_count": final_snap["valid_040_count"],
            "valid_curve_pos_count": final_snap["valid_curve_pos_count"],
            "valid_curve_neg_count": final_snap["valid_curve_neg_count"],
            "rejected_command_count": final_snap["rejected_command_count"],
            "invalid_ros_cmd_count": final_snap["invalid_count"],
            "source_mismatch_count": final_snap["source_mismatch_count"],
            "max_cmd_publishers_seen": final_snap["max_cmd_publishers_seen"],
            "expected_source_frame_id": EXPECTED_SOURCE_FRAME_ID,
            "watchdog_steps": watchdog_steps,
            "fresh_steps": fresh_steps,
            "watchdog_after_fresh": final_snap["watchdog_after_fresh"],
            "effective_040_steps": accepted_040_effective_steps,
            "effective_curve_pos_steps": accepted_curve_pos_effective_steps,
            "effective_curve_neg_steps": accepted_curve_neg_effective_steps,
            "effective_zero_steps": accepted_zero_effective_steps,
            "router_state_steps": dict(router_state_steps),
            "active_policy_steps": dict(active_policy_steps),
            "router_transitions": list(router_transitions),
            "router_transition_pairs": transition_pairs,
            "completed_brake_lengths_steps": list(completed_brake_lengths),
            "canceled_brake_lengths_steps": list(canceled_brake_lengths),
            "final_router_state": router.state.name,
            "max_action_abs": max_action_abs,
            "max_root_lin_speed_body": max_root_lin_speed_body,
            "max_root_ang_speed_body": max_root_ang_speed_body,
            "min_root_z": None if min_root_z == float("inf") else min_root_z,
            "max_root_z": None if max_root_z == float("-inf") else max_root_z,
        }
    )

    profile = args_cli.validation_profile
    profile_guards = {
        "profile_received_any_command": final_snap["valid_rx_count"] > 0,
    }

    if profile == "straight":
        profile_guards.update(
            {
                "profile_received_zero": final_snap["valid_zero_count"] > 0,
                "profile_received_straight": final_snap["valid_040_count"] > 0,
                "profile_effective_straight": accepted_040_effective_steps > 0,
                "profile_router_used_loco": router_state_steps["LOCO"] > 0,
                "profile_router_used_brake": router_state_steps["BRAKE"] >= BRAKE_STEPS,
                "profile_router_returned_stand": router.state is RouterState.STAND,
                "profile_brake_completed_18_steps": router_completed_brakes_ok,
            }
        )
    elif profile == "curve-pos":
        profile_guards.update(
            {
                "profile_received_zero": final_snap["valid_zero_count"] > 0,
                "profile_received_curve_pos": final_snap["valid_curve_pos_count"] > 0,
                "profile_effective_curve_pos": accepted_curve_pos_effective_steps > 0,
                "profile_router_used_loco": router_state_steps["LOCO"] > 0,
                "profile_router_used_brake": router_state_steps["BRAKE"] >= BRAKE_STEPS,
                "profile_router_returned_stand": router.state is RouterState.STAND,
                "profile_brake_completed_18_steps": router_completed_brakes_ok,
            }
        )
    elif profile == "curve-neg":
        profile_guards.update(
            {
                "profile_received_zero": final_snap["valid_zero_count"] > 0,
                "profile_received_curve_neg": final_snap["valid_curve_neg_count"] > 0,
                "profile_effective_curve_neg": accepted_curve_neg_effective_steps > 0,
                "profile_router_used_loco": router_state_steps["LOCO"] > 0,
                "profile_router_used_brake": router_state_steps["BRAKE"] >= BRAKE_STEPS,
                "profile_router_returned_stand": router.state is RouterState.STAND,
                "profile_brake_completed_18_steps": router_completed_brakes_ok,
            }
        )
    elif profile == "full-soak":
        profile_guards.update(
            {
                "profile_received_zero": final_snap["valid_zero_count"] > 0,
                "profile_received_straight": final_snap["valid_040_count"] > 0,
                "profile_received_curve_pos": final_snap["valid_curve_pos_count"] > 0,
                "profile_received_curve_neg": final_snap["valid_curve_neg_count"] > 0,
                "profile_effective_straight": accepted_040_effective_steps > 0,
                "profile_effective_curve_pos": accepted_curve_pos_effective_steps > 0,
                "profile_effective_curve_neg": accepted_curve_neg_effective_steps > 0,
                "profile_router_used_loco": router_state_steps["LOCO"] > 0,
                "profile_router_used_brake": router_state_steps["BRAKE"] >= BRAKE_STEPS,
                "profile_router_returned_stand": router.state is RouterState.STAND,
                "profile_brake_completed_18_steps": router_completed_brakes_ok,
            }
        )

    guards = {
        "stand_checkpoint_sha_pass": bool(report["stand_checkpoint_sha_pass"]),
        "loco_checkpoint_sha_pass": bool(report["loco_checkpoint_sha_pass"]),
        "checkpoint_sha_pass": bool(report["checkpoint_sha_pass"]),
        "stand_policy_loaded": bool(report["stand_policy_loaded"]),
        "loco_policy_loaded": bool(report["loco_policy_loaded"]),
        "policy_loaded": bool(report["policy_loaded"]),
        "step_dt_50hz": abs(dt - EXPECTED_STEP_DT) <= 1.0e-9,
        "obs_dim_123": int(policy_obs.shape[1]) == EXPECTED_OBS_DIM,
        "action_dim_37": int(uenv.action_manager.total_action_dim) == EXPECTED_ACTION_DIM,
        "startup_root_velocity_zero": bool(report.get("startup_root_velocity_zero_pass", False)),
        "router_frozen_thresholds": (
            abs(router.enter - ENTER_LOCO_SPEED) <= 1.0e-12
            and abs(router.exit - EXIT_LOCO_SPEED) <= 1.0e-12
        ),
        "router_frozen_brake_18": router.brake_steps == BRAKE_STEPS,
        "no_blending": True,
        "no_rejected_commands": final_snap["rejected_command_count"] == 0,
        "no_invalid_ros_commands": final_snap["invalid_count"] == 0,
        "no_source_mismatch": final_snap["source_mismatch_count"] == 0,
        "single_cmd_publisher_max": final_snap["max_cmd_publishers_seen"] <= 1,
        "no_environment_termination": reset_count == 0,
        "finite_policy_actions": nonfinite_action_count == 0,
        "meaningful_rollout": step_count >= 50,
        "reward_bypass_active": report.get("reward_bypass_active") is True,
        **profile_guards,
    }
    if args_cli.capture_trajectory:
        guards.update(
            {
                "trajectory_capture_started": bool(trajectory_capture_started),
                "trajectory_capture_saved": trajectory_path is not None and trajectory_path.is_file(),
                "trajectory_capture_shape_pass": bool(trajectory_shape_pass),
                "trajectory_capture_50hz_dt": abs(dt - EXPECTED_STEP_DT) <= 1.0e-9,
                "trajectory_capture_meaningful": trajectory_frame_count >= 50,
            }
        )

    missed_deadline_pct = (
        100.0 * missed_deadlines / step_count if step_count > 0 else None
    )
    timing_guards = {
        "compute_median_under_20ms": compute_stats is not None and compute_stats["median_ms"] < 20.0,
        "compute_p95_under_20ms": compute_stats is not None and compute_stats["p95_ms"] < 20.0,
        "compute_overrun_rate_le_5pct": overrun_pct is not None and overrun_pct <= 5.0,
        "missed_deadline_rate_le_5pct": missed_deadline_pct is not None and missed_deadline_pct <= 5.0,
        "achieved_loop_rate_ge_49hz": achieved_loop_hz is not None and achieved_loop_hz >= 49.0,
    }
    report["guardrails"] = guards
    report["timing_guardrails"] = timing_guards
    report["functional_status"] = "PASS" if all(guards.values()) else "FAIL"
    report["timing_status"] = "PASS" if all(timing_guards.values()) else "FAIL"
    report["status"] = (
        "PASS"
        if report["functional_status"] == "PASS" and report["timing_status"] == "PASS"
        else "FAIL"
    )

    if args_cli.local_preview_sequence:
        _local_preview_pass = (
            report.get("stop_reason") == "local preview sequence complete"
            and int(nonfinite_action_count) == 0
            and int(reset_count) == 0
        )
        report["functional_status"] = "NOT_APPLICABLE_LOCAL_PREVIEW"
        report["timing_status"] = "NOT_CANONICAL_LOCAL_PREVIEW"
        report["status"] = "VISUAL_PREVIEW_PASS" if _local_preview_pass else "VISUAL_PREVIEW_FAIL"

except Exception as exc:
    report["errors"].append(repr(exc))
    report["status"] = "FAIL"

finally:
    safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in POLICY_LABEL).strip("._") or "policy"
    json_path = OUT_DIR / f"g1_dual_policy_deployment_{safe_label}.json"
    txt_path = OUT_DIR / f"g1_dual_policy_deployment_{safe_label}.txt"

    if report.get("status") == "PASS":
        final_marker = "G1_VIDEO_MASTER_V6_DUAL_POLICY_PASS"
    elif report.get("status") == "VISUAL_PREVIEW_PASS":
        final_marker = "G1_VIDEO_MASTER_V6_DUAL_POLICY_LOCAL_PREVIEW_PASS"
    else:
        final_marker = "G1_VIDEO_MASTER_V6_DUAL_POLICY_FAIL"
    report["final_marker"] = final_marker

    try:
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        lines = [
            "G1 ROS ↔ ISAAC — VIDEO MASTER RUNNER V6 DUAL-POLICY",
            "=" * 112,
            f"Status                              : {report.get('status')}",
            f"Task ID                             : {TASK_ID}",
            f"Validation profile                  : {args_cli.validation_profile}",
            f"Combined checkpoint SHA PASS        : {report.get('checkpoint_sha_pass')}",
            f"v2-G checkpoint SHA PASS            : {report.get('stand_checkpoint_sha_pass')}",
            f"v4-N checkpoint SHA PASS            : {report.get('loco_checkpoint_sha_pass')}",
            f"Combined policy loaded              : {report.get('policy_loaded')}",
            f"v2-G policy loaded                  : {report.get('stand_policy_loaded')}",
            f"v4-N policy loaded                  : {report.get('loco_policy_loaded')}",
            f"Router EXIT / ENTER m/s             : {EXIT_LOCO_SPEED} / {ENTER_LOCO_SPEED}",
            f"BRAKE dwell                         : {BRAKE_S} s / {BRAKE_STEPS} steps",
            f"Router final state                  : {report.get('final_router_state')}",
            f"Router state steps                  : {report.get('router_state_steps')}",
            f"Active policy steps                 : {report.get('active_policy_steps')}",
            f"Completed BRAKE lengths             : {report.get('completed_brake_lengths_steps')}",
            f"Canceled BRAKE lengths              : {report.get('canceled_brake_lengths_steps')}",
            f"Runtime s                           : {report.get('runtime_s')}",
            f"Step count                          : {report.get('step_count')}",
            f"Trajectory capture requested        : {report.get('trajectory_capture_requested')}",
            f"Trajectory frames / duration s      : {report.get('trajectory_frame_count')} / {report.get('trajectory_duration_s')}",
            f"Trajectory shape PASS               : {report.get('trajectory_shape_pass')}",
            f"Trajectory path                     : {report.get('trajectory_path')}",
            f"Trajectory SHA256                   : {report.get('trajectory_sha256')}",
            f"Reset count                         : {report.get('reset_count')}",
            f"Valid zero commands                 : {report.get('valid_zero_count')}",
            f"Valid .40 straight commands         : {report.get('valid_040_count')}",
            f"Valid +curve commands               : {report.get('valid_curve_pos_count')}",
            f"Valid -curve commands               : {report.get('valid_curve_neg_count')}",
            f"Effective .40 straight steps        : {report.get('effective_040_steps')}",
            f"Effective +curve steps              : {report.get('effective_curve_pos_steps')}",
            f"Effective -curve steps              : {report.get('effective_curve_neg_steps')}",
            f"Rejected commands                   : {report.get('rejected_command_count')}",
            f"Invalid ROS commands                : {report.get('invalid_ros_cmd_count')}",
            f"Source mismatch commands            : {report.get('source_mismatch_count')}",
            f"Max cmd publishers seen             : {report.get('max_cmd_publishers_seen')}",
            f"Max action abs                      : {report.get('max_action_abs')}",
            f"Max body linear speed               : {report.get('max_root_lin_speed_body')}",
            f"Max body angular speed              : {report.get('max_root_ang_speed_body')}",
            f"Min / max root z                    : {report.get('min_root_z')} / {report.get('max_root_z')}",
            f"Loop overruns                       : {report.get('loop_overruns')} ({report.get('loop_overrun_pct')}%)",
            f"Missed deadlines                    : {report.get('missed_deadlines')} ({report.get('missed_deadline_pct')}%)",
            f"Spin margin ms                      : {report.get('spin_margin_ms')}",
            f"Achieved loop Hz                    : {report.get('achieved_loop_hz')}",
            f"Compute cycle stats ms              : {report.get('compute_cycle_ms')}",
            f"Loop period stats ms                : {report.get('loop_period_ms')}",
            f"Functional status                   : {report.get('functional_status')}",
            f"Timing status                       : {report.get('timing_status')}",
            f"Stop reason                         : {report.get('stop_reason')}",
            "Training run                        : NO",
            "Repository source modified          : NO",
            "Checkpoint modified                 : NO",
            "",
            "GUARDRAILS",
            "-" * 112,
        ]
        for k, v in report.get("guardrails", {}).items():
            lines.append(f"{k:42s}: {'PASS' if v else 'FAIL'}")
        lines += ["", "TIMING GUARDRAILS", "-" * 112]
        for k, v in report.get("timing_guardrails", {}).items():
            lines.append(f"{k:42s}: {'PASS' if v else 'FAIL'}")
        lines += ["", "ERRORS", "-" * 112]
        if report["errors"]:
            lines.extend(f"- {e}" for e in report["errors"])
        else:
            lines.append("NONE")
        txt_path.write_text("\n".join(lines), encoding="utf-8")

        print()
        print("=" * 112)
        print("DUAL-POLICY DEPLOYMENT WINDOWS SUMMARY")
        print("=" * 112)
        print(f"Status                              : {report.get('status')}")
        print(f"Validation profile                  : {args_cli.validation_profile}")
        print(f"Combined checkpoint SHA PASS        : {report.get('checkpoint_sha_pass')}")
        print(f"v2-G / v4-N SHA PASS                : {report.get('stand_checkpoint_sha_pass')} / {report.get('loco_checkpoint_sha_pass')}")
        print(f"Combined policy loaded              : {report.get('policy_loaded')}")
        print(f"v2-G / v4-N loaded                  : {report.get('stand_policy_loaded')} / {report.get('loco_policy_loaded')}")
        print(f"Router final state                  : {report.get('final_router_state')}")
        print(f"Router transitions                  : {report.get('router_transition_pairs')}")
        print(f"Completed BRAKE lengths             : {report.get('completed_brake_lengths_steps')}")
        print(f"Step count                          : {report.get('step_count')}")
        print(f"Trajectory capture requested        : {report.get('trajectory_capture_requested')}")
        print(f"Trajectory frames / duration s      : {report.get('trajectory_frame_count')} / {report.get('trajectory_duration_s')}")
        print(f"Trajectory shape PASS               : {report.get('trajectory_shape_pass')}")
        print(f"Trajectory path                     : {report.get('trajectory_path')}")
        print(f"Trajectory SHA256                   : {report.get('trajectory_sha256')}")
        print(f"Reset count                         : {report.get('reset_count')}")
        print(f"Valid zero commands                 : {report.get('valid_zero_count')}")
        print(f"Valid .40 straight commands         : {report.get('valid_040_count')}")
        print(f"Valid +curve commands               : {report.get('valid_curve_pos_count')}")
        print(f"Valid -curve commands               : {report.get('valid_curve_neg_count')}")
        print(f"Effective .40 straight steps        : {report.get('effective_040_steps')}")
        print(f"Effective +curve steps              : {report.get('effective_curve_pos_steps')}")
        print(f"Effective -curve steps              : {report.get('effective_curve_neg_steps')}")
        print(f"Rejected commands                   : {report.get('rejected_command_count')}")
        print(f"Source mismatch commands            : {report.get('source_mismatch_count')}")
        print(f"Max cmd publishers seen             : {report.get('max_cmd_publishers_seen')}")
        print(f"Achieved loop Hz                    : {report.get('achieved_loop_hz')}")
        print(f"Compute median ms                   : {(report.get('compute_cycle_ms') or {}).get('median_ms')}")
        print(f"Compute p95 ms                      : {(report.get('compute_cycle_ms') or {}).get('p95_ms')}")
        print(f"Loop period median ms               : {(report.get('loop_period_ms') or {}).get('median_ms')}")
        print(f"Loop overruns                       : {report.get('loop_overruns')} ({report.get('loop_overrun_pct')}%)")
        print(f"Missed deadlines                    : {report.get('missed_deadlines')} ({report.get('missed_deadline_pct')}%)")
        print(f"Functional status                   : {report.get('functional_status')}")
        print(f"Timing status                       : {report.get('timing_status')}")
        print(f"Stop reason                         : {report.get('stop_reason')}")
        print(f"Errors                              : {len(report.get('errors', []))}")
        print(f"TXT                                 : {txt_path}")
        print(f"JSON                                : {json_path}")
        print("Repository source modified          : NO")
        print("Checkpoint modified                 : NO")
        print(final_marker, flush=True)
    except Exception as persist_exc:
        print(f"[ERROR] Could not persist V6 dual-policy report: {persist_exc!r}", flush=True)

    if ros_executor is not None:
        try:
            ros_executor.shutdown(timeout_sec=1.0)
        except Exception:
            pass
    if rclpy is not None:
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
    if ros_thread is not None:
        ros_thread.join(timeout=2.0)
    if ros_node is not None:
        try:
            ros_node.destroy_node()
        except Exception:
            pass
    if env is not None:
        try:
            env.close()
        except Exception:
            pass
    try:
        simulation_app.close()
    except Exception:
        pass
