from __future__ import annotations

import argparse
import gc
import traceback

parser = argparse.ArgumentParser(
    description=(
        "Read-only audit of the velocity-command configuration actually instantiated "
        "for frozen v2-G and frozen v4-N environments. No policy inference, no env.step, "
        "no scene.update, no physics advance."
    )
)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import humanoid_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


V2G_TASK = "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-v0"
V4N_TASK = "Humanoid-RL-Lab-G1-Flat-V4N-Symmetry-Augmentation-v0"

PASS_MARKER = "V2G_V4N_TRAINING_COMMAND_SUPPORT_AUDIT_PASS"


def fmt(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, (str, int, float, bool)):
        return repr(value)
    if isinstance(value, dict):
        return repr(value)
    if hasattr(value, "__dict__"):
        items = {}
        for k, v in vars(value).items():
            if k.startswith("_"):
                continue
            if isinstance(v, (str, int, float, bool, type(None), tuple, list, dict)):
                items[k] = v
        return repr(items)
    return repr(value)


def print_command_cfg(task_name: str, label: str) -> None:
    cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=1)
    cmd = cfg.commands.base_velocity

    print("-" * 132)
    print(label)
    print("-" * 132)
    print(f"task_id                  : {task_name}")
    print(f"cfg_type                 : {type(cfg).__module__}.{type(cfg).__qualname__}")
    print(f"command_type             : {type(cmd).__module__}.{type(cmd).__qualname__}")

    for name in (
        "resampling_time_range",
        "rel_standing_envs",
        "rel_heading_envs",
        "heading_command",
        "heading_control_stiffness",
        "debug_vis",
    ):
        print(f"{name:<25}: {fmt(getattr(cmd, name, None))}")

    ranges = getattr(cmd, "ranges", None)
    print(f"{'ranges':<25}: {fmt(ranges)}")

    if ranges is not None:
        print(f"{'ranges.lin_vel_x':<25}: {fmt(getattr(ranges, 'lin_vel_x', None))}")
        print(f"{'ranges.lin_vel_y':<25}: {fmt(getattr(ranges, 'lin_vel_y', None))}")
        print(f"{'ranges.ang_vel_z':<25}: {fmt(getattr(ranges, 'ang_vel_z', None))}")
        print(f"{'ranges.heading':<25}: {fmt(getattr(ranges, 'heading', None))}")

    for name in (
        "asset_name",
        "simple_heading",
    ):
        if hasattr(cmd, name):
            print(f"{name:<25}: {fmt(getattr(cmd, name))}")

    print("")


def main() -> None:
    print("=" * 132)
    print("v2-G / v4-N TRAINING COMMAND SUPPORT AUDIT")
    print("=" * 132)
    print("policy inference          : ABSENT")
    print("env instantiation         : ABSENT")
    print("env.step                  : ABSENT")
    print("scene.update              : ABSENT")
    print("physics advance           : ABSENT")
    print("purpose                   : establish training command support before router-threshold calibration")
    print("")

    print_command_cfg(V2G_TASK, "FROZEN v2-G STAND TASK")
    print_command_cfg(V4N_TASK, "FROZEN v4-N LOCOMOTION TASK")

    print("=" * 132)
    print("INTERPRETATION RULE")
    print("=" * 132)
    print(
        "Use these ranges only to define where calibration experiments are scientifically admissible. "
        "Do NOT infer a final ENTER/EXIT threshold from configuration alone."
    )
    print(
        "Next dynamic calibration should remain inside the locomotion policy's trained command support "
        "and should use development seeds only; holdout remains untouched."
    )
    print(PASS_MARKER)


try:
    main()
except BaseException as exc:
    print("SCRIPT_RUNTIME_FAIL", flush=True)
    print(f"EXCEPTION_TYPE={type(exc).__name__}", flush=True)
    print(f"EXCEPTION_MESSAGE={exc}", flush=True)
    traceback.print_exc()
    raise
finally:
    gc.collect()
    simulation_app.close()
