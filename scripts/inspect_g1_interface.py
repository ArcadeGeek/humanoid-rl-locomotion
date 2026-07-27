# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""Inspect the runtime interface of a registered Isaac Lab G1 task."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect an Isaac Lab G1 task interface.")
    parser.add_argument(
        "--task",
        type=str,
        default="Humanoid-RL-Lab-G1-Flat-Play-v0",
        help="Registered Gym task identifier.",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
    parser.add_argument(
        "--report",
        type=str,
        default="",
        help="Optional report path. Defaults to Desktop\\humanoid_rl_lab_g1_interface_report.txt.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def shape_of(value: Any) -> str:
    shape = getattr(value, "shape", None)
    return str(tuple(shape)) if shape is not None else "<no shape>"


def safe_get(obj: Any, name: str, default: Any = "<unavailable>") -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def header(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def print_manager(title: str, manager: Any) -> None:
    header(title)
    print(manager)

    for attr in (
        "active_terms",
        "total_action_dim",
        "group_obs_dim",
        "group_obs_term_dim",
        "_group_obs_term_names",
        "_group_obs_term_dim",
    ):
        value = safe_get(manager, attr, None)
        if value is not None:
            print(f"{attr}: {value}")

    terms = safe_get(manager, "_terms", None)
    if terms is None:
        return

    iterable = terms.items() if isinstance(terms, dict) else enumerate(terms)
    print("runtime_terms:")
    for key, term in iterable:
        print(f"  - key={key!r}")
        print(f"    class={type(term).__module__}.{type(term).__name__}")
        for attr in ("action_dim", "joint_ids", "joint_names", "_joint_ids", "_joint_names"):
            value = safe_get(term, attr, None)
            if value is not None:
                print(f"    {attr}={value}")
        cfg = safe_get(term, "cfg", None)
        if cfg is not None:
            print(f"    cfg={cfg}")


def print_cfg(title: str, obj: Any) -> None:
    header(title)
    if obj is None:
        print("<none>")
        return
    if hasattr(obj, "to_dict"):
        try:
            print(obj.to_dict())
            return
        except Exception:
            pass
    print(obj)


def main() -> int:
    parser = build_parser()
    args_cli = parser.parse_args()

    if args_cli.num_envs < 1:
        parser.error("--num_envs must be at least 1")

    if args_cli.report:
        report_path = Path(args_cli.report).expanduser().resolve()
    else:
        desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
        report_path = desktop / "humanoid_rl_lab_g1_interface_report.txt"

    report_path.parent.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    env = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        with report_path.open("w", encoding="utf-8", newline="\n") as report:
            with contextlib.redirect_stdout(Tee(original_stdout, report)), contextlib.redirect_stderr(
                Tee(original_stderr, report)
            ):
                import gymnasium as gym
                import torch

                import humanoid_rl_lab.tasks  # noqa: F401
                from isaaclab_tasks.utils import parse_env_cfg

                header("HUMANOID RL LAB — G1 RUNTIME INTERFACE REPORT")
                print(f"task: {args_cli.task}")
                print(f"requested_num_envs: {args_cli.num_envs}")
                print(f"report: {report_path}")
                print(f"python: {sys.executable}")
                print(f"torch: {torch.__version__}")
                print(f"cuda_available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"gpu: {torch.cuda.get_device_name(0)}")

                env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

                header("CONFIGURATION CLASSES")
                print(f"env_cfg_class: {type(env_cfg).__module__}.{type(env_cfg).__name__}")
                print(f"scene_cfg_class: {type(env_cfg.scene).__module__}.{type(env_cfg.scene).__name__}")
                print(f"actions_cfg_class: {type(env_cfg.actions).__module__}.{type(env_cfg.actions).__name__}")
                print(f"observations_cfg_class: {type(env_cfg.observations).__module__}.{type(env_cfg.observations).__name__}")
                print(f"commands_cfg_class: {type(env_cfg.commands).__module__}.{type(env_cfg.commands).__name__}")

                sim_dt = float(env_cfg.sim.dt)
                decimation = int(env_cfg.decimation)
                header("TIMING")
                print(f"sim_dt_s: {sim_dt}")
                print(f"physics_hz: {1.0 / sim_dt}")
                print(f"decimation: {decimation}")
                print(f"policy_dt_s: {sim_dt * decimation}")
                print(f"policy_hz: {1.0 / (sim_dt * decimation)}")
                print(f"episode_length_s: {env_cfg.episode_length_s}")
                print(f"scene_num_envs: {env_cfg.scene.num_envs}")
                print(f"scene_env_spacing: {env_cfg.scene.env_spacing}")
                print(f"device: {env_cfg.sim.device}")

                print_cfg("ACTIONS CONFIGURATION", env_cfg.actions)
                print_cfg("OBSERVATIONS CONFIGURATION", env_cfg.observations)
                print_cfg("COMMANDS CONFIGURATION", env_cfg.commands)

                env = gym.make(args_cli.task, cfg=env_cfg)
                base_env = env.unwrapped

                header("GYM SPACES")
                print(f"observation_space: {env.observation_space}")
                print(f"action_space: {env.action_space}")
                print(f"base_env_class: {type(base_env).__module__}.{type(base_env).__name__}")
                print(f"base_env_device: {safe_get(base_env, 'device')}")
                print(f"base_env_num_envs: {safe_get(base_env, 'num_envs')}")

                obs, info = env.reset()

                header("RESET OBSERVATIONS")
                print(f"observation_container_type: {type(obs).__module__}.{type(obs).__name__}")
                if isinstance(obs, dict):
                    for key, value in obs.items():
                        print(f"{key}: shape={shape_of(value)}, dtype={getattr(value, 'dtype', '<n/a>')}")
                else:
                    print(f"shape={shape_of(obs)}, dtype={getattr(obs, 'dtype', '<n/a>')}")
                print(f"info_type: {type(info).__module__}.{type(info).__name__}")

                print_manager("ACTION MANAGER", base_env.action_manager)
                print_manager("OBSERVATION MANAGER", base_env.observation_manager)
                print_manager("COMMAND MANAGER", base_env.command_manager)

                robot = base_env.scene["robot"]
                header("ROBOT ARTICULATION")
                print(f"robot_class: {type(robot).__module__}.{type(robot).__name__}")
                joint_names = safe_get(robot, "joint_names", None)
                if joint_names is None:
                    joint_names = safe_get(safe_get(robot, "data", None), "joint_names", None)
                print(f"joint_count: {len(joint_names) if joint_names is not None else '<unavailable>'}")
                print(f"joint_names: {joint_names}")

                default_joint_pos = safe_get(safe_get(robot, "data", None), "default_joint_pos", None)
                if default_joint_pos is not None:
                    print(f"default_joint_pos_shape: {shape_of(default_joint_pos)}")
                    try:
                        print(f"default_joint_pos_env0: {default_joint_pos[0].detach().cpu().tolist()}")
                    except Exception:
                        print(f"default_joint_pos: {default_joint_pos}")

                header("ZERO-ACTION STEP")
                action_dim = int(base_env.action_manager.total_action_dim)
                num_envs = int(base_env.num_envs)
                actions = torch.zeros((num_envs, action_dim), device=base_env.device)
                print(f"zero_action_shape: {tuple(actions.shape)}")
                step_result = env.step(actions)
                print(f"step_result_items: {len(step_result)}")
                next_obs = step_result[0]
                if isinstance(next_obs, dict):
                    for key, value in next_obs.items():
                        print(f"next_obs[{key!r}]: shape={shape_of(value)}")
                else:
                    print(f"next_obs_shape: {shape_of(next_obs)}")
                print("zero_action_step: OK")

                header("SUMMARY")
                policy_obs = obs.get("policy") if isinstance(obs, dict) else obs
                print(f"policy_observation_shape: {shape_of(policy_obs)}")
                print(f"total_action_dim: {action_dim}")
                print(f"robot_joint_count: {len(joint_names) if joint_names is not None else '<unavailable>'}")
                print("inspection_status: SUCCESS")

        original_stdout.write(f"\nReport saved to: {report_path}\n")
        original_stdout.flush()
        return 0

    except Exception:
        with report_path.open("a", encoding="utf-8", newline="\n") as report:
            report.write("\n" + "=" * 88 + "\n")
            report.write("INSPECTION FAILED\n")
            report.write("=" * 88 + "\n")
            traceback.print_exc(file=report)
        traceback.print_exc(file=original_stderr)
        original_stderr.write(f"\nFailure report saved to: {report_path}\n")
        original_stderr.flush()
        return 1

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
