# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

import warnings

warnings.warn(
    "scripts/reinforcement_learning/rsl_rl/play.py is deprecated. Use "
    "`./isaaclab.sh play --rl_library rsl_rl --task <TASK>` instead. "
    "Example: `./isaaclab.sh play --rl_library rsl_rl --task Isaac-Cartpole-v0`.",
    DeprecationWarning,
    stacklevel=1,
)

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import humanoid_rl_lab.tasks  # noqa: F401  # project-owned task registry
from isaaclab_tasks.utils import (
    add_launcher_args,
    get_checkpoint_path,
    launch_simulation,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import sys
sys.path.insert(0, r"C:\IsaacLab\scripts\reinforcement_learning\rsl_rl")
import cli_args  # isort: skip

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.video:
    args_cli.enable_cameras = True


# Call an external callback if requested. This gives opportunity to external code to register the environments
# The function is expected to return a list of arguments that were not consumed by the callback.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# clear out sys.argv for Hydra
# The remaining arguments are the arguments that were not consumed by both this scripts
# argparser and (optionally) the external callback function. Both sides of this
# intersection are pre-fold (the callback reads the user's original sys.argv), so
# preset tokens like ``physics=NAME`` compare correctly here. Fold runs after.
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

# Check for installed RSL-RL version
installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        # grab task name for checkpoint path
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        # handle deprecated configurations
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed

        # ==========================================================================================
        # THESIS CONTROLLED A/B PROBE — OFFICIAL NVIDIA G1 POLICY AT FIXED SCIENTIFIC COMMAND
        # Presentation/evaluation-only copy. The Isaac Lab source tree and checkpoint are untouched.
        # ==========================================================================================
        _cmd = env_cfg.commands.base_velocity
        _cmd.heading_command = False
        _cmd.rel_heading_envs = 0.0
        _cmd.ranges.lin_vel_x = (0.40, 0.40)
        _cmd.ranges.lin_vel_y = (0.0, 0.0)
        _cmd.ranges.ang_vel_z = (0.0, 0.0)

        print("=" * 100)
        print("G1 FROZEN V2-G FIXED-0.40 PLAY-EQUIVALENT CONTROLLED VISUAL PROBE")
        print("Command                             : [0.40, 0.00, 0.00]")
        print("heading_command                     : False")
        print("Policy                              : user-supplied frozen v2-G checkpoint")
        print("Purpose                             : controlled qualitative A/B vs frozen v2-G")
        print("Isaac Lab source modified           : NO")
        print("=" * 100)
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # create isaac environment

        # ==========================================================================================
        # PLAY-EQUIVALENT EVALUATION CONDITIONS
        # Match official G1 Flat PLAY evaluation semantics:
        #   - disable observation corruption
        #   - disable external-force randomization
        #   - disable random pushes
        # Policy/checkpoint/rewards/actions/command remain unchanged.
        # ==========================================================================================
        if hasattr(env_cfg.observations.policy, "enable_corruption"):
            env_cfg.observations.policy.enable_corruption = False
        if hasattr(env_cfg, "events"):
            if hasattr(env_cfg.events, "base_external_force_torque"):
                env_cfg.events.base_external_force_torque = None
            if hasattr(env_cfg.events, "push_robot"):
                env_cfg.events.push_robot = None

        print("=" * 100)
        print("V2-G PLAY-EQUIVALENT EVALUATION OVERRIDES")
        print("observation corruption              : DISABLED")
        print("base external-force randomization   : DISABLED")
        print("random pushes                       : DISABLED")
        print("checkpoint / policy / rewards       : UNCHANGED")
        print("=" * 100)

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        # configure_seed must be called after runner construction so that PyTorch deterministic settings
        # do not interfere with the runner's internal initialization.
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)
        runner.load(resume_path)

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # export the trained policy to JIT and ONNX formats
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

        if version.parse(installed_version) >= version.parse("4.0.0"):
            # use the new export functions for rsl-rl >= 4.0.0
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            policy_nn = None  # Not needed for rsl-rl >= 4.0.0
        else:
            # extract the neural network for rsl-rl < 4.0.0
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

            # extract the normalizer
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            # export to JIT and ONNX
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        dt = env.unwrapped.step_dt

        # reset environment
        obs = env.get_observations()
        timestep = 0

        # ==========================================================================================
        # THESIS A/B GAIT PROBE — logging only; policy, physics, rewards and checkpoint untouched.
        # 3 s fixed-.40 warm-up are discarded, then exactly 600 samples (12.00 s at step_dt=.02).
        # ==========================================================================================
        import csv as _probe_csv
        import hashlib as _probe_hashlib
        import statistics as _probe_statistics
        from pathlib import Path as _ProbePath

        _PROBE_WARMUP_STEPS = 150
        _PROBE_RECORD_STEPS = 600
        _PROBE_TOTAL_STEPS = _PROBE_WARMUP_STEPS + _PROBE_RECORD_STEPS
        _probe_total_step = 0
        _probe_rows = []
        _probe_reset_count = 0

        _uenv = env.unwrapped
        _robot = _uenv.scene["robot"]
        _contact = _uenv.scene.sensors["contact_forces"]

        def _probe_torch(value):
            return value.torch if hasattr(value, "torch") else value

        _joint_names = list(_robot.joint_names)
        _robot_body_names = list(_robot.body_names)
        _contact_body_names = list(_contact.body_names)

        _left_body_name = "left_ankle_roll_link"
        _right_body_name = "right_ankle_roll_link"
        if _left_body_name not in _robot_body_names or _right_body_name not in _robot_body_names:
            raise RuntimeError(
                f"ankle bodies missing from robot body_names: left={_left_body_name in _robot_body_names}, "
                f"right={_right_body_name in _robot_body_names}"
            )
        if _left_body_name not in _contact_body_names or _right_body_name not in _contact_body_names:
            raise RuntimeError(
                f"ankle bodies missing from contact sensor body_names: left={_left_body_name in _contact_body_names}, "
                f"right={_right_body_name in _contact_body_names}"
            )

        _li = _robot_body_names.index(_left_body_name)
        _ri = _robot_body_names.index(_right_body_name)
        _lci = _contact_body_names.index(_left_body_name)
        _rci = _contact_body_names.index(_right_body_name)

        _probe_out_dir = _ProbePath(
            r"C:\humanoid_rl_lab\logs\thesis\gait_probe\frozen_v2g_fixed040_playeq"
        )
        _probe_out_dir.mkdir(parents=True, exist_ok=True)
        _probe_csv_path = _probe_out_dir / "frozen_v2g_fixed040_playeq_12s_trajectory.csv"
        _probe_meta_path = _probe_out_dir / "frozen_v2g_fixed040_playeq_12s_meta.txt"

        def _probe_sha256(path):
            h = _probe_hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        print("=" * 100)
        print("G1 FROZEN V2-G FIXED-0.40 PLAY-EQUIVALENT — QUANTITATIVE GAIT PROBE")
        print(f"step_dt                             : {dt}")
        print(f"warm-up                             : {_PROBE_WARMUP_STEPS} steps = {_PROBE_WARMUP_STEPS * dt:.2f} s")
        print(f"record                              : {_PROBE_RECORD_STEPS} steps = {_PROBE_RECORD_STEPS * dt:.2f} s")
        print(f"output CSV                          : {_probe_csv_path}")
        print("Evidence                            : FROZEN V2-G / CONTROLLED FIXED COMMAND / LOGGING ONLY")
        print("=" * 100)
        # simulate environment
        try:
            while True:
                start_time = time.time()
                # run everything in inference mode
                with torch.inference_mode():
                    # agent stepping
                    actions = policy(obs)
                    # env stepping
                    obs, _, dones, _ = env.step(actions)

                    _probe_total_step += 1
                    _probe_reset_count += int(torch.count_nonzero(dones).item())

                    if _probe_total_step > _PROBE_WARMUP_STEPS:
                        _sample_idx = _probe_total_step - _PROBE_WARMUP_STEPS

                        _root_pos = _probe_torch(_robot.data.root_pos_w)[0]
                        _root_quat = _probe_torch(_robot.data.root_quat_w)[0]
                        _root_lin_b = _probe_torch(_robot.data.root_lin_vel_b)[0]
                        _root_ang_b = _probe_torch(_robot.data.root_ang_vel_b)[0]
                        _joint_pos = _probe_torch(_robot.data.joint_pos)[0]
                        _joint_vel = _probe_torch(_robot.data.joint_vel)[0]
                        _body_pos = _probe_torch(_robot.data.body_pos_w)[0]
                        _body_lin_vel = _probe_torch(_robot.data.body_lin_vel_w)[0]
                        _contact_time = _probe_torch(_contact.data.current_contact_time)[0]
                        _cmd = _probe_torch(_uenv.command_manager.get_command("base_velocity"))[0]
                        _act = actions[0]

                        row = {
                            "sample": int(_sample_idx),
                            "sim_time_s": float(_sample_idx * dt),
                            "cmd_vx": float(_cmd[0].item()),
                            "cmd_vy": float(_cmd[1].item()),
                            "cmd_wz": float(_cmd[2].item()),
                            "root_x_w": float(_root_pos[0].item()),
                            "root_y_w": float(_root_pos[1].item()),
                            "root_z_w": float(_root_pos[2].item()),
                            "root_qw": float(_root_quat[0].item()),
                            "root_qx": float(_root_quat[1].item()),
                            "root_qy": float(_root_quat[2].item()),
                            "root_qz": float(_root_quat[3].item()),
                            "root_vx_b": float(_root_lin_b[0].item()),
                            "root_vy_b": float(_root_lin_b[1].item()),
                            "root_vz_b": float(_root_lin_b[2].item()),
                            "root_wx_b": float(_root_ang_b[0].item()),
                            "root_wy_b": float(_root_ang_b[1].item()),
                            "root_wz_b": float(_root_ang_b[2].item()),
                            "left_foot_x_w": float(_body_pos[_li, 0].item()),
                            "left_foot_y_w": float(_body_pos[_li, 1].item()),
                            "left_foot_z_w": float(_body_pos[_li, 2].item()),
                            "right_foot_x_w": float(_body_pos[_ri, 0].item()),
                            "right_foot_y_w": float(_body_pos[_ri, 1].item()),
                            "right_foot_z_w": float(_body_pos[_ri, 2].item()),
                            "left_foot_vx_w": float(_body_lin_vel[_li, 0].item()),
                            "left_foot_vy_w": float(_body_lin_vel[_li, 1].item()),
                            "left_foot_vz_w": float(_body_lin_vel[_li, 2].item()),
                            "right_foot_vx_w": float(_body_lin_vel[_ri, 0].item()),
                            "right_foot_vy_w": float(_body_lin_vel[_ri, 1].item()),
                            "right_foot_vz_w": float(_body_lin_vel[_ri, 2].item()),
                            "left_contact": int(float(_contact_time[_lci].item()) > 0.0),
                            "right_contact": int(float(_contact_time[_rci].item()) > 0.0),
                            "left_contact_time_s": float(_contact_time[_lci].item()),
                            "right_contact_time_s": float(_contact_time[_rci].item()),
                        }

                        for _j, _name in enumerate(_joint_names):
                            row[f"q::{_name}"] = float(_joint_pos[_j].item())
                            row[f"qd::{_name}"] = float(_joint_vel[_j].item())
                            if _j < int(_act.shape[0]):
                                row[f"action::{_name}"] = float(_act[_j].item())

                        _probe_rows.append(row)

                    if _probe_total_step >= _PROBE_TOTAL_STEPS:
                        if len(_probe_rows) != _PROBE_RECORD_STEPS:
                            raise RuntimeError(
                                f"probe row count mismatch: {len(_probe_rows)} != {_PROBE_RECORD_STEPS}"
                            )

                        with _probe_csv_path.open("w", newline="", encoding="utf-8") as _f:
                            _writer = _probe_csv.DictWriter(_f, fieldnames=list(_probe_rows[0].keys()))
                            _writer.writeheader()
                            _writer.writerows(_probe_rows)

                        _vx_mean = _probe_statistics.fmean(r["root_vx_b"] for r in _probe_rows)
                        _z_mean = _probe_statistics.fmean(r["root_z_w"] for r in _probe_rows)
                        _cmd_vx_min = min(r["cmd_vx"] for r in _probe_rows)
                        _cmd_vx_max = max(r["cmd_vx"] for r in _probe_rows)
                        _csv_sha = _probe_sha256(_probe_csv_path)
                        _checkpoint_sha = _probe_sha256(resume_path)

                        _meta = "\n".join([
                            "G1 FROZEN V2-G FIXED-0.40 PLAY-EQUIVALENT QUANTITATIVE GAIT PROBE",
                            f"task={args_cli.task}",
                            f"checkpoint={resume_path}",
                            f"checkpoint_sha256={_checkpoint_sha}",
                            f"step_dt={dt}",
                            f"warmup_steps={_PROBE_WARMUP_STEPS}",
                            f"record_steps={_PROBE_RECORD_STEPS}",
                            f"record_duration_s={_PROBE_RECORD_STEPS * dt}",
                            f"cmd_vx_min={_cmd_vx_min}",
                            f"cmd_vx_max={_cmd_vx_max}",
                            f"mean_root_vx_b={_vx_mean}",
                            f"mean_root_z_w={_z_mean}",
                            f"reset_count={_probe_reset_count}",
                            f"csv={_probe_csv_path}",
                            f"csv_sha256={_csv_sha}",
                        ]) + "\n"
                        _probe_meta_path.write_text(_meta, encoding="utf-8")

                        print("=" * 100)
                        print("G1 FROZEN V2-G FIXED-0.40 PLAY-EQUIVALENT — PROBE COMPLETE")
                        print(f"Recorded samples                    : {len(_probe_rows)}")
                        print(f"Recorded simulation duration        : {_PROBE_RECORD_STEPS * dt:.2f} s")
                        print(f"Effective command vx range          : [{_cmd_vx_min:.6f}, {_cmd_vx_max:.6f}]")
                        print(f"Mean body vx                        : {_vx_mean:.6f} m/s")
                        print(f"Mean root z                         : {_z_mean:.6f} m")
                        print(f"Environment resets                  : {_probe_reset_count}")
                        print(f"Trajectory SHA256                   : {_csv_sha}")
                        print(f"Checkpoint SHA256                   : {_checkpoint_sha}")
                        print(f"CSV                                 : {_probe_csv_path}")
                        print(f"Metadata                            : {_probe_meta_path}")
                        print("G1_OFFICIAL_FIXED040_GAIT_PROBE_PASS")
                        print("=" * 100)
                        break
                    # reset recurrent states for episodes that have terminated
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)
                if args_cli.video:
                    timestep += 1
                    if timestep == args_cli.video_length:
                        break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            # close the simulator
            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
