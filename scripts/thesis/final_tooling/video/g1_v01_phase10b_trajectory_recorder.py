#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import statistics
import sys
import threading
import time
from pathlib import Path

parser = argparse.ArgumentParser(
    description=(
        "Phase 9B: end-to-end ROS 2 -> frozen v2-G straight locomotion deployment rehearsal. "
        "Only zero and the scientific command [0.40, 0, 0] are accepted."
    )
)
parser.add_argument("--max-runtime-s", type=float, default=180.0)
parser.add_argument("--watchdog-s", type=float, default=0.50)
parser.add_argument("--status-hz", type=float, default=5.0)
parser.add_argument("--telemetry-hz", type=float, default=20.0)
parser.add_argument("--real-time", action="store_true", default=True)
parser.add_argument("--spin-margin-ms", type=float, default=1.0)
parser.add_argument(
    "--trajectory-out-dir",
    default=r"C:\humanoid_rl_lab\logs\thesis\video_final\V01_v2g_scientific_straight_canonical_50hz",
)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

TASK_ID = "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-v0"
CHECKPOINT = Path(
    r"C:\humanoid_rl_lab\checkpoints\local_g1_yaw_direct_wide_yaw_flat_orientation_v2g\model_600.pt"
)
EXPECTED_SHA256 = "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3"

OUT_DIR = Path(
    r"C:\humanoid_rl_lab\logs\thesis\g1_ros_deployment_audit\phase10b_ethernet_canonical_soak"
)
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

report = {
    "phase": "10B",
    "purpose": "120 s end-to-end Ethernet final soak with scientific straight and engineering forward+yaw segments",
    "scientific_command": list(SCI_CMD),
    "task_id": TASK_ID,
    "checkpoint": str(CHECKPOINT),
    "expected_sha256": EXPECTED_SHA256,
    "training_run": False,
    "repository_source_modified": False,
    "checkpoint_modified": False,
    "accepted_commands_only": [list(ZERO_CMD), list(SCI_CMD), list(CURVE_POS_CMD), list(CURVE_NEG_CMD)],
    "scientific_vs_engineering": {"straight_040": "scientific", "curve_pos_040_025": "engineering", "curve_neg_040_m025": "engineering"},
    "classification": "final deployment soak; straight .40 is scientific protocol, forward+yaw ±.25 are engineering checks",
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
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Canonical checkpoint missing: {CHECKPOINT}")

    actual_sha = sha256_file(CHECKPOINT)
    report["checkpoint_sha256"] = actual_sha
    report["checkpoint_sha_pass"] = actual_sha == EXPECTED_SHA256
    if actual_sha != EXPECTED_SHA256:
        raise RuntimeError(
            f"Checkpoint SHA mismatch: expected {EXPECTED_SHA256}, got {actual_sha}"
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

    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

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
    agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")

    installed_rsl = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl)

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        if hasattr(agent_cfg, "device"):
            agent_cfg.device = args_cli.device

    if hasattr(env_cfg, "seed"):
        env_cfg.seed = 42
    if hasattr(agent_cfg, "seed"):
        agent_cfg.seed = 42

    env_cfg.scene.num_envs = 1
    env_cfg.log_dir = str(OUT_DIR)

    # Deployment-only in-memory guardrails, already validated in Phase 6D.
    env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    reset_base = getattr(env_cfg.events, "reset_base", None)
    if reset_base is not None:
        pose_range = reset_base.params.get("pose_range", {})
        velocity_range = reset_base.params.get("velocity_range", {})
        for key in list(pose_range.keys()):
            pose_range[key] = (0.0, 0.0)
        for key in list(velocity_range.keys()):
            velocity_range[key] = (0.0, 0.0)

    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.ranges.lin_vel_x = (0.0, 0.0)
    cmd_cfg.ranges.lin_vel_y = (0.0, 0.0)
    cmd_cfg.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(cmd_cfg, "debug_vis"):
        cmd_cfg.debug_vis = False

    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    uenv = env.unwrapped

    env_ids = torch.tensor([0], dtype=torch.int32, device=uenv.device)
    uenv.command_manager.reset(env_ids=env_ids)
    cmd_tensor = uenv.command_manager.get_command("base_velocity")
    cmd_tensor.zero_()

    if agent_cfg.class_name != "OnPolicyRunner":
        raise RuntimeError(f"Expected OnPolicyRunner, got {agent_cfg.class_name!r}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(CHECKPOINT))
    policy = runner.get_inference_policy(device=uenv.device)

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

    if abs(float(uenv.step_dt) - 0.02) > 1.0e-9:
        raise RuntimeError(f"Unexpected step_dt: {uenv.step_dt}")
    if int(uenv.action_manager.total_action_dim) != 37:
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
            super().__init__("g1_phase10_windows_ethernet_final_soak_runner")
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

    step_count = 0
    # G1_V01_CANONICAL_TRAJECTORY_RECORDER
    # Buffer replay state only in RAM; disk I/O occurs after the live loop.
    trajectory_rows = []
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

    status_period = 1.0 / args_cli.status_hz
    telemetry_period = 1.0 / args_cli.telemetry_hz
    last_status_pub = 0.0
    last_telemetry_pub = 0.0

    start_wall = time.perf_counter()
    next_deadline = start_wall
    spin_margin_s = max(0.0, args_cli.spin_margin_ms / 1000.0)
    obs = env.get_observations()

    print("=" * 112)
    print("G1 ROS ↔ ISAAC — PHASE 10B ETHERNET CANONICAL 120 s SOAK")
    print("=" * 112)
    print(f"Checkpoint SHA                      : PASS ({actual_sha})")
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

        age = None if snap["last_valid_rx"] is None else loop_start - snap["last_valid_rx"]
        watchdog = age is None or age > args_cli.watchdog_s

        with ros_state.lock:
            if not watchdog:
                ros_state.have_ever_been_fresh = True
            elif ros_state.have_ever_been_fresh:
                ros_state.watchdog_after_fresh = True

        effective = ZERO_CMD if watchdog else snap["latest_cmd"]

        cmd_tensor[0, 0] = effective[0]
        cmd_tensor[0, 1] = effective[1]
        cmd_tensor[0, 2] = effective[2]

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

        obs = env.get_observations()

        with torch.inference_mode():
            actions = policy(obs)

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

        # Canonical 50 Hz trajectory snapshot (RAM only).
        root_lin_w = robot_vec(robot, "root_lin_vel_w")
        root_ang_w = robot_vec(robot, "root_ang_vel_w")
        joint_pos_row = robot.data.joint_pos[0].detach().cpu().tolist()
        joint_vel_row = robot.data.joint_vel[0].detach().cpu().tolist()
        action_row = actions[0].detach().cpu().tolist()
        trajectory_rows.append(
            [
                int(step_count),
                float(step_count * dt),
                float(time.perf_counter() - start_wall),
                float(effective[0]),
                float(effective[1]),
                float(effective[2]),
                *[float(x) for x in root_pos[:3].detach().cpu().tolist()],
                *[float(x) for x in root_quat[:4].detach().cpu().tolist()],
                *[float(x) for x in root_lin_w[:3].detach().cpu().tolist()],
                *[float(x) for x in root_ang_w[:3].detach().cpu().tolist()],
                *[float(x) for x in joint_pos_row],
                *[float(x) for x in joint_vel_row],
                *[float(x) for x in action_row],
            ]
        )

        root_z = float(root_pos[2].item())
        lin_speed_b = float(torch.linalg.norm(root_lin_b[:3]).item())
        ang_speed_b = float(torch.linalg.norm(root_ang_b[:3]).item())

        min_root_z = min(min_root_z, root_z)
        max_root_z = max(max_root_z, root_z)
        max_root_lin_speed_body = max(max_root_lin_speed_body, lin_speed_b)
        max_root_ang_speed_body = max(max_root_ang_speed_body, ang_speed_b)

        now = time.perf_counter()

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
                "phase": "10B",
                "policy_loaded": True,
                "checkpoint_sha_pass": True,
                "scientific_command": list(SCI_CMD),
                "engineering_curve_pos_command": list(CURVE_POS_CMD),
                "engineering_curve_neg_command": list(CURVE_NEG_CMD),
                "step_count": step_count,
                "runtime_s": now - start_wall,
                "watchdog_active": bool(
                    age_ms is None or age_ms > args_cli.watchdog_s * 1000.0
                ),
                "cmd_age_ms": age_ms,
                "effective_cmd": list(effective),
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

    trajectory_out_dir = Path(args_cli.trajectory_out_dir)
    trajectory_out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_csv = trajectory_out_dir / "g1_v01_canonical_50hz_trajectory.csv"
    trajectory_meta = trajectory_out_dir / "g1_v01_canonical_50hz_trajectory_meta.json"

    joint_count = 37
    action_count = 37
    if trajectory_rows:
        expected_cols = 19 + joint_count + joint_count + action_count
        if len(trajectory_rows[0]) != expected_cols:
            raise RuntimeError(
                f"Unexpected trajectory width: got {len(trajectory_rows[0])}, expected {expected_cols}"
            )

    header = [
        "step", "sim_time_s", "wall_elapsed_s",
        "cmd_vx", "cmd_vy", "cmd_wz",
        "root_x", "root_y", "root_z",
        "root_qw", "root_qx", "root_qy", "root_qz",
        "root_vx_w", "root_vy_w", "root_vz_w",
        "root_wx_w", "root_wy_w", "root_wz_w",
    ]
    header += [f"joint_pos_{i:02d}" for i in range(joint_count)]
    header += [f"joint_vel_{i:02d}" for i in range(joint_count)]
    header += [f"action_{i:02d}" for i in range(action_count)]

    with trajectory_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(trajectory_rows)

    trajectory_sha256 = hashlib.sha256(trajectory_csv.read_bytes()).hexdigest()

    try:
        joint_names = list(robot.joint_names)
    except Exception:
        joint_names = [f"joint_{i:02d}" for i in range(joint_count)]

    trajectory_meta_payload = {
        "classification": "canonical 50 Hz ROS deployment rollout state capture for deterministic visual replay",
        "task_id": TASK_ID,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": actual_sha,
        "step_dt_s": float(dt),
        "sample_rate_hz": float(1.0 / dt),
        "sample_count": int(len(trajectory_rows)),
        "joint_count": int(joint_count),
        "action_count": int(action_count),
        "joint_names": joint_names,
        "source_frame_id": EXPECTED_SOURCE_FRAME_ID,
        "physics_modified": False,
        "policy_modified": False,
        "repository_modified": False,
        "per_step_disk_io": False,
        "trajectory_csv": str(trajectory_csv),
        "trajectory_csv_sha256": trajectory_sha256,
    }
    trajectory_meta.write_text(
        json.dumps(trajectory_meta_payload, indent=2),
        encoding="utf-8",
    )

    print()
    print("G1 V01 CANONICAL 50 HZ TRAJECTORY RECORDER")
    print("-" * 112)
    print(f"Samples                             : {len(trajectory_rows)}")
    print(f"Step dt                             : {dt}")
    print(f"Trajectory CSV                      : {trajectory_csv}")
    print(f"Trajectory SHA256                   : {trajectory_sha256}")
    print(f"Trajectory metadata                 : {trajectory_meta}")
    print("Per-step disk I/O                   : NO")
    print("Physics modified                    : NO")
    print("Policy modified                     : NO")
    print("G1_V01_CANONICAL_TRAJECTORY_CAPTURE_PASS")

    final_snap = ros_state.snapshot()

    compute_stats = summarize_ms(compute_ms_samples)
    period_stats = summarize_ms(loop_period_ms_samples)
    overrun_pct = (100.0 * loop_overruns / step_count) if step_count > 0 else None
    achieved_loop_hz = (
        1000.0 / period_stats["mean_ms"]
        if period_stats is not None and period_stats["mean_ms"] > 0.0
        else None
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
            "max_action_abs": max_action_abs,
            "max_root_lin_speed_body": max_root_lin_speed_body,
            "max_root_ang_speed_body": max_root_ang_speed_body,
            "min_root_z": None if min_root_z == float("inf") else min_root_z,
            "max_root_z": None if max_root_z == float("-inf") else max_root_z,
        }
    )

    guards = {
        "checkpoint_sha_pass": report["checkpoint_sha_pass"],
        "policy_loaded": report["policy_loaded"],
        "step_dt_50hz": abs(dt - 0.02) <= 1.0e-9,
        "received_zero_commands": final_snap["valid_zero_count"] > 0,
        "received_scientific_040_commands": final_snap["valid_040_count"] > 0,
        "received_curve_pos_commands": final_snap["valid_curve_pos_count"] > 0,
        "received_curve_neg_commands": final_snap["valid_curve_neg_count"] > 0,
        "effective_040_applied": accepted_040_effective_steps > 0,
        "effective_curve_pos_applied": accepted_curve_pos_effective_steps > 0,
        "effective_curve_neg_applied": accepted_curve_neg_effective_steps > 0,
        "no_rejected_commands": final_snap["rejected_command_count"] == 0,
        "no_invalid_ros_commands": final_snap["invalid_count"] == 0,
        "no_source_mismatch": final_snap["source_mismatch_count"] == 0,
        "single_cmd_publisher_max": final_snap["max_cmd_publishers_seen"] <= 1,
        "no_environment_termination": reset_count == 0,
        "finite_policy_actions": nonfinite_action_count == 0,
        "meaningful_rollout": step_count >= 5500,
        "reward_bypass_active": report.get("reward_bypass_active") is True,
    }
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

except Exception as exc:
    report["errors"].append(repr(exc))
    report["status"] = "FAIL"

finally:
    json_path = OUT_DIR / "g1_ros_phase10b_ethernet_canonical_soak.json"
    txt_path = OUT_DIR / "g1_ros_phase10b_ethernet_canonical_soak.txt"

    try:
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        lines = [
            "G1 ROS ↔ ISAAC — PHASE 10B ETHERNET CANONICAL 120 s SOAK",
            "=" * 112,
            f"Status                              : {report.get('status')}",
            f"Task ID                             : {TASK_ID}",
            f"Checkpoint SHA PASS                 : {report.get('checkpoint_sha_pass')}",
            f"Policy loaded                       : {report.get('policy_loaded')}",
            f"Runtime s                           : {report.get('runtime_s')}",
            f"Step count                          : {report.get('step_count')}",
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
        print("PHASE 10B WINDOWS FINAL SUMMARY")
        print("=" * 112)
        print(f"Status                              : {report.get('status')}")
        print(f"Checkpoint SHA PASS                 : {report.get('checkpoint_sha_pass')}")
        print(f"Policy loaded                       : {report.get('policy_loaded')}")
        print(f"Step count                          : {report.get('step_count')}")
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
        print(flush=True)
    except Exception as persist_exc:
        print(f"[ERROR] Could not persist Phase 10B report: {persist_exc!r}", flush=True)

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
