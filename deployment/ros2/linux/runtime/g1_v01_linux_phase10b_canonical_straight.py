#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

parser = argparse.ArgumentParser(
    description=(
        "V01 final-video Linux command source derived from the validated Phase10B contract. "
        "Sequence: ZERO -> scientific straight [0.40,0,0] -> ZERO."
    )
)
parser.add_argument("--rate-hz", type=float, default=50.0)
parser.add_argument("--discovery-timeout-s", type=float, default=30.0)
parser.add_argument("--spin-margin-ms", type=float, default=0.5)
parser.add_argument("--zero-before-s", type=float, default=3.0)
parser.add_argument("--straight-s", type=float, default=12.0)
parser.add_argument("--zero-after-s", type=float, default=3.0)
parser.add_argument(
    "--out-dir",
    default=str(
        Path.home()
        / "g1_ros_deployment_audit"
        / "video_final"
        / "V01_v2g_scientific_straight_linux"
    ),
)
args = parser.parse_args()

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Bool, String

CMD_TOPIC = "/g1/deployment/cmd_vel"
ECHO_TOPIC = "/g1/deployment/cmd_echo"
STATUS_TOPIC = "/g1/deployment/status"
BASE_TWIST_BODY_TOPIC = "/g1/deployment/base_twist_body"
BASE_POSE_TOPIC = "/g1/deployment/base_pose"
STOP_TOPIC = "/g1/deployment/stop"

CANONICAL_FRAME_ID = "g1_cmd_source_linux_phase10b_canonical"

ZERO = (0.0, 0.0, 0.0)
STRAIGHT = (0.40, 0.0, 0.0)
CMD_TOL = 1.0e-4

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


def near(vals, target):
    return max(abs(a - b) for a, b in zip(vals, target)) <= CMD_TOL


def stats(values):
    vals = [float(x) for x in values]
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }


class V01Source(Node):
    def __init__(self):
        super().__init__("g1_v01_linux_phase10b_canonical_straight")
        self.pub = self.create_publisher(TwistStamped, CMD_TOPIC, qos_cmd)
        self.stop_pub = self.create_publisher(Bool, STOP_TOPIC, qos_status)
        self.echo_sub = self.create_subscription(
            TwistStamped, ECHO_TOPIC, self.on_echo, qos_cmd
        )
        self.status_sub = self.create_subscription(
            String, STATUS_TOPIC, self.on_status, qos_status
        )
        self.twist_sub = self.create_subscription(
            TwistStamped, BASE_TWIST_BODY_TOPIC, self.on_twist, qos_cmd
        )
        self.pose_sub = self.create_subscription(
            PoseStamped, BASE_POSE_TOPIC, self.on_pose, qos_cmd
        )

        self.lock = threading.Lock()
        self.sent = {"zero": 0, "straight": 0}
        self.echo = {"zero": 0, "straight": 0, "other": 0}
        self.statuses = []
        self.twists = []
        self.poses = []
        self.invalid_status = 0
        self.send_lateness_ms = []

    def publish_cmd(self, cmd, scheduled_t=None):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = CANONICAL_FRAME_ID
        msg.twist.linear.x = float(cmd[0])
        msg.twist.linear.y = float(cmd[1])
        msg.twist.angular.z = float(cmd[2])
        self.pub.publish(msg)

        if scheduled_t is not None:
            self.send_lateness_ms.append(
                max(0.0, (time.perf_counter() - scheduled_t) * 1000.0)
            )

        with self.lock:
            if near(cmd, ZERO):
                self.sent["zero"] += 1
            elif near(cmd, STRAIGHT):
                self.sent["straight"] += 1

    def on_echo(self, msg):
        vals = (
            float(msg.twist.linear.x),
            float(msg.twist.linear.y),
            float(msg.twist.angular.z),
        )
        with self.lock:
            if near(vals, ZERO):
                self.echo["zero"] += 1
            elif near(vals, STRAIGHT):
                self.echo["straight"] += 1
            else:
                self.echo["other"] += 1

    def on_status(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            with self.lock:
                self.invalid_status += 1
            return
        with self.lock:
            self.statuses.append((time.perf_counter(), payload))

    def on_twist(self, msg):
        with self.lock:
            self.twists.append(
                (
                    time.perf_counter(),
                    float(msg.twist.linear.x),
                    float(msg.twist.linear.y),
                    float(msg.twist.linear.z),
                    float(msg.twist.angular.x),
                    float(msg.twist.angular.y),
                    float(msg.twist.angular.z),
                )
            )

    def on_pose(self, msg):
        with self.lock:
            self.poses.append(
                (
                    time.perf_counter(),
                    float(msg.pose.position.x),
                    float(msg.pose.position.y),
                    float(msg.pose.position.z),
                )
            )


# --- Read-only Phase10B network/FastDDS preflight -----------------------------

profile = os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE")
if not profile:
    raise SystemExit("FASTRTPS_DEFAULT_PROFILES_FILE is not set.")

profile_path = Path(profile)
if not profile_path.is_file():
    raise SystemExit(f"FastDDS profile missing: {profile_path}")

if "10.42.0.2" not in profile_path.read_text(encoding="utf-8", errors="ignore"):
    raise SystemExit(
        "FastDDS profile does not whitelist expected Linux Ethernet IP 10.42.0.2."
    )

route = subprocess.run(
    ["ip", "route", "get", "10.42.0.1"],
    capture_output=True,
    text=True,
    check=False,
).stdout.strip()
route_pass = "dev eno1" in route and "src 10.42.0.2" in route

wifi = subprocess.run(
    ["ip", "-br", "addr", "show", "wlo1"],
    capture_output=True,
    text=True,
    check=False,
).stdout.strip()
wifi_down = "DOWN" in wifi

if not route_pass:
    raise SystemExit(f"FAIL: unexpected Ethernet route: {route}")
if not wifi_down:
    raise SystemExit(f"FAIL: Wi-Fi wlo1 is not DOWN: {wifi}")


# --- ROS ---------------------------------------------------------------------

rclpy.init()
node = V01Source()
executor = SingleThreadedExecutor()
executor.add_node(node)
spin_errors = []


def spin_worker():
    try:
        executor.spin()
    except Exception as exc:
        if rclpy.ok():
            spin_errors.append(repr(exc))


thread = threading.Thread(target=spin_worker, daemon=False)
thread.start()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
stage_rows = []

try:
    print("=" * 116)
    print("G1 FINAL VIDEO V01 — LINUX PHASE10B CANONICAL SCIENTIFIC STRAIGHT")
    print("=" * 116)
    print(f"Frame id                            : {CANONICAL_FRAME_ID}")
    print(f"Sequence                            : ZERO {args.zero_before_s:.1f}s -> "
          f"[0.40,0,0] {args.straight_s:.1f}s -> ZERO {args.zero_after_s:.1f}s")
    print(f"Command rate                        : {args.rate_hz:.3f} Hz")
    print("Command QoS                         : BEST_EFFORT / KEEP_LAST(1)")
    print(f"FastDDS profile                     : {profile_path}")
    print(f"Ethernet route                      : {route}")
    print(f"Wi-Fi wlo1                          : {wifi}")
    print()

    deadline = time.perf_counter() + args.discovery_timeout_s
    while time.perf_counter() < deadline:
        if (
            node.count_subscribers(CMD_TOPIC) >= 1
            and node.count_publishers(STATUS_TOPIC) >= 1
            and node.count_publishers(ECHO_TOPIC) >= 1
            and node.count_publishers(BASE_TWIST_BODY_TOPIC) >= 1
            and node.count_publishers(BASE_POSE_TOPIC) >= 1
        ):
            break
        time.sleep(0.1)

    endpoint_counts = {
        "cmd_subscribers": int(node.count_subscribers(CMD_TOPIC)),
        "status_publishers": int(node.count_publishers(STATUS_TOPIC)),
        "echo_publishers": int(node.count_publishers(ECHO_TOPIC)),
        "twist_publishers": int(node.count_publishers(BASE_TWIST_BODY_TOPIC)),
        "pose_publishers": int(node.count_publishers(BASE_POSE_TOPIC)),
    }
    discovery_pass = all(v == 1 for v in endpoint_counts.values())
    print(
        f"Endpoint discovery                  : "
        f"{'PASS' if discovery_pass else 'FAIL'} {endpoint_counts}"
    )
    if not discovery_pass:
        raise RuntimeError("Required Windows final-video endpoints not discovered.")

    period = 1.0 / args.rate_hz
    spin_margin_s = max(0.0, args.spin_margin_ms / 1000.0)

    def stream(stage_name, classification, cmd, seconds):
        start = time.perf_counter()
        deadline = start
        end = start + seconds
        print(
            f"{stage_name:20s}: {seconds:4.1f} s  cmd={cmd}  [{classification}]"
        )
        while deadline < end:
            now = time.perf_counter()
            sleep_until = deadline - spin_margin_s
            if sleep_until > now:
                time.sleep(sleep_until - now)
            while time.perf_counter() < deadline:
                pass
            node.publish_cmd(cmd, scheduled_t=deadline)
            deadline += period
        finish = time.perf_counter()
        stage_rows.append(
            {
                "stage": stage_name,
                "classification": classification,
                "command": list(cmd),
                "start_s": start,
                "end_s": finish,
                "duration_wall_s": finish - start,
            }
        )

    rollout_start = time.perf_counter()
    stream("zero_before", "neutral", ZERO, args.zero_before_s)
    stream("straight", "scientific", STRAIGHT, args.straight_s)
    stream("zero_after", "neutral", ZERO, args.zero_after_s)
    rollout_end = time.perf_counter()

    # Allow final telemetry/status to arrive, then issue the canonical stop.
    time.sleep(0.5)
    stop = Bool()
    stop.data = True
    for _ in range(10):
        node.stop_pub.publish(stop)
        time.sleep(0.1)

finally:
    time.sleep(0.5)

    with node.lock:
        sent = dict(node.sent)
        echo = dict(node.echo)
        statuses = list(node.statuses)
        twists = list(node.twists)
        poses = list(node.poses)
        invalid_status = node.invalid_status
        send_lateness_ms = list(node.send_lateness_ms)

    def stage_contains(t, stage, margin=0.75):
        a = stage["start_s"] + margin
        b = stage["end_s"] - margin
        return a <= t <= b if b >= a else stage["start_s"] <= t <= stage["end_s"]

    straight_stage = next((s for s in stage_rows if s["stage"] == "straight"), None)
    straight_rows = []
    if straight_stage is not None:
        straight_rows = [r for r in twists if stage_contains(r[0], straight_stage)]

    straight_vx = stats([r[1] for r in straight_rows])

    first_stage_start = min((s["start_s"] for s in stage_rows), default=0.0)
    last_stage_end = max((s["end_s"] for s in stage_rows), default=0.0)
    rollout_statuses = [
        p for t, p in statuses if first_stage_start <= t <= last_stage_end
    ]

    straight_statuses = []
    if straight_stage is not None:
        straight_statuses = [
            p
            for t, p in statuses
            if (straight_stage["start_s"] + 1.0) <= t <= straight_stage["end_s"]
        ]

    policy_loaded_all = bool(rollout_statuses) and all(
        bool(p.get("policy_loaded", False)) for p in rollout_statuses
    )
    sha_pass_all = bool(rollout_statuses) and all(
        bool(p.get("checkpoint_sha_pass", False)) for p in rollout_statuses
    )
    no_resets = bool(rollout_statuses) and max(
        int(p.get("reset_count", 0)) for p in rollout_statuses
    ) == 0
    no_rejected = bool(rollout_statuses) and max(
        int(p.get("rejected_command_count", 0)) for p in rollout_statuses
    ) == 0
    no_invalid = bool(rollout_statuses) and max(
        int(p.get("invalid_count", 0)) for p in rollout_statuses
    ) == 0
    no_watchdog_during_straight = bool(straight_statuses) and not any(
        bool(p.get("watchdog_active", True)) for p in straight_statuses
    )

    effective_zero_seen = any(
        near(tuple(p.get("effective_cmd", [999, 999, 999])), ZERO)
        for p in rollout_statuses
    )
    effective_straight_seen = any(
        near(tuple(p.get("effective_cmd", [999, 999, 999])), STRAIGHT)
        for p in rollout_statuses
    )

    total_sent = sum(sent.values())
    total_echo = sum(echo.values())
    echo_deficit = max(0, total_sent - total_echo)
    echo_deficit_pct = (
        100.0 * echo_deficit / total_sent if total_sent else None
    )

    guards = {
        "ethernet_route_expected": route_pass,
        "wifi_down": wifi_down,
        "endpoint_discovery": discovery_pass,
        "single_windows_endpoint_each": all(v == 1 for v in endpoint_counts.values()),
        "status_received": len(statuses) > 0,
        "body_twist_received": len(twists) > 0,
        "pose_received": len(poses) > 0,
        "zero_echo_seen": echo["zero"] > 0,
        "straight_echo_seen": echo["straight"] > 0,
        "no_unexpected_echo": echo["other"] == 0,
        "echo_deficit_le_1pct": (
            echo_deficit_pct is not None and echo_deficit_pct <= 1.0
        ),
        "policy_loaded": policy_loaded_all,
        "checkpoint_sha_pass": sha_pass_all,
        "effective_zero_seen": effective_zero_seen,
        "effective_straight_seen": effective_straight_seen,
        "straight_forward_motion": (
            straight_vx is not None and straight_vx["median"] > 0.10
        ),
        "no_environment_resets": no_resets,
        "no_rejected_commands": no_rejected,
        "no_invalid_commands": no_invalid,
        "no_watchdog_during_straight": no_watchdog_during_straight,
        "invalid_status_zero": invalid_status == 0,
        "linux_executor": len(spin_errors) == 0,
    }

    status = "PASS" if all(guards.values()) else "FAIL"

    twist_csv = out_dir / "g1_v01_body_twist.csv"
    with twist_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "linux_perf_counter_s",
                "body_vx",
                "body_vy",
                "body_vz",
                "body_wx",
                "body_wy",
                "body_wz",
            ]
        )
        writer.writerows(twists)

    pose_csv = out_dir / "g1_v01_base_pose.csv"
    with pose_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["linux_perf_counter_s", "world_x", "world_y", "world_z"]
        )
        writer.writerows(poses)

    stage_csv = out_dir / "g1_v01_stage_timeline.csv"
    with stage_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stage",
                "classification",
                "vx",
                "vy",
                "wz",
                "start_s",
                "end_s",
                "duration_wall_s",
            ]
        )
        for s in stage_rows:
            writer.writerow(
                [
                    s["stage"],
                    s["classification"],
                    *s["command"],
                    s["start_s"],
                    s["end_s"],
                    s["duration_wall_s"],
                ]
            )

    status_jsonl = out_dir / "g1_v01_status.jsonl"
    with status_jsonl.open("w", encoding="utf-8") as f:
        for t, payload in statuses:
            f.write(
                json.dumps(
                    {"linux_perf_counter_s": t, "status": payload},
                    separators=(",", ":"),
                )
                + "\n"
            )

    summary_json = out_dir / "g1_v01_linux_summary.json"
    summary = {
        "status": status,
        "classification": (
            "final video V01; exact scientific straight command; "
            "canonical Phase10B ROS source contract"
        ),
        "frame_id": CANONICAL_FRAME_ID,
        "sequence": [
            ["zero_before", list(ZERO), args.zero_before_s],
            ["straight", list(STRAIGHT), args.straight_s],
            ["zero_after", list(ZERO), args.zero_after_s],
        ],
        "rate_hz": args.rate_hz,
        "route": route,
        "wifi": wifi,
        "fastdds_profile": str(profile_path),
        "endpoint_counts_at_discovery": endpoint_counts,
        "sent": sent,
        "echo": echo,
        "echo_deficit": echo_deficit,
        "echo_deficit_pct": echo_deficit_pct,
        "status_messages": len(statuses),
        "twist_messages": len(twists),
        "pose_messages": len(poses),
        "send_lateness_ms": stats(send_lateness_ms),
        "straight_body_vx": straight_vx,
        "guardrails": guards,
        "spin_errors": spin_errors,
        "rollout_wall_s": (
            rollout_end - rollout_start
            if "rollout_start" in globals() and "rollout_end" in globals()
            else None
        ),
        "stages": stage_rows,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("-" * 116)
    print("V01 LINUX FINAL SUMMARY")
    print("-" * 116)
    print(f"Status                              : {status}")
    print(f"Frame id                            : {CANONICAL_FRAME_ID}")
    print(f"Sent zero / straight                : {sent['zero']} / {sent['straight']}")
    print(f"Echo zero / straight / other        : {echo['zero']} / {echo['straight']} / {echo['other']}")
    print(
        f"Echo deficit                        : {echo_deficit} "
        f"({echo_deficit_pct:.3f}%)"
        if echo_deficit_pct is not None
        else "Echo deficit                        : n/a"
    )
    print(f"Status/twist/pose messages          : {len(statuses)} / {len(twists)} / {len(poses)}")
    print(f"Straight body vx                    : {straight_vx}")
    print(f"No environment resets               : {no_resets}")
    print(f"No watchdog during straight         : {no_watchdog_during_straight}")
    print(f"Linux executor errors               : {len(spin_errors)}")
    print(f"CSV body twist                      : {twist_csv}")
    print(f"CSV base pose                       : {pose_csv}")
    print(f"CSV stage timeline                  : {stage_csv}")
    print(f"JSONL status                        : {status_jsonl}")
    print(f"JSON summary                        : {summary_json}")
    print()
    print("GUARDRAILS")
    print("-" * 116)
    for k, v in guards.items():
        print(f"{k:42s}: {'PASS' if v else 'FAIL'}")
    print()
    print("G1_V01_LINUX_PHASE10B_CANONICAL_PASS" if status == "PASS"
          else "G1_V01_LINUX_PHASE10B_CANONICAL_FAIL")

    try:
        executor.shutdown(timeout_sec=2.0)
    except Exception:
        pass
    try:
        thread.join(timeout=3.0)
    except Exception:
        pass
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass
