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
        "Phase 10 Linux Ethernet final soak: 3 x 40 s cycles. "
        "Scientific straight [0.40,0,0]; engineering forward+yaw [0.40,0,±0.25]."
    )
)
parser.add_argument("--rate-hz", type=float, default=50.0)
parser.add_argument("--discovery-timeout-s", type=float, default=30.0)
parser.add_argument("--spin-margin-ms", type=float, default=0.5)
parser.add_argument(
    "--out-dir",
    default=str(Path.home() / "g1_ros_deployment_audit" / "phase10b_ethernet_canonical_soak"),
)
args = parser.parse_args()

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Bool, String

CMD_TOPIC = "/g1/deployment/cmd_vel"
ECHO_TOPIC = "/g1/deployment/cmd_echo"
STATUS_TOPIC = "/g1/deployment/status"
BASE_TWIST_BODY_TOPIC = "/g1/deployment/base_twist_body"
BASE_POSE_TOPIC = "/g1/deployment/base_pose"
STOP_TOPIC = "/g1/deployment/stop"

ZERO = (0.0, 0.0, 0.0)
STRAIGHT = (0.40, 0.0, 0.0)       # scientific protocol
CURVE_POS = (0.40, 0.0, 0.25)     # engineering deployment check
CURVE_NEG = (0.40, 0.0, -0.25)    # engineering deployment check
CMD_TOL = 1.0e-4

# Exactly 40 s/cycle; repeated 3 times = 120 s.
CYCLE = [
    ("zero_start", ZERO, 4.0, "neutral"),
    ("straight_1", STRAIGHT, 6.0, "scientific"),
    ("curve_pos", CURVE_POS, 6.0, "engineering"),
    ("straight_2", STRAIGHT, 6.0, "scientific"),
    ("curve_neg", CURVE_NEG, 6.0, "engineering"),
    ("straight_3", STRAIGHT, 6.0, "scientific"),
    ("zero_end", ZERO, 6.0, "neutral"),
]
REPEATS = 3
EXPECTED_SOAK_S = sum(x[2] for x in CYCLE) * REPEATS

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


class Probe(Node):
    def __init__(self):
        super().__init__("g1_phase10b_linux_ethernet_final_soak_probe")
        self.pub = self.create_publisher(TwistStamped, CMD_TOPIC, qos_cmd)
        self.stop_pub = self.create_publisher(Bool, STOP_TOPIC, qos_status)
        self.echo_sub = self.create_subscription(TwistStamped, ECHO_TOPIC, self.on_echo, qos_cmd)
        self.status_sub = self.create_subscription(String, STATUS_TOPIC, self.on_status, qos_status)
        self.twist_sub = self.create_subscription(TwistStamped, BASE_TWIST_BODY_TOPIC, self.on_twist, qos_cmd)
        self.pose_sub = self.create_subscription(PoseStamped, BASE_POSE_TOPIC, self.on_pose, qos_cmd)

        self.lock = threading.Lock()
        self.sent = {"zero": 0, "straight": 0, "curve_pos": 0, "curve_neg": 0}
        self.echo = {"zero": 0, "straight": 0, "curve_pos": 0, "curve_neg": 0, "other": 0}
        self.statuses = []
        self.twists = []
        self.poses = []
        self.invalid_status = 0
        self.send_lateness_ms = []

    def publish_cmd(self, cmd, scheduled_t=None):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "g1_cmd_source_linux_phase10b_canonical"
        msg.twist.linear.x = float(cmd[0])
        msg.twist.linear.y = float(cmd[1])
        msg.twist.angular.z = float(cmd[2])
        self.pub.publish(msg)

        if scheduled_t is not None:
            self.send_lateness_ms.append(max(0.0, (time.perf_counter() - scheduled_t) * 1000.0))

        with self.lock:
            if near(cmd, ZERO):
                self.sent["zero"] += 1
            elif near(cmd, STRAIGHT):
                self.sent["straight"] += 1
            elif near(cmd, CURVE_POS):
                self.sent["curve_pos"] += 1
            elif near(cmd, CURVE_NEG):
                self.sent["curve_neg"] += 1

    def on_echo(self, msg):
        vals = (float(msg.twist.linear.x), float(msg.twist.linear.y), float(msg.twist.angular.z))
        with self.lock:
            if near(vals, ZERO):
                self.echo["zero"] += 1
            elif near(vals, STRAIGHT):
                self.echo["straight"] += 1
            elif near(vals, CURVE_POS):
                self.echo["curve_pos"] += 1
            elif near(vals, CURVE_NEG):
                self.echo["curve_neg"] += 1
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


# Read-only Ethernet/FastDDS preflight.
profile = os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE")
if not profile:
    raise SystemExit("FASTRTPS_DEFAULT_PROFILES_FILE is not set.")
profile_path = Path(profile)
if not profile_path.is_file():
    raise SystemExit(f"FastDDS profile missing: {profile_path}")
if "10.42.0.2" not in profile_path.read_text(encoding="utf-8", errors="ignore"):
    raise SystemExit("FastDDS profile does not whitelist expected Linux Ethernet IP 10.42.0.2.")

route = subprocess.run(
    ["ip", "route", "get", "10.42.0.1"],
    capture_output=True, text=True, check=False
).stdout.strip()
route_pass = "dev eno1" in route and "src 10.42.0.2" in route

wifi = subprocess.run(
    ["ip", "-br", "addr", "show", "wlo1"],
    capture_output=True, text=True, check=False
).stdout.strip()
wifi_down = "DOWN" in wifi

rclpy.init()
node = Probe()
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
    print("G1 ROS ↔ ISAAC — PHASE 10B LINUX ETHERNET CANONICAL 120 s SOAK")
    print("=" * 116)
    print(f"Profile                             : 3 x 40 s cycles; total {EXPECTED_SOAK_S:.1f} s")
    print("Scientific command                  : [0.40, 0, 0]")
    print("Engineering curve commands          : [0.40, 0, +0.25], [0.40, 0, -0.25]")
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
    print(f"Endpoint discovery                  : {'PASS' if discovery_pass else 'FAIL'} {endpoint_counts}")
    if not discovery_pass:
        raise RuntimeError("Required Windows Phase 10 endpoints not discovered.")

    period = 1.0 / args.rate_hz
    spin_margin_s = max(0.0, args.spin_margin_ms / 1000.0)

    def stream(cmd, seconds):
        start = time.perf_counter()
        deadline = start
        end = start + seconds
        while deadline < end:
            now = time.perf_counter()
            sleep_until = deadline - spin_margin_s
            if sleep_until > now:
                time.sleep(sleep_until - now)
            while time.perf_counter() < deadline:
                pass
            node.publish_cmd(cmd, scheduled_t=deadline)
            deadline += period

    soak_start = time.perf_counter()
    for cycle_idx in range(1, REPEATS + 1):
        print(f"Cycle {cycle_idx}/{REPEATS}")
        for stage_name, cmd, seconds, classification in CYCLE:
            start = time.perf_counter()
            label = f"c{cycle_idx}_{stage_name}"
            print(f"  {label:20s}: {seconds:4.1f} s  cmd={cmd}  [{classification}]")
            stream(cmd, seconds)
            end = time.perf_counter()
            stage_rows.append({
                "cycle": cycle_idx,
                "stage": stage_name,
                "label": label,
                "classification": classification,
                "command": list(cmd),
                "start_s": start,
                "end_s": end,
                "duration_wall_s": end - start,
            })
    soak_end = time.perf_counter()

    time.sleep(1.0)
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

    def stage_contains(t, stage, margin=1.0):
        return (stage["start_s"] + margin) <= t <= (stage["end_s"] - margin)

    def rows_for(predicate):
        selected = []
        for st in stage_rows:
            if predicate(st):
                selected.extend([r for r in twists if stage_contains(r[0], st)])
        return selected

    zero_rows = rows_for(lambda s: s["classification"] == "neutral")
    straight_rows = rows_for(lambda s: s["classification"] == "scientific")
    pos_rows = rows_for(lambda s: s["stage"] == "curve_pos")
    neg_rows = rows_for(lambda s: s["stage"] == "curve_neg")

    zero_vx = stats([r[1] for r in zero_rows])
    straight_vx = stats([r[1] for r in straight_rows])
    pos_vx = stats([r[1] for r in pos_rows])
    pos_wz = stats([r[6] for r in pos_rows])
    neg_vx = stats([r[1] for r in neg_rows])
    neg_wz = stats([r[6] for r in neg_rows])

    first_stage_start = min((s["start_s"] for s in stage_rows), default=0.0)
    last_stage_end = max((s["end_s"] for s in stage_rows), default=0.0)
    rollout_statuses = [
        p for t, p in statuses
        if first_stage_start <= t <= last_stage_end
    ]
    settled_statuses = [
        p for t, p in statuses
        if first_stage_start + 1.0 <= t <= last_stage_end
    ]

    policy_loaded_all = bool(rollout_statuses) and all(bool(p.get("policy_loaded", False)) for p in rollout_statuses)
    sha_pass_all = bool(rollout_statuses) and all(bool(p.get("checkpoint_sha_pass", False)) for p in rollout_statuses)
    no_resets = bool(rollout_statuses) and max(int(p.get("reset_count", 0)) for p in rollout_statuses) == 0
    no_rejected = bool(rollout_statuses) and max(int(p.get("rejected_command_count", 0)) for p in rollout_statuses) == 0
    no_invalid = bool(rollout_statuses) and max(int(p.get("invalid_count", 0)) for p in rollout_statuses) == 0
    no_watchdog_after_settle = bool(settled_statuses) and not any(bool(p.get("watchdog_active", True)) for p in settled_statuses)

    effective_seen = {
        "zero": any(near(tuple(p.get("effective_cmd", [999,999,999])), ZERO) for p in rollout_statuses),
        "straight": any(near(tuple(p.get("effective_cmd", [999,999,999])), STRAIGHT) for p in rollout_statuses),
        "curve_pos": any(near(tuple(p.get("effective_cmd", [999,999,999])), CURVE_POS) for p in rollout_statuses),
        "curve_neg": any(near(tuple(p.get("effective_cmd", [999,999,999])), CURVE_NEG) for p in rollout_statuses),
    }

    total_sent = sum(sent.values())
    total_echo = sum(echo.values())
    echo_deficit = max(0, total_sent - total_echo)
    echo_deficit_pct = 100.0 * echo_deficit / total_sent if total_sent else None

    guards = {
        "ethernet_route_expected": route_pass,
        "wifi_down": wifi_down,
        "endpoint_discovery": discovery_pass,
        "single_windows_endpoint_each": all(v == 1 for v in endpoint_counts.values()),
        "status_received": len(statuses) > 0,
        "body_twist_received": len(twists) > 0,
        "pose_received": len(poses) > 0,
        "all_command_echo_classes_seen": all(echo[k] > 0 for k in ("zero","straight","curve_pos","curve_neg")),
        "no_unexpected_echo": echo["other"] == 0,
        "echo_deficit_le_1pct": echo_deficit_pct is not None and echo_deficit_pct <= 1.0,
        "policy_loaded": policy_loaded_all,
        "checkpoint_sha_pass": sha_pass_all,
        "all_effective_commands_seen": all(effective_seen.values()),
        "straight_forward_motion": straight_vx is not None and straight_vx["median"] > 0.10,
        "curve_pos_forward_motion": pos_vx is not None and pos_vx["median"] > 0.10,
        "curve_neg_forward_motion": neg_vx is not None and neg_vx["median"] > 0.10,
        "curve_pos_correct_sign_wz": pos_wz is not None and pos_wz["median"] > 0.05,
        "curve_neg_correct_sign_wz": neg_wz is not None and neg_wz["median"] < -0.05,
        "no_environment_resets": no_resets,
        "no_rejected_commands": no_rejected,
        "no_invalid_commands": no_invalid,
        "no_watchdog_after_settle": no_watchdog_after_settle,
        "invalid_status_zero": invalid_status == 0,
        "linux_executor": len(spin_errors) == 0,
        "soak_duration_ge_118s": "soak_start" in globals() and "soak_end" in globals() and (soak_end - soak_start) >= 118.0,
    }

    status = "PASS" if all(guards.values()) else "FAIL"

    # Raw evidence outside source Git.
    twist_csv = out_dir / "g1_phase10b_body_twist.csv"
    with twist_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["linux_perf_counter_s","body_vx","body_vy","body_vz","body_wx","body_wy","body_wz"])
        writer.writerows(twists)

    pose_csv = out_dir / "g1_phase10b_base_pose.csv"
    with pose_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["linux_perf_counter_s","world_x","world_y","world_z"])
        writer.writerows(poses)

    stage_csv = out_dir / "g1_phase10b_stage_timeline.csv"
    with stage_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cycle","stage","label","classification","vx","vy","wz","start_s","end_s","duration_wall_s"])
        for s in stage_rows:
            writer.writerow([
                s["cycle"], s["stage"], s["label"], s["classification"],
                *s["command"], s["start_s"], s["end_s"], s["duration_wall_s"]
            ])

    status_jsonl = out_dir / "g1_phase10b_status.jsonl"
    with status_jsonl.open("w", encoding="utf-8") as f:
        for t, payload in statuses:
            f.write(json.dumps({"linux_perf_counter_s": t, "status": payload}, separators=(",", ":")) + "\n")

    summary_json = out_dir / "g1_phase10b_linux_summary.json"
    summary = {
        "status": status,
        "classification": "final deployment soak; .40 straight scientific, forward+yaw ±.25 engineering",
        "expected_soak_s": EXPECTED_SOAK_S,
        "actual_soak_s": (soak_end - soak_start) if "soak_start" in globals() and "soak_end" in globals() else None,
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
        "zero_body_vx": zero_vx,
        "straight_body_vx": straight_vx,
        "curve_pos_body_vx": pos_vx,
        "curve_pos_body_wz": pos_wz,
        "curve_neg_body_vx": neg_vx,
        "curve_neg_body_wz": neg_wz,
        "effective_seen": effective_seen,
        "guardrails": guards,
        "spin_errors": spin_errors,
        "stages": stage_rows,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("-" * 116)
    print("PHASE 10B LINUX FINAL SUMMARY")
    print("-" * 116)
    print(f"Status                              : {status}")
    print(f"Actual soak duration s              : {summary['actual_soak_s']}")
    print(f"Sent zero/straight/+curve/-curve    : {sent['zero']} / {sent['straight']} / {sent['curve_pos']} / {sent['curve_neg']}")
    print(f"Echo zero/straight/+curve/-curve    : {echo['zero']} / {echo['straight']} / {echo['curve_pos']} / {echo['curve_neg']}")
    print(f"Unexpected echoes                   : {echo['other']}")
    print(f"Echo deficit                        : {echo_deficit} ({echo_deficit_pct:.3f}%)" if echo_deficit_pct is not None else "Echo deficit                        : n/a")
    print(f"Status/twist/pose messages          : {len(statuses)} / {len(twists)} / {len(poses)}")
    print(f"Straight body vx                    : {straight_vx}")
    print(f"+curve body vx / wz                 : {pos_vx} / {pos_wz}")
    print(f"-curve body vx / wz                 : {neg_vx} / {neg_wz}")
    print(f"Zero body vx                        : {zero_vx}")
    print(f"Effective command classes seen      : {effective_seen}")
    print(f"No environment resets               : {no_resets}")
    print(f"No watchdog after settle            : {no_watchdog_after_settle}")
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
