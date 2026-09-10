#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

CMD_TOPIC = "/g1/deployment/cmd_vel"
STOP_TOPIC = "/g1/deployment/stop"
FRAME_ID = "g1_cmd_source_linux_phase10b_canonical"


def parse_args():
    p = argparse.ArgumentParser(
        description="Canonical Linux ROS 2 command sequence for G1 final video V01."
    )
    p.add_argument("--zero-before-s", type=float, default=2.0)
    p.add_argument("--walk-s", type=float, default=12.0)
    p.add_argument("--zero-after-s", type=float, default=2.0)
    p.add_argument("--vx", type=float, default=0.40)
    p.add_argument("--rate-hz", type=float, default=50.0)
    p.add_argument("--wait-subscriber-s", type=float, default=20.0)
    return p.parse_args()


class VideoV01Publisher(Node):
    def __init__(self):
        super().__init__("g1_final_video_v01_canonical_publisher")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.cmd_pub = self.create_publisher(TwistStamped, CMD_TOPIC, qos)
        self.stop_pub = self.create_publisher(Bool, STOP_TOPIC, 1)

    def publish_cmd(self, vx: float, vy: float = 0.0, wz: float = 0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME_ID
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def publish_stop(self):
        msg = Bool()
        msg.data = True
        self.stop_pub.publish(msg)


def run_segment(node: VideoV01Publisher, duration_s: float, vx: float, rate_hz: float, label: str):
    if duration_s <= 0.0:
        return
    period = 1.0 / rate_hz
    deadline = time.monotonic() + duration_s
    next_tick = time.monotonic()
    count = 0
    print(f"[V01] {label}: {duration_s:.1f}s, vx={vx:.2f} m/s")
    while True:
        now = time.monotonic()
        if now >= deadline:
            break
        node.publish_cmd(vx)
        rclpy.spin_once(node, timeout_sec=0.0)
        count += 1
        next_tick += period
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_tick = time.monotonic()
    print(f"[V01] {label}: published {count} commands")


def main():
    args = parse_args()

    if args.rate_hz <= 0:
        raise SystemExit("--rate-hz must be > 0")
    if abs(args.vx - 0.40) > 1.0e-9:
        raise SystemExit("V01 is the scientific reference video: --vx must remain exactly 0.40 m/s.")

    rclpy.init()
    node = VideoV01Publisher()
    try:
        print("=" * 96)
        print("G1 FINAL VIDEO V01 — CANONICAL ROS 2 STRAIGHT-LOCOMOTION SEQUENCE")
        print("=" * 96)
        print(f"Command topic : {CMD_TOPIC}")
        print(f"Stop topic    : {STOP_TOPIC}")
        print(f"Frame ID      : {FRAME_ID}")
        print("QoS           : BEST_EFFORT / KEEP_LAST(1)")
        print(f"Rate          : {args.rate_hz:.1f} Hz")
        print(
            f"Sequence      : zero {args.zero_before_s:.1f}s -> "
            f"vx=0.40 {args.walk_s:.1f}s -> zero {args.zero_after_s:.1f}s -> stop"
        )

        wait_deadline = time.monotonic() + args.wait_subscriber_s
        while node.count_subscribers(CMD_TOPIC) < 1 and time.monotonic() < wait_deadline:
            rclpy.spin_once(node, timeout_sec=0.10)

        subscribers = node.count_subscribers(CMD_TOPIC)
        if subscribers < 1:
            raise RuntimeError(
                f"No subscriber detected on {CMD_TOPIC} within {args.wait_subscriber_s:.1f}s. "
                "Start the Windows final-video runner first."
            )

        print(f"[V01] Windows subscriber detected: {subscribers}")
        print("[V01] START SEQUENCE")

        run_segment(node, args.zero_before_s, 0.0, args.rate_hz, "ZERO BEFORE")
        run_segment(node, args.walk_s, 0.40, args.rate_hz, "SCIENTIFIC STRAIGHT")
        run_segment(node, args.zero_after_s, 0.0, args.rate_hz, "ZERO AFTER")

        for _ in range(10):
            node.publish_cmd(0.0)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(1.0 / args.rate_hz)

        node.publish_stop()
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.15)
        node.publish_stop()

        print("[V01] STOP sent")
        print("[V01] COMPLETE")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
