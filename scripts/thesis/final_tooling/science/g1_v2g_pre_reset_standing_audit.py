from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

CSV_PATH = Path(
    r"C:\humanoid_rl_lab\logs\thesis\standing_probe"
    r"\frozen_v2g_zero_command_playeq_seed42_20s"
    r"\frozen_v2g_zero_command_playeq_seed42_20s_samples.csv"
)

EXPECTED_ROWS = 1000
EXPECTED_DT = 0.02

LEG_JOINTS = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    alpha = pos - lo
    return values[lo] * (1.0 - alpha) + values[hi] * alpha


def quat_angle_between(q1, q2):
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def root_tilt_rad(row):
    qx = float(row["root_qx"])
    qy = float(row["root_qy"])
    rzz = 1.0 - 2.0 * (qx * qx + qy * qy)
    rzz = max(-1.0, min(1.0, rzz))
    return math.acos(rzz)


def root_yaw_rad(row):
    qw = float(row["root_qw"])
    qx = float(row["root_qx"])
    qy = float(row["root_qy"])
    qz = float(row["root_qz"])
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def unwrap(values):
    out = [values[0]]
    for value in values[1:]:
        candidate = value
        delta = candidate - out[-1]
        while delta > math.pi:
            candidate -= 2.0 * math.pi
            delta = candidate - out[-1]
        while delta < -math.pi:
            candidate += 2.0 * math.pi
            delta = candidate - out[-1]
        out.append(candidate)
    return out


def load_rows():
    if not CSV_PATH.is_file():
        raise RuntimeError(f"missing CSV: {CSV_PATH}")

    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))

    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_ROWS} rows, got {len(rows)}"
        )

    names = set(rows[0].keys())
    required = {
        "sample",
        "sim_time_s",
        "cmd_vx",
        "cmd_vy",
        "cmd_wz",
        "root_x_w",
        "root_y_w",
        "root_z_w",
        "root_qw",
        "root_qx",
        "root_qy",
        "root_qz",
        "root_vx_b",
        "root_vy_b",
        "root_wz_b",
        "left_foot_x_w",
        "left_foot_y_w",
        "right_foot_x_w",
        "right_foot_y_w",
        "left_foot_vx_w",
        "left_foot_vy_w",
        "right_foot_vx_w",
        "right_foot_vy_w",
        "left_contact",
        "right_contact",
    }
    for joint in LEG_JOINTS:
        required.add(f"q::{joint}")
        required.add(f"qd::{joint}")

    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"missing columns: {missing}")

    commands = [
        (
            float(r["cmd_vx"]),
            float(r["cmd_vy"]),
            float(r["cmd_wz"]),
        )
        for r in rows
    ]
    if any(
        max(abs(vx), abs(vy), abs(wz)) > 1e-12
        for vx, vy, wz in commands
    ):
        raise RuntimeError("CSV is not a strict zero-command run")

    times = [float(r["sim_time_s"]) for r in rows]
    if len(times) >= 2:
        dts = [b - a for a, b in zip(times, times[1:])]
        if max(abs(dt - EXPECTED_DT) for dt in dts) > 1e-9:
            raise RuntimeError(
                "unexpected/nonuniform CSV time step; "
                f"min={min(dts):.12f}, max={max(dts):.12f}"
            )

    return rows


def transition_features(rows):
    features = []
    for i in range(1, len(rows)):
        a = rows[i - 1]
        b = rows[i]

        dx = float(b["root_x_w"]) - float(a["root_x_w"])
        dy = float(b["root_y_w"]) - float(a["root_y_w"])
        dz = float(b["root_z_w"]) - float(a["root_z_w"])

        q1 = tuple(float(a[k]) for k in ("root_qw", "root_qx", "root_qy", "root_qz"))
        q2 = tuple(float(b[k]) for k in ("root_qw", "root_qx", "root_qy", "root_qz"))

        joint_rms = math.sqrt(
            statistics.fmean(
                (
                    float(b[f"q::{joint}"])
                    - float(a[f"q::{joint}"])
                ) ** 2
                for joint in LEG_JOINTS
            )
        )

        features.append(
            {
                "to_index": i,
                "xy_jump": math.hypot(dx, dy),
                "z_jump": abs(dz),
                "quat_jump_deg": math.degrees(quat_angle_between(q1, q2)),
                "joint_rms_jump": joint_rms,
            }
        )

    return features


def detect_reset(features):
    # Reset is a simulator teleport, so require a conspicuous discontinuity.
    # Multiple independent channels are printed for auditability.
    candidates = []
    for f in features:
        score = 0
        if f["xy_jump"] >= 0.15:
            score += 1
        if f["z_jump"] >= 0.15:
            score += 1
        if f["quat_jump_deg"] >= 20.0:
            score += 1
        if f["joint_rms_jump"] >= 0.30:
            score += 1

        if score >= 2 or f["xy_jump"] >= 0.30 or f["z_jump"] >= 0.30:
            candidates.append((score, f))

    if not candidates:
        # Fall back to the largest normalized discontinuity for diagnosis,
        # but do not silently call it a confirmed reset.
        ranked = sorted(
            features,
            key=lambda f: (
                f["xy_jump"] / 0.15
                + f["z_jump"] / 0.15
                + f["quat_jump_deg"] / 20.0
                + f["joint_rms_jump"] / 0.30
            ),
            reverse=True,
        )
        return None, ranked[:5]

    candidates.sort(key=lambda item: item[1]["to_index"])
    return candidates[0][1], [item[1] for item in candidates]


def summarize(rows, label):
    if len(rows) < 2:
        raise RuntimeError(f"{label}: too few rows: {len(rows)}")

    x0 = float(rows[0]["root_x_w"])
    y0 = float(rows[0]["root_y_w"])

    xy_exc = [
        math.hypot(
            float(r["root_x_w"]) - x0,
            float(r["root_y_w"]) - y0,
        )
        for r in rows
    ]

    planar = [
        math.hypot(
            float(r["root_vx_b"]),
            float(r["root_vy_b"]),
        )
        for r in rows
    ]

    abs_wz = [abs(float(r["root_wz_b"])) for r in rows]
    tilt = [root_tilt_rad(r) for r in rows]

    foot_speed = [
        0.5
        * (
            math.hypot(
                float(r["left_foot_vx_w"]),
                float(r["left_foot_vy_w"]),
            )
            + math.hypot(
                float(r["right_foot_vx_w"]),
                float(r["right_foot_vy_w"]),
            )
        )
        for r in rows
    ]

    leg_qd = [
        statistics.fmean(
            abs(float(r[f"qd::{joint}"]))
            for joint in LEG_JOINTS
        )
        for r in rows
    ]

    both_contact = 100.0 * statistics.fmean(
        1.0
        if (
            int(float(r["left_contact"])) != 0
            and int(float(r["right_contact"])) != 0
        )
        else 0.0
        for r in rows
    )

    stance_width = [
        abs(
            float(r["left_foot_y_w"])
            - float(r["right_foot_y_w"])
        )
        for r in rows
    ]

    fore_aft = [
        abs(
            float(r["left_foot_x_w"])
            - float(r["right_foot_x_w"])
        )
        for r in rows
    ]

    yaw = unwrap([root_yaw_rad(r) for r in rows])

    heights = [float(r["root_z_w"]) for r in rows]

    return {
        "label": label,
        "n": len(rows),
        "duration_s": (len(rows) - 1) * EXPECTED_DT,
        "drift_m": math.hypot(
            float(rows[-1]["root_x_w"]) - x0,
            float(rows[-1]["root_y_w"]) - y0,
        ),
        "max_exc_m": max(xy_exc),
        "vxy_mean": statistics.fmean(planar),
        "vxy_p95": percentile(planar, 0.95),
        "wz_mean": statistics.fmean(abs_wz),
        "tilt_mean_deg": math.degrees(statistics.fmean(tilt)),
        "tilt_p95_deg": math.degrees(percentile(tilt, 0.95)),
        "tilt_max_deg": math.degrees(max(tilt)),
        "foot_mean": statistics.fmean(foot_speed),
        "foot_p95": percentile(foot_speed, 0.95),
        "leg_qd_mean": statistics.fmean(leg_qd),
        "double_support_pct": both_contact,
        "stance_width_cm": 100.0 * statistics.fmean(stance_width),
        "fore_aft_cm": 100.0 * statistics.fmean(fore_aft),
        "height_mean": statistics.fmean(heights),
        "height_std": statistics.pstdev(heights),
        "heading_net_deg": math.degrees(yaw[-1] - yaw[0]),
    }


def print_summary(s):
    print(f"SEGMENT={s['label']}")
    print(f"samples                         : {s['n']}")
    print(f"duration_s                      : {s['duration_s']:.3f}")
    print(f"drift_m                         : {s['drift_m']:.6f}")
    print(f"max_xy_excursion_m              : {s['max_exc_m']:.6f}")
    print(f"planar_speed_mean_mps           : {s['vxy_mean']:.6f}")
    print(f"planar_speed_p95_mps            : {s['vxy_p95']:.6f}")
    print(f"abs_yaw_rate_mean_radps         : {s['wz_mean']:.6f}")
    print(f"root_tilt_mean_deg              : {s['tilt_mean_deg']:.6f}")
    print(f"root_tilt_p95_deg               : {s['tilt_p95_deg']:.6f}")
    print(f"root_tilt_max_deg               : {s['tilt_max_deg']:.6f}")
    print(f"foot_xy_speed_mean_mps          : {s['foot_mean']:.6f}")
    print(f"foot_xy_speed_p95_mps           : {s['foot_p95']:.6f}")
    print(f"leg_joint_speed_mean_radps      : {s['leg_qd_mean']:.6f}")
    print(f"double_support_pct              : {s['double_support_pct']:.2f}")
    print(f"stance_width_mean_cm            : {s['stance_width_cm']:.3f}")
    print(f"foot_fore_aft_sep_mean_cm       : {s['fore_aft_cm']:.3f}")
    print(f"root_height_mean_m              : {s['height_mean']:.6f}")
    print(f"root_height_std_m               : {s['height_std']:.6f}")
    print(f"heading_net_deg                 : {s['heading_net_deg']:.6f}")


def main():
    rows = load_rows()
    features = transition_features(rows)
    reset, candidates = detect_reset(features)

    print("=" * 104)
    print("V2-G ZERO-COMMAND PRE-RESET STANDING AUDIT")
    print("=" * 104)
    print(f"CSV                             : {CSV_PATH}")
    print(f"rows                            : {len(rows)}")
    print(f"dt                              : {EXPECTED_DT:.5f} s")
    print()

    print("TOP RESET/DISCONTINUITY CANDIDATES")
    for f in sorted(
        features,
        key=lambda x: (
            x["xy_jump"] / 0.15
            + x["z_jump"] / 0.15
            + x["quat_jump_deg"] / 20.0
            + x["joint_rms_jump"] / 0.30
        ),
        reverse=True,
    )[:8]:
        print(
            f"to_index={f['to_index']:4d} "
            f"time={f['to_index'] * EXPECTED_DT:7.3f}s "
            f"xy_jump={f['xy_jump']:.4f}m "
            f"z_jump={f['z_jump']:.4f}m "
            f"quat_jump={f['quat_jump_deg']:.2f}deg "
            f"joint_rms_jump={f['joint_rms_jump']:.4f}rad"
        )

    print()

    if reset is None:
        print("RESET_DETECTION=UNCONFIRMED")
        print(
            "No transition met the conservative reset threshold. "
            "Do not interpret a pre-reset segment automatically."
        )
        print("V2G_PRE_RESET_STANDING_AUDIT_NEEDS_MANUAL_REVIEW")
        return

    reset_index = reset["to_index"]
    pre = rows[:reset_index]

    print("RESET_DETECTION=CONFIRMED_BY_POSE_DISCONTINUITY")
    print(f"reset_transition_to_index       : {reset_index}")
    print(f"reset_transition_time_s         : {reset_index * EXPECTED_DT:.3f}")
    print(f"pre_reset_samples               : {len(pre)}")
    print(f"pre_reset_duration_s            : {(len(pre) - 1) * EXPECTED_DT:.3f}")
    print(
        "reset_jump                    : "
        f"xy={reset['xy_jump']:.4f}m, "
        f"z={reset['z_jump']:.4f}m, "
        f"quat={reset['quat_jump_deg']:.2f}deg, "
        f"joint_rms={reset['joint_rms_jump']:.4f}rad"
    )
    print()

    print_summary(summarize(pre, "ALL_PRE_RESET"))

    # Diagnose deterioration approaching the reset.
    print()
    print("=" * 104)
    print("PRE-RESET TIME WINDOWS")
    print("=" * 104)

    for seconds in (5.0, 3.0, 2.0, 1.0):
        count = int(round(seconds / EXPECTED_DT))
        if len(pre) >= count:
            segment = pre[-count:]
            print()
            print_summary(
                summarize(
                    segment,
                    f"LAST_{seconds:.0f}S_BEFORE_RESET",
                )
            )

    # Also inspect the earliest stable portion after the 3 s warmup.
    for seconds in (5.0, 10.0):
        count = int(round(seconds / EXPECTED_DT))
        if len(pre) >= count:
            print()
            print_summary(
                summarize(
                    pre[:count],
                    f"FIRST_{seconds:.0f}S_OF_RECORDED_PRE_RESET",
                )
            )

    print()
    print("=" * 104)
    print("V2G_PRE_RESET_STANDING_AUDIT_PASS")
    print("=" * 104)


if __name__ == "__main__":
    main()
