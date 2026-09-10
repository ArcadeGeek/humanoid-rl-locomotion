from __future__ import annotations

import hashlib
from pathlib import Path


SOURCE = Path(
    r"C:\humanoid_rl_lab\scripts\thesis\baselines\v2g"
    r"\g1_v2g_fixed040_playeq_gait_probe.py"
)
OUTPUT = Path(
    r"C:\Users\Mirko\Desktop\g1_v2g_zero_command_standing_probe.py"
)
CHECKPOINT = Path(
    r"C:\humanoid_rl_lab\checkpoints"
    r"\local_g1_yaw_direct_wide_yaw_flat_orientation_v2g"
    r"\model_600.pt"
)

EXPECTED_SOURCE_SHA = (
    "6594EE372C0CD15D8F0BE3CC1CBB258208CB59AC32373CD83D73A8BA4736F569"
)
EXPECTED_CHECKPOINT_SHA = (
    "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one occurrence, found {count}: {old!r}"
        )
    return text.replace(old, new, 1)


def require(text: str, token: str, label: str) -> None:
    if token not in text:
        raise RuntimeError(f"{label}: missing token: {token!r}")


def main() -> None:
    if not SOURCE.is_file():
        raise RuntimeError(f"canonical source probe missing: {SOURCE}")

    if not CHECKPOINT.is_file():
        raise RuntimeError(f"frozen checkpoint missing: {CHECKPOINT}")

    source_sha = sha256(SOURCE)
    checkpoint_sha = sha256(CHECKPOINT)

    if source_sha != EXPECTED_SOURCE_SHA:
        raise RuntimeError(
            "canonical probe SHA mismatch\n"
            f"expected={EXPECTED_SOURCE_SHA}\n"
            f"actual  ={source_sha}"
        )

    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError(
            "frozen checkpoint SHA mismatch\n"
            f"expected={EXPECTED_CHECKPOINT_SHA}\n"
            f"actual  ={checkpoint_sha}"
        )

    if OUTPUT.exists():
        raise RuntimeError(
            f"output already exists; refusing to overwrite: {OUTPUT}"
        )

    text = SOURCE.read_text(encoding="utf-8")

    for token, label in (
        ("_cmd.heading_command = False", "direct-yaw mode"),
        ("_cmd.ranges.lin_vel_x = (0.40, 0.40)", "fixed-vx command"),
        ("_cmd.ranges.lin_vel_y = (0.0, 0.0)", "zero-vy command"),
        ("_cmd.ranges.ang_vel_z = (0.0, 0.0)", "zero-wz command"),
        (
            "env_cfg.observations.policy.enable_corruption = False",
            "observation corruption disabled",
        ),
        (
            "env_cfg.events.base_external_force_torque = None",
            "external forces disabled",
        ),
        ("env_cfg.events.push_robot = None", "pushes disabled"),
        ("_PROBE_WARMUP_STEPS = 150", "warmup"),
        ("_PROBE_RECORD_STEPS = 600", "record length"),
        ("obs = env.get_observations()", "initial observation"),
        ("actions = policy(obs)", "policy inference"),
        ("obs, _, dones, _ = env.step(actions)", "environment stepping"),
    ):
        require(text, token, label)

    text = replace_once(
        text,
        "_cmd.ranges.lin_vel_x = (0.40, 0.40)",
        "_cmd.ranges.lin_vel_x = (0.0, 0.0)",
        "zero-command vx",
    )

    text = replace_once(
        text,
        'print("Command                             : [0.40, 0.00, 0.00]")',
        'print("Command                             : [0.00, 0.00, 0.00]")',
        "console command label",
    )

    text = replace_once(
        text,
        "_PROBE_RECORD_STEPS = 600",
        "_PROBE_RECORD_STEPS = 1000",
        "20-second record length",
    )

    text = replace_once(
        text,
        r'r"C:\humanoid_rl_lab\logs\thesis\gait_probe\frozen_v2g_fixed040_playeq"',
        r'r"C:\humanoid_rl_lab\logs\thesis\standing_probe\frozen_v2g_zero_command_playeq_seed42_20s"',
        "standing output directory",
    )
    text = replace_once(
        text,
        '"frozen_v2g_fixed040_playeq_12s_trajectory.csv"',
        '"frozen_v2g_zero_command_playeq_seed42_20s_samples.csv"',
        "standing CSV name",
    )
    text = replace_once(
        text,
        '"frozen_v2g_fixed040_playeq_12s_meta.txt"',
        '"frozen_v2g_zero_command_playeq_seed42_20s_meta.txt"',
        "standing metadata name",
    )

    text = replace_once(
        text,
        "        import hashlib as _probe_hashlib\n",
        "        import hashlib as _probe_hashlib\n"
        "        import math as _probe_math\n",
        "math import",
    )

    metric_needle = (
        "                        _checkpoint_sha = _probe_sha256(resume_path)\n"
    )

    metric_block = '''                        _checkpoint_sha = _probe_sha256(resume_path)

                        # ZERO-COMMAND STANDING SUMMARY - offline metrics only.
                        def _probe_percentile(values, q):
                            values = sorted(values)
                            pos = (len(values) - 1) * q
                            lo = int(_probe_math.floor(pos))
                            hi = int(_probe_math.ceil(pos))
                            if lo == hi:
                                return values[lo]
                            alpha = pos - lo
                            return values[lo] * (1.0 - alpha) + values[hi] * alpha

                        def _probe_root_tilt_rad(row):
                            qx = row["root_qx"]
                            qy = row["root_qy"]
                            rzz = 1.0 - 2.0 * (qx * qx + qy * qy)
                            rzz = max(-1.0, min(1.0, rzz))
                            return _probe_math.acos(rzz)

                        def _probe_root_yaw_rad(row):
                            qw = row["root_qw"]
                            qx = row["root_qx"]
                            qy = row["root_qy"]
                            qz = row["root_qz"]
                            siny = 2.0 * (qw * qz + qx * qy)
                            cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
                            return _probe_math.atan2(siny, cosy)

                        _x0 = _probe_rows[0]["root_x_w"]
                        _y0 = _probe_rows[0]["root_y_w"]

                        _xy_exc = [
                            _probe_math.hypot(
                                r["root_x_w"] - _x0,
                                r["root_y_w"] - _y0,
                            )
                            for r in _probe_rows
                        ]
                        _planar_speed = [
                            _probe_math.hypot(r["root_vx_b"], r["root_vy_b"])
                            for r in _probe_rows
                        ]
                        _abs_wz = [
                            abs(r["root_wz_b"])
                            for r in _probe_rows
                        ]

                        _left_foot_speed = [
                            _probe_math.hypot(
                                r["left_foot_vx_w"],
                                r["left_foot_vy_w"],
                            )
                            for r in _probe_rows
                        ]
                        _right_foot_speed = [
                            _probe_math.hypot(
                                r["right_foot_vx_w"],
                                r["right_foot_vy_w"],
                            )
                            for r in _probe_rows
                        ]
                        _foot_xy_mean_per_sample = [
                            0.5 * (left + right)
                            for left, right in zip(
                                _left_foot_speed,
                                _right_foot_speed,
                            )
                        ]

                        _root_tilt = [
                            _probe_root_tilt_rad(r)
                            for r in _probe_rows
                        ]

                        _both_contact_pct = 100.0 * _probe_statistics.fmean(
                            1.0
                            if (r["left_contact"] and r["right_contact"])
                            else 0.0
                            for r in _probe_rows
                        )

                        _stance_width = [
                            abs(r["left_foot_y_w"] - r["right_foot_y_w"])
                            for r in _probe_rows
                        ]
                        _fore_aft = [
                            abs(r["left_foot_x_w"] - r["right_foot_x_w"])
                            for r in _probe_rows
                        ]

                        _leg_joints = (
                            "left_hip_pitch_joint",
                            "right_hip_pitch_joint",
                            "left_knee_joint",
                            "right_knee_joint",
                            "left_ankle_pitch_joint",
                            "right_ankle_pitch_joint",
                        )

                        for _joint_name in _leg_joints:
                            if f"q::{_joint_name}" not in _probe_rows[0]:
                                raise RuntimeError(
                                    f"standing metric joint position missing: {_joint_name}"
                                )
                            if f"qd::{_joint_name}" not in _probe_rows[0]:
                                raise RuntimeError(
                                    f"standing metric joint velocity missing: {_joint_name}"
                                )

                        _leg_qd = [
                            sum(
                                abs(r[f"qd::{name}"])
                                for name in _leg_joints
                            )
                            / len(_leg_joints)
                            for r in _probe_rows
                        ]

                        _yaw_raw = [
                            _probe_root_yaw_rad(r)
                            for r in _probe_rows
                        ]
                        _yaw_unwrapped = [_yaw_raw[0]]
                        for _yaw_value in _yaw_raw[1:]:
                            _candidate = _yaw_value
                            _delta = _candidate - _yaw_unwrapped[-1]
                            while _delta > _probe_math.pi:
                                _candidate -= 2.0 * _probe_math.pi
                                _delta = _candidate - _yaw_unwrapped[-1]
                            while _delta < -_probe_math.pi:
                                _candidate += 2.0 * _probe_math.pi
                                _delta = _candidate - _yaw_unwrapped[-1]
                            _yaw_unwrapped.append(_candidate)

                        _stand_drift_m = _probe_math.hypot(
                            _probe_rows[-1]["root_x_w"] - _x0,
                            _probe_rows[-1]["root_y_w"] - _y0,
                        )
                        _stand_max_xy_m = max(_xy_exc)
                        _stand_vxy_mean = _probe_statistics.fmean(
                            _planar_speed
                        )
                        _stand_vxy_p95 = _probe_percentile(
                            _planar_speed, 0.95
                        )
                        _stand_abs_wz_mean = _probe_statistics.fmean(
                            _abs_wz
                        )
                        _stand_tilt_mean_deg = _probe_math.degrees(
                            _probe_statistics.fmean(_root_tilt)
                        )
                        _stand_tilt_p95_deg = _probe_math.degrees(
                            _probe_percentile(_root_tilt, 0.95)
                        )
                        _stand_foot_xy_mean = _probe_statistics.fmean(
                            _foot_xy_mean_per_sample
                        )
                        _stand_foot_xy_p95 = _probe_percentile(
                            _foot_xy_mean_per_sample, 0.95
                        )
                        _stand_leg_qd_mean = _probe_statistics.fmean(
                            _leg_qd
                        )
                        _stand_width_mean_m = _probe_statistics.fmean(
                            _stance_width
                        )
                        _stand_fore_aft_mean_m = _probe_statistics.fmean(
                            _fore_aft
                        )
                        _stand_heading_net_deg = _probe_math.degrees(
                            _yaw_unwrapped[-1] - _yaw_unwrapped[0]
                        )

                        _stand_joint_mean_std = {}
                        for _joint_name in _leg_joints:
                            _values = [
                                r[f"q::{_joint_name}"]
                                for r in _probe_rows
                            ]
                            _stand_joint_mean_std[_joint_name] = (
                                _probe_statistics.fmean(_values),
                                _probe_statistics.pstdev(_values),
                            )
'''

    text = replace_once(
        text,
        metric_needle,
        metric_block,
        "standing metric computation",
    )

    old_final_marker = (
        '                        print("G1_OFFICIAL_FIXED040_GAIT_PROBE_PASS")'
    )

    new_final_block = '''                        print()
                        print("=" * 100)
                        print("V2-G ZERO-COMMAND STANDING SUMMARY")
                        print("=" * 100)
                        print(f"drift_m                            : {_stand_drift_m:.6f}")
                        print(f"max_xy_excursion_m                 : {_stand_max_xy_m:.6f}")
                        print(f"planar_speed_mean_mps              : {_stand_vxy_mean:.6f}")
                        print(f"planar_speed_p95_mps               : {_stand_vxy_p95:.6f}")
                        print(f"abs_yaw_rate_mean_radps            : {_stand_abs_wz_mean:.6f}")
                        print(f"root_tilt_mean_deg                 : {_stand_tilt_mean_deg:.6f}")
                        print(f"root_tilt_p95_deg                  : {_stand_tilt_p95_deg:.6f}")
                        print(f"foot_xy_speed_mean_mps             : {_stand_foot_xy_mean:.6f}")
                        print(f"foot_xy_speed_p95_mps              : {_stand_foot_xy_p95:.6f}")
                        print(f"leg_joint_speed_mean_radps         : {_stand_leg_qd_mean:.6f}")
                        print(f"double_support_pct                 : {_both_contact_pct:.2f}")
                        print(f"stance_width_mean_cm               : {100.0 * _stand_width_mean_m:.3f}")
                        print(f"foot_fore_aft_separation_mean_cm   : {100.0 * _stand_fore_aft_mean_m:.3f}")
                        print(f"heading_net_deg                    : {_stand_heading_net_deg:.6f}")
                        print("joint mean +/- std [rad]")
                        for _joint_name in _leg_joints:
                            _mean, _std = _stand_joint_mean_std[_joint_name]
                            print(
                                f"  {_joint_name:<35} "
                                f"{_mean:+.6f} +/- {_std:.6f}"
                            )
                        print("=" * 100)
                        print("G1_V2G_ZERO_COMMAND_STANDING_PROBE_PASS")'''

    text = replace_once(
        text,
        old_final_marker,
        new_final_block,
        "final standing summary marker",
    )

    text = text.replace("FIXED-0.40", "ZERO-COMMAND")
    text = text.replace("fixed-.40", "zero-command")
    text = text.replace(
        "CONTROLLED FIXED COMMAND",
        "CONTROLLED ZERO COMMAND",
    )

    for token in (
        "_cmd.ranges.lin_vel_x = (0.0, 0.0)",
        "_cmd.ranges.lin_vel_y = (0.0, 0.0)",
        "_cmd.ranges.ang_vel_z = (0.0, 0.0)",
        "_PROBE_RECORD_STEPS = 1000",
        "frozen_v2g_zero_command_playeq_seed42_20s",
        "G1_V2G_ZERO_COMMAND_STANDING_PROBE_PASS",
    ):
        require(text, token, "generated derivative")

    if "_cmd.ranges.lin_vel_x = (0.40, 0.40)" in text:
        raise RuntimeError(
            "generated derivative still contains fixed vx=0.40"
        )

    compile(text, str(OUTPUT), "exec")

    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    output_sha = sha256(OUTPUT)

    print(f"SOURCE_SHA256={source_sha}")
    print(f"CHECKPOINT_SHA256={checkpoint_sha}")
    print(f"OUTPUT={OUTPUT}")
    print(f"OUTPUT_SHA256={output_sha}")
    print("V2G_ZERO_COMMAND_STANDING_PROBE_BUILD_PASS")


if __name__ == "__main__":
    main()
