#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(r"C:\humanoid_rl_lab\logs\thesis\video_final\V06_historical_progression")
OUTPUT_JSON = ROOT / "replay" / "V06_selected_video_source_freeze_audit_v1.json"

EXPECTED = {
    "V2-A": {
        "video": ROOT / "video" / "V06_V2A_historical_50fps.mp4",
        "manifest": ROOT / "replay" / "V2A_final_render_manifest.json",
        "trajectory_sha256": "C87B8FF38287308D735DFA7953B18174A849367FE07C3F00C45F83449EB998CB",
        "renderer_sha256": "6434F676BF854D42E8E5F34062A85C69E7DF830DB26ED093A1A49F82E88E6611",
        "role": "direct-yaw genuine-scratch baseline",
    },
    "V2-G": {
        "video": ROOT / "video" / "V06_V2G_historical_50fps.mp4",
        "manifest": ROOT / "replay" / "V2G_final_render_manifest.json",
        "trajectory_sha256": "0ECEE5C3D480C105078E2DB6E8F8FB95592DB8D83D508871D6F6DA19A42FF027",
        "renderer_sha256": "091C318BCC3827945D6F3D2A0D5E9FC9C2CE25D7AE00884FA39286650CA63F96",
        "role": "flat-orientation refinement / frozen STAND controller",
    },
    "V3-C": {
        "video": ROOT / "video" / "V06_V3C_historical_50fps.mp4",
        "manifest": ROOT / "replay" / "V3C_final_render_manifest.json",
        "trajectory_sha256": "E88F1AA00B21632FC6BFD884D62EFC0B91EDDB1BC616CAC3983CA8F85B3C7B77",
        "renderer_sha256": "E09F351A47504D947D354642E3A667F05FA931E50B3BD68E173B388AD1956519",
        "role": "exploratory swing-clearance refinement / NOT promoted",
    },
    "V4-A": {
        "video": ROOT / "video" / "V06_V4A_historical_50fps.mp4",
        "manifest": ROOT / "replay" / "V4A_final_render_manifest.json",
        "trajectory_sha256": "FABD4FA23C2B5EE1171AEC26DB0CF4BB9BBE64F349CEA1548C0E8E05CE8E6033",
        "renderer_sha256": "C157653DE9F8A8EAC0F3D47243102D1631F763B6D02441FF4EA46E40C34E7D04",
        "role": "official G1 flat genuine-scratch reproduction",
    },
    "V4-N": {
        "video": ROOT / "video" / "V06_V4N_historical_50fps.mp4",
        "manifest": ROOT / "replay" / "V4N_final_render_manifest.json",
        "trajectory_sha256": "261FB104C6C3F5F2EC77B9718E351D6AA0E7012C8A9FE9D3FB7104A449C357D5",
        "renderer_sha256": "95BE16EBDDAA341E82912A9021A42C22CA7161E44BDB68D08E8BBFA53C72B2DC",
        "role": "symmetry augmentation / final LOCOMOTION controller",
    },
}

WIDTH = 2560
HEIGHT = 1440
FPS = 50.0
DURATION_S = 4.0
FRAMES = 200
STATE_TOL = 5.0e-4

PASS_MARKER = "G1_V06_SELECTED_VIDEO_SOURCE_FREEZE_AUDIT_V1_PASS"
FAIL_MARKER = "G1_V06_SELECTED_VIDEO_SOURCE_FREEZE_AUDIT_V1_FAIL"

CLASSIFICATION = (
    "read-only provenance/technical audit of the five selected V06 historical presentation renders; "
    "no simulation, no policy execution, no video mutation"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def run_checked(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=capture,
    )


def ffprobe_video(ffprobe: str, path: Path) -> dict:
    cp = run_checked(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,codec_name,pix_fmt,nb_read_frames:format=duration",
            "-of", "json",
            str(path),
        ],
        capture=True,
    )
    payload = json.loads(cp.stdout)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"{path.name}: expected one video stream, got {len(streams)}")
    s = streams[0]
    fps_text = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1"
    fps = float(Fraction(fps_text))
    duration = float((payload.get("format") or {}).get("duration", 0.0))
    decoded_frames = int(s.get("nb_read_frames") or 0)
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": fps,
        "duration_s": duration,
        "codec": str(s.get("codec_name") or ""),
        "pix_fmt": str(s.get("pix_fmt") or ""),
        "decoded_frames": decoded_frames,
    }


def exact_float(value, target: float, tol: float = 1.0e-9) -> bool:
    try:
        return abs(float(value) - float(target)) <= tol
    except Exception:
        return False


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def audit_manifest(case: str, spec: dict, actual_video_sha: str) -> dict:
    manifest_path = spec["manifest"]
    require(manifest_path.is_file(), f"{case}: manifest missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    require(payload.get("status") == "PASS", f"{case}: manifest status is not PASS")
    require(payload.get("encode_status") == "PASS", f"{case}: manifest encode_status is not PASS")
    require(str(payload.get("trajectory_sha256", "")).upper() == spec["trajectory_sha256"],
            f"{case}: trajectory SHA mismatch in manifest")
    require(str(payload.get("renderer_sha256", "")).upper() == spec["renderer_sha256"],
            f"{case}: renderer SHA mismatch in manifest")
    require(str(payload.get("video_sha256", "")).upper() == actual_video_sha,
            f"{case}: manifest video SHA != current MP4 SHA")

    require(int(payload.get("frames", -1)) == FRAMES, f"{case}: manifest frames mismatch")
    require(int(payload.get("fps", -1)) == int(FPS), f"{case}: manifest fps mismatch")
    require(exact_float(payload.get("duration_s"), DURATION_S), f"{case}: manifest duration mismatch")

    resolution = payload.get("resolution")
    require(resolution == [WIDTH, HEIGHT], f"{case}: manifest resolution mismatch: {resolution}")
    require(exact_float(payload.get("viewport_scale"), 1.0), f"{case}: viewport scale mismatch")

    physics = payload.get("physics_replay") or {}
    require(physics.get("real_physx_step_per_frame") is True, f"{case}: no real PhysX step-per-frame flag")
    require(physics.get("wallclock_pacing") is False, f"{case}: wallclock pacing should be false")
    require(physics.get("recorded_actions_used") is True, f"{case}: recorded-actions flag missing")
    require(physics.get("recorded_policy_commands_used") is True, f"{case}: recorded-command flag missing")

    root_err = float(physics.get("max_root_pos_abs_error", math.inf))
    quat_err = float(physics.get("max_root_quat_sign_invariant_abs_error", math.inf))
    joint_err = float(physics.get("max_joint_pos_abs_error", math.inf))
    require(root_err <= STATE_TOL, f"{case}: root-pos replay error too large: {root_err}")
    require(quat_err <= STATE_TOL, f"{case}: root-quat replay error too large: {quat_err}")
    require(joint_err <= STATE_TOL, f"{case}: joint-pos replay error too large: {joint_err}")

    visual = payload.get("visual_preset") or {}
    expected_visual = {
        "showcase_shift_x": -60.0,
        "camera_forward": -3.10,
        "camera_left": -2.90,
        "camera_eye_z": 1.58,
        "camera_target_z": 0.92,
        "camera_lookahead": 0.00,
        "camera_position_smoothing_s": 0.85,
        "camera_heading_mode": "fixed",
        "lighting": "softbox",
        "ground": "demo",
        "scenery": "rough-dark",
        "aa_op": 4,
        "motion_blur": False,
        "frame_generation": False,
    }
    for key, target in expected_visual.items():
        value = visual.get(key)
        if isinstance(target, float):
            require(exact_float(value, target), f"{case}: visual {key} mismatch: {value}")
        else:
            require(value == target, f"{case}: visual {key} mismatch: {value}")

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "trajectory_sha256": spec["trajectory_sha256"],
        "renderer_sha256": spec["renderer_sha256"],
        "max_state_errors": {
            "root_pos": root_err,
            "root_quat": quat_err,
            "joint_pos": joint_err,
        },
    }


def decode_clean(ffmpeg: str, path: Path) -> None:
    cp = subprocess.run(
        [
            ffmpeg,
            "-v", "error",
            "-i", str(path),
            "-map", "0:v:0",
            "-f", "null",
            os.devnull,
        ],
        text=True,
        capture_output=True,
    )
    if cp.returncode != 0 or cp.stderr.strip():
        raise RuntimeError(
            f"{path.name}: decode check failed rc={cp.returncode}: {cp.stderr[-2000:]}"
        )


def main() -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    require(bool(ffmpeg), "ffmpeg not found on PATH")
    require(bool(ffprobe), "ffprobe not found on PATH")

    print("=" * 132)
    print("G1 V06 SELECTED HISTORICAL VIDEO SOURCE FREEZE AUDIT")
    print("=" * 132)
    print(f"Classification : {CLASSIFICATION}")
    print("Selected set   : V2-A / V2-G / V3-C620 / V4-A / V4-N")
    print("Expected       : 2560x1440 | 50 fps | 4.000 s | 200 decoded frames | H.264 yuv420p")
    print("-" * 132)

    results = {}
    for case, spec in EXPECTED.items():
        video_path = spec["video"]
        require(video_path.is_file(), f"{case}: video missing: {video_path}")

        actual_sha = sha256_file(video_path)
        info = ffprobe_video(ffprobe, video_path)

        require(info["width"] == WIDTH and info["height"] == HEIGHT,
                f"{case}: resolution mismatch: {info['width']}x{info['height']}")
        require(abs(info["fps"] - FPS) <= 1.0e-6, f"{case}: fps mismatch: {info['fps']}")
        require(abs(info["duration_s"] - DURATION_S) <= 0.05,
                f"{case}: duration mismatch: {info['duration_s']}")
        require(info["decoded_frames"] == FRAMES,
                f"{case}: decoded frame count mismatch: {info['decoded_frames']}")
        require(info["codec"].lower() == "h264", f"{case}: codec mismatch: {info['codec']}")
        require(info["pix_fmt"].lower() == "yuv420p", f"{case}: pixel format mismatch: {info['pix_fmt']}")

        decode_clean(ffmpeg, video_path)
        manifest_info = audit_manifest(case, spec, actual_sha)

        results[case] = {
            "role": spec["role"],
            "video": str(video_path),
            "video_sha256": actual_sha,
            "video_bytes": int(video_path.stat().st_size),
            "ffprobe": info,
            **manifest_info,
            "decode_clean": True,
        }

        print(
            f"{case:4s} : PASS | SHA={actual_sha} | "
            f"{info['width']}x{info['height']} @ {info['fps']:.3f} | "
            f"{info['duration_s']:.3f}s | frames={info['decoded_frames']} | decode=PASS"
        )

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "PASS",
        "classification": CLASSIFICATION,
        "selected_milestones": ["V2-A", "V2-G", "V3-C", "V4-A", "V4-N"],
        "omitted_from_video": ["V2-D", "V2-E", "V2-F", "V4-B"],
        "video_plan": "five-milestone historical/project progression; not one continuously improving policy",
        "sources": results,
        "source_mutation": False,
        "simulation_or_policy_execution": False,
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 132)
    print("V06 SELECTED VIDEO SOURCE FREEZE SUMMARY")
    print("=" * 132)
    print("Status                              : PASS")
    print("Selected videos                     : 5 / 5")
    print("Manifest / trajectory provenance    : PASS")
    print("Physics replay state guards         : PASS")
    print("Resolution / FPS / frames           : 2560x1440 / 50 / 200 each")
    print("Full H.264 decode                    : PASS 5 / 5")
    print("Source videos modified              : NO")
    print("Simulation / policy execution       : NO")
    print(f"Output JSON                         : {OUTPUT_JSON}")
    print(PASS_MARKER)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(FAIL_MARKER, file=sys.stderr)
        print(f"EXCEPTION_TYPE={type(exc).__name__}", file=sys.stderr)
        print(f"EXCEPTION_MESSAGE={exc}", file=sys.stderr)
        raise
