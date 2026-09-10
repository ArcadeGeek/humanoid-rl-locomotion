#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT_DEFAULT = Path(r"C:\humanoid_rl_lab\logs\thesis\video_final\V06_historical_progression")

EXPECTED_SOURCES = {
    "V2-A": {
        "rel": Path(r"video\V06_V2A_historical_50fps.mp4"),
        "sha256": "11BBBF6341F68940751E7C7E2B1765F3AC1AC0AC2E95EFE28C88E9ACFD689D07",
        "title": "V2-A | BASELINE",
        "subtitle": "Direct-yaw genuine scratch",
    },
    "V2-G": {
        "rel": Path(r"video\V06_V2G_historical_50fps.mp4"),
        "sha256": "98505B1631F4DFEB7898A0600BCB61626659A6837E1A99EE1C6B1614FFEF30AD",
        "title": "V2-G | FINAL V2",
        "subtitle": "Flat-orientation refinement | final STAND specialist",
    },
    "V3-C": {
        "rel": Path(r"video\V06_V3C_historical_50fps.mp4"),
        "sha256": "3D85522F9E70D905E0A3035271AF720926FBE610765BEFBE5C529E354BB78DA4",
        "title": "V3-C | EXPLORATORY",
        "subtitle": "Swing-clearance refinement | NOT promoted",
    },
    "V4-A": {
        "rel": Path(r"video\V06_V4A_historical_50fps.mp4"),
        "sha256": "7CB97BF1657ADC6A04762B34FF0A668A5E8BFD527ED706DD4BD0E38FAF230D67",
        "title": "V4-A | NEW SCRATCH CAMPAIGN",
        "subtitle": "Official G1 Flat genuine-scratch baseline",
    },
    "V4-N": {
        "rel": Path(r"video\V06_V4N_historical_50fps.mp4"),
        "sha256": "59DBE6D72E7F866AE9FADF551572A1DFCE8CF23A4824D270B5B454DE312B5F8E",
        "title": "V4-N | FINAL LOCOMOTION",
        "subtitle": "Symmetry augmentation | promoted controller",
    },
}

ORDER = ["V2-A", "V2-G", "V3-C", "V4-A", "V4-N"]

WIDTH = 2560
HEIGHT = 1440
FPS = 50.0
SOURCE_DURATION_S = 4.0
SOURCE_FRAMES = 200
EXPECTED_DURATION_S = 20.0
EXPECTED_OUTPUT_FRAMES = 1000
INTERNAL_DIP_S = 0.16  # 8 frames at 50 fps; zero cross-policy overlap

PASS_MARKER = "G1_V06_HISTORICAL_PROGRESSION_FINAL_V2_SOFT_TRANSITIONS_PASS"
FAIL_MARKER = "G1_V06_HISTORICAL_PROGRESSION_FINAL_V2_SOFT_TRANSITIONS_FAIL"

CLASSIFICATION = (
    "presentation-only historical/project progression assembled from five frozen deterministic historical renders; "
    "NOT a single continuously improving policy, NOT new scientific evidence and NOT ROS deployment evidence"
)

NARRATIVE = (
    "V2-A baseline -> V2-G final v2/STAND specialist -> V3-C exploratory refinement (not promoted) -> "
    "V4-A new genuine-scratch campaign -> V4-N final promoted LOCOMOTION controller"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def run_checked(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=capture)


def ffprobe_video(ffprobe: str, path: Path, *, count_frames: bool = False) -> dict:
    entries = "stream=width,height,avg_frame_rate,r_frame_rate,codec_name,pix_fmt"
    if count_frames:
        entries += ",nb_read_frames"
    entries += ":format=duration"
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
    ]
    if count_frames:
        cmd.append("-count_frames")
    cmd += ["-show_entries", entries, "-of", "json", str(path)]
    cp = run_checked(cmd, capture=True)
    payload = json.loads(cp.stdout)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"expected exactly one primary video stream in {path}, got {len(streams)}")
    s = streams[0]
    rate_text = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1"
    result = {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": float(Fraction(rate_text)),
        "duration_s": float((payload.get("format") or {}).get("duration", 0.0)),
        "codec": str(s.get("codec_name") or ""),
        "pix_fmt": str(s.get("pix_fmt") or ""),
    }
    if count_frames:
        result["decoded_frames"] = int(s.get("nb_read_frames") or 0)
    return result


def decode_clean(ffmpeg: str, path: Path) -> None:
    cp = subprocess.run(
        [
            ffmpeg, "-v", "error",
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
            f"full decode failed for {path.name}: rc={cp.returncode}, stderr={cp.stderr[-2000:]}"
        )


def qtext(text: str) -> str:
    # Frozen labels contain no apostrophes.
    return text.replace("\\", r"\\").replace(":", r"\:").replace("%", r"\%")


def label_filters(font_spec: str, line1: str, line2: str, *, enable: str | None = None) -> str:
    # Exactly the compact panel geometry approved in V05 v4.
    en = f":enable='{enable}'" if enable else ""
    box = f"drawbox=x=60:y=ih-220:w=900:h=130:color=black@0.55:t=fill{en}"
    t1 = (
        f"drawtext=fontfile='{font_spec}':text='{qtext(line1)}':fontcolor=white:fontsize=46:"
        f"x=90:y=h-198{en}"
    )
    t2 = (
        f"drawtext=fontfile='{font_spec}':text='{qtext(line2)}':fontcolor=white:fontsize=30:"
        f"x=92:y=h-137{en}"
    )
    return ",".join((box, t1, t2))


def build_filter_script(font_spec: str) -> str:
    lines: list[str] = []

    # All five sources are already exactly 4 s / 200 frames / 50 fps.
    # Internal dip-to-black transitions soften cuts while preserving zero cross-policy overlap.
    for idx, case in enumerate(ORDER):
        spec = EXPECTED_SOURCES[case]
        filters = [
            f"[{idx}:v]trim=start=0:end=4.0",
            "setpts=PTS-STARTPTS",
            "fps=50",
            "format=yuv420p",
        ]

        # Soft transition without ever blending two distinct policies together.
        # Each internal cut reaches black, then the next clip fades in from black.
        if idx == 0:
            filters.append("fade=t=in:st=0:d=0.30")
        else:
            filters.append(f"fade=t=in:st=0:d={INTERNAL_DIP_S:.2f}")

        if idx < len(ORDER) - 1:
            filters.append(
                f"fade=t=out:st={SOURCE_DURATION_S - INTERNAL_DIP_S:.2f}:d={INTERNAL_DIP_S:.2f}"
            )

        if case != "V4-N":
            filters.append(label_filters(font_spec, spec["title"], spec["subtitle"]))
        else:
            # First 3.1 s identify the final locomotion specialist.
            filters.append(
                label_filters(
                    font_spec,
                    spec["title"],
                    spec["subtitle"],
                    enable="between(t,0,3.099)",
                )
            )
            # Final 0.9 s closes the project narrative without adding a sixth simulation clip.
            filters.append(
                label_filters(
                    font_spec,
                    "FINAL DUAL-POLICY ARCHITECTURE",
                    "v2-G STAND + v4-N LOCOMOTION",
                    enable="gte(t,3.10)",
                )
            )
            filters.append("fade=t=out:st=3.70:d=0.30")

        lines.append(",".join(filters) + f"[s{idx}];")

    lines.append("[s0][s1][s2][s3][s4]concat=n=5:v=1:a=0[outv]\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the final V06 five-milestone historical/project progression from the already frozen "
            "V2-A, V2-G, V3-C, V4-A and V4-N deterministic MP4 renders. No Isaac Sim run occurs."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DEFAULT / "video" / "V06_historical_progression_final_v2_soft_transitions_50fps.mp4",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT_DEFAULT / "replay" / "V06_historical_progression_final_v2_soft_transitions_manifest.json",
    )
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--preset", default="slow")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe must both be available on PATH")

    filters_available = run_checked([ffmpeg, "-hide_banner", "-filters"], capture=True).stdout
    if "drawtext" not in filters_available:
        raise RuntimeError("this ffmpeg build does not provide the drawtext filter")

    font = Path(r"C:\Windows\Fonts\segoeui.ttf")
    if not font.is_file():
        font = Path(r"C:\Windows\Fonts\arial.ttf")
    if not font.is_file():
        raise RuntimeError("neither Segoe UI nor Arial is available under C:\\Windows\\Fonts")
    font_spec = str(font).replace("\\", "/").replace(":", r"\:")

    source_paths: dict[str, Path] = {}
    source_info: dict[str, dict] = {}

    print("=" * 132)
    print("G1 V06 HISTORICAL PROGRESSION FINAL — SOURCE AUDIT")
    print("=" * 132)
    print(f"Classification : {CLASSIFICATION}")
    print(f"Narrative      : {NARRATIVE}")
    print("Editing        : 0.16 s dip-to-black at internal boundaries; no crossfade/overlap between policies")
    print("-" * 132)

    for case in ORDER:
        spec = EXPECTED_SOURCES[case]
        p = args.root / spec["rel"]
        if not p.is_file():
            raise RuntimeError(f"{case}: missing frozen source video: {p}")

        actual_sha = sha256_file(p)
        if actual_sha != spec["sha256"]:
            raise RuntimeError(f"{case}: SHA mismatch: expected {spec['sha256']}, got {actual_sha}")

        info = ffprobe_video(ffprobe, p, count_frames=True)
        if info["width"] != WIDTH or info["height"] != HEIGHT:
            raise RuntimeError(f"{case}: resolution mismatch: {info['width']}x{info['height']}")
        if abs(info["fps"] - FPS) > 1e-6:
            raise RuntimeError(f"{case}: fps mismatch: {info['fps']}")
        if abs(info["duration_s"] - SOURCE_DURATION_S) > 0.05:
            raise RuntimeError(f"{case}: duration mismatch: {info['duration_s']}")
        if info["decoded_frames"] != SOURCE_FRAMES:
            raise RuntimeError(f"{case}: frame-count mismatch: {info['decoded_frames']}")
        if info["codec"].lower() != "h264" or info["pix_fmt"].lower() != "yuv420p":
            raise RuntimeError(f"{case}: codec/pixfmt mismatch: {info['codec']} / {info['pix_fmt']}")

        source_paths[case] = p
        source_info[case] = {
            **info,
            "sha256": actual_sha,
            "path": str(p),
            "title": spec["title"],
            "subtitle": spec["subtitle"],
        }
        print(
            f"{case:4s}: SHA PASS | {info['width']}x{info['height']} @ {info['fps']:.3f} | "
            f"{info['duration_s']:.3f}s | frames={info['decoded_frames']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    filter_path = args.manifest.parent / "V06_historical_progression_final_v2_soft_transitions_filter_complex.txt"
    filter_path.write_text(build_filter_script(font_spec), encoding="utf-8")

    cmd = [ffmpeg, "-hide_banner", "-y"]
    for case in ORDER:
        cmd += ["-i", str(source_paths[case])]
    cmd += [
        "-filter_complex_script", str(filter_path),
        "-map", "[outv]",
        "-an",
        "-c:v", "libx264",
        "-preset", str(args.preset),
        "-crf", str(args.crf),
        "-pix_fmt", "yuv420p",
        "-r", "50",
        "-movflags", "+faststart",
        str(args.output),
    ]

    print("-" * 132)
    print("Building V06 presentation-only historical progression...")
    run_checked(cmd)

    final_info = ffprobe_video(ffprobe, args.output, count_frames=True)
    if final_info["width"] != WIDTH or final_info["height"] != HEIGHT:
        raise RuntimeError(f"output resolution mismatch: {final_info}")
    if abs(final_info["fps"] - FPS) > 1e-6:
        raise RuntimeError(f"output FPS mismatch: {final_info['fps']}")
    if abs(final_info["duration_s"] - EXPECTED_DURATION_S) > 0.05:
        raise RuntimeError(
            f"output duration mismatch: expected {EXPECTED_DURATION_S:.3f}s, got {final_info['duration_s']:.3f}s"
        )
    if final_info["decoded_frames"] != EXPECTED_OUTPUT_FRAMES:
        raise RuntimeError(
            f"output decoded-frame mismatch: expected {EXPECTED_OUTPUT_FRAMES}, got {final_info['decoded_frames']}"
        )
    if final_info["codec"].lower() != "h264" or final_info["pix_fmt"].lower() != "yuv420p":
        raise RuntimeError(f"output codec/pixfmt mismatch: {final_info}")

    # Full decode before declaring the derivative output frozen.
    decode_clean(ffmpeg, args.output)

    output_sha = sha256_file(args.output)
    manifest = {
        "status": "PASS",
        "classification": CLASSIFICATION,
        "narrative": NARRATIVE,
        "selected_milestones": ORDER,
        "omitted_from_video": ["V2-D", "V2-E", "V2-F", "V4-B"],
        "editing_semantics": {
            "source_clip_duration_s": SOURCE_DURATION_S,
            "all_sources_use_same_zero_then_straight_protocol": True,
            "hard_cuts_between_policies": False,
            "dip_to_black_between_policies": True,
            "internal_dip_s_each_side": INTERNAL_DIP_S,
            "internal_dip_frames_each_side": int(round(INTERNAL_DIP_S * FPS)),
            "crossfade_between_policies": False,
            "cross_policy_frame_overlap": False,
            "opening_fade_s": 0.30,
            "closing_fade_s": 0.30,
            "final_architecture_overlay_start_local_v4n_s": 3.10,
            "transition_reason": (
                "soften presentation cuts while preserving temporal comparability and avoiding any visual overlap "
                "between distinct policies"
            ),
        },
        "output": str(args.output),
        "output_sha256": output_sha,
        "output_video": final_info,
        "expected_duration_s": EXPECTED_DURATION_S,
        "expected_decoded_frames": EXPECTED_OUTPUT_FRAMES,
        "sources": source_info,
        "labels": {
            case: {
                "title": EXPECTED_SOURCES[case]["title"],
                "subtitle": EXPECTED_SOURCES[case]["subtitle"],
            }
            for case in ORDER
        },
        "final_architecture_label": "v2-G STAND + v4-N LOCOMOTION",
        "rendering": (
            "FFmpeg derivative edit only; no new simulation, no policy inference, no trajectory mutation, "
            "no interpolation of simulated states"
        ),
        "filter_script": str(filter_path),
        "filter_script_sha256": sha256_file(filter_path),
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "encoder": {
            "codec": "libx264",
            "preset": args.preset,
            "crf": args.crf,
            "pix_fmt": "yuv420p",
            "fps": 50,
        },
        "full_decode_pass": True,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 132)
    print("V06 HISTORICAL PROGRESSION FINAL SUMMARY")
    print("=" * 132)
    print("Status                              : PASS")
    print(f"Classification                      : {CLASSIFICATION}")
    print("Selected milestones                 : V2-A / V2-G / V3-C / V4-A / V4-N")
    print("Transitions                         : 0.16 s DIP-TO-BLACK / no policy overlap")
    print(f"Resolution / FPS                    : {final_info['width']}x{final_info['height']} / {final_info['fps']:.3f}")
    print(f"Duration / decoded frames           : {final_info['duration_s']:.3f} s / {final_info['decoded_frames']}")
    print("Full output decode                  : PASS")
    print(f"Video                               : {args.output}")
    print(f"Video SHA256                        : {output_sha}")
    print(f"Manifest                            : {args.manifest}")
    print("New simulation / policy execution   : NO")
    print("Source videos / trajectories changed: NO")
    print(PASS_MARKER)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(FAIL_MARKER, file=sys.stderr)
        print(f"EXCEPTION_TYPE={type(exc).__name__}", file=sys.stderr)
        print(f"EXCEPTION_MESSAGE={exc}", file=sys.stderr)
        raise
