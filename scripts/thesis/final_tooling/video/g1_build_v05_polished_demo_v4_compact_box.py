#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

EXPECTED_SOURCES = {
    "V01": {
        "rel": Path(r"V01_scientific_straight\video\V01_scientific_straight_50fps.mp4"),
        "sha256": "0C24DE3B2F85204801BFD7FC76AE958A8FDB33D1A7091FF8B4498F750CB07FF0",
    },
    "V02": {
        "rel": Path(r"V02_curve_pos\video\V02_curve_pos_50fps.mp4"),
        "sha256": "7714B4321DF05E9944C562C6FD3F54706D2193780B9B90BD3259EAE51F78A9C1",
    },
    "V03": {
        "rel": Path(r"V03_curve_neg\video\V03_curve_neg_50fps.mp4"),
        "sha256": "B83616EEDDF39BE033319483F75504571DC960DD4A2C23CD7D194A74EC612C41",
    },
    "V04": {
        "rel": Path(r"V04_integrated_sequence\video\V04_integrated_sequence_50fps.mp4"),
        "sha256": "C73E96B24619B5020906608E8A2B482726F639D5E9C0E0E15E71CA40F957D751",
    },
}

WIDTH = 2560
HEIGHT = 1440
FPS = 50.0
EXPECTED_DURATION_S = 33.2
PASS_MARKER = "G1_V05_POLISHED_PRESENTATION_DEMO_V4_PASS"
FAIL_MARKER = "G1_V05_POLISHED_PRESENTATION_DEMO_V4_FAIL"
CLASSIFICATION = (
    "presentation-only derivative edit assembled from final deterministic V01-V04 renders; "
    "NOT new scientific evidence and NOT ROS deployment evidence"
)

SEGMENTS = [
    {
        "source": "V01",
        "start": 0.0,
        "end": 3.2,
        "label1": "G1 DUAL-POLICY CONTROLLER",
        "label2": "STAND | v2-G",
    },
    {
        "source": "V01",
        "start": 3.2,
        "end": 9.2,
        "label1": "STRAIGHT | SCIENTIFIC REFERENCE",
        "label2": "cmd [0.40, 0, 0] | v4-N",
    },
    {
        "source": "V02",
        "start": 4.0,
        "end": 10.0,
        "label1": "CURVE + | ENGINEERING / DEMO",
        "label2": "cmd [0.40, 0, +0.25] | v4-N",
    },
    {
        "source": "V03",
        "start": 4.0,
        "end": 10.0,
        "label1": "CURVE - | ENGINEERING / DEMO",
        "label2": "cmd [0.40, 0, -0.25] | v4-N",
    },
    {
        "source": "V04",
        "start": 28.0,
        "end": 40.0,
        "dynamic": True,
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
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
            "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,codec_name,pix_fmt:format=duration",
            "-of", "json",
            str(path),
        ],
        capture=True,
    )
    payload = json.loads(cp.stdout)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"expected exactly one primary video stream in {path}, got {len(streams)}")
    s = streams[0]
    rate_text = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1"
    fps = float(Fraction(rate_text))
    duration = float((payload.get("format") or {}).get("duration", 0.0))
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": fps,
        "duration_s": duration,
        "codec": s.get("codec_name"),
        "pix_fmt": s.get("pix_fmt"),
    }


def qtext(text: str) -> str:
    # No apostrophes are used in frozen labels. Escape characters that have filter meaning.
    return text.replace("\\", r"\\").replace(":", r"\:").replace("%", r"\%")


def label_filters(font_spec: str, line1: str, line2: str, *, enable: str | None = None) -> str:
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

    # Segment 0: opening stand. A short fade-in is presentation-only.
    s = SEGMENTS[0]
    lines.append(
        f"[0:v]trim=start={s['start']}:end={s['end']},setpts=PTS-STARTPTS,fps=50,format=yuv420p,"
        f"fade=t=in:st=0:d=0.40,{label_filters(font_spec, s['label1'], s['label2'])}[s0];"
    )

    # Segment 1: straight reference from V01.
    s = SEGMENTS[1]
    lines.append(
        f"[0:v]trim=start={s['start']}:end={s['end']},setpts=PTS-STARTPTS,fps=50,format=yuv420p,"
        f"{label_filters(font_spec, s['label1'], s['label2'])}[s1];"
    )

    # Segment 2: positive engineering curve from V02.
    s = SEGMENTS[2]
    lines.append(
        f"[1:v]trim=start={s['start']}:end={s['end']},setpts=PTS-STARTPTS,fps=50,format=yuv420p,"
        f"{label_filters(font_spec, s['label1'], s['label2'])}[s2];"
    )

    # Segment 3: negative engineering curve from V03.
    s = SEGMENTS[3]
    lines.append(
        f"[2:v]trim=start={s['start']}:end={s['end']},setpts=PTS-STARTPTS,fps=50,format=yuv420p,"
        f"{label_filters(font_spec, s['label1'], s['label2'])}[s3];"
    )

    # Segment 4: V04 source PTS 28..40 s. Local PTS 6.00 = BRAKE start,
    # local PTS 6.36 = STAND start. These are presentation labels only; underlying pixels are unchanged.
    dyn0 = label_filters(font_spec, "INTEGRATED RETURN", "STRAIGHT | v4-N", enable="between(t,0,5.999)")
    dyn1 = label_filters(font_spec, "DUAL-POLICY HANDOFF", "BRAKE 0.36 s | v4-N", enable="between(t,6.0,6.359)")
    dyn2 = label_filters(font_spec, "DUAL-POLICY HANDOFF", "STAND | v2-G", enable="gte(t,6.36)")
    lines.append(
        "[3:v]trim=start=28.0:end=40.0,setpts=PTS-STARTPTS,fps=50,format=yuv420p,"
        f"{dyn0},{dyn1},{dyn2},fade=t=out:st=11.50:d=0.50[s4];"
    )

    lines.append("[s0][s1][s2][s3][s4]concat=n=5:v=1:a=0[outv]\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build V05 polished presentation-only demo from the already-final V01-V04 MP4 files. "
            "No Isaac Sim run and no policy execution occurs."
        )
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path(r"C:\humanoid_rl_lab\logs\thesis\video_final"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"C:\humanoid_rl_lab\logs\thesis\video_final\V05_polished_demo\video\V05_polished_demo_50fps.mp4"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(r"C:\humanoid_rl_lab\logs\thesis\video_final\V05_polished_demo\replay\V05_polished_demo_manifest.json"),
    )
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--preset", default="slow")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe must both be available on PATH")

    filters = run_checked([ffmpeg, "-hide_banner", "-filters"], capture=True).stdout
    if " drawtext " not in filters and "drawtext" not in filters:
        raise RuntimeError("this ffmpeg build does not provide the drawtext filter")

    font = Path(r"C:\Windows\Fonts\segoeui.ttf")
    if not font.is_file():
        font = Path(r"C:\Windows\Fonts\arial.ttf")
    if not font.is_file():
        raise RuntimeError("neither Segoe UI nor Arial font file is available in C:\\Windows\\Fonts")
    # FFmpeg filter parser on Windows requires an escaped drive colon and forward slashes.
    font_spec = str(font).replace("\\", "/").replace(":", r"\:")

    source_paths: dict[str, Path] = {}
    source_info: dict[str, dict] = {}
    print("=" * 118)
    print("G1 V05 POLISHED PRESENTATION DEMO — SOURCE AUDIT")
    print("=" * 118)
    print(f"Classification : {CLASSIFICATION}")
    for name, spec in EXPECTED_SOURCES.items():
        p = args.video_root / spec["rel"]
        if not p.is_file():
            raise RuntimeError(f"missing final source video: {p}")
        actual_sha = sha256_file(p)
        if actual_sha != spec["sha256"]:
            raise RuntimeError(f"{name} SHA mismatch: expected {spec['sha256']}, got {actual_sha}")
        info = ffprobe_video(ffprobe, p)
        if info["width"] != WIDTH or info["height"] != HEIGHT:
            raise RuntimeError(f"{name} resolution mismatch: {info['width']}x{info['height']}")
        if abs(info["fps"] - FPS) > 1e-6:
            raise RuntimeError(f"{name} FPS mismatch: {info['fps']}")
        source_paths[name] = p
        source_info[name] = {**info, "sha256": actual_sha, "path": str(p)}
        print(f"{name}: SHA PASS | {info['width']}x{info['height']} @ {info['fps']:.3f} fps | {info['duration_s']:.3f} s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    filter_path = args.manifest.parent / "V05_polished_demo_filter_complex.txt"
    filter_path.write_text(build_filter_script(font_spec), encoding="utf-8")

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-i", str(source_paths["V01"]),
        "-i", str(source_paths["V02"]),
        "-i", str(source_paths["V03"]),
        "-i", str(source_paths["V04"]),
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

    print("-" * 118)
    print("Building presentation-only derivative MP4...")
    run_checked(cmd)

    final_info = ffprobe_video(ffprobe, args.output)
    if final_info["width"] != WIDTH or final_info["height"] != HEIGHT:
        raise RuntimeError(f"output resolution mismatch: {final_info}")
    if abs(final_info["fps"] - FPS) > 1e-6:
        raise RuntimeError(f"output FPS mismatch: {final_info['fps']}")
    if abs(final_info["duration_s"] - EXPECTED_DURATION_S) > 0.12:
        raise RuntimeError(
            f"output duration mismatch: expected {EXPECTED_DURATION_S:.3f}s, got {final_info['duration_s']:.3f}s"
        )

    output_sha = sha256_file(args.output)
    manifest = {
        "status": "PASS",
        "classification": CLASSIFICATION,
        "output": str(args.output),
        "output_sha256": output_sha,
        "output_video": final_info,
        "expected_duration_s": EXPECTED_DURATION_S,
        "sources": source_info,
        "segments": SEGMENTS,
        "labels": {
            "straight": "SCIENTIFIC REFERENCE",
            "curve_pos": "ENGINEERING / DEMO",
            "curve_neg": "ENGINEERING / DEMO",
            "handoff": "BRAKE 0.36 s v4-N -> STAND v2-G",
        },
        "rendering": "derivative ffmpeg edit only; no new simulation, no new policy inference, no altered source trajectories",
        "filter_script": str(filter_path),
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "encoder": {"codec": "libx264", "preset": args.preset, "crf": args.crf, "pix_fmt": "yuv420p"},
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 118)
    print("V05 POLISHED PRESENTATION DEMO SUMMARY")
    print("=" * 118)
    print("Status                              : PASS")
    print(f"Classification                      : {CLASSIFICATION}")
    print(f"Resolution / FPS                    : {final_info['width']}x{final_info['height']} / {final_info['fps']:.3f}")
    print(f"Duration                            : {final_info['duration_s']:.3f} s")
    print(f"Video                               : {args.output}")
    print(f"Video SHA256                        : {output_sha}")
    print(f"Manifest                            : {args.manifest}")
    print("New simulation / policy execution   : NO")
    print("Source trajectories modified        : NO")
    print(PASS_MARKER)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(FAIL_MARKER, file=sys.stderr)
        print(f"EXCEPTION_TYPE={type(exc).__name__}", file=sys.stderr)
        print(f"EXCEPTION_MESSAGE={exc}", file=sys.stderr)
        raise
