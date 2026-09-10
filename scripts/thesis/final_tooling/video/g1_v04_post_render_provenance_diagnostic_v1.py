#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone

VIDEO_ROOT = Path(r"C:\humanoid_rl_lab\logs\thesis\video_final")
V04_DIR = VIDEO_ROOT / "V04_integrated_sequence"
VIDEO = V04_DIR / "video" / "V04_integrated_sequence_50fps.mp4"
MANIFEST = V04_DIR / "replay" / "V04_final_render_manifest.json"
TRAJECTORY = V04_DIR / "capture" / "V04_master_trajectory_2000steps_50hz.pt"
FRAMES_DIR = V04_DIR / "frames"

CANONICAL_VIDEO_SHA = "6859E7477BA73B28F7FA361BD1E7FF24C002EF106D8DBA543903282C2B173129"
CANONICAL_TRAJECTORY_SHA = "5CD504EA5E1ED58FE426AADF56F94AA8F91A66BB21541CF1AB8C4C4441020333"
EXPECTED_FRAMES = 2000
EXPECTED_WIDTH = 2560
EXPECTED_HEIGHT = 1440
EXPECTED_FPS = 50.0
EXPECTED_DURATION = 40.0

PASS_MARKER = "G1_V04_POST_RENDER_PROVENANCE_DIAGNOSTIC_V1_PASS"
FAIL_MARKER = "G1_V04_POST_RENDER_PROVENANCE_DIAGNOSTIC_V1_FAIL"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def ffprobe_info(ffprobe: str, video: Path) -> dict:
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames:format=duration,size",
        "-of", "json",
        str(video),
    ]
    proc = run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed rc={proc.returncode}: {proc.stderr[-2000:]}")
    payload = json.loads(proc.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"expected exactly one video stream, got {len(streams)}")
    stream = streams[0]
    fmt = payload.get("format", {})

    def rate_to_float(text: str) -> float:
        n, d = text.split("/")
        return float(n) / float(d)

    return {
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "pix_fmt": stream.get("pix_fmt"),
        "r_fps": rate_to_float(stream["r_frame_rate"]),
        "avg_fps": rate_to_float(stream["avg_frame_rate"]),
        "nb_frames": int(stream["nb_frames"]) if stream.get("nb_frames") not in (None, "N/A") else None,
        "nb_read_frames": int(stream["nb_read_frames"]) if stream.get("nb_read_frames") not in (None, "N/A") else None,
        "duration_s": float(fmt["duration"]),
        "size_bytes_probe": int(fmt["size"]),
    }


def main() -> None:
    for p, label in [
        (VIDEO, "V04 MP4"),
        (MANIFEST, "V04 manifest"),
        (TRAJECTORY, "V04 trajectory"),
        (FRAMES_DIR, "V04 frames directory"),
    ]:
        if not p.exists():
            raise RuntimeError(f"missing {label}: {p}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    current_video_sha = sha256_file(VIDEO)
    current_traj_sha = sha256_file(TRAJECTORY)

    manifest_video_sha = str(manifest.get("video_sha256", "")).upper()
    manifest_traj_sha = str(manifest.get("trajectory_sha256", "")).upper()
    manifest_video_bytes = manifest.get("video_bytes")
    manifest_encode = manifest.get("encode_status")

    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        raise RuntimeError("ffmpeg and ffprobe must both be available on PATH")

    info = ffprobe_info(ffprobe, VIDEO)

    frame_paths = sorted(FRAMES_DIR.glob("frame_*.png"))
    frame_count = len(frame_paths)
    frame_name_pass = (
        frame_count == EXPECTED_FRAMES
        and frame_paths[0].name == "frame_000000.png"
        and frame_paths[-1].name == "frame_001999.png"
    )

    decode = run([ffmpeg, "-v", "error", "-i", str(VIDEO), "-f", "null", "-"])
    decode_pass = decode.returncode == 0 and not decode.stderr.strip()

    video_mtime = VIDEO.stat().st_mtime
    manifest_mtime = MANIFEST.stat().st_mtime
    video_newer_than_manifest = video_mtime > manifest_mtime + 0.5

    guards = {
        "manifest_status_pass": manifest.get("status") == "PASS",
        "manifest_encode_pass": manifest_encode == "PASS",
        "manifest_video_sha_is_original_canonical": manifest_video_sha == CANONICAL_VIDEO_SHA,
        "manifest_trajectory_sha_is_canonical": manifest_traj_sha == CANONICAL_TRAJECTORY_SHA,
        "current_trajectory_sha_is_canonical": current_traj_sha == CANONICAL_TRAJECTORY_SHA,
        "current_video_sha_matches_manifest": current_video_sha == manifest_video_sha,
        "current_video_bytes_match_manifest": int(VIDEO.stat().st_size) == int(manifest_video_bytes or -1),
        "resolution_2560x1440": info["width"] == EXPECTED_WIDTH and info["height"] == EXPECTED_HEIGHT,
        "fps_50": abs(info["avg_fps"] - EXPECTED_FPS) < 1e-9,
        "duration_40s": abs(info["duration_s"] - EXPECTED_DURATION) < 1e-3,
        "decoded_frame_count_2000": info["nb_read_frames"] == EXPECTED_FRAMES,
        "canonical_png_sequence_present": frame_name_pass,
        "video_decode_clean": decode_pass,
    }

    if manifest_video_sha == CANONICAL_VIDEO_SHA and current_video_sha != manifest_video_sha:
        diagnosis = "POST_RENDER_MP4_CHANGE_CONFIRMED"
    elif manifest_video_sha == current_video_sha == CANONICAL_VIDEO_SHA:
        diagnosis = "NO_CURRENT_SHA_MISMATCH"
    elif manifest_video_sha != CANONICAL_VIDEO_SHA:
        diagnosis = "MANIFEST_DOES_NOT_MATCH_REPORTED_CANONICAL_SHA"
    else:
        diagnosis = "UNCLASSIFIED"

    print("=" * 124)
    print("G1 V04 POST-RENDER PROVENANCE DIAGNOSTIC — READ ONLY")
    print("=" * 124)
    print(f"Video path                         : {VIDEO}")
    print(f"Manifest path                      : {MANIFEST}")
    print(f"Trajectory path                    : {TRAJECTORY}")
    print()
    print(f"Reported canonical video SHA       : {CANONICAL_VIDEO_SHA}")
    print(f"Manifest video SHA                 : {manifest_video_sha or '<missing>'}")
    print(f"Current video SHA                  : {current_video_sha}")
    print(f"Manifest video bytes               : {manifest_video_bytes}")
    print(f"Current video bytes                : {VIDEO.stat().st_size}")
    print()
    print(f"Canonical trajectory SHA           : {CANONICAL_TRAJECTORY_SHA}")
    print(f"Manifest trajectory SHA            : {manifest_traj_sha or '<missing>'}")
    print(f"Current trajectory SHA             : {current_traj_sha}")
    print()
    print(f"Video mtime UTC                    : {iso_mtime(VIDEO)}")
    print(f"Manifest mtime UTC                 : {iso_mtime(MANIFEST)}")
    print(f"Video newer than manifest          : {video_newer_than_manifest}")
    print()
    print(f"Codec / pix_fmt                    : {info['codec']} / {info['pix_fmt']}")
    print(f"Resolution                         : {info['width']}x{info['height']}")
    print(f"FPS r/avg                          : {info['r_fps']:.9f} / {info['avg_fps']:.9f}")
    print(f"Duration                           : {info['duration_s']:.6f} s")
    print(f"Frames declared / decoded          : {info['nb_frames']} / {info['nb_read_frames']}")
    print(f"PNG sequence count                 : {frame_count}")
    print(f"Decode clean                       : {decode_pass}")
    print()
    print(f"DIAGNOSIS                          : {diagnosis}")
    print("-" * 124)
    for key, value in guards.items():
        print(f"{key:42s}: {'PASS' if value else 'FAIL'}")

    report = {
        "diagnosis": diagnosis,
        "canonical_video_sha": CANONICAL_VIDEO_SHA,
        "manifest_video_sha": manifest_video_sha,
        "current_video_sha": current_video_sha,
        "manifest_video_bytes": manifest_video_bytes,
        "current_video_bytes": VIDEO.stat().st_size,
        "canonical_trajectory_sha": CANONICAL_TRAJECTORY_SHA,
        "manifest_trajectory_sha": manifest_traj_sha,
        "current_trajectory_sha": current_traj_sha,
        "video_mtime_utc": iso_mtime(VIDEO),
        "manifest_mtime_utc": iso_mtime(MANIFEST),
        "video_newer_than_manifest": video_newer_than_manifest,
        "video_info": info,
        "png_count": frame_count,
        "decode_pass": decode_pass,
        "guards": guards,
    }
    report_path = V04_DIR / "replay" / "V04_post_render_provenance_diagnostic_v1.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Diagnostic JSON                    : {report_path}")

    core_source_pass = all([
        guards["manifest_status_pass"],
        guards["manifest_encode_pass"],
        guards["manifest_video_sha_is_original_canonical"],
        guards["manifest_trajectory_sha_is_canonical"],
        guards["current_trajectory_sha_is_canonical"],
        guards["resolution_2560x1440"],
        guards["fps_50"],
        guards["duration_40s"],
        guards["decoded_frame_count_2000"],
        guards["canonical_png_sequence_present"],
        guards["video_decode_clean"],
    ])
    if not core_source_pass:
        raise RuntimeError("V04 provenance/source integrity diagnostic found a core failure; do not build V05 yet.")

    print(PASS_MARKER)


try:
    main()
except Exception as exc:
    print(FAIL_MARKER)
    print(f"EXCEPTION_TYPE={type(exc).__name__}")
    print(f"EXCEPTION_MESSAGE={exc}")
    raise
