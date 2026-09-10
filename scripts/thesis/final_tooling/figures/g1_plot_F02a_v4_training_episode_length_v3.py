#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(r"C:\humanoid_rl_lab")
DEFAULT_ROOT = REPO / r"logs\thesis\figures_final\02_v4_campaign"
DEFAULT_PNG_DIR = DEFAULT_ROOT / "png"
DEFAULT_PDF_DIR = DEFAULT_ROOT / "pdf"
DEFAULT_DATA_DIR = DEFAULT_ROOT / "data"

CASES = {
    "V4-A": {
        "event": REPO / (
            r"logs\rsl_rl\humanoid_rl_lab_g1_v4a_official_baseline_scratch"
            r"\2026-09-01_14-49-13_v4a_genuine_scratch_seed42_official1500"
            r"\events.out.tfevents.1788267047.DESKTOP-25EV483.85020.0"
        ),
        "sha256": "FA7BF7548C0D4EB1405A4D09525CFCB0FE760F138829476E309FD7C031AF1A9B",
        "color": "#4C78A8",
    },
    "V4-B": {
        "event": REPO / (
            r"logs\rsl_rl\humanoid_rl_lab_g1_v4b_direct_yaw_scratch"
            r"\2026-09-01_20-08-18_v4b_genuine_scratch_seed42_official1500"
            r"\events.out.tfevents.1788286190.DESKTOP-25EV483.73332.0"
        ),
        "sha256": "78ED806268C647336B9AC95CBEF24093001E95B9B9EF5DB9F2BB0F02140F4AC3",
        "color": "#54A24B",
    },
    "V4-N": {
        "event": REPO / (
            r"logs\rsl_rl\humanoid_rl_lab_g1_v4n_symmetry_augmentation_scratch"
            r"\2026-09-04_13-58-13_v4n_genuine_scratch_seed42_official1500"
            r"\events.out.tfevents.1788523186.DESKTOP-25EV483.64012.0"
        ),
        "sha256": "5BA7A9DEB190DE94192A131530198644AA265053E8B0D996570CAEA854AEE358",
        "color": "#F58518",  # frozen thesis convention: v4-N = orange
    },
}

TAG = "Train/mean_episode_length"
WINDOW = 25
EXPECTED_POINTS = 1500
ZOOM_MAX_ITERATION = 600
REFERENCE_STEPS = 980.0

PNG_NAME = "F02a_v4_training_episode_length_v3.png"
PDF_NAME = "F02a_v4_training_episode_length_v3.pdf"
CSV_NAME = "F02a_v4_training_episode_length_data_v3.csv"
PASS_MARKER = "G1_THESIS_F02A_V4_TRAINING_EPISODE_LENGTH_V3_PASS"


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


# ---------------------------------------------------------------------------
# Minimal TensorBoard event reader
# ---------------------------------------------------------------------------
# TensorBoard event files are TFRecord streams containing protobuf Event
# messages. For this figure we only need:
#
# Event.step                    field 2, varint
# Event.summary                 field 5, length-delimited
# Summary.value                 field 1, repeated length-delimited
# Summary.Value.tag             field 1, string
# Summary.Value.simple_value    field 2, fixed32 float
#
# Reading those wire fields directly avoids optional dependencies such as
# tensorboardX, tensorboard, or tensorflow and does not alter the source logs.


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if pos >= len(data):
            fail("truncated protobuf varint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, pos
        shift += 7
        if shift > 70:
            fail("invalid protobuf varint")


def _iter_proto_fields(data: bytes):
    pos = 0
    n = len(data)
    while pos < n:
        key, pos = _read_varint(data, pos)
        field_number = key >> 3
        wire_type = key & 0x07

        if field_number <= 0:
            fail("invalid protobuf field number")

        if wire_type == 0:  # varint
            value, pos = _read_varint(data, pos)

        elif wire_type == 1:  # fixed64
            if pos + 8 > n:
                fail("truncated protobuf fixed64")
            value = data[pos : pos + 8]
            pos += 8

        elif wire_type == 2:  # length-delimited
            length, pos = _read_varint(data, pos)
            if pos + length > n:
                fail("truncated protobuf length-delimited field")
            value = data[pos : pos + length]
            pos += length

        elif wire_type == 5:  # fixed32
            if pos + 4 > n:
                fail("truncated protobuf fixed32")
            value = data[pos : pos + 4]
            pos += 4

        else:
            fail(f"unsupported protobuf wire type {wire_type}")

        yield field_number, wire_type, value


def _parse_summary_value(message: bytes) -> tuple[str | None, float | None]:
    tag = None
    simple_value = None

    for field_number, wire_type, value in _iter_proto_fields(message):
        if field_number == 1 and wire_type == 2:
            tag = value.decode("utf-8")
        elif field_number == 2 and wire_type == 5:
            simple_value = struct.unpack("<f", value)[0]

    return tag, simple_value


def _parse_summary(message: bytes, target_tag: str) -> list[float]:
    values = []

    for field_number, wire_type, value in _iter_proto_fields(message):
        if field_number == 1 and wire_type == 2:
            tag, scalar = _parse_summary_value(value)
            if tag == target_tag and scalar is not None:
                values.append(float(scalar))

    return values


def _parse_event(message: bytes, target_tag: str) -> tuple[int, list[float]]:
    # Protobuf default for an omitted int64 field is zero. Event step 0 is
    # therefore legitimately absent on the wire in the first logged event.
    step = 0
    values = []

    for field_number, wire_type, value in _iter_proto_fields(message):
        if field_number == 2 and wire_type == 0:
            step = int(value)
        elif field_number == 5 and wire_type == 2:
            values.extend(_parse_summary(value, target_tag))

    return step, values


def read_scalar_tag(path: Path, target_tag: str) -> tuple[np.ndarray, np.ndarray]:
    steps: list[int] = []
    values: list[float] = []

    with path.open("rb") as fh:
        record_index = 0

        while True:
            length_bytes = fh.read(8)
            if len(length_bytes) == 0:
                break
            if len(length_bytes) != 8:
                fail(f"{path}: truncated TFRecord length at record {record_index}")

            record_length = struct.unpack("<Q", length_bytes)[0]

            # TFRecord CRC fields are framing metadata; they are consumed here.
            length_crc = fh.read(4)
            payload = fh.read(record_length)
            payload_crc = fh.read(4)

            if len(length_crc) != 4:
                fail(f"{path}: truncated TFRecord length CRC at record {record_index}")
            if len(payload) != record_length:
                fail(f"{path}: truncated TFRecord payload at record {record_index}")
            if len(payload_crc) != 4:
                fail(f"{path}: truncated TFRecord payload CRC at record {record_index}")

            step, scalars = _parse_event(payload, target_tag)
            for scalar in scalars:
                steps.append(step)
                values.append(scalar)

            record_index += 1

    if not values:
        fail(f"{path}: tag not found: {target_tag}")

    step_array = np.asarray(steps, dtype=int)
    value_array = np.asarray(values, dtype=float)

    order = np.argsort(step_array, kind="stable")
    return step_array[order], value_array[order]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    if len(values) < window:
        return result

    kernel = np.ones(window, dtype=float) / float(window)
    result[window - 1 :] = np.convolve(values, kernel, mode="valid")
    return result


def first_crossing(steps: np.ndarray, smoothed: np.ndarray, threshold: float):
    indices = np.flatnonzero(np.isfinite(smoothed) & (smoothed >= threshold))
    if len(indices) == 0:
        return None
    return int(steps[int(indices[0])])


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Segoe UI", "DejaVu Sans"],
            "font.size": 15,
            "axes.titlesize": 22,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.linewidth": 1.25,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.28,
            "legend.fontsize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thesis F02a from the three canonical v4 TensorBoard runs."
    )
    parser.add_argument("--event-v4a", type=Path, default=CASES["V4-A"]["event"])
    parser.add_argument("--event-v4b", type=Path, default=CASES["V4-B"]["event"])
    parser.add_argument("--event-v4n", type=Path, default=CASES["V4-N"]["event"])
    parser.add_argument("--png-dir", type=Path, default=DEFAULT_PNG_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    event_paths = {
        "V4-A": args.event_v4a,
        "V4-B": args.event_v4b,
        "V4-N": args.event_v4n,
    }

    datasets = {}
    crossings = {}

    for case in ("V4-A", "V4-B", "V4-N"):
        path = event_paths[case]

        if not path.is_file():
            fail(f"{case}: missing TensorBoard event file: {path}")

        actual_sha = sha256_file(path)
        if actual_sha != CASES[case]["sha256"]:
            fail(
                f"{case}: TensorBoard SHA mismatch: "
                f"expected {CASES[case]['sha256']}, got {actual_sha}"
            )

        steps, raw = read_scalar_tag(path, TAG)

        if len(raw) != EXPECTED_POINTS:
            fail(f"{case}: expected {EXPECTED_POINTS} points, got {len(raw)}")
        if not np.all(np.isfinite(raw)):
            fail(f"{case}: non-finite episode-length data")
        if len(np.unique(steps)) != EXPECTED_POINTS:
            fail(f"{case}: duplicate/missing TensorBoard step indices")
        if not np.array_equal(steps, np.arange(EXPECTED_POINTS, dtype=int)):
            fail(f"{case}: expected TensorBoard steps 0..1499")

        smooth = moving_average(raw, WINDOW)
        crossing = first_crossing(steps, smooth, REFERENCE_STEPS)

        datasets[case] = {
            "steps": steps,
            "raw": raw,
            "smooth": smooth,
        }
        crossings[case] = crossing

    args.png_dir.mkdir(parents=True, exist_ok=True)
    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    reference_grid = datasets["V4-A"]["steps"]
    for case in ("V4-B", "V4-N"):
        if not np.array_equal(datasets[case]["steps"], reference_grid):
            fail(f"{case}: TensorBoard iteration grid differs from V4-A")

    csv_path = args.data_dir / CSV_NAME
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "iteration",
            "v4a_raw",
            "v4a_ma25",
            "v4b_raw",
            "v4b_ma25",
            "v4n_raw",
            "v4n_ma25",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, step in enumerate(reference_grid):
            writer.writerow(
                {
                    "iteration": int(step),
                    "v4a_raw": datasets["V4-A"]["raw"][i],
                    "v4a_ma25": datasets["V4-A"]["smooth"][i],
                    "v4b_raw": datasets["V4-B"]["raw"][i],
                    "v4b_ma25": datasets["V4-B"]["smooth"][i],
                    "v4n_raw": datasets["V4-N"]["raw"][i],
                    "v4n_ma25": datasets["V4-N"]["smooth"][i],
                }
            )

    configure_style()
    fig, ax = plt.subplots(figsize=(11.8, 7.5))

    for case in ("V4-A", "V4-B", "V4-N"):
        data = datasets[case]
        mask = data["steps"] <= ZOOM_MAX_ITERATION

        ax.plot(
            data["steps"][mask],
            data["raw"][mask],
            linewidth=0.9,
            alpha=0.23,
            color=CASES[case]["color"],
            zorder=1,
        )

        ax.plot(
            data["steps"][mask],
            data["smooth"][mask],
            linewidth=2.8,
            color=CASES[case]["color"],
            label=case,
            zorder=3,
        )

    ax.axhline(
        REFERENCE_STEPS,
        linestyle="--",
        linewidth=1.8,
        color="#6BAED6",
        alpha=0.95,
        zorder=0,
    )
    ax.text(
        ZOOM_MAX_ITERATION - 8,
        REFERENCE_STEPS + 11,
        "980-step reference",
        ha="right",
        va="bottom",
        fontsize=13.0,
        color="#4F4F4F",
    )

    ax.set_title("Training Episode Length — v4 Campaign", pad=17, fontweight="normal")
    ax.set_xlabel("Training iteration", labelpad=11)
    ax.set_ylabel("Mean episode length [steps]", labelpad=10)

    ax.set_xlim(0, ZOOM_MAX_ITERATION)
    ax.set_ylim(0, 1030)
    ax.set_xticks(np.arange(0, ZOOM_MAX_ITERATION + 1, 100))
    ax.set_yticks(np.arange(0, 1001, 200))

    ax.grid(True, linestyle="--", zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.25)

    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#C8C8C8",
    )

    fig.text(
        0.5,
        0.030,
        "First 600 of 1500 iterations  |  training seed 42  |  25-iteration moving average",
        ha="center",
        va="bottom",
        fontsize=13.5,
        color="#555555",
    )

    fig.subplots_adjust(left=0.12, right=0.975, top=0.89, bottom=0.17)

    png_path = args.png_dir / PNG_NAME
    pdf_path = args.pdf_dir / PDF_NAME

    fig.savefig(png_path, dpi=180, bbox_inches=None)
    fig.savefig(pdf_path, format="pdf", bbox_inches=None)
    plt.close(fig)

    if not png_path.is_file() or png_path.stat().st_size <= 0:
        fail("PNG output missing or empty")
    if not pdf_path.is_file() or pdf_path.stat().st_size <= 0:
        fail("PDF output missing or empty")

    print("=" * 118)
    print("THESIS FIGURE F02a — GENERATION SUMMARY")
    print("=" * 118)
    print("Status                      : PASS")
    print("Figure                      : Training episode length — v4 campaign (visual revision v3)")
    print("Runs                        : V4-A / V4-B / V4-N canonical genuine-scratch seed42")
    print("TensorBoard tag             : Train/mean_episode_length")
    print("Event reader                : dependency-free TFRecord/Protobuf wire parser")
    print("Smoothing                   : 25-iteration moving average")
    print("Display                     : first 600 / 1500 iterations")
    print(f"980-step crossing V4-A      : {crossings['V4-A']}")
    print(f"980-step crossing V4-B      : {crossings['V4-B']}")
    print(f"980-step crossing V4-N      : {crossings['V4-N']}")
    print(f"Data freeze                 : {csv_path}")
    print(f"PNG                         : {png_path}")
    print(f"PDF vector                  : {pdf_path}")
    print("Scientific data/protocol    : UNCHANGED FROM v2")
    print("Visual revision              : full frame / larger text / visible raw traces / light-blue reference")
    print("Simulation / policy run     : NO")
    print("Repository/checkpoints      : NOT MODIFIED")
    print(PASS_MARKER)


if __name__ == "__main__":
    main()
