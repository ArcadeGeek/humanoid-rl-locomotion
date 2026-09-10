#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(r"C:\humanoid_rl_lab")
DEFAULT_SOURCE_DIR = (
    REPO
    / r"logs\thesis\router\2026-09-05_11-48-15_v2g_v4n_router_runtime_smoke_seed104"
)

DEFAULT_ROOT = REPO / r"logs\thesis\figures_final\07_integrated_router"
DEFAULT_PNG_DIR = DEFAULT_ROOT / "png"
DEFAULT_PDF_DIR = DEFAULT_ROOT / "pdf"
DEFAULT_DATA_DIR = DEFAULT_ROOT / "data"

EXPECTED_SUMMARY_SHA256 = "30CC75B3CF54F93B5506ACFE527EDA6792B49E98E9E56399AB2A8CB5BC217F2B"
EXPECTED_TRAJECTORY_SHA256 = "44DCD4AFFAFA49D292CFF677BD238C5EBB9C200CF409EAA43FF45F98BC9A4231"

EXPECTED_SEED = 104
EXPECTED_DT = 0.02
EXPECTED_ROWS = 973
EXPECTED_EXIT = 0.10
EXPECTED_ENTER = 0.20
EXPECTED_BRAKE_STEPS = 18
EXPECTED_BRAKE_S = 0.36

EXPECTED_STAND_SHA = "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3"
EXPECTED_LOCO_SHA = "367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342"

V2G_COLOR = "#2F5D7E"
V4N_COLOR = "#F58518"
COMMAND_COLOR = "#6BAED6"
MEASURED_COLOR = "#303030"
RAW_COLOR = "#8F8F8F"

MA_WINDOW_STEPS = 15  # 0.30 s at 50 Hz, display-only centered moving average.

PNG_NAME = "F07a_integrated_router_sequence_v4.png"
PDF_NAME = "F07a_integrated_router_sequence_v4.pdf"
DATA_NAME = "F07a_integrated_router_sequence_data_v4.csv"
PASS_MARKER = "G1_THESIS_F07A_INTEGRATED_ROUTER_SEQUENCE_V4_PASS"


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def parse_summary(path: Path) -> dict[str, str]:
    result = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        fail("trajectory CSV is empty")
    return rows


def centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0 or window % 2 == 0:
        fail("moving-average window must be positive and odd")
    half = window // 2
    out = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out[i] = float(np.mean(values[lo:hi]))
    return out


def segments(values: list[str]) -> list[tuple[int, int, str]]:
    result = []
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            result.append((start, i - 1, values[start]))
            start = i
    return result


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
            "legend.fontsize": 14.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thesis F07a: integrated v2-G/v4-N router sequence."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--png-dir", type=Path, default=DEFAULT_PNG_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    summary_path = args.source_dir / "summary.txt"
    trajectory_path = args.source_dir / "trajectory.csv"

    if not summary_path.is_file():
        fail(f"missing summary: {summary_path}")
    if not trajectory_path.is_file():
        fail(f"missing trajectory: {trajectory_path}")

    if sha256_file(summary_path) != EXPECTED_SUMMARY_SHA256:
        fail("summary SHA mismatch")
    if sha256_file(trajectory_path) != EXPECTED_TRAJECTORY_SHA256:
        fail("trajectory SHA mismatch")

    summary = parse_summary(summary_path)
    rows = read_rows(trajectory_path)

    required_summary = {
        "seed",
        "stand_checkpoint_sha256",
        "loco_checkpoint_sha256",
        "step_dt",
        "enter_loco_speed",
        "exit_loco_speed",
        "brake_s",
        "brake_steps",
        "transition_sequence",
        "canceled_brake_lengths",
        "completed_brake_lengths",
        "termination",
        "scientific_role",
        "blending",
    }
    missing = required_summary - set(summary)
    if missing:
        fail(f"summary missing keys: {sorted(missing)}")

    if int(summary["seed"]) != EXPECTED_SEED:
        fail("seed mismatch")
    if summary["stand_checkpoint_sha256"].upper() != EXPECTED_STAND_SHA:
        fail("v2-G checkpoint SHA mismatch")
    if summary["loco_checkpoint_sha256"].upper() != EXPECTED_LOCO_SHA:
        fail("v4-N checkpoint SHA mismatch")
    if not math.isclose(float(summary["step_dt"]), EXPECTED_DT, abs_tol=1e-12):
        fail("step_dt mismatch")
    if not math.isclose(float(summary["enter_loco_speed"]), EXPECTED_ENTER, abs_tol=1e-12):
        fail("ENTER threshold mismatch")
    if not math.isclose(float(summary["exit_loco_speed"]), EXPECTED_EXIT, abs_tol=1e-12):
        fail("EXIT threshold mismatch")
    if not math.isclose(float(summary["brake_s"]), EXPECTED_BRAKE_S, abs_tol=1e-12):
        fail("brake duration mismatch")
    if int(summary["brake_steps"]) != EXPECTED_BRAKE_STEPS:
        fail("brake step-count mismatch")
    if summary["termination"] != "NO":
        fail("router integration run terminated unexpectedly")
    if summary["scientific_role"] != "FUNCTIONAL_INTEGRATION_ONLY":
        fail("unexpected scientific role")
    if summary["blending"] != "ABSENT":
        fail("unexpected blending status")

    if summary["transition_sequence"] != (
        "[('STAND', 'LOCO'), ('LOCO', 'BRAKE'), ('BRAKE', 'LOCO'), "
        "('LOCO', 'BRAKE'), ('BRAKE', 'STAND')]"
    ):
        fail("transition sequence mismatch")
    if summary["canceled_brake_lengths"] != "[5]":
        fail("expected one canceled BRAKE of 5 steps")
    if summary["completed_brake_lengths"] != "[18]":
        fail("expected one completed BRAKE of 18 steps")

    if len(rows) != EXPECTED_ROWS:
        fail(f"trajectory row count mismatch: {len(rows)}")

    required_cols = {
        "step",
        "time_s",
        "request_phase",
        "requested_vx_mps",
        "router_state",
        "active_policy",
        "policy_command_vx_mps",
        "brake_index",
        "root_vxy_mps",
        "root_tilt_deg",
        "double_support",
        "done",
    }
    missing_cols = required_cols - set(rows[0])
    if missing_cols:
        fail(f"trajectory missing columns: {sorted(missing_cols)}")

    time_s = np.asarray([float(r["time_s"]) for r in rows], dtype=float)
    requested = np.asarray([float(r["requested_vx_mps"]) for r in rows], dtype=float)
    policy_cmd = np.asarray([float(r["policy_command_vx_mps"]) for r in rows], dtype=float)
    root_vxy = np.asarray([float(r["root_vxy_mps"]) for r in rows], dtype=float)
    done = np.asarray([int(r["done"]) for r in rows], dtype=int)
    router_state = [r["router_state"] for r in rows]
    active_policy = [r["active_policy"] for r in rows]
    request_phase = [r["request_phase"] for r in rows]

    if not np.all(np.isfinite(time_s)):
        fail("non-finite time values")
    if not np.all(np.isfinite(requested)):
        fail("non-finite requested command")
    if not np.all(np.isfinite(root_vxy)):
        fail("non-finite planar speed")
    if not np.allclose(policy_cmd, requested, rtol=0.0, atol=1e-12):
        fail("policy command differs from requested command in this router smoke")
    if np.any(done != 0):
        fail("done flag asserted in trajectory")

    expected_request_segments = [
        (0, 149, "REQ_STAND_PRE", 0.00),
        (150, 549, "REQ_LOCO_MAIN", 0.40),
        (550, 599, "REQ_HYSTERESIS_BAND", 0.15),
        (600, 604, "REQ_BRAKE_CANCEL_WINDOW", 0.00),
        (605, 704, "REQ_LOCO_RESUME", 0.40),
        (705, 972, "REQ_FINAL_STOP", 0.00),
    ]
    for start, end, phase, vx in expected_request_segments:
        if request_phase[start : end + 1] != [phase] * (end - start + 1):
            fail(f"request phase mismatch for {phase}")
        if not np.allclose(requested[start : end + 1], vx, rtol=0.0, atol=1e-12):
            fail(f"requested vx mismatch for {phase}")

    expected_router_segments = [
        (0, 149, "STAND", "v2-G"),
        (150, 599, "LOCO", "v4-N"),
        (600, 604, "BRAKE", "v4-N"),
        (605, 704, "LOCO", "v4-N"),
        (705, 722, "BRAKE", "v4-N"),
        (723, 972, "STAND", "v2-G"),
    ]
    for start, end, state, policy in expected_router_segments:
        if router_state[start : end + 1] != [state] * (end - start + 1):
            fail(f"router state mismatch for {state} segment {start}-{end}")
        if active_policy[start : end + 1] != [policy] * (end - start + 1):
            fail(f"active policy mismatch for {state} segment {start}-{end}")

    # Hysteresis behavior guard: 0.15 m/s lies between EXIT and ENTER and LOCO is retained.
    if not (EXPECTED_EXIT < 0.15 < EXPECTED_ENTER):
        fail("0.15 m/s is not inside hysteresis band")
    if set(router_state[550:600]) != {"LOCO"}:
        fail("LOCO not retained through hysteresis-band request")

    # BRAKE cancellation/completion exact guards.
    if router_state[600:605] != ["BRAKE"] * 5:
        fail("canceled BRAKE is not exactly 5 steps")
    if router_state[605] != "LOCO":
        fail("BRAKE cancellation does not return to LOCO")
    if router_state[705:723] != ["BRAKE"] * 18:
        fail("final BRAKE is not exactly 18 steps")
    if router_state[723] != "STAND":
        fail("completed BRAKE does not transition to STAND")

    root_vxy_ma = centered_moving_average(root_vxy, MA_WINDOW_STEPS)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = args.data_dir / DATA_NAME
    with freeze_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_s",
                "requested_vx_mps",
                "root_vxy_mps_raw",
                "root_vxy_mps_ma030",
                "router_state",
                "active_policy",
                "request_phase",
            ],
        )
        writer.writeheader()
        for i in range(len(rows)):
            writer.writerow(
                {
                    "time_s": time_s[i],
                    "requested_vx_mps": requested[i],
                    "root_vxy_mps_raw": root_vxy[i],
                    "root_vxy_mps_ma030": root_vxy_ma[i],
                    "router_state": router_state[i],
                    "active_policy": active_policy[i],
                    "request_phase": request_phase[i],
                }
            )

    configure_style()
    fig, ax = plt.subplots(figsize=(11.8, 7.5))

    # Router-state shading. V2-G STAND stays blue; all v4-N-controlled
    # intervals remain orange, with BRAKE slightly stronger.
    state_spans = [
        (0.02, 3.00, "STAND", V2G_COLOR, 0.085),
        (3.02, 12.00, "LOCO", V4N_COLOR, 0.085),
        (12.02, 12.10, "BRAKE", V4N_COLOR, 0.18),
        (12.12, 14.10, "LOCO", V4N_COLOR, 0.085),
        (14.12, 14.46, "BRAKE", V4N_COLOR, 0.18),
        (14.48, 19.46, "STAND", V2G_COLOR, 0.085),
    ]
    for x0, x1, state, color, alpha in state_spans:
        ax.axvspan(x0, x1, color=color, alpha=alpha, zorder=0)

    # Raw physical response + display-only smoothing.
    ax.plot(
        time_s,
        root_vxy,
        color=RAW_COLOR,
        linewidth=1.05,
        alpha=0.42,
        label="Planar speed (raw)",
        zorder=1,
    )
    ax.plot(
        time_s,
        root_vxy_ma,
        color=MEASURED_COLOR,
        linewidth=2.6,
        label="Planar speed (0.30 s MA)",
        zorder=3,
    )
    ax.step(
        time_s,
        requested,
        where="post",
        color=COMMAND_COLOR,
        linestyle="--",
        linewidth=2.2,
        label="Requested $v_x$",
        zorder=4,
    )

    # State labels are kept short and placed where the long states leave room.
    ax.text(
        1.50, 0.462, "STAND · v2-G",
        ha="center", va="center",
        fontsize=12.5, color=V2G_COLOR,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#C8C8C8", alpha=0.92),
        zorder=7,
    )
    ax.text(
        7.00, 0.462, "LOCO · v4-N",
        ha="center", va="center",
        fontsize=12.5, color=V4N_COLOR,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#C8C8C8", alpha=0.92),
        zorder=7,
    )
    ax.text(
        16.85, 0.462, "STAND · v2-G",
        ha="center", va="center",
        fontsize=12.5, color=V2G_COLOR,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#C8C8C8", alpha=0.92),
        zorder=7,
    )

    ax.text(
        14.29, 0.425, "BRAKE",
        ha="center", va="center",
        rotation=90,
        fontsize=10.8, fontweight="normal",
        color="#A9570A",
        zorder=6,
    )

    # Functional-router events. These annotations describe the already-frozen
    # integration sequence; they are not threshold-calibration evidence.
    ax.text(
        8.35, 0.185,
        "0.15 m/s inside hysteresis band\nLOCO retained",
        ha="center", va="center",
        fontsize=12.1, color="#202020",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                  edgecolor="#C8C8C8", alpha=0.96),
        zorder=7,
    )
    ax.text(
        12.36, 0.075,
        "BRAKE canceled\nafter 5 steps",
        ha="center", va="center",
        fontsize=12.0, color="#202020",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                  edgecolor="#C8C8C8", alpha=0.96),
        zorder=7,
    )
    ax.annotate(
        "18-step BRAKE\n→ STAND",
        xy=(14.29, 0.365),
        xytext=(16.10, 0.235),
        ha="center", va="center",
        fontsize=12.0, color="#202020",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                  edgecolor="#C8C8C8", alpha=0.96),
        arrowprops=dict(
            arrowstyle="-",
            color="#606060",
            linewidth=1.15,
            shrinkA=4,
            shrinkB=3,
        ),
        zorder=7,
    )

    ax.set_title(
        "Integrated Router Sequence — v2-G STAND / v4-N LOCOMOTION",
        pad=17,
        fontweight="normal",
    )
    ax.set_xlabel("Time [s]", labelpad=11)
    ax.set_ylabel("Forward command / planar speed [m/s]", labelpad=10)

    ax.set_xlim(0.0, 19.5)
    ax.set_ylim(0.0, 0.49)
    ax.set_xticks(np.arange(0.0, 20.0, 2.5))
    ax.set_yticks(np.arange(0.0, 0.451, 0.05))
    ax.grid(True, linestyle="--", zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.25)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.165),
        ncol=3,
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#C8C8C8",
        borderpad=0.45,
        columnspacing=1.7,
        handlelength=2.7,
    )

    fig.text(
        0.5,
        0.014,
        (
            "seed 104  |  development integration run  |  "
            "EXIT = 0.10 m/s  |  ENTER = 0.20 m/s  |  BRAKE = 0.36 s"
        ),
        ha="center",
        va="bottom",
        fontsize=13.5,
        color="#555555",
    )

    fig.subplots_adjust(left=0.12, right=0.975, top=0.87, bottom=0.245)

    args.png_dir.mkdir(parents=True, exist_ok=True)
    args.pdf_dir.mkdir(parents=True, exist_ok=True)

    png_path = args.png_dir / PNG_NAME
    pdf_path = args.pdf_dir / PDF_NAME

    fig.savefig(png_path, dpi=180, bbox_inches=None)
    fig.savefig(pdf_path, format="pdf", bbox_inches=None)
    plt.close(fig)

    if not png_path.is_file() or png_path.stat().st_size <= 0:
        fail("PNG output missing or empty")
    if not pdf_path.is_file() or pdf_path.stat().st_size <= 0:
        fail("PDF output missing or empty")

    print("=" * 124)
    print("THESIS FIGURE F07a — GENERATION SUMMARY")
    print("=" * 124)
    print("Status                              : PASS")
    print("Figure                              : integrated router sequence (final layout candidate v4)")
    print("Seed                                : 104 / development integration")
    print("Sequence                            : STAND -> LOCO -> BRAKE(cancel) -> LOCO -> BRAKE -> STAND")
    print("Hysteresis-band request             : 0.15 m/s / LOCO retained")
    print("Canceled BRAKE                      : 5 steps")
    print("Completed BRAKE                     : 18 steps / 0.36 s")
    print("Final state                         : STAND / v2-G")
    print("Termination                         : NO")
    print("Blending                            : ABSENT")
    print("Threshold role                      : integration visualization only; NOT calibration evidence")
    print("Display smoothing                   : centered 0.30 s moving average; raw signal also shown")
    print("Visual revision                     : annotation-only v4; hysteresis further left; BRAKE restored high; longer final-brake leader")
    print(f"Data freeze                         : {freeze_path}")
    print(f"PNG                                 : {png_path}")
    print(f"PDF vector                          : {pdf_path}")
    print("Simulation                          : NO (offline use of frozen trajectory)")
    print("Repository/checkpoints              : NOT MODIFIED")
    print(PASS_MARKER)


if __name__ == "__main__":
    main()
