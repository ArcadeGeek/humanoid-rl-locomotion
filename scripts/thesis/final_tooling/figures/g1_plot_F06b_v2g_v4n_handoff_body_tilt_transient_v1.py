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
DEFAULT_SOURCE_ROOT = REPO / r"logs\thesis\handoff"

DEFAULT_ROOT = REPO / r"logs\thesis\figures_final\06_handoff_brake"
DEFAULT_PNG_DIR = DEFAULT_ROOT / "png"
DEFAULT_PDF_DIR = DEFAULT_ROOT / "pdf"
DEFAULT_DATA_DIR = DEFAULT_ROOT / "data"

EXPECTED_SEEDS = [100, 101, 102, 103, 104]
EXPECTED_DT = 0.02
EXPECTED_STAND_PRE_STEPS = 150
EXPECTED_LOCO_STEPS = 400
EXPECTED_BRAKE_STEPS = 18
EXPECTED_BRAKE_S = 0.36
EXPECTED_LOCO_VX = 0.40

EXPECTED_STAND_SHA = "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3"
EXPECTED_LOCO_SHA = "367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342"

EXPECTED_FILES = {
    100: {
        "brake": {
            "summary": "E5323E310A6F5A78A4911AAD645A89ED4F15430B77C6E481F515A7A81F8F974C",
            "trajectory": "86F2B85343686C032B7ABD09CFA56404829DE31277354761E2DEA9538BCE7977",
        },
        "immediate": {
            "summary": "21239B05A0E7D52C4170D8811351EF67A1A4C5B427CE2C6E1EC5103794EA1403",
            "trajectory": "97FBA7D6117EB0230F579AD6CB9FD6133DFF9BE1ABC0FA6B774495A3E55FF11B",
        },
    },
    101: {
        "brake": {
            "summary": "77713EFD5DCC51638F7E1B8A3880760BA32C1974497D014A15C0E8D651AC8884",
            "trajectory": "08C9AA3359377B7A7E227630CDC612C67131307FEAE1829A8ECD49E560599B7F",
        },
        "immediate": {
            "summary": "9F71347FFF33A0931E98E42291C62A0DD1402FB16117BD7DD69C78AEE33F980F",
            "trajectory": "B7353E81DC60C7CF8544E016A005BFD7743A717EE8FBDDF45F9EF31246580432",
        },
    },
    102: {
        "brake": {
            "summary": "8A03104FF6662A2A7653C3E18A760CBBE7532F360FA91D007BF9E48808FAC582",
            "trajectory": "A0AB51C14B49E68024373998D3F5D0E4E7857C6368FE96FA4F8B6BFC93557A3C",
        },
        "immediate": {
            "summary": "8DA63A5289B90A2452A8EE242D09BB2AA04EAA7372A94552EEE997801C832FB9",
            "trajectory": "B42DF878163CD67AC0838D4D79F8FD45C70F8E712D3FF25AC7229858226074C3",
        },
    },
    103: {
        "brake": {
            "summary": "6EDBE92B1E432BE994C3407AF58BF76ED46C46A9073E1095B644381A7DA7A84E",
            "trajectory": "AAC2669B80A9D74461AA442B4F66D66476311C73A626896529CCB51F748008E2",
        },
        "immediate": {
            "summary": "C6393A8A27C6D288E0EEC2AEABE8ADBBC832FBBC4923C77A2A905C6E969C45F2",
            "trajectory": "9622614FAD8C72E88BCD6AC5413D0F957376D3B9A4C8F16B88A05471F06102CA",
        },
    },
    104: {
        "brake": {
            "summary": "087CC621908A027F8B9949F23C825271BE99B9D468F4FE41AE4E29DB26B6EB23",
            "trajectory": "34A0D8052A8F4CE51DDBB5FB1D1C61736AEA5114A9EE569F0A5DAF381A23846E",
        },
        "immediate": {
            "summary": "46A19F3141D96DDCB857895DD7631C2927EB55D36266AC8A9313F535F5C0F003",
            "trajectory": "0D09FCA31B10DB564C5239ABC01E641F81BA11A483854B80907E333B31805BF8",
        },
    },
}

IMMEDIATE_COLOR = "#4C78A8"
BRAKE_COLOR = "#F58518"

WINDOW_PRE_S = 0.50
WINDOW_POST_S = 1.50
WINDOW_PRE_STEPS = 25
WINDOW_POST_STEPS = 75

EXPECTED_IMMEDIATE_FIRST05_MEAN_TILT_DEG = 2.1751241245269775
EXPECTED_BRAKE_FIRST05_MEAN_TILT_DEG = 1.5296643862724304
EXPECTED_IMMEDIATE_FIRST05_MAX_TILT_DEG = 2.973895263671875
EXPECTED_BRAKE_FIRST05_MAX_TILT_DEG = 2.4768323421478273
EXPECTED_MEAN_REDUCTION_PCT = 29.67461631160542
EXPECTED_PEAK_REDUCTION_PCT = 16.71420401370569

PNG_NAME = "F06b_v2g_v4n_handoff_body_tilt_transient_v1.png"
PDF_NAME = "F06b_v2g_v4n_handoff_body_tilt_transient_v1.pdf"
DATA_NAME = "F06b_v2g_v4n_handoff_body_tilt_transient_data_v1.csv"
PASS_MARKER = "G1_THESIS_F06B_V2G_V4N_HANDOFF_BODY_TILT_TRANSIENT_V1_PASS"


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


def read_trajectory(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])

    required = {
        "step",
        "time_s",
        "phase",
        "active_policy",
        "command_vx_mps",
        "root_tilt_deg",
        "done",
    }
    missing = required - set(fields)
    if missing:
        fail(f"{path}: missing trajectory columns {sorted(missing)}")

    return fields, rows


def discover_condition_dir(root: Path, seed: int, condition: str) -> Path:
    if condition == "brake":
        suffix = f"_v2g_v4n_brake_dwell_036_zero_root_vel_start_seed{seed}"
    elif condition == "immediate":
        suffix = f"_v2g_v4n_immediate_switch_zero_root_vel_start_seed{seed}"
    else:
        fail(f"unknown condition: {condition}")

    candidates = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.endswith(suffix)
    )
    if len(candidates) != 1:
        fail(
            f"seed {seed} {condition}: expected exactly one directory ending "
            f"{suffix!r}, found {len(candidates)}"
        )
    return candidates[0]


def verify_summary(summary: dict[str, str], seed: int, condition: str) -> None:
    required = {
        "seed",
        "stand_checkpoint_sha256",
        "loco_checkpoint_sha256",
        "step_dt",
        "stand_pre_s",
        "locomotion_s",
        "brake_s",
        "stand_post_s",
        "locomotion_vx",
        "termination",
        "stand_post_first_05_mean_tilt_deg",
        "stand_post_first_05_max_tilt_deg",
    }
    missing = required - set(summary)
    if missing:
        fail(f"seed {seed} {condition}: summary missing {sorted(missing)}")

    if int(summary["seed"]) != seed:
        fail(f"seed {seed} {condition}: summary seed mismatch")
    if summary["stand_checkpoint_sha256"].upper() != EXPECTED_STAND_SHA:
        fail(f"seed {seed} {condition}: stand checkpoint SHA mismatch")
    if summary["loco_checkpoint_sha256"].upper() != EXPECTED_LOCO_SHA:
        fail(f"seed {seed} {condition}: loco checkpoint SHA mismatch")
    if not math.isclose(float(summary["step_dt"]), EXPECTED_DT, abs_tol=1e-12):
        fail(f"seed {seed} {condition}: dt mismatch")
    if not math.isclose(float(summary["stand_pre_s"]), 3.0, abs_tol=1e-12):
        fail(f"seed {seed} {condition}: stand_pre mismatch")
    if not math.isclose(float(summary["locomotion_s"]), 8.0, abs_tol=1e-12):
        fail(f"seed {seed} {condition}: locomotion duration mismatch")
    if not math.isclose(float(summary["stand_post_s"]), 8.0, abs_tol=1e-12):
        fail(f"seed {seed} {condition}: stand_post duration mismatch")
    if not math.isclose(float(summary["locomotion_vx"]), EXPECTED_LOCO_VX, abs_tol=1e-12):
        fail(f"seed {seed} {condition}: locomotion vx mismatch")
    if summary["termination"] != "NO":
        fail(f"seed {seed} {condition}: unexpected termination={summary['termination']}")

    expected_brake = EXPECTED_BRAKE_S if condition == "brake" else 0.0
    if not math.isclose(float(summary["brake_s"]), expected_brake, abs_tol=1e-12):
        fail(f"seed {seed} {condition}: brake_s mismatch")


def find_first_phase(rows: list[dict[str, str]], phase: str) -> int:
    for idx, row in enumerate(rows):
        if row["phase"] == phase:
            return idx
    fail(f"phase {phase!r} not found")


def sample_sd(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.std(values, axis=axis, ddof=1)


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
            "legend.fontsize": 14.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis F06b: paired V4-N->V2-G handoff body-tilt transient, "
            "immediate switch versus 0.36 s V4-N zero-command braking."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--png-dir", type=Path, default=DEFAULT_PNG_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    if not args.source_root.is_dir():
        fail(f"missing handoff source root: {args.source_root}")

    trajectories: dict[str, list[np.ndarray]] = {"immediate": [], "brake": []}
    summary_first05_mean: dict[str, list[float]] = {"immediate": [], "brake": []}
    summary_first05_max: dict[str, list[float]] = {"immediate": [], "brake": []}

    relative_time = (
        np.arange(-WINDOW_PRE_STEPS, WINDOW_POST_STEPS + 1, dtype=float)
        * EXPECTED_DT
    )

    for seed in EXPECTED_SEEDS:
        loaded = {}

        for condition in ("brake", "immediate"):
            directory = discover_condition_dir(args.source_root, seed, condition)
            summary_path = directory / "summary.txt"
            trajectory_path = directory / "trajectory.csv"

            if not summary_path.is_file() or not trajectory_path.is_file():
                fail(f"seed {seed} {condition}: missing summary or trajectory")

            expected = EXPECTED_FILES[seed][condition]
            if sha256_file(summary_path) != expected["summary"]:
                fail(f"seed {seed} {condition}: summary SHA mismatch")
            if sha256_file(trajectory_path) != expected["trajectory"]:
                fail(f"seed {seed} {condition}: trajectory SHA mismatch")

            summary = parse_summary(summary_path)
            verify_summary(summary, seed, condition)
            fields, rows = read_trajectory(trajectory_path)

            loaded[condition] = (fields, rows, summary)

            summary_first05_mean[condition].append(
                float(summary["stand_post_first_05_mean_tilt_deg"])
            )
            summary_first05_max[condition].append(
                float(summary["stand_post_first_05_max_tilt_deg"])
            )

        b_fields, b_rows, _ = loaded["brake"]
        i_fields, i_rows, _ = loaded["immediate"]

        if b_fields != i_fields:
            fail(f"seed {seed}: trajectory schemas differ")
        if len(b_rows) != 968:
            fail(f"seed {seed}: brake trajectory expected 968 rows, got {len(b_rows)}")
        if len(i_rows) != 950:
            fail(f"seed {seed}: immediate trajectory expected 950 rows, got {len(i_rows)}")

        prefix_len = EXPECTED_STAND_PRE_STEPS + EXPECTED_LOCO_STEPS
        if b_rows[:prefix_len] != i_rows[:prefix_len]:
            fail(f"seed {seed}: paired pre-handoff trajectory prefix is not identical")

        i_stand = find_first_phase(i_rows, "STAND_POST")
        b_brake = find_first_phase(b_rows, "BRAKE_V4N")
        b_stand = find_first_phase(b_rows, "STAND_POST")

        if i_stand != 550:
            fail(f"seed {seed}: immediate STAND_POST starts at row {i_stand}, expected 550")
        if b_brake != 550:
            fail(f"seed {seed}: BRAKE_V4N starts at row {b_brake}, expected 550")
        if b_stand != 568:
            fail(f"seed {seed}: brake STAND_POST starts at row {b_stand}, expected 568")
        if b_stand - b_brake != EXPECTED_BRAKE_STEPS:
            fail(f"seed {seed}: brake dwell is not exactly 18 steps")

        if i_rows[i_stand]["active_policy"] != "v2-G":
            fail(f"seed {seed}: immediate switch does not activate v2-G")
        if b_rows[b_brake]["active_policy"] != "v4-N":
            fail(f"seed {seed}: brake phase does not remain on v4-N")
        if b_rows[b_stand]["active_policy"] != "v2-G":
            fail(f"seed {seed}: post-brake phase does not activate v2-G")
        if not math.isclose(float(b_rows[b_brake]["command_vx_mps"]), 0.0, abs_tol=1e-12):
            fail(f"seed {seed}: brake phase command is not zero")
        if not math.isclose(float(i_rows[i_stand]["command_vx_mps"]), 0.0, abs_tol=1e-12):
            fail(f"seed {seed}: immediate STAND command is not zero")

        for condition, stand_idx, rows in (
            ("immediate", i_stand, i_rows),
            ("brake", b_stand, b_rows),
        ):
            lo = stand_idx - WINDOW_PRE_STEPS
            hi = stand_idx + WINDOW_POST_STEPS
            window_rows = rows[lo : hi + 1]
            if len(window_rows) != len(relative_time):
                fail(f"seed {seed} {condition}: wrong aligned window length")

            values = np.asarray(
                [float(r["root_tilt_deg"]) for r in window_rows],
                dtype=float,
            )
            if not np.all(np.isfinite(values)):
                fail(f"seed {seed} {condition}: non-finite tilt data")
            trajectories[condition].append(values)

            stand_first25 = np.asarray(
                [float(r["root_tilt_deg"]) for r in rows[stand_idx : stand_idx + 25]],
                dtype=float,
            )
            summary_mean = float(
                loaded[condition][2]["stand_post_first_05_mean_tilt_deg"]
            )
            summary_max = float(
                loaded[condition][2]["stand_post_first_05_max_tilt_deg"]
            )
            if not math.isclose(float(np.mean(stand_first25)), summary_mean, abs_tol=1e-12):
                fail(f"seed {seed} {condition}: first-0.5s mean tilt summary mismatch")
            if not math.isclose(float(np.max(stand_first25)), summary_max, abs_tol=1e-12):
                fail(f"seed {seed} {condition}: first-0.5s max tilt summary mismatch")

    immediate = np.vstack(trajectories["immediate"])
    brake = np.vstack(trajectories["brake"])

    immediate_mean = np.mean(immediate, axis=0)
    immediate_sd = sample_sd(immediate, axis=0)
    brake_mean = np.mean(brake, axis=0)
    brake_sd = sample_sd(brake, axis=0)

    i_first05_mean = float(np.mean(summary_first05_mean["immediate"]))
    b_first05_mean = float(np.mean(summary_first05_mean["brake"]))
    i_first05_max = float(np.mean(summary_first05_max["immediate"]))
    b_first05_max = float(np.mean(summary_first05_max["brake"]))

    mean_reduction_pct = 100.0 * (1.0 - b_first05_mean / i_first05_mean)
    peak_reduction_pct = 100.0 * (1.0 - b_first05_max / i_first05_max)

    if not math.isclose(
        i_first05_mean, EXPECTED_IMMEDIATE_FIRST05_MEAN_TILT_DEG, abs_tol=1e-12
    ):
        fail(f"canonical immediate first-0.5 mean tilt changed: {i_first05_mean}")
    if not math.isclose(
        b_first05_mean, EXPECTED_BRAKE_FIRST05_MEAN_TILT_DEG, abs_tol=1e-12
    ):
        fail(f"canonical brake first-0.5 mean tilt changed: {b_first05_mean}")
    if not math.isclose(
        i_first05_max, EXPECTED_IMMEDIATE_FIRST05_MAX_TILT_DEG, abs_tol=1e-12
    ):
        fail(f"canonical immediate first-0.5 max tilt changed: {i_first05_max}")
    if not math.isclose(
        b_first05_max, EXPECTED_BRAKE_FIRST05_MAX_TILT_DEG, abs_tol=1e-12
    ):
        fail(f"canonical brake first-0.5 max tilt changed: {b_first05_max}")
    if not math.isclose(
        mean_reduction_pct, EXPECTED_MEAN_REDUCTION_PCT, abs_tol=1e-9
    ):
        fail(f"canonical mean tilt reduction changed: {mean_reduction_pct}")
    if not math.isclose(
        peak_reduction_pct, EXPECTED_PEAK_REDUCTION_PCT, abs_tol=1e-9
    ):
        fail(f"canonical peak tilt reduction changed: {peak_reduction_pct}")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = args.data_dir / DATA_NAME
    with freeze_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_relative_to_v2g_engagement_s",
                "immediate_mean_tilt_deg",
                "immediate_sd_tilt_deg",
                "brake036_mean_tilt_deg",
                "brake036_sd_tilt_deg",
            ],
        )
        writer.writeheader()
        for idx in range(len(relative_time)):
            writer.writerow(
                {
                    "time_relative_to_v2g_engagement_s": relative_time[idx],
                    "immediate_mean_tilt_deg": immediate_mean[idx],
                    "immediate_sd_tilt_deg": immediate_sd[idx],
                    "brake036_mean_tilt_deg": brake_mean[idx],
                    "brake036_sd_tilt_deg": brake_sd[idx],
                }
            )

    configure_style()
    fig, ax = plt.subplots(figsize=(11.8, 7.5))

    ax.axvspan(
        -EXPECTED_BRAKE_S,
        0.0,
        color=BRAKE_COLOR,
        alpha=0.08,
        zorder=0,
    )

    ax.fill_between(
        relative_time,
        immediate_mean - immediate_sd,
        immediate_mean + immediate_sd,
        color=IMMEDIATE_COLOR,
        alpha=0.14,
        linewidth=0.0,
        zorder=1,
    )
    ax.fill_between(
        relative_time,
        brake_mean - brake_sd,
        brake_mean + brake_sd,
        color=BRAKE_COLOR,
        alpha=0.14,
        linewidth=0.0,
        zorder=1,
    )

    ax.plot(
        relative_time,
        immediate_mean,
        color=IMMEDIATE_COLOR,
        linewidth=2.7,
        label="Immediate switch",
        zorder=3,
    )
    ax.plot(
        relative_time,
        brake_mean,
        color=BRAKE_COLOR,
        linewidth=2.7,
        label="0.36 s V4-N brake",
        zorder=3,
    )

    ax.axvline(
        0.0,
        color="#4F4F4F",
        linestyle="--",
        linewidth=1.4,
        zorder=2,
    )

    ax.text(
        -0.34,
        0.875,
        "Brake condition:\nV4-N zero command",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=12.5,
        color="#202020",
        zorder=6,
    )
    ax.text(
        0.035,
        0.96,
        "v2-G engagement",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=12.5,
        color="#303030",
        zorder=6,
    )

    ax.text(
        0.985,
        0.055,
        "Lower is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13.5,
        color="#555555",
        zorder=7,
    )
    ax.text(
        0.980,
        0.955,
        (
            f"First 0.5 s after v2-G engagement\n"
            f"Mean body tilt: {mean_reduction_pct:.0f}% lower\n"
            f"Peak body tilt: {peak_reduction_pct:.0f}% lower"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13.2,
        color="#202020",
        zorder=7,
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": "white",
            "edgecolor": "#C8C8C8",
            "linewidth": 1.0,
            "alpha": 0.96,
        },
    )

    ax.set_title(
        "V4-N → V2-G Handoff — Body-Tilt Transient",
        pad=17,
        fontweight="normal",
    )
    ax.set_xlabel("Time relative to v2-G engagement [s]", labelpad=11)
    ax.set_ylabel("Body tilt [deg]", labelpad=10)

    ax.set_xlim(-WINDOW_PRE_S, WINDOW_POST_S)
    ax.set_ylim(0.0, 3.4)
    ax.set_xticks(np.arange(-0.5, 1.51, 0.25))
    ax.set_yticks(np.arange(0.0, 3.401, 0.5))

    ax.grid(True, linestyle="--", zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.25)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.165),
        ncol=2,
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#C8C8C8",
        borderpad=0.45,
        columnspacing=2.0,
        handlelength=2.8,
    )

    fig.text(
        0.5,
        0.014,
        "seeds 100–104  |  paired pre-handoff trajectories  |  mean ± 1 SD  |  50 Hz",
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
    print("THESIS FIGURE F06b — GENERATION SUMMARY")
    print("=" * 124)
    print("Status                              : PASS")
    print("Figure                              : V4-N -> V2-G handoff body-tilt transient")
    print("Seeds                               : 100-104")
    print("Paired pre-handoff trajectory       : EXACT / first 550 rows identical for every seed")
    print("Immediate condition                 : v2-G engaged immediately at zero command")
    print("Brake condition                     : v4-N zero command for 0.36 s / 18 steps, then v2-G")
    print(f"Immediate first0.5 mean tilt [deg]  : {i_first05_mean:.12f}")
    print(f"Brake first0.5 mean tilt [deg]      : {b_first05_mean:.12f}")
    print(f"Immediate first0.5 peak tilt [deg]  : {i_first05_max:.12f}")
    print(f"Brake first0.5 peak tilt [deg]      : {b_first05_max:.12f}")
    print(f"Mean-tilt reduction [%]             : {mean_reduction_pct:.9f}")
    print(f"Peak-tilt reduction [%]             : {peak_reduction_pct:.9f}")
    print("Mechanism interpretation            : improved post-switch posture after pre-switch state conditioning")
    print("Termination                         : NONE in all 10 trials")
    print(f"Data freeze                         : {freeze_path}")
    print(f"PNG                                 : {png_path}")
    print(f"PDF vector                          : {pdf_path}")
    print("Simulation                          : NO (offline use of frozen trajectories)")
    print("Repository/checkpoints              : NOT MODIFIED")
    print(PASS_MARKER)


if __name__ == "__main__":
    main()
