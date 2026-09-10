#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
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

EXPECTED_SEEDS = list(range(100, 110))
COMMAND_VX = 0.40
DURATION_S = 20.0

COLORS = {
    "V4-B": "#54A24B",
    "V4-N": "#F58518",  # frozen thesis convention: v4-N = orange
}

FILE_SHA256 = {
    "V4-B": {
        100: "7E74313387B5D4D0A9A26ED7EEF2B7EAE6D8F9E10E777F35A6B97E36ADF3347E",
        101: "F1DE259D0E89F37B5A86634BC9784706F1A01B201DDF84D9F428123E7AF1B5EF",
        102: "272A509306DDCBBF2C6B81EC42B12B58B3A3CD2917A1665C33FD4AA2098FEE87",
        103: "6A5A933FC6108F8901DE72C250DF62DE1FF0CC05910EABBDB003924E87A38BD9",
        104: "6AF602536D8B131DDF8A1204B6C426878D0EF1B75B135D17E575BC81F76DF9CB",
        105: "114888168F48CCB1FDE30623EFDAB1EDE5037AB0C0EFA9F5A0C6794018F2F451",
        106: "3B499ABE1178A91F72D4DAA2AA5F5F371DE166114187AF4DA07150EC9D279C2D",
        107: "115838311192AEECE0B99552E1EF0656C4AD38961FF6AD7C79CE8D342EDE5633",
        108: "4DF0A493CC622157091C47627B72318F665C5B77CA6C4ED65704ABA17551B076",
        109: "9D5A1379D45251E13B675E6F486D58D31AAC03F335384791A1468D5422485E77",
    },
    "V4-N": {
        100: "F6B7D901D6AB9D88DD5973F9B9A651D0E608ADAB100D18792ADDAB5F9E4C05C7",
        101: "947FC28048BE0B378F90956390FA3D38BBD676F2C6835C239DA86F9BB64F07E4",
        102: "FB08F03C1E488478E59922B9E8317CEDD3443C1A8A7E8356AE6E7F7D18484106",
        103: "D1B75A2FDE4F9440DA894DCCC0E4A78776CA01C3259EDA3DD75245DAEE161CC3",
        104: "4A33B9E2B8DCCD501183D77306AECB275DC050F190AD0C65BF0A10FE6105B8CD",
        105: "89B29A0528FE800F2D709817CB64E40479D1642D1477FEDAE6EF16B1F4016E41",
        106: "96893CBFDB0F400CFA478124257BFE4380608EB8CE4AF2E5EAABB8DEC3187C66",
        107: "97A14BAFCB4A520B86132D49DCA19B437462AFF766F198B29B73B62F0848FAD1",
        108: "2AC25510C89C892C7D3BD36D799D95B6F739BDA0C8C54479DAD06456EF63A96D",
        109: "020CD609C91E7CDAF1EB4FC31CC5854F06530A572C3FE959EA55B76786670474",
    },
}

PNG_NAME = "F02c_v4b_v4n_knee_kinematic_asymmetry_v3.png"
PDF_NAME = "F02c_v4b_v4n_knee_kinematic_asymmetry_v3.pdf"
DATA_NAME = "F02c_v4b_v4n_knee_kinematic_asymmetry_data_v3.csv"
PASS_MARKER = "G1_THESIS_F02C_V4B_V4N_KNEE_KINEMATIC_ASYMMETRY_V3_PASS"


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def source_path(case: str, seed: int) -> Path:
    slug = "v4b" if case == "V4-B" else "v4n"
    return (
        REPO
        / "logs"
        / "evaluations"
        / "g1"
        / f"{slug}_model1499_play_equivalent_seed{seed}_vx040_20s_direction_dev"
        / "gait_probe_analysis.json"
    )


def asymmetry_index(left: float, right: float) -> float:
    denominator = (abs(left) + abs(right)) / 2.0
    if denominator <= 0.0:
        fail("cannot compute normalized asymmetry with zero denominator")
    return abs(left - right) / denominator


def load_seed(case: str, seed: int) -> dict:
    path = source_path(case, seed)
    if not path.is_file():
        fail(f"{case} seed {seed}: missing gait probe: {path}")

    actual_sha = sha256_file(path)
    expected_sha = FILE_SHA256[case][seed]
    if actual_sha != expected_sha:
        fail(
            f"{case} seed {seed}: SHA mismatch: "
            f"expected {expected_sha}, got {actual_sha}"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))

    protocol = payload.get("protocol", {})
    command = protocol.get("command")
    if command != [COMMAND_VX, 0.0, 0.0]:
        fail(f"{case} seed {seed}: unexpected command {command}")
    if not math.isclose(float(protocol.get("duration_s")), DURATION_S, abs_tol=1e-12):
        fail(f"{case} seed {seed}: unexpected duration")

    knees = payload.get("knees_rad")
    if not isinstance(knees, dict):
        fail(f"{case} seed {seed}: knees_rad missing")

    required = (
        "left_stance_mean",
        "right_stance_mean",
        "left_swing_mean",
        "right_swing_mean",
    )
    for key in required:
        if key not in knees:
            fail(f"{case} seed {seed}: missing {key}")
        if not math.isfinite(float(knees[key])):
            fail(f"{case} seed {seed}: non-finite {key}")

    left_stance = float(knees["left_stance_mean"])
    right_stance = float(knees["right_stance_mean"])
    left_swing = float(knees["left_swing_mean"])
    right_swing = float(knees["right_swing_mean"])

    return {
        "policy": case,
        "seed": seed,
        "left_stance_mean_rad": left_stance,
        "right_stance_mean_rad": right_stance,
        "left_swing_mean_rad": left_swing,
        "right_swing_mean_rad": right_swing,
        "stance_asymmetry_fraction": asymmetry_index(left_stance, right_stance),
        "swing_asymmetry_fraction": asymmetry_index(left_swing, right_swing),
        "source_sha256": actual_sha,
    }


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
        description="Generate thesis F02c: V4-B vs V4-N left/right knee kinematic asymmetry."
    )
    parser.add_argument("--png-dir", type=Path, default=DEFAULT_PNG_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    rows = []
    for case in ("V4-B", "V4-N"):
        for seed in EXPECTED_SEEDS:
            rows.append(load_seed(case, seed))

    stats = {}
    for case in ("V4-B", "V4-N"):
        case_rows = [r for r in rows if r["policy"] == case]

        stance_pct = [100.0 * r["stance_asymmetry_fraction"] for r in case_rows]
        swing_pct = [100.0 * r["swing_asymmetry_fraction"] for r in case_rows]

        stats[case] = {
            "stance_mean_pct": statistics.fmean(stance_pct),
            "stance_sd_pct": statistics.stdev(stance_pct),
            "swing_mean_pct": statistics.fmean(swing_pct),
            "swing_sd_pct": statistics.stdev(swing_pct),
        }

    stance_reduction_pct = (
        100.0
        * (stats["V4-B"]["stance_mean_pct"] - stats["V4-N"]["stance_mean_pct"])
        / stats["V4-B"]["stance_mean_pct"]
    )
    swing_reduction_pct = (
        100.0
        * (stats["V4-B"]["swing_mean_pct"] - stats["V4-N"]["swing_mean_pct"])
        / stats["V4-B"]["swing_mean_pct"]
    )

    args.data_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = args.data_dir / DATA_NAME

    with freeze_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "policy",
            "seed",
            "left_stance_mean_rad",
            "right_stance_mean_rad",
            "left_swing_mean_rad",
            "right_swing_mean_rad",
            "stance_asymmetry_pct",
            "swing_asymmetry_pct",
            "source_sha256",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "policy": row["policy"],
                    "seed": row["seed"],
                    "left_stance_mean_rad": row["left_stance_mean_rad"],
                    "right_stance_mean_rad": row["right_stance_mean_rad"],
                    "left_swing_mean_rad": row["left_swing_mean_rad"],
                    "right_swing_mean_rad": row["right_swing_mean_rad"],
                    "stance_asymmetry_pct": 100.0 * row["stance_asymmetry_fraction"],
                    "swing_asymmetry_pct": 100.0 * row["swing_asymmetry_fraction"],
                    "source_sha256": row["source_sha256"],
                }
            )

    configure_style()

    fig, ax = plt.subplots(figsize=(10.3, 7.3))

    categories = ["Stance", "Swing"]
    x = np.arange(len(categories), dtype=float)
    width = 0.27

    means_b = np.array(
        [stats["V4-B"]["stance_mean_pct"], stats["V4-B"]["swing_mean_pct"]]
    )
    sds_b = np.array(
        [stats["V4-B"]["stance_sd_pct"], stats["V4-B"]["swing_sd_pct"]]
    )
    means_n = np.array(
        [stats["V4-N"]["stance_mean_pct"], stats["V4-N"]["swing_mean_pct"]]
    )
    sds_n = np.array(
        [stats["V4-N"]["stance_sd_pct"], stats["V4-N"]["swing_sd_pct"]]
    )

    bars_b = ax.bar(
        x - width / 2,
        means_b,
        width,
        yerr=sds_b,
        capsize=6,
        label="V4-B",
        color=COLORS["V4-B"],
        edgecolor="black",
        linewidth=1.1,
        error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": "black"},
        zorder=3,
    )

    bars_n = ax.bar(
        x + width / 2,
        means_n,
        width,
        yerr=sds_n,
        capsize=6,
        label="V4-N",
        color=COLORS["V4-N"],
        edgecolor="black",
        linewidth=1.1,
        error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": "black"},
        zorder=3,
    )

    # Place each numeric label just above the corresponding +1 SD cap.
    label_gap = 0.12

    for bar, value, sd in zip(bars_b, means_b, sds_b):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + sd + label_gap,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=14.5,
            color="#202020",
            zorder=6,
        )

    for bar, value, sd in zip(bars_n, means_n, sds_n):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + sd + label_gap,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=14.5,
            color="#202020",
            zorder=6,
        )


    ax.set_title(
        "Knee Kinematic Asymmetry — V4-B vs V4-N",
        pad=17,
        fontweight="normal",
    )
    ax.set_ylabel("Knee L/R asymmetry [%]", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)

    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.0, 6.35)
    ax.set_yticks(np.arange(0.0, 6.1, 1.0))

    ax.grid(True, axis="y", linestyle="--", zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.25)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.018, 0.985),
        ncol=1,
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#C8C8C8",
    )

    ax.text(
        0.985,
        0.965,
        "Lower is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13.5,
        color="#555555",
        zorder=6,
    )

    fig.text(
        0.5,
        0.022,
        "$v_x$ command = 0.40 m/s  |  seeds 100–109  |  20 s  |  mean ± 1 SD",
        ha="center",
        va="bottom",
        fontsize=13.5,
        color="#555555",
    )

    fig.subplots_adjust(left=0.13, right=0.97, top=0.87, bottom=0.125)

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

    print("=" * 118)
    print("THESIS FIGURE F02c — GENERATION SUMMARY")
    print("=" * 118)
    print("Status                              : PASS")
    print("Figure                              : Knee kinematic asymmetry (final visual v3)")
    print("Policies                            : V4-B / V4-N")
    print("Protocol                            : vx=0.40 m/s, 20 s, seeds 100-109")
    print("Asymmetry definition                : |L-R| / ((|L|+|R|)/2)")
    print(
        "V4-B stance asymmetry mean ± SD [%] : "
        f"{stats['V4-B']['stance_mean_pct']:.9f} ± "
        f"{stats['V4-B']['stance_sd_pct']:.9f}"
    )
    print(
        "V4-N stance asymmetry mean ± SD [%] : "
        f"{stats['V4-N']['stance_mean_pct']:.9f} ± "
        f"{stats['V4-N']['stance_sd_pct']:.9f}"
    )
    print(
        "V4-B swing asymmetry mean ± SD [%]  : "
        f"{stats['V4-B']['swing_mean_pct']:.9f} ± "
        f"{stats['V4-B']['swing_sd_pct']:.9f}"
    )
    print(
        "V4-N swing asymmetry mean ± SD [%]  : "
        f"{stats['V4-N']['swing_mean_pct']:.9f} ± "
        f"{stats['V4-N']['swing_sd_pct']:.9f}"
    )
    print(f"V4-N stance asymmetry reduction [%] : {stance_reduction_pct:.6f}")
    print(f"V4-N swing asymmetry reduction [%]  : {swing_reduction_pct:.6f}")
    print(
        "Interpretation guard                : knee kinematic L/R symmetry; "
        "not a universal whole-gait symmetry score"
    )
    print("Visual revision                     : restored Lower is better annotation at top-right")
    print(f"Data freeze                         : {freeze_path}")
    print(f"PNG                                 : {png_path}")
    print(f"PDF vector                          : {pdf_path}")
    print("Simulation / policy run             : NO")
    print("Repository/checkpoints              : NOT MODIFIED")
    print(PASS_MARKER)


if __name__ == "__main__":
    main()
