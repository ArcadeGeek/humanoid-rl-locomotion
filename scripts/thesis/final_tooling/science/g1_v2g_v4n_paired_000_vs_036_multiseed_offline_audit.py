from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

ROOT = Path(r"C:\humanoid_rl_lab\logs\thesis\handoff")
SEEDS = (100, 101, 102, 103, 104)

DWELL_SHA = "F34793DA13231216B01BB7DA32CD6224274C2514D99BE01ACA891883C0A682A3"
IMMEDIATE_SHA = "5161CECAE1F6247F32F6F9521ABCC523C7D7ED0502D3C7C560C0D6117726D9FB"

DWELL_DIR_SUFFIX = "_v2g_v4n_brake_dwell_036_zero_root_vel_start_seed{seed}"
IMMEDIATE_DIR_SUFFIX = "_v2g_v4n_immediate_switch_zero_root_vel_start_seed{seed}"

PASS_MARKER = "V2G_V4N_PAIRED_000_VS_036_MULTISEED_OFFLINE_AUDIT_PASS"

# Metrics selected before reading the paired results.
# direction: "lower" means smaller is favorable for dwell; "higher" means larger is favorable.
METRICS = (
    ("stand_post_first_05_mean_vxy_mps", "lower"),
    ("stand_post_first_05_max_vxy_mps", "lower"),
    ("stand_post_first_05_mean_tilt_deg", "lower"),
    ("stand_post_first_05_max_tilt_deg", "lower"),
    ("stand_post_first_05_mean_abs_wz_radps", "lower"),
    ("stand_post_first_05_max_abs_wz_radps", "lower"),
    ("stand_post_first_05_double_support_pct", "higher"),

    ("stand_post_first_10_mean_vxy_mps", "lower"),
    ("stand_post_first_10_max_vxy_mps", "lower"),
    ("stand_post_first_10_mean_tilt_deg", "lower"),
    ("stand_post_first_10_max_tilt_deg", "lower"),
    ("stand_post_first_10_mean_abs_wz_radps", "lower"),
    ("stand_post_first_10_max_abs_wz_radps", "lower"),
    ("stand_post_first_10_double_support_pct", "higher"),

    ("stand_post_first_20_mean_vxy_mps", "lower"),
    ("stand_post_first_20_max_vxy_mps", "lower"),
    ("stand_post_first_20_mean_tilt_deg", "lower"),
    ("stand_post_first_20_max_tilt_deg", "lower"),
    ("stand_post_first_20_mean_abs_wz_radps", "lower"),
    ("stand_post_first_20_max_abs_wz_radps", "lower"),
    ("stand_post_first_20_double_support_pct", "higher"),

    ("stand_post_xy_drift_m", "lower"),
    ("stand_post_mean_vxy_mps", "lower"),
    ("stand_post_max_vxy_mps", "lower"),
    ("stand_post_mean_tilt_deg", "lower"),
    ("stand_post_max_tilt_deg", "lower"),
    ("stand_post_mean_abs_wz_radps", "lower"),
    ("stand_post_double_support_pct", "higher"),

    ("stand_post_final2s_mean_vxy_mps", "lower"),
    ("stand_post_final2s_max_vxy_mps", "lower"),
    ("stand_post_final2s_mean_tilt_deg", "lower"),
    ("stand_post_final2s_max_tilt_deg", "lower"),
    ("stand_post_final2s_mean_abs_wz_radps", "lower"),
    ("stand_post_final2s_double_support_pct", "higher"),
)

ACTION_JUMP_KEYS = {
    "dwell_rms": "brake_to_stand_same_obs_action_rms_delta",
    "dwell_max": "brake_to_stand_same_obs_action_max_abs_delta",
    "immediate_rms": "loco_to_stand_same_obs_action_rms_delta",
    "immediate_max": "loco_to_stand_same_obs_action_max_abs_delta",
}


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def parse_summary(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def latest_matching_dir(suffix: str) -> Path:
    matches = sorted(
        (p for p in ROOT.iterdir() if p.is_dir() and p.name.endswith(suffix)),
        key=lambda p: p.name,
    )
    if not matches:
        fail(f"missing result directory matching *{suffix}")
    return matches[-1]


def load_condition(seed: int, kind: str) -> tuple[Path, dict[str, str]]:
    if kind == "dwell":
        suffix = DWELL_DIR_SUFFIX.format(seed=seed)
        expected_sha = DWELL_SHA
        expected_brake = 0.36
    elif kind == "immediate":
        suffix = IMMEDIATE_DIR_SUFFIX.format(seed=seed)
        expected_sha = IMMEDIATE_SHA
        expected_brake = 0.0
    else:
        fail(f"unknown condition: {kind}")

    result_dir = latest_matching_dir(suffix)
    summary_path = result_dir / "summary.txt"
    if not summary_path.is_file():
        fail(f"missing summary.txt: {summary_path}")

    data = parse_summary(summary_path)

    if data.get("seed") != str(seed):
        fail(f"{kind} seed mismatch in {summary_path}: {data.get('seed')}")
    if data.get("script_sha256", "").upper() != expected_sha:
        fail(
            f"{kind} script SHA mismatch seed{seed}: "
            f"{data.get('script_sha256')} != {expected_sha}"
        )
    if data.get("termination") != "NO":
        fail(f"{kind} seed{seed} did not complete with termination=NO")

    try:
        brake_s = float(data["brake_s"])
    except Exception as exc:
        fail(f"{kind} seed{seed}: invalid/missing brake_s: {exc}")
    if abs(brake_s - expected_brake) > 1e-9:
        fail(
            f"{kind} seed{seed}: brake_s={brake_s} expected {expected_brake}"
        )

    startup = data.get("startup_root_velocity_test", "")
    for axis in ("'x': (0.0, 0.0)", "'y': (0.0, 0.0)", "'z': (0.0, 0.0)",
                 "'roll': (0.0, 0.0)", "'pitch': (0.0, 0.0)", "'yaw': (0.0, 0.0)"):
        if axis not in startup:
            fail(f"{kind} seed{seed}: zero-root-velocity startup guardrail missing: {axis}")

    return result_dir, data


def f(data: dict[str, str], key: str) -> float:
    if key not in data:
        fail(f"missing metric: {key}")
    try:
        value = float(data[key])
    except ValueError:
        fail(f"non-numeric metric {key}={data[key]!r}")
    if not math.isfinite(value):
        fail(f"non-finite metric {key}={value}")
    return value


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        fail("empty values")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) >= 2 else 0.0
    return mean, sd


def pct_change(immediate: float, dwell: float, direction: str) -> float:
    # Positive means favorable for dwell.
    if abs(immediate) < 1e-12:
        return float("nan")
    if direction == "lower":
        return 100.0 * (immediate - dwell) / abs(immediate)
    if direction == "higher":
        return 100.0 * (dwell - immediate) / abs(immediate)
    fail(f"bad direction: {direction}")


def favorable(immediate: float, dwell: float, direction: str, tol: float = 1e-12) -> bool:
    if direction == "lower":
        return dwell < immediate - tol
    if direction == "higher":
        return dwell > immediate + tol
    fail(f"bad direction: {direction}")


def main() -> None:
    if not ROOT.is_dir():
        fail(f"handoff log root missing: {ROOT}")

    paired: dict[int, dict[str, object]] = {}
    for seed in SEEDS:
        dwell_dir, dwell = load_condition(seed, "dwell")
        immediate_dir, immediate = load_condition(seed, "immediate")
        paired[seed] = {
            "dwell_dir": dwell_dir,
            "dwell": dwell,
            "immediate_dir": immediate_dir,
            "immediate": immediate,
        }

    out_dir = ROOT / "paired_000_vs_036_zero_root_vel_start_seeds100_104"
    out_dir.mkdir(parents=True, exist_ok=True)

    long_csv = out_dir / "paired_metrics_long.csv"
    aggregate_csv = out_dir / "paired_metrics_aggregate.csv"
    report_path = out_dir / "paired_report.txt"

    long_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []

    for metric, direction in METRICS:
        immediate_values: list[float] = []
        dwell_values: list[float] = []
        favorable_flags: list[bool] = []
        pct_values: list[float] = []

        for seed in SEEDS:
            entry = paired[seed]
            imm = f(entry["immediate"], metric)  # type: ignore[arg-type]
            dw = f(entry["dwell"], metric)       # type: ignore[arg-type]
            delta = dw - imm
            pct = pct_change(imm, dw, direction)
            fav = favorable(imm, dw, direction)

            immediate_values.append(imm)
            dwell_values.append(dw)
            favorable_flags.append(fav)
            if math.isfinite(pct):
                pct_values.append(pct)

            long_rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "direction": direction,
                    "immediate_0p00": imm,
                    "dwell_0p36": dw,
                    "dwell_minus_immediate": delta,
                    "favorable_pct_change": pct,
                    "dwell_favorable": int(fav),
                }
            )

        imm_mean, imm_sd = mean_sd(immediate_values)
        dw_mean, dw_sd = mean_sd(dwell_values)
        paired_delta = [d - i for i, d in zip(immediate_values, dwell_values)]
        delta_mean, delta_sd = mean_sd(paired_delta)
        mean_pct = statistics.fmean(pct_values) if pct_values else float("nan")

        aggregate_rows.append(
            {
                "metric": metric,
                "direction": direction,
                "immediate_mean": imm_mean,
                "immediate_sd": imm_sd,
                "dwell_mean": dw_mean,
                "dwell_sd": dw_sd,
                "paired_delta_mean": delta_mean,
                "paired_delta_sd": delta_sd,
                "mean_favorable_pct_change": mean_pct,
                "favorable_seed_count": sum(favorable_flags),
                "seed_count": len(SEEDS),
            }
        )

    with long_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(long_rows[0].keys()))
        writer.writeheader()
        writer.writerows(long_rows)

    with aggregate_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    # Separate action-jump analysis because "smaller" is not the mechanism hypothesis.
    jump_rows = []
    for seed in SEEDS:
        entry = paired[seed]
        imm = entry["immediate"]  # type: ignore[assignment]
        dw = entry["dwell"]       # type: ignore[assignment]
        jump_rows.append(
            {
                "seed": seed,
                "immediate_rms": f(imm, ACTION_JUMP_KEYS["immediate_rms"]),
                "dwell_rms": f(dw, ACTION_JUMP_KEYS["dwell_rms"]),
                "immediate_max": f(imm, ACTION_JUMP_KEYS["immediate_max"]),
                "dwell_max": f(dw, ACTION_JUMP_KEYS["dwell_max"]),
            }
        )

    imm_rms = [r["immediate_rms"] for r in jump_rows]
    dw_rms = [r["dwell_rms"] for r in jump_rows]
    imm_max = [r["immediate_max"] for r in jump_rows]
    dw_max = [r["dwell_max"] for r in jump_rows]

    imm_rms_mean, imm_rms_sd = mean_sd(imm_rms)
    dw_rms_mean, dw_rms_sd = mean_sd(dw_rms)
    imm_max_mean, imm_max_sd = mean_sd(imm_max)
    dw_max_mean, dw_max_sd = mean_sd(dw_max)

    report: list[str] = []
    report.append("=" * 132)
    report.append("v2-G / v4-N PAIRED OFFLINE COMPARISON: IMMEDIATE 0.00 s vs BRAKING DWELL 0.36 s")
    report.append("=" * 132)
    report.append("seeds                    : 100,101,102,103,104")
    report.append("startup conditioning     : zero root reset velocity, engineering-only")
    report.append("policies                  : frozen v2-G + frozen v4-N")
    report.append("comparison               : paired, same-seed")
    report.append("simulator execution      : ABSENT (offline summary audit)")
    report.append("holdout use              : ABSENT")
    report.append("")
    report.append("[INPUT VALIDATION]")
    for seed in SEEDS:
        entry = paired[seed]
        report.append(
            f"seed {seed}: "
            f"immediate={Path(entry['immediate_dir']).name} | "
            f"dwell={Path(entry['dwell_dir']).name} | "
            "termination=NO/NO"
        )
    report.append("")
    report.append("[SELECTED PAIRED AGGREGATES]")
    selected = {
        "stand_post_first_05_mean_vxy_mps",
        "stand_post_first_05_max_vxy_mps",
        "stand_post_first_05_max_tilt_deg",
        "stand_post_first_05_max_abs_wz_radps",
        "stand_post_first_05_double_support_pct",
        "stand_post_first_20_mean_vxy_mps",
        "stand_post_xy_drift_m",
        "stand_post_mean_vxy_mps",
        "stand_post_final2s_mean_vxy_mps",
        "stand_post_final2s_mean_tilt_deg",
        "stand_post_final2s_double_support_pct",
    }
    for row in aggregate_rows:
        if row["metric"] not in selected:
            continue
        report.append(
            f"{row['metric']:<45} "
            f"immediate={row['immediate_mean']:.9f}±{row['immediate_sd']:.9f}  "
            f"dwell={row['dwell_mean']:.9f}±{row['dwell_sd']:.9f}  "
            f"fav_seeds={row['favorable_seed_count']}/{row['seed_count']}  "
            f"mean_fav_change={row['mean_favorable_pct_change']:.3f}%"
        )

    report.append("")
    report.append("[SAME-OBS ACTION DISCONTINUITY — MECHANISM DIAGNOSTIC]")
    report.append(
        f"RMS immediate             : {imm_rms_mean:.9f} ± {imm_rms_sd:.9f}"
    )
    report.append(
        f"RMS dwell                 : {dw_rms_mean:.9f} ± {dw_rms_sd:.9f}"
    )
    report.append(
        f"MAX immediate             : {imm_max_mean:.9f} ± {imm_max_sd:.9f}"
    )
    report.append(
        f"MAX dwell                 : {dw_max_mean:.9f} ± {dw_max_sd:.9f}"
    )
    report.append(
        "interpretation rule       : large transient improvement without commensurate action-jump reduction "
        "supports pre-switch state conditioning, not action smoothing, as the dwell mechanism."
    )

    core_metrics = (
        "stand_post_first_05_mean_vxy_mps",
        "stand_post_first_05_max_vxy_mps",
        "stand_post_first_05_max_tilt_deg",
        "stand_post_first_05_max_abs_wz_radps",
        "stand_post_first_05_double_support_pct",
        "stand_post_xy_drift_m",
    )
    agg_by_name = {r["metric"]: r for r in aggregate_rows}
    core_all_5 = sum(
        int(agg_by_name[m]["favorable_seed_count"] == len(SEEDS))
        for m in core_metrics
    )

    report.append("")
    report.append("[DESCRIPTIVE CONSISTENCY]")
    report.append(
        f"core metrics favorable on all 5 seeds : {core_all_5}/{len(core_metrics)}"
    )
    report.append(
        "NOTE                     : no post-hoc acceptance threshold is introduced here; "
        "this report gives paired effect sizes and directional consistency only."
    )
    report.append(PASS_MARKER)

    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report), flush=True)
    print(f"\nlong_csv                 : {long_csv}", flush=True)
    print(f"aggregate_csv            : {aggregate_csv}", flush=True)
    print(f"report                    : {report_path}", flush=True)


if __name__ == "__main__":
    main()
