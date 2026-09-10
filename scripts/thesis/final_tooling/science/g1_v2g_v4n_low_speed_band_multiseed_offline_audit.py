from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

ROOT = Path(r"C:\humanoid_rl_lab\logs\thesis\router")
SEEDS = (100, 101, 102, 103, 104)

POINTS = (
    {
        "vx": 0.10,
        "tag": "vx010",
        "suffix": "_vx010_paired_competence_seed{seed}",
        "script_sha256": "F78FBB21A26794228A86E62F4878E3235CFF27CE8C8E4548924D9AC9264E5E09",
    },
    {
        "vx": 0.15,
        "tag": "vx015",
        "suffix": "_vx015_paired_competence_seed{seed}",
        "script_sha256": "92A4A88BFF5A0F71A7806CF18628AA70A13F556422C0107E6E63E60BC270DB75",
    },
    {
        "vx": 0.20,
        "tag": "vx020",
        "suffix": "_vx020_paired_competence_seed{seed}",
        "script_sha256": "5FBB74817BEA5CA8981B774C5C2A5C8F47E103DEB38B173CA6CDDA3339A3AF7E",
    },
)

PASS_MARKER = "V2G_V4N_LOW_SPEED_BAND_MULTISEED_OFFLINE_AUDIT_PASS"

TAIL_KEYS = (
    "tail3_mean_abs_tracking_error_mps",
    "tail3_mean_tilt_deg",
    "tail3_mean_abs_wz_radps",
    "tail3_double_support_pct",
)


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


def num(data: dict[str, str], key: str) -> float:
    if key not in data:
        fail(f"missing key {key}")
    try:
        value = float(data[key])
    except ValueError:
        fail(f"non-numeric {key}={data[key]!r}")
    if not math.isfinite(value):
        fail(f"non-finite {key}={value}")
    return value


def mean_sd(values: list[float]) -> tuple[float, float]:
    return (
        statistics.fmean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def load_point(seed: int, spec: dict[str, object]) -> tuple[Path, dict[str, str]]:
    suffix = str(spec["suffix"]).format(seed=seed)
    result_dir = latest_matching_dir(suffix)
    summary = result_dir / "summary.txt"
    if not summary.is_file():
        fail(f"missing summary: {summary}")

    data = parse_summary(summary)

    if data.get("seed") != str(seed):
        fail(f"seed mismatch in {summary}: {data.get('seed')}")

    expected_vx = float(spec["vx"])
    actual_vx = num(data, "command_vx")
    if abs(actual_vx - expected_vx) > 1e-12:
        fail(f"command mismatch seed{seed} {spec['tag']}: {actual_vx}")

    if data.get("v2g_termination") != "NO":
        fail(f"v2-G termination seed{seed} {spec['tag']}")
    if data.get("v4n_termination") != "NO":
        fail(f"v4-N termination seed{seed} {spec['tag']}")

    reset_delta = num(data, "reset_fingerprint_max_delta")
    if reset_delta > 1e-6:
        fail(f"paired reset mismatch seed{seed} {spec['tag']}: {reset_delta}")

    if data.get("router") != "ABSENT":
        fail(f"router unexpectedly present seed{seed} {spec['tag']}")
    if data.get("handoff") != "ABSENT":
        fail(f"handoff unexpectedly present seed{seed} {spec['tag']}")
    if data.get("threshold_acceptance") != "ABSENT":
        fail(f"threshold acceptance unexpectedly present seed{seed} {spec['tag']}")

    return result_dir, data


def main() -> None:
    if not ROOT.is_dir():
        fail(f"router log root missing: {ROOT}")

    records = []
    source_dirs = []

    for seed in SEEDS:
        for spec in POINTS:
            result_dir, data = load_point(seed, spec)
            source_dirs.append((seed, spec["tag"], result_dir))

            record = {
                "seed": seed,
                "vx": float(spec["vx"]),
                "tag": str(spec["tag"]),
                "result_dir": str(result_dir),
            }

            for policy in ("v2g", "v4n"):
                for key in TAIL_KEYS:
                    record[f"{policy}_{key}"] = num(data, f"{policy}_{key}")

            record["tracking_advantage_v4n_mps"] = (
                record["v2g_tail3_mean_abs_tracking_error_mps"]
                - record["v4n_tail3_mean_abs_tracking_error_mps"]
            )
            record["tilt_advantage_v2g_deg"] = (
                record["v4n_tail3_mean_tilt_deg"]
                - record["v2g_tail3_mean_tilt_deg"]
            )
            record["yawrate_advantage_v4n_radps"] = (
                record["v2g_tail3_mean_abs_wz_radps"]
                - record["v4n_tail3_mean_abs_wz_radps"]
            )
            record["double_support_advantage_v2g_pp"] = (
                record["v2g_tail3_double_support_pct"]
                - record["v4n_tail3_double_support_pct"]
            )
            records.append(record)

    out_dir = ROOT / "low_speed_band_multiseed_offline_audit_seeds100_104"
    out_dir.mkdir(parents=True, exist_ok=True)

    long_csv = out_dir / "low_speed_band_long.csv"
    agg_csv = out_dir / "low_speed_band_aggregate.csv"
    report_path = out_dir / "low_speed_band_report.txt"

    with long_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    aggregate = []
    for spec in POINTS:
        vx = float(spec["vx"])
        rows = [r for r in records if abs(r["vx"] - vx) < 1e-12]

        row = {"vx": vx, "tag": spec["tag"], "seed_count": len(rows)}
        for key in (
            "tracking_advantage_v4n_mps",
            "tilt_advantage_v2g_deg",
            "yawrate_advantage_v4n_radps",
            "double_support_advantage_v2g_pp",
        ):
            vals = [float(r[key]) for r in rows]
            mean, sd = mean_sd(vals)
            row[f"{key}_mean"] = mean
            row[f"{key}_sd"] = sd
            row[f"{key}_positive_seeds"] = sum(v > 0.0 for v in vals)

        for policy in ("v2g", "v4n"):
            for key in TAIL_KEYS:
                vals = [float(r[f"{policy}_{key}"]) for r in rows]
                mean, sd = mean_sd(vals)
                row[f"{policy}_{key}_mean"] = mean
                row[f"{policy}_{key}_sd"] = sd

        aggregate.append(row)

    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)

    # Per-seed ordering diagnostic:
    # Does v4-N tracking advantage increase from 0.10 -> 0.15 -> 0.20?
    monotonic_tracking_seeds = 0
    per_seed_order = {}
    for seed in SEEDS:
        sr = sorted(
            (r for r in records if r["seed"] == seed),
            key=lambda r: float(r["vx"]),
        )
        adv = [float(r["tracking_advantage_v4n_mps"]) for r in sr]
        monotonic = adv[0] < adv[1] < adv[2]
        monotonic_tracking_seeds += int(monotonic)
        per_seed_order[seed] = (adv, monotonic)

    # Descriptive boundary structure counts.
    # No threshold is accepted here; these are directional summaries only.
    low_rows = [r for r in records if abs(float(r["vx"]) - 0.10) < 1e-12]
    mid_rows = [r for r in records if abs(float(r["vx"]) - 0.15) < 1e-12]
    high_rows = [r for r in records if abs(float(r["vx"]) - 0.20) < 1e-12]

    low_v2g_more_static = sum(
        (float(r["tilt_advantage_v2g_deg"]) > 0.0)
        and (float(r["double_support_advantage_v2g_pp"]) > 0.0)
        for r in low_rows
    )
    mid_both_no_term = len(mid_rows)  # already validated no termination above
    high_v4n_tracking_better = sum(
        float(r["tracking_advantage_v4n_mps"]) > 0.0 for r in high_rows
    )
    high_v4n_yaw_better = sum(
        float(r["yawrate_advantage_v4n_radps"]) > 0.0 for r in high_rows
    )

    report = []
    report.append("=" * 136)
    report.append("v2-G / v4-N LOW-SPEED BAND MULTISEED OFFLINE AUDIT")
    report.append("=" * 136)
    report.append("seeds                    : 100,101,102,103,104")
    report.append("commands                 : vx=0.10, 0.15, 0.20 m/s")
    report.append("comparison               : paired, same seed, same reset fingerprint")
    report.append("startup conditioning     : zero root reset velocity")
    report.append("router / handoff         : ABSENT / ABSENT")
    report.append("holdout                  : UNTOUCHED")
    report.append("simulator execution      : ABSENT (offline summaries only)")
    report.append("")
    report.append("[INPUT VALIDATION]")
    for seed, tag, result_dir in source_dirs:
        report.append(f"seed {seed} {tag}: {result_dir.name} | paired reset PASS | termination NO/NO")

    report.append("")
    report.append("[AGGREGATE TAIL-3s POLICY DIFFERENCES]")
    report.append(
        "Positive tracking_advantage_v4n means v4-N has LOWER tracking error; "
        "positive tilt/double_support advantage means v2-G is more stand-like."
    )
    for row in aggregate:
        report.append(
            f"vx={row['vx']:.2f} | "
            f"tracking_adv_v4N={row['tracking_advantage_v4n_mps_mean']:.9f}"
            f"±{row['tracking_advantage_v4n_mps_sd']:.9f} m/s "
            f"({row['tracking_advantage_v4n_mps_positive_seeds']}/5) | "
            f"tilt_adv_v2G={row['tilt_advantage_v2g_deg_mean']:.9f}"
            f"±{row['tilt_advantage_v2g_deg_sd']:.9f} deg "
            f"({row['tilt_advantage_v2g_deg_positive_seeds']}/5) | "
            f"yaw_adv_v4N={row['yawrate_advantage_v4n_radps_mean']:.9f}"
            f"±{row['yawrate_advantage_v4n_radps_sd']:.9f} rad/s "
            f"({row['yawrate_advantage_v4n_radps_positive_seeds']}/5) | "
            f"DS_adv_v2G={row['double_support_advantage_v2g_pp_mean']:.3f}"
            f"±{row['double_support_advantage_v2g_pp_sd']:.3f} pp "
            f"({row['double_support_advantage_v2g_pp_positive_seeds']}/5)"
        )

    report.append("")
    report.append("[ORDERED STRUCTURE — DESCRIPTIVE, NO ACCEPTANCE THRESHOLD]")
    for seed in SEEDS:
        adv, monotonic = per_seed_order[seed]
        report.append(
            f"seed {seed}: v4-N tracking advantage "
            f"[0.10,0.15,0.20]=[{adv[0]:.9f},{adv[1]:.9f},{adv[2]:.9f}] "
            f"strictly_increasing={int(monotonic)}"
        )
    report.append(
        f"tracking advantage strictly increases 0.10->0.15->0.20 : "
        f"{monotonic_tracking_seeds}/5 seeds"
    )
    report.append(
        f"vx=0.10 v2-G lower tilt AND higher DS                   : "
        f"{low_v2g_more_static}/5 seeds"
    )
    report.append(
        f"vx=0.15 both policies complete without termination       : "
        f"{mid_both_no_term}/5 seeds"
    )
    report.append(
        f"vx=0.20 v4-N lower tracking error                        : "
        f"{high_v4n_tracking_better}/5 seeds"
    )
    report.append(
        f"vx=0.20 v4-N lower |wz|                                  : "
        f"{high_v4n_yaw_better}/5 seeds"
    )

    report.append("")
    report.append("[INTERPRETATION RULE]")
    report.append(
        "This audit may establish reproducible low-speed overlap structure, but it does not "
        "post-hoc optimize or automatically freeze ENTER/EXIT thresholds."
    )
    report.append(
        "If the same ordered structure is coherent across seeds 100-104, 0.10/0.20 may be "
        "promoted as development-calibrated engineering thresholds before untouched holdout validation."
    )
    report.append(PASS_MARKER)

    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report), flush=True)
    print(f"\nlong_csv                 : {long_csv}")
    print(f"aggregate_csv            : {agg_csv}")
    print(f"report                    : {report_path}")


if __name__ == "__main__":
    main()
