from __future__ import annotations

import ast
import csv
import hashlib
import math
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(r"C:\humanoid_rl_lab")

CANONICAL_PROBE = (
    REPO
    / "scripts"
    / "thesis"
    / "v4"
    / "g1_v4b_contact_gait_probe.py"
)

EXPECTED_PROBE_SHA256 = (
    "40948E95BA457FCC1AFD2708619054FE734CCEFF515130A997AFFF3958A84848"
)

TASK = "Humanoid-RL-Lab-G1-Flat-V4N-Symmetry-Augmentation-v0"

RUN_DIR = Path(
    r"C:\humanoid_rl_lab\logs\rsl_rl\humanoid_rl_lab_g1_v4n_symmetry_augmentation_scratch"
    r"\2026-09-04_13-58-13_v4n_genuine_scratch_seed42_official1500"
)

CHECKPOINT = RUN_DIR / "model_1499.pt"

EXPECTED_CHECKPOINT_SHA256 = (
    "367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342"
)

SEEDS = list(range(100, 110))

BASE_OUT = (
    REPO
    / "logs"
    / "evaluations"
    / "g1"
    / "v4n_multiseed_direction_dev_100_109_vx040_20s"
)

SUMMARY_CSV = BASE_OUT / "multiseed_direction_summary.csv"

OLD_OUTPUT_LABEL = (
    "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
)

OLD_PASS = "V4B_CONTACT_GAIT_PROBE_CANONICAL_PASS"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def wrap_pi(x: float) -> float:
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def avg(xs):
    return sum(xs) / len(xs)


def sample_std(xs):
    if len(xs) < 2:
        return 0.0
    m = avg(xs)
    return math.sqrt(
        sum((x - m) ** 2 for x in xs)
        / (len(xs) - 1)
    )


def find_simple_assignment(tree: ast.Module, name: str) -> ast.Assign:
    matches = []

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue

        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            matches.append(node)

    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one top-level assignment for {name}, "
            f"found {len(matches)}"
        )

    return matches[0]


class ProbeTransformer(ast.NodeTransformer):
    def __init__(
        self,
        seed: int,
        output_label: str,
        pass_marker: str,
    ) -> None:
        self.seed = seed
        self.output_label = output_label
        self.pass_marker = pass_marker
        self.output_replacements = 0
        self.pass_replacements = 0
        self.title_replacements = 0

    def visit_Constant(self, node: ast.Constant):
        if not isinstance(node.value, str):
            return node

        if node.value == OLD_OUTPUT_LABEL:
            self.output_replacements += 1
            return ast.copy_location(
                ast.Constant(value=self.output_label),
                node,
            )

        if node.value == OLD_PASS:
            self.pass_replacements += 1
            return ast.copy_location(
                ast.Constant(value=self.pass_marker),
                node,
            )

        if node.value == "V4-B MODEL_1499 — CONTACT-AWARE GAIT RESULTS":
            self.title_replacements += 1
            return ast.copy_location(
                ast.Constant(
                    value=(
                        f"V4-N MODEL_1499 — CONTACT-AWARE GAIT RESULTS "
                        f"— EVAL SEED {self.seed}"
                    )
                ),
                node,
            )

        return node


def transform_probe(
    source: str,
    seed: int,
    output_label: str,
    pass_marker: str,
) -> str:
    tree = ast.parse(source)

    task_node = find_simple_assignment(tree, "TASK")
    run_node = find_simple_assignment(tree, "RUN_DIR")
    checkpoint_node = find_simple_assignment(tree, "CHECKPOINT")
    seed_node = find_simple_assignment(tree, "SEED")

    task_node.value = ast.Constant(value=TASK)

    run_node.value = ast.Call(
        func=ast.Name(id="Path", ctx=ast.Load()),
        args=[ast.Constant(value=str(RUN_DIR))],
        keywords=[],
    )

    checkpoint_node.value = ast.BinOp(
        left=ast.Name(id="RUN_DIR", ctx=ast.Load()),
        op=ast.Div(),
        right=ast.Constant(value="model_1499.pt"),
    )

    seed_node.value = ast.Constant(value=seed)

    transformer = ProbeTransformer(
        seed=seed,
        output_label=output_label,
        pass_marker=pass_marker,
    )
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    if transformer.output_replacements != 1:
        raise RuntimeError(
            f"seed {seed}: expected exactly one output-label replacement, "
            f"found {transformer.output_replacements}"
        )

    if transformer.pass_replacements != 1:
        raise RuntimeError(
            f"seed {seed}: expected exactly one PASS-marker replacement, "
            f"found {transformer.pass_replacements}"
        )

    transformed = ast.unparse(tree) + "\n"

    compile(
        transformed,
        f"<v4n-eval-seed-{seed}>",
        "exec",
    )

    check_tree = ast.parse(transformed)

    if ast.literal_eval(
        find_simple_assignment(check_tree, "TASK").value
    ) != TASK:
        raise RuntimeError(
            f"seed {seed}: TASK replacement validation failed"
        )

    if ast.literal_eval(
        find_simple_assignment(check_tree, "SEED").value
    ) != seed:
        raise RuntimeError(
            f"seed {seed}: SEED replacement validation failed"
        )

    if OLD_OUTPUT_LABEL in transformed:
        raise RuntimeError(
            f"seed {seed}: stale canonical output label survived"
        )

    if OLD_PASS in transformed:
        raise RuntimeError(
            f"seed {seed}: stale canonical PASS marker survived"
        )

    if transformed.count(output_label) != 1:
        raise RuntimeError(
            f"seed {seed}: output label occurrence count != 1"
        )

    if transformed.count(pass_marker) != 1:
        raise RuntimeError(
            f"seed {seed}: PASS marker occurrence count != 1"
        )

    return transformed


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(
            f"samples.csv missing: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) < 2:
        raise RuntimeError(
            f"insufficient samples in {path}"
        )

    required = {
        "cmd_vx_mps",
        "cmd_vy_mps",
        "cmd_wz_radps",
        "base_vx_mps",
        "base_vy_mps",
        "base_wz_radps",
        "heading_rad",
        "world_x_m",
        "world_y_m",
    }

    missing = required.difference(rows[0].keys())

    if missing:
        raise RuntimeError(
            f"CSV missing columns: {sorted(missing)}"
        )

    return rows


def analyse_seed(
    seed: int,
    samples: Path,
) -> dict[str, float | int | str]:
    rows = read_rows(samples)

    def col(name: str):
        return [float(r[name]) for r in rows]

    cmd_vx = col("cmd_vx_mps")
    cmd_vy = col("cmd_vy_mps")
    cmd_wz = col("cmd_wz_radps")

    if max(abs(v - 0.4) for v in cmd_vx) > 1.0e-9:
        raise RuntimeError(
            f"seed {seed}: cmd_vx not fixed at 0.4"
        )

    if max(abs(v) for v in cmd_vy) > 1.0e-9:
        raise RuntimeError(
            f"seed {seed}: cmd_vy not fixed at 0"
        )

    if max(abs(v) for v in cmd_wz) > 1.0e-9:
        raise RuntimeError(
            f"seed {seed}: cmd_wz not fixed at 0"
        )

    base_vx = col("base_vx_mps")
    base_vy = col("base_vy_mps")
    base_wz = col("base_wz_radps")
    heading = col("heading_rad")
    world_x = col("world_x_m")
    world_y = col("world_y_m")

    h0 = heading[0]

    heading_errors = [
        wrap_pi(h - h0)
        for h in heading
    ]

    abs_heading_errors = [
        abs(e)
        for e in heading_errors
    ]

    dx = world_x[-1] - world_x[0]
    dy = world_y[-1] - world_y[0]

    c = math.cos(h0)
    s = math.sin(h0)

    along = c * dx + s * dy
    cross = -s * dx + c * dy

    return {
        "seed": seed,
        "samples": len(rows),
        "mean_base_vx_mps": avg(base_vx),
        "mean_base_vy_mps": avg(base_vy),
        "mean_base_wz_radps": avg(base_wz),
        "initial_heading_deg": math.degrees(h0),
        "final_heading_error_deg": math.degrees(
            heading_errors[-1]
        ),
        "mean_abs_heading_error_deg": math.degrees(
            avg(abs_heading_errors)
        ),
        "max_abs_heading_error_deg": math.degrees(
            max(abs_heading_errors)
        ),
        "global_dx_m": dx,
        "global_dy_m": dy,
        "along_track_initial_heading_m": along,
        "cross_track_initial_heading_m": cross,
        "output_dir": str(samples.parent),
    }


def main() -> int:
    if not CANONICAL_PROBE.is_file():
        raise SystemExit(
            f"ABORT: canonical v4-B probe missing: "
            f"{CANONICAL_PROBE}"
        )

    actual_probe_sha = sha256(CANONICAL_PROBE)

    if actual_probe_sha != EXPECTED_PROBE_SHA256:
        raise SystemExit(
            "ABORT: canonical v4-B probe SHA mismatch.\n"
            f"expected={EXPECTED_PROBE_SHA256}\n"
            f"actual  ={actual_probe_sha}"
        )

    if not CHECKPOINT.is_file():
        raise SystemExit(
            f"ABORT: v4-N checkpoint missing: {CHECKPOINT}"
        )

    actual_checkpoint_sha = sha256(CHECKPOINT)

    if actual_checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit(
            "ABORT: v4-N checkpoint SHA mismatch.\n"
            f"expected={EXPECTED_CHECKPOINT_SHA256}\n"
            f"actual  ={actual_checkpoint_sha}"
        )

    if BASE_OUT.exists():
        raise SystemExit(
            f"ABORT: summary output directory already exists: "
            f"{BASE_OUT}\n"
            "Refusing to overwrite/reuse ambiguous evidence."
        )

    source = CANONICAL_PROBE.read_text(
        encoding="utf-8"
    )

    # Preflight every transformation before creating outputs.
    for seed in SEEDS:
        output_label = (
            f"v4n_model1499_play_equivalent_seed{seed}"
            f"_vx040_20s_direction_dev"
        )
        pass_marker = (
            f"V4N_CONTACT_GAIT_PROBE_SEED_{seed}_PASS"
        )
        transform_probe(
            source,
            seed,
            output_label,
            pass_marker,
        )

    print("=" * 126)
    print(
        "V4-N MULTI-SEED DIRECTION AUDIT — DEVELOPMENT SEEDS 100..109"
    )
    print("=" * 126)
    print("Canonical probe       :", CANONICAL_PROBE)
    print("Canonical probe SHA   :", actual_probe_sha)
    print("v4-N checkpoint       :", CHECKPOINT)
    print("Checkpoint SHA        :", actual_checkpoint_sha)
    print("Command               : [0.40, 0.00, 0.00]")
    print("Play-equivalent       : unchanged")
    print("Duration              : 20 s (+ canonical warmup)")
    print("Seeds                 :", SEEDS)
    print(
        "Classification        : DEVELOPMENT/TUNING DIAGNOSTIC — NOT HOLDOUT"
    )
    print("=" * 126)

    BASE_OUT.mkdir(
        parents=True,
        exist_ok=False,
    )

    summary_rows = []

    with tempfile.TemporaryDirectory(
        prefix="v4n_multiseed_"
    ) as td:
        tmpdir = Path(td)

        for seed in SEEDS:
            output_label = (
                f"v4n_model1499_play_equivalent_seed{seed}"
                f"_vx040_20s_direction_dev"
            )

            pass_marker = (
                f"V4N_CONTACT_GAIT_PROBE_SEED_{seed}_PASS"
            )

            expected_out = (
                REPO
                / "logs"
                / "evaluations"
                / "g1"
                / output_label
            )

            if expected_out.exists():
                raise SystemExit(
                    f"ABORT: output directory already exists "
                    f"for seed {seed}: {expected_out}\n"
                    "Refusing to overwrite/reuse ambiguous evidence."
                )

            transformed = transform_probe(
                source,
                seed,
                output_label,
                pass_marker,
            )

            temp_probe = (
                tmpdir
                / f"g1_v4n_contact_gait_probe_seed{seed}.py"
            )

            temp_probe.write_text(
                transformed,
                encoding="utf-8",
                newline="\n",
            )

            print()
            print(
                f"[SEED {seed}] START",
                flush=True,
            )

            proc = subprocess.run(
                [sys.executable, str(temp_probe)],
                cwd=REPO,
                text=True,
                check=False,
            )

            if proc.returncode != 0:
                raise SystemExit(
                    f"ABORT: canonical-derived v4-N probe "
                    f"failed for seed {seed}, "
                    f"code {proc.returncode}."
                )

            samples = expected_out / "samples.csv"

            row = analyse_seed(
                seed,
                samples,
            )

            summary_rows.append(row)

            print(
                f"[SEED {seed}] "
                f"vx={row['mean_base_vx_mps']:+.4f} "
                f"vy={row['mean_base_vy_mps']:+.4f} "
                f"wz={row['mean_base_wz_radps']:+.4f} "
                f"cross={row['cross_track_initial_heading_m']:+.4f} m "
                f"mean|herr|={row['mean_abs_heading_error_deg']:.3f} deg "
                f"PASS",
                flush=True,
            )

    with SUMMARY_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    vx_means = [
        float(r["mean_base_vx_mps"])
        for r in summary_rows
    ]
    vy_means = [
        float(r["mean_base_vy_mps"])
        for r in summary_rows
    ]
    wz_means = [
        float(r["mean_base_wz_radps"])
        for r in summary_rows
    ]
    crosses = [
        float(r["cross_track_initial_heading_m"])
        for r in summary_rows
    ]
    heading_mae = [
        float(r["mean_abs_heading_error_deg"])
        for r in summary_rows
    ]

    neg_vy_count = sum(v < 0.0 for v in vy_means)
    pos_vy_count = sum(v > 0.0 for v in vy_means)

    print()
    print("=" * 126)
    print("V4-N MULTI-SEED DIRECTION SUMMARY")
    print("=" * 126)
    print(
        f"seeds                                  : "
        f"{SEEDS[0]}..{SEEDS[-1]} "
        f"({len(SEEDS)} runs)"
    )
    print(
        f"mean base vx                           : "
        f"{avg(vx_means):+.6f} m/s  "
        f"(std {sample_std(vx_means):.6f})"
    )
    print(
        f"mean base vy                           : "
        f"{avg(vy_means):+.6f} m/s  "
        f"(std {sample_std(vy_means):.6f})"
    )
    print(
        f"base vy sign count                     : "
        f"negative={neg_vy_count}, "
        f"positive={pos_vy_count}, "
        f"zero={len(vy_means)-neg_vy_count-pos_vy_count}"
    )
    print(
        f"mean base wz                           : "
        f"{avg(wz_means):+.6f} rad/s "
        f"(std {sample_std(wz_means):.6f})"
    )
    print(
        f"mean initial-heading cross-track       : "
        f"{avg(crosses):+.4f} m    "
        f"(std {sample_std(crosses):.4f})"
    )
    print(
        f"mean absolute cross-track              : "
        f"{avg([abs(v) for v in crosses]):.4f} m    "
        f"(std {sample_std([abs(v) for v in crosses]):.4f})"
    )
    print(
        f"mean per-run mean |heading error|      : "
        f"{avg(heading_mae):.3f} deg  "
        f"(std {sample_std(heading_mae):.3f})"
    )
    print(
        f"summary CSV                            : "
        f"{SUMMARY_CSV}"
    )
    print()
    print("INTERPRETATION GUARD")
    print(
        "  Seeds 100..109 are DEVELOPMENT/TUNING evidence."
    )
    print(
        "  They are already non-holdout and are used here "
        "only for matched v4-B/v4-N comparison."
    )
    print()
    print(
        "V4N_MULTISEED_DIRECTION_AUDIT_PASS"
    )
    print("=" * 126)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
