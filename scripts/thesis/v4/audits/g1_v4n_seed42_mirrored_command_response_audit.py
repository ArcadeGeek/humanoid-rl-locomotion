from __future__ import annotations

import ast
import csv
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(r"C:\humanoid_rl_lab")

SOURCE_PROBE = (
    REPO
    / "scripts"
    / "thesis"
    / "v4"
    / "g1_v4b_contact_gait_probe.py"
)

EXPECTED_SOURCE_SHA = (
    "40948E95BA457FCC1AFD2708619054FE734CCEFF515130A997AFFF3958A84848"
)

TASK = "Humanoid-RL-Lab-G1-Flat-V4N-Symmetry-Augmentation-v0"

RUN_DIR = Path(
    r"C:\humanoid_rl_lab\logs\rsl_rl\humanoid_rl_lab_g1_v4n_symmetry_augmentation_scratch"
    r"\2026-09-04_13-58-13_v4n_genuine_scratch_seed42_official1500"
)

MODEL = RUN_DIR / "model_1499.pt"

EXPECTED_MODEL_SHA = (
    "367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342"
)

OLD_OUT = "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
OLD_PASS = "V4B_CONTACT_GAIT_PROBE_CANONICAL_PASS"

CASES = (
    ("baseline", 0.40, 0.00, 0.00),
    ("vy_p020", 0.40, +0.20, 0.00),
    ("vy_m020", 0.40, -0.20, 0.00),
    ("wz_p020", 0.40, 0.00, +0.20),
    ("wz_m020", 0.40, 0.00, -0.20),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def output_name(case_name: str) -> str:
    return (
        "v4n_seed42_mirrored_command_response_"
        + case_name
        + "_play_equivalent_evalseed42_20s_v1"
    )


def read_means(csv_path: Path) -> dict[str, float]:
    required = (
        "cmd_vx_mps",
        "cmd_vy_mps",
        "cmd_wz_radps",
        "base_vx_mps",
        "base_vy_mps",
        "base_wz_radps",
    )

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)

        if reader.fieldnames is None:
            raise RuntimeError(
                f"CSV has no header: {csv_path}"
            )

        missing = [
            name
            for name in required
            if name not in reader.fieldnames
        ]

        if missing:
            raise RuntimeError(
                f"missing columns in {csv_path}: {missing}"
            )

        rows = list(reader)

    if len(rows) != 1000:
        raise RuntimeError(
            f"expected 1000 samples in {csv_path}, got {len(rows)}"
        )

    sums = {
        name: 0.0
        for name in required
    }

    for row in rows:
        for name in required:
            sums[name] += float(row[name])

    return {
        name: sums[name] / len(rows)
        for name in required
    }


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


class CaseTransformer(ast.NodeTransformer):
    def __init__(
        self,
        case_name: str,
        out_name: str,
        vx: float,
        vy: float,
        wz: float,
    ) -> None:
        self.case_name = case_name
        self.out_name = out_name
        self.vx = vx
        self.vy = vy
        self.wz = wz
        self.output_replacements = 0
        self.pass_replacements = 0
        self.title_replacements = 0

    def visit_Constant(self, node: ast.Constant):
        if not isinstance(node.value, str):
            return node

        if node.value == OLD_OUT:
            self.output_replacements += 1
            return ast.copy_location(
                ast.Constant(value=self.out_name),
                node,
            )

        if node.value == OLD_PASS:
            self.pass_replacements += 1
            return ast.copy_location(
                ast.Constant(
                    value=(
                        f"V4N_CONTACT_GAIT_PROBE_MIRROR_{self.case_name}_PASS"
                    )
                ),
                node,
            )

        if node.value == "V4-B MODEL_1499 — CONTACT-AWARE GAIT RESULTS":
            self.title_replacements += 1
            return ast.copy_location(
                ast.Constant(
                    value=(
                        "V4-N MODEL_1499 — CONTACT-AWARE GAIT RESULTS "
                        f"— MIRROR CASE {self.case_name}"
                    )
                ),
                node,
            )

        return node


def derive_case_source(
    base_source: str,
    case_name: str,
    out_name: str,
    vx: float,
    vy: float,
    wz: float,
) -> str:
    tree = ast.parse(base_source)

    task_node = find_simple_assignment(tree, "TASK")
    run_node = find_simple_assignment(tree, "RUN_DIR")
    checkpoint_node = find_simple_assignment(tree, "CHECKPOINT")
    seed_node = find_simple_assignment(tree, "SEED")
    vx_node = find_simple_assignment(tree, "VX")
    vy_node = find_simple_assignment(tree, "VY")
    wz_node = find_simple_assignment(tree, "WZ")

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

    seed_node.value = ast.Constant(value=42)
    vx_node.value = ast.Constant(value=float(vx))
    vy_node.value = ast.Constant(value=float(vy))
    wz_node.value = ast.Constant(value=float(wz))

    transformer = CaseTransformer(
        case_name=case_name,
        out_name=out_name,
        vx=vx,
        vy=vy,
        wz=wz,
    )

    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    if transformer.output_replacements != 1:
        raise RuntimeError(
            f"{case_name}: expected one output-label replacement, "
            f"found {transformer.output_replacements}"
        )

    if transformer.pass_replacements != 1:
        raise RuntimeError(
            f"{case_name}: expected one PASS-marker replacement, "
            f"found {transformer.pass_replacements}"
        )

    transformed = ast.unparse(tree) + "\n"

    compile(
        transformed,
        f"<v4n-mirror-{case_name}>",
        "exec",
    )

    check_tree = ast.parse(transformed)

    checks = {
        "TASK": TASK,
        "SEED": 42,
        "VX": float(vx),
        "VY": float(vy),
        "WZ": float(wz),
    }

    for name, expected in checks.items():
        actual = ast.literal_eval(
            find_simple_assignment(check_tree, name).value
        )
        if actual != expected:
            raise RuntimeError(
                f"{case_name}: {name} replacement failed: "
                f"expected {expected!r}, got {actual!r}"
            )

    if OLD_OUT in transformed:
        raise RuntimeError(
            f"{case_name}: stale v4-B output label survived"
        )

    if OLD_PASS in transformed:
        raise RuntimeError(
            f"{case_name}: stale v4-B PASS marker survived"
        )

    if transformed.count(out_name) != 1:
        raise RuntimeError(
            f"{case_name}: output label occurrence count != 1"
        )

    return transformed


if not SOURCE_PROBE.is_file():
    raise SystemExit(
        f"ABORT: frozen v4-B canonical probe missing: {SOURCE_PROBE}"
    )

source_sha = sha256(SOURCE_PROBE)

if source_sha != EXPECTED_SOURCE_SHA:
    raise SystemExit(
        "ABORT: frozen v4-B probe SHA mismatch.\n"
        f"Expected: {EXPECTED_SOURCE_SHA}\n"
        f"Actual  : {source_sha}"
    )

if not MODEL.is_file():
    raise SystemExit(
        f"ABORT: frozen v4-N model missing: {MODEL}"
    )

model_sha = sha256(MODEL)

if model_sha != EXPECTED_MODEL_SHA:
    raise SystemExit(
        "ABORT: frozen v4-N model SHA mismatch.\n"
        f"Expected: {EXPECTED_MODEL_SHA}\n"
        f"Actual  : {model_sha}"
    )

base_source = SOURCE_PROBE.read_text(
    encoding="utf-8"
)

for token in (
    "SEED = 42",
    "VX = 0.40",
    "VY = 0.0",
    "WZ = 0.0",
    '"--deterministic"',
    "env_cfg.observations.policy.enable_corruption = False",
    "env_cfg.events.base_external_force_torque = None",
    "env_cfg.events.push_robot = None",
):
    if token not in base_source:
        raise RuntimeError(
            f"frozen protocol token missing: {token}"
        )

# Preflight every derived case before creating any evidence.
for case_name, vx, vy, wz in CASES:
    derive_case_source(
        base_source,
        case_name,
        output_name(case_name),
        vx,
        vy,
        wz,
    )

print("=" * 132)
print("v4-N SEED42 MIRRORED COMMAND RESPONSE AUDIT")
print("=" * 132)
print(f"Frozen probe SHA : {source_sha} — PASS")
print(f"Frozen model SHA : {model_sha} — PASS")
print("Evaluation seed  : 42 for every case")
print("Duration         : 20 s / 1000 samples per case")
print("Protocol         : frozen Play-equivalent evaluator")
print("Cases            : baseline, vy ±0.20, wz ±0.20")
print()

results = {}

with tempfile.TemporaryDirectory(
    prefix="g1_v4n_mirror_cmd_"
) as temp_dir:
    temp_dir = Path(temp_dir)

    for case_name, vx, vy, wz in CASES:
        out_name = output_name(case_name)

        output_dir = (
            REPO
            / "logs"
            / "evaluations"
            / "g1"
            / out_name
        )

        if output_dir.exists():
            raise SystemExit(
                f"ABORT: output already exists: {output_dir}"
            )

        source = derive_case_source(
            base_source,
            case_name,
            out_name,
            vx,
            vy,
            wz,
        )

        derived = (
            temp_dir
            / f"derived_{case_name}.py"
        )

        derived.write_text(
            source,
            encoding="utf-8",
            newline="\n",
        )

        print(
            f"RUN {case_name:8s} : "
            f"[{vx:+.2f}, {vy:+.2f}, {wz:+.2f}]"
        )

        completed = subprocess.run(
            [sys.executable, str(derived)],
            cwd=str(REPO),
            check=False,
        )

        if completed.returncode != 0:
            raise SystemExit(
                f"ABORT: {case_name} evaluator failed "
                f"with exit code {completed.returncode}"
            )

        samples_csv = (
            output_dir
            / "samples.csv"
        )

        if not samples_csv.is_file():
            raise SystemExit(
                f"ABORT: samples.csv missing for {case_name}: "
                f"{samples_csv}"
            )

        means = read_means(
            samples_csv
        )

        tolerance = 1.0e-12

        if (
            abs(means["cmd_vx_mps"] - vx) > tolerance
            or abs(means["cmd_vy_mps"] - vy) > tolerance
            or abs(means["cmd_wz_radps"] - wz) > tolerance
        ):
            raise RuntimeError(
                f"{case_name}: measured command does not match requested command"
            )

        results[case_name] = means

print()
print("-" * 132)
print("MEAN BODY-FRAME RESPONSE")
print("-" * 132)
print(
    "case      | cmd vy | mean vy | vy error | "
    "cmd wz | mean wz | wz error | mean vx"
)

for case_name, _, _, _ in CASES:
    m = results[case_name]

    vy_error = (
        m["base_vy_mps"]
        - m["cmd_vy_mps"]
    )

    wz_error = (
        m["base_wz_radps"]
        - m["cmd_wz_radps"]
    )

    print(
        f"{case_name:9s} | "
        f"{m['cmd_vy_mps']:+.4f} | "
        f"{m['base_vy_mps']:+.6f} | "
        f"{vy_error:+.6f} | "
        f"{m['cmd_wz_radps']:+.4f} | "
        f"{m['base_wz_radps']:+.6f} | "
        f"{wz_error:+.6f} | "
        f"{m['base_vx_mps']:+.6f}"
    )

vp = results["vy_p020"]
vm = results["vy_m020"]
wp = results["wz_p020"]
wm = results["wz_m020"]
b = results["baseline"]

vy_response_center = 0.5 * (
    vp["base_vy_mps"]
    + vm["base_vy_mps"]
)

vy_response_halfspan = 0.5 * (
    vp["base_vy_mps"]
    - vm["base_vy_mps"]
)

vy_error_plus = (
    vp["base_vy_mps"]
    - vp["cmd_vy_mps"]
)

vy_error_minus = (
    vm["base_vy_mps"]
    - vm["cmd_vy_mps"]
)

vy_error_center = 0.5 * (
    vy_error_plus
    + vy_error_minus
)

wz_response_center = 0.5 * (
    wp["base_wz_radps"]
    + wm["base_wz_radps"]
)

wz_response_halfspan = 0.5 * (
    wp["base_wz_radps"]
    - wm["base_wz_radps"]
)

wz_error_plus = (
    wp["base_wz_radps"]
    - wp["cmd_wz_radps"]
)

wz_error_minus = (
    wm["base_wz_radps"]
    - wm["cmd_wz_radps"]
)

wz_error_center = 0.5 * (
    wz_error_plus
    + wz_error_minus
)

print()
print("-" * 132)
print("MIRROR-SYMMETRY CLOSURE")
print("-" * 132)
print(
    f"baseline mean vy                     : "
    f"{b['base_vy_mps']:+.6f} m/s"
)
print(
    f"vy mirrored response center          : "
    f"{vy_response_center:+.6f} m/s"
)
print(
    f"vy mirrored response half-span       : "
    f"{vy_response_halfspan:+.6f} m/s"
)
print(
    f"vy mirrored tracking-error center    : "
    f"{vy_error_center:+.6f} m/s"
)
print(
    f"baseline mean wz                     : "
    f"{b['base_wz_radps']:+.6f} rad/s"
)
print(
    f"wz mirrored response center          : "
    f"{wz_response_center:+.6f} rad/s"
)
print(
    f"wz mirrored response half-span       : "
    f"{wz_response_halfspan:+.6f} rad/s"
)
print(
    f"wz mirrored tracking-error center    : "
    f"{wz_error_center:+.6f} rad/s"
)
print()
print(
    "Interpretation: for a mirror-symmetric closed-loop response, "
    "the positive/negative response centers should be near zero. "
    "A persistent signed center indicates directional asymmetry "
    "under otherwise mirrored commands."
)
print()
print("V4N_SEED42_MIRRORED_COMMAND_RESPONSE_AUDIT_PASS")
print("=" * 132)
