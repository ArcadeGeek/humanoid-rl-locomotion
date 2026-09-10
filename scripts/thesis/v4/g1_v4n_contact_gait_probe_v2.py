from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(r"C:\humanoid_rl_lab")

CANONICAL = (
    REPO
    / "scripts"
    / "thesis"
    / "v4"
    / "g1_v4b_contact_gait_probe.py"
)

CANONICAL_SHA256 = "40948E95BA457FCC1AFD2708619054FE734CCEFF515130A997AFFF3958A84848"

TASK = "Humanoid-RL-Lab-G1-Flat-V4N-Symmetry-Augmentation-v0"

RUN_DIR = Path(
    r"C:\humanoid_rl_lab\logs\rsl_rl\humanoid_rl_lab_g1_v4n_symmetry_augmentation_scratch"
    r"\2026-09-04_13-58-13_v4n_genuine_scratch_seed42_official1500"
)

CHECKPOINT = RUN_DIR / "model_1499.pt"

CHECKPOINT_SHA256 = "367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342"

OLD_OUTPUT_LABEL = "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
NEW_OUTPUT_LABEL = "v4n_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


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
            f"expected exactly one top-level assignment for {name}, found {len(matches)}"
        )

    return matches[0]


class FrozenEvaluatorTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.output_label_replacements = 0
        self.result_title_replacements = 0

    def visit_Constant(self, node: ast.Constant):
        if not isinstance(node.value, str):
            return node

        value = node.value

        if value == OLD_OUTPUT_LABEL:
            self.output_label_replacements += 1
            return ast.copy_location(
                ast.Constant(value=NEW_OUTPUT_LABEL),
                node,
            )

        if value == "V4-B MODEL_1499 — CONTACT-AWARE GAIT RESULTS":
            self.result_title_replacements += 1
            return ast.copy_location(
                ast.Constant(
                    value="V4-N MODEL_1499 — CONTACT-AWARE GAIT RESULTS"
                ),
                node,
            )

        return node


def transform_canonical(source: str) -> str:
    tree = ast.parse(source)

    task_node = find_simple_assignment(tree, "TASK")
    run_node = find_simple_assignment(tree, "RUN_DIR")
    checkpoint_node = find_simple_assignment(tree, "CHECKPOINT")

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

    transformer = FrozenEvaluatorTransformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    if transformer.output_label_replacements != 1:
        raise RuntimeError(
            "expected exactly one canonical v4-B output-label replacement, "
            f"found {transformer.output_label_replacements}"
        )

    transformed = ast.unparse(tree) + "\n"
    compile(
        transformed,
        "<v4n-derived-frozen-evaluator>",
        "exec",
    )

    check_tree = ast.parse(transformed)

    task_check = find_simple_assignment(check_tree, "TASK")
    run_check = find_simple_assignment(check_tree, "RUN_DIR")
    checkpoint_check = find_simple_assignment(
        check_tree,
        "CHECKPOINT",
    )

    if ast.literal_eval(task_check.value) != TASK:
        raise RuntimeError("TASK replacement validation failed")

    if not (
        isinstance(run_check.value, ast.Call)
        and isinstance(run_check.value.func, ast.Name)
        and run_check.value.func.id == "Path"
        and len(run_check.value.args) == 1
        and ast.literal_eval(run_check.value.args[0]) == str(RUN_DIR)
    ):
        raise RuntimeError("RUN_DIR replacement validation failed")

    checkpoint_dump = ast.dump(
        checkpoint_check.value,
        include_attributes=False,
    )
    expected_checkpoint_dump = ast.dump(
        ast.BinOp(
            left=ast.Name(id="RUN_DIR", ctx=ast.Load()),
            op=ast.Div(),
            right=ast.Constant(value="model_1499.pt"),
        ),
        include_attributes=False,
    )

    if checkpoint_dump != expected_checkpoint_dump:
        raise RuntimeError(
            "CHECKPOINT replacement validation failed"
        )

    if OLD_OUTPUT_LABEL in transformed:
        raise RuntimeError(
            "old v4-B output label still present after transformation"
        )

    if transformed.count(NEW_OUTPUT_LABEL) != 1:
        raise RuntimeError(
            "new v4-N output label must appear exactly once"
        )

    return transformed


def main() -> int:
    if not CANONICAL.is_file():
        raise SystemExit(
            f"ABORT: frozen v4-B evaluator missing: {CANONICAL}"
        )

    canonical_sha = sha256(CANONICAL)

    if canonical_sha != CANONICAL_SHA256:
        raise SystemExit(
            "ABORT: frozen v4-B evaluator SHA mismatch.\n"
            f"Expected: {CANONICAL_SHA256}\n"
            f"Actual  : {canonical_sha}"
        )

    if not CHECKPOINT.is_file():
        raise SystemExit(
            f"ABORT: v4-N model_1499 checkpoint missing: {CHECKPOINT}"
        )

    checkpoint_sha = sha256(CHECKPOINT)

    if checkpoint_sha != CHECKPOINT_SHA256:
        raise SystemExit(
            "ABORT: v4-N checkpoint SHA mismatch.\n"
            f"Expected: {CHECKPOINT_SHA256}\n"
            f"Actual  : {checkpoint_sha}"
        )

    source = CANONICAL.read_text(encoding="utf-8")
    transformed = transform_canonical(source)

    print("=" * 112, flush=True)
    print(
        "v4-N DERIVED FROZEN CONTACT-GAIT PROBE v2",
        flush=True,
    )
    print("=" * 112, flush=True)
    print(
        f"Canonical evaluator : {CANONICAL}",
        flush=True,
    )
    print(
        f"Canonical SHA256    : {canonical_sha}",
        flush=True,
    )
    print(
        f"Task                : {TASK}",
        flush=True,
    )
    print(
        f"Checkpoint          : {CHECKPOINT}",
        flush=True,
    )
    print(
        f"Checkpoint SHA256   : {checkpoint_sha}",
        flush=True,
    )
    print(
        f"Output label        : {NEW_OUTPUT_LABEL}",
        flush=True,
    )
    print(
        "Methodology         : frozen v4-B evaluator; only task/run/checkpoint/evidence label changed",
        flush=True,
    )
    print("=" * 112, flush=True)

    with tempfile.TemporaryDirectory(
        prefix="g1_v4n_frozen_probe_"
    ) as tmp:
        derived = (
            Path(tmp)
            / "g1_v4n_derived_contact_gait_probe.py"
        )

        derived.write_text(
            transformed,
            encoding="utf-8",
            newline="\n",
        )

        proc = subprocess.run(
            [sys.executable, str(derived)],
            cwd=str(REPO),
            check=False,
        )

        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

    print()
    print(
        "V4N_DERIVED_FROZEN_CONTACT_GAIT_PROBE_V2_PASS",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
