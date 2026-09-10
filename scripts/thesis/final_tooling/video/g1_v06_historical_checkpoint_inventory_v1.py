#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\humanoid_rl_lab")
ROOTS = [
    REPO / "logs" / "rsl_rl",
    REPO / "checkpoints",
]
OUT_DIR = REPO / "logs" / "thesis" / "video_final" / "V06_historical_progression" / "replay"
OUT_JSON = OUT_DIR / "V06_checkpoint_inventory_v1.json"

MODEL_RE = re.compile(r"^model_(\d+)\.pt$", re.IGNORECASE)

# We only need checkpoint families relevant to the thesis progression.
KEYWORDS = (
    "upright",
    "yaw",
    "v2",
    "v3",
    "v4",
    "g1",
)

PASS_MARKER = "G1_V06_HISTORICAL_CHECKPOINT_INVENTORY_V1_PASS"
FAIL_MARKER = "G1_V06_HISTORICAL_CHECKPOINT_INVENTORY_V1_FAIL"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def iteration(path: Path):
    m = MODEL_RE.match(path.name)
    return int(m.group(1)) if m else None


def family_from_path(path: Path) -> str:
    s = str(path).lower()
    if "v4n" in s or "v4n_" in s:
        return "V4-N"
    if "v4b" in s or "v4b_" in s:
        return "V4-B"
    if "v4a" in s or "v4a_" in s:
        return "V4-A"
    if "v4" in s:
        return "V4-other"
    if "v3c" in s or "v3c_" in s or "v3c-" in s:
        return "V3-C"
    if "v3d" in s or "v3d_" in s or "v3d-" in s:
        return "V3-D"
    if "v3b" in s or "v3b_" in s or "v3b-" in s:
        return "V3-B"
    if "v3a" in s or "v3a_" in s or "v3a-" in s:
        return "V3-A"
    if "v3" in s:
        return "V3-other"
    if "flat_orientation" in s:
        return "V2-G"
    if "wide_yaw" in s:
        return "V2-F-or-G"
    if "heavy_tail" in s:
        return "V2-E"
    if "tight_yaw" in s:
        return "V2-B"
    if "upright_yaw_direct" in s:
        return "V2-family"
    return "UNCLASSIFIED"


def relevant(path: Path) -> bool:
    s = str(path).lower()
    return "g1" in s and any(k in s for k in KEYWORDS)


def run_label(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return str(path.parent)
    parts = rel.parts
    if len(parts) <= 1:
        return str(path.parent)
    # Group by checkpoint parent/run directory, not by individual file.
    return str(root / Path(*parts[:-1]))


def main() -> None:
    found = []
    missing_roots = []

    for root in ROOTS:
        if not root.exists():
            missing_roots.append(str(root))
            continue
        for p in root.rglob("*.pt"):
            if not p.is_file() or not relevant(p):
                continue
            found.append(p)

    # De-duplicate exact paths.
    found = sorted(set(found), key=lambda p: str(p).lower())

    groups = defaultdict(list)
    for p in found:
        owner_root = next((r for r in ROOTS if r in p.parents or p.parent == r), REPO)
        groups[run_label(p, owner_root)].append(p)

    rows = []
    print("=" * 132)
    print("G1 V06 HISTORICAL PROGRESSION — READ-ONLY CHECKPOINT INVENTORY")
    print("=" * 132)
    print("Purpose        : discover what historical checkpoints actually still exist before any new rendering")
    print("Mutation       : NONE (only writes this inventory JSON)")
    print(f"Search roots   : {', '.join(str(r) for r in ROOTS)}")
    print(f"PT files found : {len(found)}")
    if missing_roots:
        print(f"Missing roots  : {missing_roots}")
    print()

    if not found:
        raise RuntimeError("No relevant G1 thesis checkpoints found under configured roots.")

    family_counts = defaultdict(int)

    for group_path in sorted(groups, key=str.lower):
        pts = sorted(
            groups[group_path],
            key=lambda p: (
                iteration(p) is None,
                iteration(p) if iteration(p) is not None else 10**12,
                p.name.lower(),
            ),
        )
        fam = family_from_path(Path(group_path))
        family_counts[fam] += len(pts)

        iters = [iteration(p) for p in pts if iteration(p) is not None]
        min_it = min(iters) if iters else None
        max_it = max(iters) if iters else None

        # SHA only scientifically useful anchor candidates:
        # first/final, model_600, model_620, model_699, model_1499.
        anchor_paths = set()
        if pts:
            anchor_paths.add(pts[0])
            anchor_paths.add(pts[-1])
        for p in pts:
            if iteration(p) in {600, 620, 699, 1499}:
                anchor_paths.add(p)

        anchors = []
        for p in sorted(anchor_paths, key=lambda x: str(x).lower()):
            anchors.append(
                {
                    "path": str(p),
                    "iteration": iteration(p),
                    "bytes": p.stat().st_size,
                    "mtime_utc": iso_mtime(p),
                    "sha256": sha256_file(p),
                }
            )

        row = {
            "family_guess": fam,
            "group": group_path,
            "checkpoint_count": len(pts),
            "iteration_min": min_it,
            "iteration_max": max_it,
            "checkpoint_names": [p.name for p in pts],
            "anchors": anchors,
        }
        rows.append(row)

        print("-" * 132)
        print(f"FAMILY  : {fam}")
        print(f"GROUP   : {group_path}")
        print(f"COUNT   : {len(pts)}")
        print(f"ITER    : {min_it} .. {max_it}")
        print("MODELS  : " + ", ".join(p.name for p in pts))
        print("ANCHORS :")
        for a in anchors:
            print(f"  iter={str(a['iteration']):>5s}  SHA={a['sha256']}  {a['path']}")

    print()
    print("=" * 132)
    print("FAMILY COUNTS")
    print("=" * 132)
    for fam in sorted(family_counts):
        print(f"{fam:16s}: {family_counts[fam]} checkpoint files")

    # Explicit checks for known final anchors.
    known = {
        "v2g_final_model600": {
            "path": REPO / "checkpoints" / "local_g1_yaw_direct_wide_yaw_flat_orientation_v2g" / "model_600.pt",
            "expected_sha": "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3",
        },
        "v4n_final_model1499": {
            "path": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_v4n_symmetry_augmentation_scratch"
                    / "2026-09-04_13-58-13_v4n_genuine_scratch_seed42_official1500" / "model_1499.pt",
            "expected_sha": "367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342",
        },
    }
    known_status = {}
    print()
    print("=" * 132)
    print("KNOWN FINAL ANCHORS")
    print("=" * 132)
    for name, spec in known.items():
        p = spec["path"]
        exists = p.is_file()
        actual = sha256_file(p) if exists else None
        sha_pass = exists and actual == spec["expected_sha"]
        known_status[name] = {
            "path": str(p),
            "exists": exists,
            "expected_sha": spec["expected_sha"],
            "actual_sha": actual,
            "sha_pass": sha_pass,
        }
        print(f"{name:24s}: {'PASS' if sha_pass else 'FAIL'}")
        print(f"  {p}")
        print(f"  SHA={actual}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "classification": (
            "read-only historical checkpoint availability inventory for presentation planning; "
            "NOT scientific re-evaluation and NOT new evidence"
        ),
        "search_roots": [str(r) for r in ROOTS],
        "missing_roots": missing_roots,
        "pt_file_count": len(found),
        "family_counts": dict(family_counts),
        "groups": rows,
        "known_final_anchors": known_status,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not all(v["sha_pass"] for v in known_status.values()):
        raise RuntimeError("Known final anchor SHA guard failed.")

    print()
    print(f"Inventory JSON : {OUT_JSON}")
    print(PASS_MARKER)


try:
    main()
except Exception as exc:
    print(FAIL_MARKER)
    print(f"EXCEPTION_TYPE={type(exc).__name__}")
    print(f"EXCEPTION_MESSAGE={exc}")
    raise
