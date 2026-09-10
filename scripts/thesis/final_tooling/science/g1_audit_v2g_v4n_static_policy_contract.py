from __future__ import annotations

import json
from collections.abc import Mapping

import humanoid_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


TASK_STAND = (
    "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-"
    "Wide-Yaw-Flat-Orientation-v0"
)
TASK_LOCO = "Humanoid-RL-Lab-G1-Flat-V4N-Symmetry-Augmentation-v0"


def to_plain(value):
    """Convert Isaac Lab config objects into deterministic JSON-safe data."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()

    if isinstance(value, Mapping):
        return {
            str(k): to_plain(v)
            for k, v in value.items()
        }

    if isinstance(value, tuple):
        return [to_plain(v) for v in value]

    if isinstance(value, list):
        return [to_plain(v) for v in value]

    if isinstance(value, set):
        return sorted(to_plain(v) for v in value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        return f"{value.__module__}:{value.__qualname__}"

    return repr(value)


def diff_paths(a, b, path=""):
    """Return compact leaf-level differences between two JSON-like trees."""
    out = []

    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b))
        for key in keys:
            child = f"{path}.{key}" if path else str(key)
            if key not in a:
                out.append((child, "<MISSING>", b[key]))
            elif key not in b:
                out.append((child, a[key], "<MISSING>"))
            else:
                out.extend(diff_paths(a[key], b[key], child))
        return out

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((f"{path}.length", len(a), len(b)))
        for idx, (av, bv) in enumerate(zip(a, b)):
            out.extend(diff_paths(av, bv, f"{path}[{idx}]"))
        return out

    if a != b:
        out.append((path, a, b))

    return out


def top_level_keys(value):
    plain = to_plain(value)
    if isinstance(plain, dict):
        return list(plain.keys())
    return []


def select_inference_fields(agent_plain):
    """Extract agent fields that can matter to inference plumbing."""
    selected = {}

    def walk(value, path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                key_l = key.lower()
                if (
                    "clip" in key_l
                    or "normaliz" in key_l
                    or "obs_group" in key_l
                    or "observation_group" in key_l
                ):
                    selected[child_path] = child
                walk(child, child_path)
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                walk(child, f"{path}[{idx}]")

    walk(agent_plain)
    return selected


def print_diffs(title, diffs, max_items=40):
    print(title)
    if not diffs:
        print("  EXACT_MATCH")
        return

    print(f"  DIFFERENCE_COUNT={len(diffs)}")
    for path, a, b in diffs[:max_items]:
        print(f"  {path}")
        print(f"    v2-G : {a!r}")
        print(f"    v4-N : {b!r}")
    if len(diffs) > max_items:
        print(f"  ... {len(diffs) - max_items} more differences omitted")


def main():
    stand_env = parse_env_cfg(
        TASK_STAND,
        device="cpu",
        num_envs=1,
    )
    loco_env = parse_env_cfg(
        TASK_LOCO,
        device="cpu",
        num_envs=1,
    )

    stand_agent = load_cfg_from_registry(
        TASK_STAND,
        "rsl_rl_cfg_entry_point",
    )
    loco_agent = load_cfg_from_registry(
        TASK_LOCO,
        "rsl_rl_cfg_entry_point",
    )

    stand_obs = to_plain(stand_env.observations.policy)
    loco_obs = to_plain(loco_env.observations.policy)

    stand_actions = to_plain(stand_env.actions)
    loco_actions = to_plain(loco_env.actions)

    stand_commands = to_plain(stand_env.commands)
    loco_commands = to_plain(loco_env.commands)

    stand_agent_plain = to_plain(stand_agent)
    loco_agent_plain = to_plain(loco_agent)

    obs_diffs = diff_paths(stand_obs, loco_obs)
    action_diffs = diff_paths(stand_actions, loco_actions)
    command_diffs = diff_paths(stand_commands, loco_commands)

    stand_inference = select_inference_fields(stand_agent_plain)
    loco_inference = select_inference_fields(loco_agent_plain)
    inference_diffs = diff_paths(
        stand_inference,
        loco_inference,
    )

    print("=" * 110)
    print("v2-G STAND <-> v4-N LOCOMOTION STATIC POLICY-CONTRACT AUDIT")
    print("=" * 110)
    print(f"STAND task : {TASK_STAND}")
    print(f"LOCO task  : {TASK_LOCO}")
    print()

    print("POLICY OBSERVATION TOP-LEVEL ORDER")
    print(f"  v2-G : {top_level_keys(stand_env.observations.policy)}")
    print(f"  v4-N : {top_level_keys(loco_env.observations.policy)}")
    print()

    print("ACTION TOP-LEVEL ORDER")
    print(f"  v2-G : {top_level_keys(stand_env.actions)}")
    print(f"  v4-N : {top_level_keys(loco_env.actions)}")
    print()

    print_diffs(
        "POLICY OBSERVATION CONFIG COMPARISON",
        obs_diffs,
    )
    print()
    print_diffs(
        "ACTION CONFIG COMPARISON",
        action_diffs,
    )
    print()
    print_diffs(
        "RSL-RL INFERENCE-RELEVANT CONFIG COMPARISON",
        inference_diffs,
    )
    print()

    print("COMMAND CONFIG DIFFERENCES")
    if not command_diffs:
        print("  EXACT_MATCH")
    else:
        print(f"  DIFFERENCE_COUNT={len(command_diffs)}")
        for path, a, b in command_diffs[:40]:
            print(f"  {path}")
            print(f"    v2-G : {a!r}")
            print(f"    v4-N : {b!r}")
        if len(command_diffs) > 40:
            print(
                f"  ... {len(command_diffs) - 40} more differences omitted"
            )

    print()
    print("=" * 110)

    hard_failures = []

    if obs_diffs:
        hard_failures.append(
            f"policy observation config differs ({len(obs_diffs)} leaves)"
        )

    if action_diffs:
        hard_failures.append(
            f"action config differs ({len(action_diffs)} leaves)"
        )

    if inference_diffs:
        hard_failures.append(
            f"inference-relevant RSL-RL config differs "
            f"({len(inference_diffs)} leaves)"
        )

    if hard_failures:
        print("STATIC_COMPATIBILITY=NOT_PROVEN")
        for failure in hard_failures:
            print(f"FAIL: {failure}")
        print(
            "No router/handoff implementation should be attempted "
            "until these differences are classified."
        )
        print("V2G_V4N_STATIC_POLICY_CONTRACT_AUDIT_REVIEW")
    else:
        print("STATIC_COMPATIBILITY=PASS_AT_CONFIG_LEVEL")
        print("Policy observation config : EXACT MATCH")
        print("Action config             : EXACT MATCH")
        print("Inference plumbing config : EXACT MATCH")
        print(
            "Command-config differences, if any, are reported above "
            "but are not part of the policy I/O contract."
        )
        print(
            "NEXT_REQUIRED_CHECK: runtime-resolved joint/action/observation "
            "ordering and checkpoint handoff behavior."
        )
        print("V2G_V4N_STATIC_POLICY_CONTRACT_AUDIT_PASS")

    print("=" * 110)


if __name__ == "__main__":
    main()
