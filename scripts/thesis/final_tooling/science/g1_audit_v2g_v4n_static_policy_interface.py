from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
from typing import Any

import humanoid_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry


TASK_V2G = (
    "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-"
    "Wide-Yaw-Flat-Orientation-v0"
)
TASK_V4N = "Humanoid-RL-Lab-G1-Flat-V4N-Symmetry-Augmentation-v0"

CHECKPOINT_V2G = Path(
    r"C:\humanoid_rl_lab\checkpoints"
    r"\local_g1_yaw_direct_wide_yaw_flat_orientation_v2g"
    r"\model_600.pt"
)
CHECKPOINT_V4N = Path(
    r"C:\humanoid_rl_lab\logs\rsl_rl"
    r"\humanoid_rl_lab_g1_v4n_symmetry_augmentation_scratch"
    r"\2026-09-04_13-58-13_v4n_genuine_scratch_seed42_official1500"
    r"\model_1499.pt"
)

EXPECTED_V2G_SHA = (
    "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3"
)
EXPECTED_V4N_SHA = (
    "367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def qualname(value: Any) -> str:
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", None)
    if module and name:
        return f"{module}:{name}"
    return repr(value)


def canonical(value: Any, depth: int = 0) -> Any:
    if depth > 30:
        raise RuntimeError("canonicalization depth exceeded")

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(k): canonical(v, depth + 1)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [canonical(v, depth + 1) for v in value]

    if isinstance(value, set):
        return sorted(
            (canonical(v, depth + 1) for v in value),
            key=lambda x: json.dumps(x, sort_keys=True, default=str),
        )

    if isinstance(value, type) or callable(value):
        return {"__callable__": qualname(value)}

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return canonical(to_dict(), depth + 1)

    if hasattr(value, "__dict__"):
        data = {}
        for key, item in vars(value).items():
            if key.startswith("_"):
                continue
            if callable(item):
                data[key] = {"__callable__": qualname(item)}
            else:
                data[key] = canonical(item, depth + 1)
        return data

    return {
        "__type__": qualname(type(value)),
        "__repr__": repr(value),
    }


def json_text(value: Any) -> str:
    return json.dumps(
        canonical(value),
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
    )


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json_text(value).encode("utf-8")
    ).hexdigest().upper()


def cfg_dict(value: Any) -> dict[str, Any]:
    result = canonical(value)
    if not isinstance(result, dict):
        raise RuntimeError(
            f"expected dict-like config, got {type(result).__name__}"
        )
    return result


def observation_term_names(policy_dict: dict[str, Any]) -> list[str]:
    names = []
    for key, value in policy_dict.items():
        if not isinstance(value, dict):
            continue
        if "func" in value:
            names.append(key)
    return names


def action_term_names(actions_dict: dict[str, Any]) -> list[str]:
    names = []
    for key, value in actions_dict.items():
        if not isinstance(value, dict):
            continue
        # Action term configs expose class_type and/or asset_name.
        if "class_type" in value or "asset_name" in value:
            names.append(key)
    return names


def robot_default_joint_payload(env_cfg: Any) -> dict[str, Any]:
    init_state = cfg_dict(env_cfg.scene.robot.init_state)
    return {
        "joint_pos": init_state.get("joint_pos"),
        "joint_vel": init_state.get("joint_vel"),
    }


def selected_runner_payload(runner_cfg: Any) -> dict[str, Any]:
    data = cfg_dict(runner_cfg)

    payload = {
        "clip_actions": data.get("clip_actions"),
        "empirical_normalization": data.get("empirical_normalization"),
        "class_name": data.get("class_name"),
    }

    policy = data.get("policy")
    if isinstance(policy, dict):
        payload["policy"] = {
            key: policy.get(key)
            for key in (
                "class_name",
                "init_noise_std",
                "actor_hidden_dims",
                "critic_hidden_dims",
                "activation",
            )
            if key in policy
        }

    return payload


def print_diff(
    label: str,
    left: Any,
    right: Any,
    max_lines: int = 120,
) -> None:
    left_lines = json_text(left).splitlines()
    right_lines = json_text(right).splitlines()
    diff = list(
        difflib.unified_diff(
            left_lines,
            right_lines,
            fromfile="v2-G",
            tofile="v4-N",
            lineterm="",
        )
    )

    print(f"{label}_DIFF_LINES={len(diff)}")
    for line in diff[:max_lines]:
        print(line)

    if len(diff) > max_lines:
        print(
            f"... diff truncated: "
            f"{len(diff) - max_lines} additional lines"
        )


def verify_checkpoint(
    path: Path,
    expected_sha: str,
    label: str,
) -> str:
    if not path.is_file():
        raise RuntimeError(f"{label} checkpoint missing: {path}")

    actual = sha256(path)
    if actual != expected_sha:
        raise RuntimeError(
            f"{label} checkpoint SHA mismatch\n"
            f"expected={expected_sha}\n"
            f"actual  ={actual}"
        )
    return actual


def main() -> None:
    v2_sha = verify_checkpoint(
        CHECKPOINT_V2G,
        EXPECTED_V2G_SHA,
        "v2-G",
    )
    v4_sha = verify_checkpoint(
        CHECKPOINT_V4N,
        EXPECTED_V4N_SHA,
        "v4-N",
    )

    env_v2 = load_cfg_from_registry(
        TASK_V2G,
        "env_cfg_entry_point",
    )
    env_v4 = load_cfg_from_registry(
        TASK_V4N,
        "env_cfg_entry_point",
    )
    runner_v2 = load_cfg_from_registry(
        TASK_V2G,
        "rsl_rl_cfg_entry_point",
    )
    runner_v4 = load_cfg_from_registry(
        TASK_V4N,
        "rsl_rl_cfg_entry_point",
    )

    obs_v2 = cfg_dict(env_v2.observations.policy)
    obs_v4 = cfg_dict(env_v4.observations.policy)

    actions_v2 = cfg_dict(env_v2.actions)
    actions_v4 = cfg_dict(env_v4.actions)

    robot_default_v2 = robot_default_joint_payload(env_v2)
    robot_default_v4 = robot_default_joint_payload(env_v4)

    runner_sel_v2 = selected_runner_payload(runner_v2)
    runner_sel_v4 = selected_runner_payload(runner_v4)

    obs_names_v2 = observation_term_names(obs_v2)
    obs_names_v4 = observation_term_names(obs_v4)

    action_names_v2 = action_term_names(actions_v2)
    action_names_v4 = action_term_names(actions_v4)

    obs_equal = obs_v2 == obs_v4
    obs_order_equal = obs_names_v2 == obs_names_v4
    actions_equal = actions_v2 == actions_v4
    action_order_equal = action_names_v2 == action_names_v4
    robot_default_equal = robot_default_v2 == robot_default_v4

    print("=" * 112)
    print("v2-G <-> v4-N STATIC POLICY INTERFACE AUDIT")
    print("=" * 112)
    print(f"TASK_V2G                         : {TASK_V2G}")
    print(f"TASK_V4N                         : {TASK_V4N}")
    print(f"V2G_CHECKPOINT_SHA               : {v2_sha}")
    print(f"V4N_CHECKPOINT_SHA               : {v4_sha}")
    print()

    print("OBSERVATION POLICY")
    print(f"v2-G term order                  : {obs_names_v2}")
    print(f"v4-N term order                  : {obs_names_v4}")
    print(f"term order equal                 : {obs_order_equal}")
    print(f"full policy obs config equal     : {obs_equal}")
    print(f"v2-G obs config SHA              : {json_hash(obs_v2)}")
    print(f"v4-N obs config SHA              : {json_hash(obs_v4)}")
    if not obs_equal:
        print_diff("OBS_CONFIG", obs_v2, obs_v4)
    print()

    print("ACTION INTERFACE")
    print(f"v2-G action term order           : {action_names_v2}")
    print(f"v4-N action term order           : {action_names_v4}")
    print(f"action term order equal          : {action_order_equal}")
    print(f"full action config equal         : {actions_equal}")
    print(f"v2-G action config SHA           : {json_hash(actions_v2)}")
    print(f"v4-N action config SHA           : {json_hash(actions_v4)}")
    print("v2-G resolved action config:")
    print(json_text(actions_v2))
    print("v4-N resolved action config:")
    print(json_text(actions_v4))
    if not actions_equal:
        print_diff("ACTION_CONFIG", actions_v2, actions_v4)
    print()

    print("ROBOT DEFAULT JOINT STATE")
    print(f"default joint state equal         : {robot_default_equal}")
    print(f"v2-G default joint SHA            : {json_hash(robot_default_v2)}")
    print(f"v4-N default joint SHA            : {json_hash(robot_default_v4)}")
    if not robot_default_equal:
        print_diff(
            "ROBOT_DEFAULT_JOINT_STATE",
            robot_default_v2,
            robot_default_v4,
        )
    print()

    print("RUNNER / INFERENCE-RELEVANT CONFIG")
    print("v2-G:")
    print(json_text(runner_sel_v2))
    print("v4-N:")
    print(json_text(runner_sel_v4))
    print(
        "runner selected config equal       : "
        f"{runner_sel_v2 == runner_sel_v4}"
    )
    if runner_sel_v2 != runner_sel_v4:
        print_diff(
            "RUNNER_SELECTED",
            runner_sel_v2,
            runner_sel_v4,
        )
    print()

    hard_pass = all(
        (
            obs_order_equal,
            obs_equal,
            action_order_equal,
            actions_equal,
            robot_default_equal,
        )
    )

    print("=" * 112)
    if hard_pass:
        print("STATIC_INTERFACE_COMPATIBILITY=PASS")
        print("V2G_V4N_STATIC_INTERFACE_AUDIT_PASS")
    else:
        print("STATIC_INTERFACE_COMPATIBILITY=REVIEW_REQUIRED")
        print("V2G_V4N_STATIC_INTERFACE_AUDIT_REVIEW_REQUIRED")
    print("=" * 112)


if __name__ == "__main__":
    main()
