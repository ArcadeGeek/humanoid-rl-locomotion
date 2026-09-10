from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

import humanoid_rl_lab.tasks  # noqa: F401


V4B_TASK = "Humanoid-RL-Lab-G1-Flat-V4B-Direct-Yaw-v0"
V4I_TASK = "Humanoid-RL-Lab-G1-Flat-V4I-Delayed-Bounded-Zero-Command-Full-Motion-v0"

EXPECTED_EXPERIMENT = "humanoid_rl_lab_g1_v4i_delayed_bounded_zero_command_full_motion_scratch"
EXPECTED_REWARD_NAME = "zero_command_bounded_full_motion"
EXPECTED_CURRICULUM_NAME = "zero_command_bounded_full_motion_weight"
EXPECTED_INITIAL_WEIGHT = 0.0
EXPECTED_TARGET_WEIGHT = -5.0
EXPECTED_NUM_STEPS = 4800

NUM_ENVS = 4
DEVICE = "cuda:0"


MISSING = object()


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def callable_name(obj: Any) -> str:
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return repr(obj)


def plain(obj: Any) -> Any:
    """Convert Isaac Lab config values into stable recursively comparable data."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]

    if callable(obj):
        return {"__callable__": callable_name(obj)}

    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return plain(to_dict())

    if hasattr(obj, "__dict__"):
        return {
            str(k): plain(v)
            for k, v in vars(obj).items()
            if not str(k).startswith("_")
        }

    return repr(obj)


def diff(a: Any, b: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    """Leaf/key-aware recursive diff. A missing subtree is reported once."""
    diffs: list[tuple[str, Any, Any]] = []

    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b))
        for key in keys:
            child = f"{path}.{key}" if path else str(key)
            av = a.get(key, MISSING)
            bv = b.get(key, MISSING)
            if av is MISSING or bv is MISSING:
                diffs.append(
                    (
                        child,
                        "<MISSING>" if av is MISSING else av,
                        "<MISSING>" if bv is MISSING else bv,
                    )
                )
            else:
                diffs.extend(diff(av, bv, child))
        return diffs

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((path, a, b))
            return diffs
        for idx, (av, bv) in enumerate(zip(a, b, strict=True)):
            diffs.extend(diff(av, bv, f"{path}[{idx}]"))
        return diffs

    if a != b:
        diffs.append((path, a, b))
    return diffs


def configured_reward_names(rewards_cfg: Any) -> list[str]:
    data = plain(rewards_cfg)
    if not isinstance(data, dict):
        fail(f"rewards config did not normalize to dict: {type(data)!r}")

    names: list[str] = []
    for name, value in data.items():
        if not isinstance(value, dict):
            continue
        if "weight" in value and value["weight"] is not None:
            names.append(name)
    return sorted(names)


def nonzero_reward_names(rewards_cfg: Any) -> list[str]:
    data = plain(rewards_cfg)
    if not isinstance(data, dict):
        fail(f"rewards config did not normalize to dict: {type(data)!r}")

    names: list[str] = []
    for name, value in data.items():
        if not isinstance(value, dict):
            continue
        weight = value.get("weight", MISSING)
        if weight is MISSING or weight is None:
            continue
        if float(weight) != 0.0:
            names.append(name)
    return sorted(names)

def expect_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        fail(f"{label}: expected {expected!r}, got {actual!r}")


def expect_close(actual: float, expected: float, label: str, tol: float = 1.0e-12) -> None:
    if abs(float(actual) - float(expected)) > tol:
        fail(f"{label}: expected {expected!r}, got {actual!r}")


# Resolve both environment configs through the task registry using identical parse arguments.
b_env = parse_env_cfg(V4B_TASK, device=DEVICE, num_envs=NUM_ENVS)
i_env = parse_env_cfg(V4I_TASK, device=DEVICE, num_envs=NUM_ENVS)

# Resolve both PPO configs through the same compatibility path used by train.py.
installed_version = metadata.version("rsl-rl-lib")

b_agent = load_cfg_from_registry(V4B_TASK, "rsl_rl_cfg_entry_point")
i_agent = load_cfg_from_registry(V4I_TASK, "rsl_rl_cfg_entry_point")
b_agent.device = DEVICE
i_agent.device = DEVICE

b_agent = handle_deprecated_rsl_rl_cfg(b_agent, installed_version)
i_agent = handle_deprecated_rsl_rl_cfg(i_agent, installed_version)

# --------------------------------------------------------------------------------------------------
# Exact v4-I scientific semantics.
# --------------------------------------------------------------------------------------------------
reward = getattr(i_env.rewards, EXPECTED_REWARD_NAME, None)
if reward is None:
    fail(f"v4-I reward missing: {EXPECTED_REWARD_NAME}")

expect_close(reward.weight, EXPECTED_INITIAL_WEIGHT, "v4-I initial reward weight")
expect_equal(callable_name(reward.func).split(".")[-1], EXPECTED_REWARD_NAME, "v4-I reward function name")

params = reward.params
expect_equal(params["command_name"], "base_velocity", "command_name")
expect_close(params["command_threshold"], 0.06, "command_threshold")
expect_close(params["base_planar_scale"], 1.0, "base_planar_scale")
expect_close(params["yaw_sigma"], 0.20, "yaw_sigma")
expect_close(params["foot_xy_scale"], 0.25, "foot_xy_scale")

asset_cfg = params["asset_cfg"]
expect_equal(asset_cfg.name, "robot", "asset_cfg.name")
expect_equal(
    list(asset_cfg.body_names),
    ["left_ankle_roll_link", "right_ankle_roll_link"],
    "asset_cfg.body_names",
)
expect_equal(bool(asset_cfg.preserve_order), True, "asset_cfg.preserve_order")

curriculum = getattr(i_env.curriculum, EXPECTED_CURRICULUM_NAME, None)
if curriculum is None:
    fail(f"v4-I curriculum missing: {EXPECTED_CURRICULUM_NAME}")

expect_equal(
    callable_name(curriculum.func).split(".")[-1],
    "modify_reward_weight",
    "v4-I curriculum function name",
)
curriculum_params = curriculum.params
expect_equal(curriculum_params["term_name"], EXPECTED_REWARD_NAME, "curriculum term_name")
expect_close(curriculum_params["weight"], EXPECTED_TARGET_WEIGHT, "curriculum target weight")
expect_equal(int(curriculum_params["num_steps"]), EXPECTED_NUM_STEPS, "curriculum num_steps")

# Parent v4-B command/tracking semantics must remain unchanged.
cmd = i_env.commands.base_velocity
expect_equal(bool(cmd.heading_command), False, "heading_command")
expect_close(cmd.rel_standing_envs, 0.02, "rel_standing_envs")
expect_equal(tuple(cmd.ranges.lin_vel_x), (0.0, 1.0), "vx range")
expect_equal(tuple(cmd.ranges.lin_vel_y), (-0.5, 0.5), "vy range")
expect_equal(tuple(cmd.ranges.ang_vel_z), (-1.0, 1.0), "wz range")

expect_close(
    i_env.rewards.track_lin_vel_xy_exp.params["std"],
    0.5,
    "track_lin_vel_xy std",
)
expect_close(i_env.actions.joint_pos.scale, 0.5, "joint position action scale")
expect_equal(
    bool(i_env.observations.policy.enable_corruption),
    True,
    "training observation corruption",
)

# v4-B flat curriculum stays disabled; v4-I only adds its delayed reward term.
expect_equal(getattr(i_env.curriculum, "terrain_levels", None), None, "terrain_levels")

# Explicitly reject inheritance of previous failed experimental reward terms.
for forbidden_name in (
    "zero_command_motion",
    "zero_command_full_motion",
    "stand_still_joint_deviation",
):
    value = getattr(i_env.rewards, forbidden_name, None)
    if value is not None:
        fail(f"forbidden prior-experiment reward present in v4-I: {forbidden_name}")

# --------------------------------------------------------------------------------------------------
# Full resolved environment diff: exactly two subtrees versus v4-B:
#   1) the configured zero-weight reward term
#   2) its native curriculum term
# --------------------------------------------------------------------------------------------------
b_env_plain = plain(b_env)
i_env_plain = plain(i_env)
env_diffs = diff(b_env_plain, i_env_plain)

expected_env_paths = sorted(
    [
        f"rewards.{EXPECTED_REWARD_NAME}",
        f"curriculum.{EXPECTED_CURRICULUM_NAME}",
    ]
)
actual_env_paths = sorted(path for path, _, _ in env_diffs)

if actual_env_paths != expected_env_paths:
    lines = "\n".join(
        f"  {path}: B={av!r} | I={iv!r}"
        for path, av, iv in env_diffs
    )
    fail(
        "resolved env differs from v4-B outside the two intended subtrees.\n"
        f"expected paths={expected_env_paths!r}\n"
        f"actual paths={actual_env_paths!r}\n{lines}"
    )

for path, b_value, i_value in env_diffs:
    expect_equal(b_value, "<MISSING>", f"v4-B side of {path}")
    if i_value == "<MISSING>":
        fail(f"v4-I side unexpectedly missing for {path}")

# Configured reward terms: 16 -> 17, but nonzero reward terms stay 16 before the switch.
b_configured_rewards = configured_reward_names(b_env.rewards)
i_configured_rewards = configured_reward_names(i_env.rewards)
b_nonzero_rewards = nonzero_reward_names(b_env.rewards)
i_nonzero_rewards = nonzero_reward_names(i_env.rewards)

expect_equal(len(b_configured_rewards), 16, "v4-B configured reward count")
expect_equal(len(i_configured_rewards), 17, "v4-I configured reward count")
expect_equal(
    sorted(set(i_configured_rewards) - set(b_configured_rewards)),
    [EXPECTED_REWARD_NAME],
    "added configured rewards",
)
expect_equal(
    sorted(set(b_configured_rewards) - set(i_configured_rewards)),
    [],
    "removed configured rewards",
)

expect_equal(len(b_nonzero_rewards), 16, "v4-B nonzero reward count")
expect_equal(len(i_nonzero_rewards), 16, "v4-I pre-switch nonzero reward count")
expect_equal(
    sorted(set(i_nonzero_rewards) - set(b_nonzero_rewards)),
    [],
    "pre-switch added nonzero rewards",
)
expect_equal(
    sorted(set(b_nonzero_rewards) - set(i_nonzero_rewards)),
    [],
    "pre-switch removed nonzero rewards",
)

# --------------------------------------------------------------------------------------------------
# Full migrated PPO diff: experiment_name only.
# --------------------------------------------------------------------------------------------------
b_agent_plain = plain(b_agent)
i_agent_plain = plain(i_agent)
ppo_diffs = diff(b_agent_plain, i_agent_plain)

if len(ppo_diffs) != 1:
    lines = "\n".join(
        f"  {path}: B={av!r} | I={iv!r}"
        for path, av, iv in ppo_diffs
    )
    fail(
        "resolved/migrated PPO differs from v4-B in more/less than one field.\n"
        f"diff count={len(ppo_diffs)}\n{lines}"
    )

ppo_path, ppo_b_value, ppo_i_value = ppo_diffs[0]
expect_equal(ppo_path, "experiment_name", "sole PPO diff path")
expect_equal(ppo_i_value, EXPECTED_EXPERIMENT, "v4-I experiment_name")

# No resume/checkpoint initialization: genuine scratch configuration.
for attr_name in ("resume",):
    if hasattr(i_agent, attr_name):
        expect_equal(getattr(i_agent, attr_name), False, f"PPO {attr_name}")

if hasattr(i_agent, "load_run"):
    if getattr(i_agent, "load_run") not in (None, "", ".*"):
        fail(f"unexpected PPO load_run for scratch config: {i_agent.load_run!r}")

if hasattr(i_agent, "load_checkpoint"):
    if getattr(i_agent, "load_checkpoint") not in (None, "", "model_.*.pt"):
        fail(
            "unexpected PPO load_checkpoint for scratch config: "
            f"{i_agent.load_checkpoint!r}"
        )

print("=" * 140)
print("V4-I RESOLVED-CONFIG AUDIT")
print("=" * 140)
print(f"v4-B task                  : {V4B_TASK}")
print(f"v4-I task                  : {V4I_TASK}")
print(f"rsl-rl-lib                 : {installed_version}")
print()
print("SCIENTIFIC DELTA")
print(f"env diff paths             : {actual_env_paths}")
print(f"reward func                : {callable_name(reward.func)}")
print(f"initial reward weight      : {reward.weight}")
print(f"gate threshold             : {params['command_threshold']}")
print(f"base planar scale          : {params['base_planar_scale']}")
print(f"yaw sigma                  : {params['yaw_sigma']}")
print("yaw form                   : 1 - exp(-(wz / sigma)^2)")
print(f"foot XY scale              : {params['foot_xy_scale']}")
print(f"foot bodies                : {list(asset_cfg.body_names)}")
print(f"curriculum func            : {callable_name(curriculum.func)}")
print(f"curriculum target weight   : {curriculum_params['weight']}")
print(f"curriculum num_steps       : {curriculum_params['num_steps']}")
print("switch condition           : common_step_counter > num_steps")
print()
print("INHERITED v4-B SEMANTICS")
print(f"heading command            : {cmd.heading_command}")
print(f"rel standing envs          : {cmd.rel_standing_envs}")
print(f"vx range                   : {tuple(cmd.ranges.lin_vel_x)}")
print(f"vy range                   : {tuple(cmd.ranges.lin_vel_y)}")
print(f"wz range                   : {tuple(cmd.ranges.ang_vel_z)}")
print(f"track lin std              : {i_env.rewards.track_lin_vel_xy_exp.params['std']}")
print(f"action scale               : {i_env.actions.joint_pos.scale}")
print(f"obs corruption             : {i_env.observations.policy.enable_corruption}")
print(f"configured rewards         : {len(b_configured_rewards)} -> {len(i_configured_rewards)}")
print(f"pre-switch nonzero rewards : {len(b_nonzero_rewards)} -> {len(i_nonzero_rewards)}")
print("pre-switch behavioral reward set identical to v4-B: YES")
print()
print("PPO")
print(f"sole PPO diff              : {ppo_path}")
print(f"experiment                 : {ppo_i_value}")
print("all other migrated PPO fields identical to v4-B: YES")
print("checkpoint initialization  : NONE / genuine scratch")
print()
print("V4I_RESOLVED_CONFIG_AUDIT_PASS")
print("=" * 140)
