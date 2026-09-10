from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

import humanoid_rl_lab.tasks  # noqa: F401


V4B_TASK = "Humanoid-RL-Lab-G1-Flat-V4B-Direct-Yaw-v0"
V4H_TASK = "Humanoid-RL-Lab-G1-Flat-V4H-Bounded-Zero-Command-Full-Motion-v0"

EXPECTED_EXPERIMENT = "humanoid_rl_lab_g1_v4h_bounded_zero_command_full_motion_scratch"
EXPECTED_REWARD_NAME = "zero_command_bounded_full_motion"

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


def active_reward_names(rewards_cfg: Any) -> list[str]:
    data = plain(rewards_cfg)
    if not isinstance(data, dict):
        fail(f"rewards config did not normalize to dict: {type(data)!r}")

    names: list[str] = []
    for name, value in data.items():
        if not isinstance(value, dict):
            continue
        weight = value.get("weight", MISSING)
        if weight is MISSING:
            continue
        if weight is not None and float(weight) != 0.0:
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
h_env = parse_env_cfg(V4H_TASK, device=DEVICE, num_envs=NUM_ENVS)

# Resolve both PPO configs through the same compatibility path used by train.py.
installed_version = metadata.version("rsl-rl-lib")

b_agent = load_cfg_from_registry(V4B_TASK, "rsl_rl_cfg_entry_point")
h_agent = load_cfg_from_registry(V4H_TASK, "rsl_rl_cfg_entry_point")
b_agent.device = DEVICE
h_agent.device = DEVICE

b_agent = handle_deprecated_rsl_rl_cfg(b_agent, installed_version)
h_agent = handle_deprecated_rsl_rl_cfg(h_agent, installed_version)

# --------------------------------------------------------------------------------------------------
# Exact v4-H scientific semantics.
# --------------------------------------------------------------------------------------------------
reward = getattr(h_env.rewards, EXPECTED_REWARD_NAME, None)
if reward is None:
    fail(f"v4-H reward missing: {EXPECTED_REWARD_NAME}")

expect_close(reward.weight, -5.0, "v4-H reward weight")
expect_equal(callable_name(reward.func).split(".")[-1], EXPECTED_REWARD_NAME, "v4-H reward function name")

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

# Parent v4-B command/tracking semantics must remain unchanged.
cmd = h_env.commands.base_velocity
expect_equal(bool(cmd.heading_command), False, "heading_command")
expect_close(cmd.rel_standing_envs, 0.02, "rel_standing_envs")
expect_equal(tuple(cmd.ranges.lin_vel_x), (0.0, 1.0), "vx range")
expect_equal(tuple(cmd.ranges.lin_vel_y), (-0.5, 0.5), "vy range")
expect_equal(tuple(cmd.ranges.ang_vel_z), (-1.0, 1.0), "wz range")

expect_close(
    h_env.rewards.track_lin_vel_xy_exp.params["std"],
    0.5,
    "track_lin_vel_xy std",
)

expect_close(h_env.actions.joint_pos.scale, 0.5, "joint position action scale")
expect_equal(
    bool(h_env.observations.policy.enable_corruption),
    True,
    "training observation corruption",
)

# Explicitly reject inheritance of the previous failed experimental reward terms.
for forbidden_name in (
    "zero_command_motion",
    "zero_command_full_motion",
    "stand_still_joint_deviation",
):
    value = getattr(h_env.rewards, forbidden_name, None)
    if value is not None:
        fail(f"forbidden prior-experiment reward present in v4-H: {forbidden_name}")

# --------------------------------------------------------------------------------------------------
# Full resolved environment diff: exactly one key must be added under rewards.
# --------------------------------------------------------------------------------------------------
b_env_plain = plain(b_env)
h_env_plain = plain(h_env)
env_diffs = diff(b_env_plain, h_env_plain)

if len(env_diffs) != 1:
    lines = "\n".join(
        f"  {path}: B={av!r} | H={hv!r}"
        for path, av, hv in env_diffs
    )
    fail(
        "resolved env differs from v4-B in more/less than one subtree.\n"
        f"diff count={len(env_diffs)}\n{lines}"
    )

env_diff_path, env_b_value, env_h_value = env_diffs[0]
expect_equal(
    env_diff_path,
    f"rewards.{EXPECTED_REWARD_NAME}",
    "sole resolved env diff path",
)
expect_equal(env_b_value, "<MISSING>", "v4-B side of new reward diff")
if env_h_value == "<MISSING>":
    fail("v4-H side of new reward diff is unexpectedly missing")

# Active reward counts should be 16 -> 17.
b_rewards = active_reward_names(b_env.rewards)
h_rewards = active_reward_names(h_env.rewards)
expect_equal(len(b_rewards), 16, "v4-B active reward count")
expect_equal(len(h_rewards), 17, "v4-H active reward count")

added_rewards = sorted(set(h_rewards) - set(b_rewards))
removed_rewards = sorted(set(b_rewards) - set(h_rewards))
expect_equal(added_rewards, [EXPECTED_REWARD_NAME], "added active rewards")
expect_equal(removed_rewards, [], "removed active rewards")

# --------------------------------------------------------------------------------------------------
# Full migrated PPO diff: experiment_name only.
# --------------------------------------------------------------------------------------------------
b_agent_plain = plain(b_agent)
h_agent_plain = plain(h_agent)
ppo_diffs = diff(b_agent_plain, h_agent_plain)

if len(ppo_diffs) != 1:
    lines = "\n".join(
        f"  {path}: B={av!r} | H={hv!r}"
        for path, av, hv in ppo_diffs
    )
    fail(
        "resolved/migrated PPO differs from v4-B in more/less than one field.\n"
        f"diff count={len(ppo_diffs)}\n{lines}"
    )

ppo_path, ppo_b_value, ppo_h_value = ppo_diffs[0]
expect_equal(ppo_path, "experiment_name", "sole PPO diff path")
expect_equal(ppo_h_value, EXPECTED_EXPERIMENT, "v4-H experiment_name")

# No resume/checkpoint initialization: genuine scratch configuration.
for attr_name in ("resume",):
    if hasattr(h_agent, attr_name):
        expect_equal(getattr(h_agent, attr_name), False, f"PPO {attr_name}")

if hasattr(h_agent, "load_run"):
    if getattr(h_agent, "load_run") not in (None, "", ".*"):
        fail(f"unexpected PPO load_run for scratch config: {h_agent.load_run!r}")

if hasattr(h_agent, "load_checkpoint"):
    if getattr(h_agent, "load_checkpoint") not in (None, "", "model_.*.pt"):
        fail(
            "unexpected PPO load_checkpoint for scratch config: "
            f"{h_agent.load_checkpoint!r}"
        )

print("=" * 132)
print("V4-H RESOLVED-CONFIG AUDIT")
print("=" * 132)
print(f"v4-B task                  : {V4B_TASK}")
print(f"v4-H task                  : {V4H_TASK}")
print(f"rsl-rl-lib                 : {installed_version}")
print()
print("SCIENTIFIC DELTA")
print(f"sole env diff              : {env_diff_path}")
print(f"reward func                : {callable_name(reward.func)}")
print(f"weight                     : {reward.weight}")
print(f"gate threshold             : {params['command_threshold']}")
print(f"base planar scale          : {params['base_planar_scale']}")
print(f"yaw sigma                  : {params['yaw_sigma']}")
print(f"yaw form                   : 1 - exp(-(wz / sigma)^2)")
print(f"foot XY scale              : {params['foot_xy_scale']}")
print(f"foot bodies                : {list(asset_cfg.body_names)}")
print()
print("INHERITED v4-B SEMANTICS")
print(f"heading command            : {cmd.heading_command}")
print(f"rel standing envs          : {cmd.rel_standing_envs}")
print(f"vx range                   : {tuple(cmd.ranges.lin_vel_x)}")
print(f"vy range                   : {tuple(cmd.ranges.lin_vel_y)}")
print(f"wz range                   : {tuple(cmd.ranges.ang_vel_z)}")
print(f"track lin std              : {h_env.rewards.track_lin_vel_xy_exp.params['std']}")
print(f"action scale               : {h_env.actions.joint_pos.scale}")
print(f"obs corruption             : {h_env.observations.policy.enable_corruption}")
print(f"active rewards             : {len(b_rewards)} -> {len(h_rewards)}")
print(f"added rewards              : {added_rewards}")
print()
print("PPO")
print(f"sole PPO diff              : {ppo_path}")
print(f"experiment                 : {ppo_h_value}")
print("all other migrated PPO fields identical to v4-B: YES")
print("checkpoint initialization  : NONE / genuine scratch")
print()
print("V4H_RESOLVED_CONFIG_AUDIT_PASS")
print("=" * 132)
