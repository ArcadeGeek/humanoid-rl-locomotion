from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import gymnasium as gym

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import humanoid_rl_lab.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402


TASK = "Humanoid-RL-Lab-G1-Flat-V4B-Direct-Yaw-v0"


def p(*args):
    print(*args, flush=True)


def counterpart(name: str) -> str | None:
    if "left_" in name:
        return name.replace("left_", "right_", 1)
    if "right_" in name:
        return name.replace("right_", "left_", 1)
    return None


def canonical(value, depth=0):
    if depth > 6:
        return "<max-depth>"

    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return round(value, 12)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(k): canonical(v, depth + 1)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }

    if isinstance(value, (list, tuple)):
        return [canonical(v, depth + 1) for v in value]

    if callable(value):
        return (
            getattr(value, "__module__", type(value).__module__)
            + "."
            + getattr(
                value,
                "__qualname__",
                getattr(value, "__name__", type(value).__qualname__),
            )
        )

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return canonical(value.to_dict(), depth + 1)
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        public = {
            k: v
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
        if public:
            return canonical(public, depth + 1)

    return repr(value)


def resolve_joint_ids(raw_ids, joint_count: int) -> list[int]:
    if raw_ids is None:
        return list(range(joint_count))

    if isinstance(raw_ids, slice):
        return list(range(joint_count))[raw_ids]

    if hasattr(raw_ids, "tolist"):
        raw_ids = raw_ids.tolist()

    if isinstance(raw_ids, int):
        return [int(raw_ids)]

    return [int(v) for v in raw_ids]


def find_live_joint_action_term(action_manager):
    terms = getattr(action_manager, "_terms", None)
    names = getattr(action_manager, "_term_names", None)

    if isinstance(terms, dict):
        for name, term in terms.items():
            if hasattr(term, "_joint_ids"):
                return name, term

    if isinstance(terms, (list, tuple)):
        if isinstance(names, (list, tuple)):
            for name, term in zip(names, terms):
                if hasattr(term, "_joint_ids"):
                    return name, term
        else:
            for index, term in enumerate(terms):
                if hasattr(term, "_joint_ids"):
                    return f"term_{index}", term

    raise RuntimeError("live joint action term not found")


def explicit_lateral_strings(obj, prefix):
    data = canonical(obj)
    hits = []

    def walk(value, path):
        if isinstance(value, str):
            lower = value.lower()
            if "left" in lower or "right" in lower:
                hits.append((path, value))
            return

        if isinstance(value, dict):
            for key, val in value.items():
                walk(val, f"{path}.{key}")
            return

        if isinstance(value, list):
            for index, val in enumerate(value):
                walk(val, f"{path}[{index}]")

    walk(data, prefix)
    return hits


env = None
failed = False

try:
    env_cfg = load_cfg_from_registry(
        TASK,
        "env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = 1

    env = gym.make(
        TASK,
        cfg=env_cfg,
        render_mode=None,
    )

    u = env.unwrapped
    robot = u.scene["robot"]

    p()
    p("=" * 148)
    p("v4-B LEFT/RIGHT STRUCTURAL SYMMETRY AUDIT v2")
    p("=" * 148)
    p(f"Task                  : {TASK}")
    p(f"device                : {u.device}")
    p(f"step_dt               : {u.step_dt}")
    p("Environment creation  : PASS")
    p()

    # ------------------------------------------------------------------------------------------
    # Actual articulation order.
    # ------------------------------------------------------------------------------------------
    joint_names = list(robot.data.joint_names)

    if len(joint_names) != 37:
        raise RuntimeError(
            f"expected 37 joints, got {len(joint_names)}"
        )

    joint_index = {
        name: index
        for index, name in enumerate(joint_names)
    }

    p("-" * 148)
    p("1) ACTUAL ARTICULATION JOINT ORDER")
    p("-" * 148)

    for index, name in enumerate(joint_names):
        p(f"joint[{index:02d}] = {name}")

    p()

    # ------------------------------------------------------------------------------------------
    # Pairing.
    # ------------------------------------------------------------------------------------------
    left_right_pairs = []
    missing_pairs = []

    for name in joint_names:
        if not name.startswith("left_"):
            continue

        other = counterpart(name)

        if other not in joint_index:
            missing_pairs.append((name, other))
        else:
            left_right_pairs.append(
                (
                    name,
                    joint_index[name],
                    other,
                    joint_index[other],
                )
            )

    p("-" * 148)
    p("2) LEFT/RIGHT JOINT PAIRS")
    p("-" * 148)

    for left, li, right, ri in left_right_pairs:
        p(
            f"{left:<30s} [{li:02d}]  <->  "
            f"{right:<30s} [{ri:02d}]"
        )

    p(f"paired joints          : {len(left_right_pairs)}")
    p(f"missing counterparts   : {len(missing_pairs)}")

    if missing_pairs:
        for pair in missing_pairs:
            p(f"MISSING: {pair}")

    p()

    # ------------------------------------------------------------------------------------------
    # Live action mapping. Handles slice(None), tensor, list, tuple or None.
    # ------------------------------------------------------------------------------------------
    action_manager = u.action_manager
    action_term_name, action_term = find_live_joint_action_term(
        action_manager
    )

    raw_joint_ids = getattr(
        action_term,
        "_joint_ids",
        None,
    )

    action_joint_ids = resolve_joint_ids(
        raw_joint_ids,
        len(joint_names),
    )

    action_joint_names = [
        joint_names[index]
        for index in action_joint_ids
    ]

    action_shape = getattr(
        env.action_space,
        "shape",
        None,
    )

    if action_shape is None:
        raise RuntimeError(
            f"action_space has no shape: {env.action_space}"
        )

    action_dim = int(action_shape[-1])

    p("-" * 148)
    p("3) LIVE ACTION MAPPING")
    p("-" * 148)
    p(f"action term            : {action_term_name}")
    p(f"raw _joint_ids type    : {type(raw_joint_ids).__name__}")
    p(f"raw _joint_ids value   : {raw_joint_ids}")
    p(f"action dimension       : {action_dim}")
    p(f"mapped joints          : {len(action_joint_names)}")

    if action_dim != 37:
        raise RuntimeError(
            f"expected action dim 37, got {action_dim}"
        )

    if len(action_joint_names) != 37:
        raise RuntimeError(
            f"expected 37 mapped joints, got {len(action_joint_names)}"
        )

    for action_id, (joint_id, joint_name) in enumerate(
        zip(action_joint_ids, action_joint_names)
    ):
        p(
            f"action[{action_id:02d}] -> "
            f"joint[{joint_id:02d}] {joint_name}"
        )

    action_index = {
        name: index
        for index, name in enumerate(action_joint_names)
    }

    p()
    p("Action left/right pair positions:")

    action_pair_missing = []

    for left, _, right, _ in left_right_pairs:
        if left not in action_index or right not in action_index:
            action_pair_missing.append((left, right))
            continue

        p(
            f"{left:<30s} action[{action_index[left]:02d}]  <->  "
            f"{right:<30s} action[{action_index[right]:02d}]"
        )

    p(f"paired joints missing from action map: {len(action_pair_missing)}")
    p()

    # ------------------------------------------------------------------------------------------
    # Default pose.
    # ------------------------------------------------------------------------------------------
    default_q = (
        robot.data.default_joint_pos[0]
        .detach()
        .cpu()
        .numpy()
    )

    p("-" * 148)
    p("4) LIVE DEFAULT JOINT POSE — LEFT/RIGHT")
    p("-" * 148)
    p(
        "pair | q_left [rad] | q_right [rad] | "
        "|abs(L)-abs(R)| | L-R"
    )

    max_abs_magnitude_delta = 0.0

    for left, li, right, ri in left_right_pairs:
        lv = float(default_q[li])
        rv = float(default_q[ri])

        mag_delta = abs(abs(lv) - abs(rv))
        direct_delta = lv - rv

        max_abs_magnitude_delta = max(
            max_abs_magnitude_delta,
            mag_delta,
        )

        p(
            f"{left:<30s} <-> {right:<30s} | "
            f"{lv:+.6f} | {rv:+.6f} | "
            f"{mag_delta:.9f} | {direct_delta:+.6f}"
        )

    p(
        "max |abs(L)-abs(R)| : "
        f"{max_abs_magnitude_delta:.9f} rad"
    )
    p()

    p("Resolved init_state.joint_pos:")
    p(canonical(env_cfg.scene.robot.init_state.joint_pos))
    p()

    # ------------------------------------------------------------------------------------------
    # Actuator configs.
    # ------------------------------------------------------------------------------------------
    p("-" * 148)
    p("5) RESOLVED ACTUATOR CONFIGURATION")
    p("-" * 148)

    actuators = env_cfg.scene.robot.actuators

    if isinstance(actuators, dict):
        actuator_items = actuators.items()
    elif hasattr(actuators, "items"):
        actuator_items = actuators.items()
    else:
        actuator_items = vars(actuators).items()

    actuator_count = 0

    for name, cfg in actuator_items:
        if name.startswith("_"):
            continue
        actuator_count += 1
        p(f"{name}: {canonical(cfg)}")

    p(f"actuator groups printed: {actuator_count}")
    p()

    # ------------------------------------------------------------------------------------------
    # Lateral references in manager configs.
    # ------------------------------------------------------------------------------------------
    p("-" * 148)
    p("6) EXPLICIT LEFT/RIGHT REFERENCES IN MANAGER CONFIGS")
    p("-" * 148)

    sections = (
        ("observations", env_cfg.observations),
        ("rewards", env_cfg.rewards),
        ("terminations", env_cfg.terminations),
        ("events", env_cfg.events),
    )

    lateral_total = 0

    for section_name, section_obj in sections:
        hits = explicit_lateral_strings(
            section_obj,
            section_name,
        )

        p(f"{section_name}: {len(hits)} explicit left/right string(s)")

        for path, value in hits:
            p(f"  {path} = {value!r}")

        lateral_total += len(hits)

    p()

    # ------------------------------------------------------------------------------------------
    # Live physics pair comparison for core joint properties exposed by robot.data.
    # ------------------------------------------------------------------------------------------
    p("-" * 148)
    p("7) LIVE JOINT-LIMIT MIRROR CHECK")
    p("-" * 148)

    limits = getattr(
        robot.data,
        "soft_joint_pos_limits",
        None,
    )

    limit_asymmetry_count = 0

    if limits is None:
        p("soft_joint_pos_limits unavailable — SKIPPED")
    else:
        limits_cpu = limits[0].detach().cpu().numpy()

        for left, li, right, ri in left_right_pairs:
            llo, lhi = map(float, limits_cpu[li])
            rlo, rhi = map(float, limits_cpu[ri])

            same = (
                abs(llo - rlo) < 1.0e-6
                and abs(lhi - rhi) < 1.0e-6
            )

            mirrored = (
                abs(llo + rhi) < 1.0e-6
                and abs(lhi + rlo) < 1.0e-6
            )

            status = "PASS" if (same or mirrored) else "CHECK"

            if status == "CHECK":
                limit_asymmetry_count += 1

            p(
                f"{status:5s} | {left:<30s} [{llo:+.4f},{lhi:+.4f}] "
                f"<-> {right:<30s} [{rlo:+.4f},{rhi:+.4f}] "
                f"| same={same} mirrored={mirrored}"
            )

    p()
    p("-" * 148)
    p("SUMMARY")
    p("-" * 148)
    p(f"joint count                         : {len(joint_names)}")
    p(f"left/right pairs                    : {len(left_right_pairs)}")
    p(f"missing joint counterparts          : {len(missing_pairs)}")
    p(f"action dim                          : {action_dim}")
    p(f"action mapped joints                : {len(action_joint_names)}")
    p(f"pair gaps in action mapping         : {len(action_pair_missing)}")
    p(
        "max default-pose |abs(L)-abs(R)| : "
        f"{max_abs_magnitude_delta:.9f} rad"
    )
    p(f"explicit lateral config strings     : {lateral_total}")
    p(f"joint-limit pairs needing check     : {limit_asymmetry_count}")
    p()
    p("V4B_LEFT_RIGHT_STRUCTURAL_SYMMETRY_AUDIT_V2_PASS")
    p("=" * 148)

except Exception:
    failed = True
    p()
    p("=" * 148)
    p("V4B_LEFT_RIGHT_STRUCTURAL_SYMMETRY_AUDIT_V2_FAIL")
    p("=" * 148)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()

finally:
    if env is not None:
        try:
            env.close()
        except Exception:
            p("WARNING: env.close() raised:")
            traceback.print_exc(file=sys.stdout)

    try:
        simulation_app.close()
    except Exception:
        p("WARNING: simulation_app.close() raised:")
        traceback.print_exc(file=sys.stdout)

if failed:
    raise SystemExit(1)
