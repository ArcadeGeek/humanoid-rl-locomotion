from __future__ import annotations

import sys
import traceback

import gymnasium as gym

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import humanoid_rl_lab.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from omni.usd import get_context  # noqa: E402
from pxr import UsdPhysics  # noqa: E402


TASK = "Humanoid-RL-Lab-G1-Flat-V4B-Direct-Yaw-v0"


def p(*args):
    print(*args, flush=True)


def counterpart(name: str) -> str | None:
    if name.startswith("left_"):
        return name.replace("left_", "right_", 1)
    if name.startswith("right_"):
        return name.replace("right_", "left_", 1)
    return None


def fmt(value) -> str:
    if value is None:
        return "None"
    try:
        return str(tuple(value))
    except TypeError:
        return str(value)


def get_joint_metadata(stage, joint_name: str) -> dict:
    matches = []
    for prim in stage.Traverse():
        if prim.GetName() == joint_name and prim.IsA(UsdPhysics.RevoluteJoint):
            matches.append(prim)

    if len(matches) != 1:
        raise RuntimeError(
            f"{joint_name}: expected exactly one RevoluteJoint prim, found {len(matches)}"
        )

    joint = UsdPhysics.RevoluteJoint(matches[0])

    return {
        "prim_path": str(matches[0].GetPath()),
        "axis": joint.GetAxisAttr().Get(),
        "lower": joint.GetLowerLimitAttr().Get(),
        "upper": joint.GetUpperLimitAttr().Get(),
        "local_pos0": joint.GetLocalPos0Attr().Get(),
        "local_rot0": joint.GetLocalRot0Attr().Get(),
        "local_pos1": joint.GetLocalPos1Attr().Get(),
        "local_rot1": joint.GetLocalRot1Attr().Get(),
        "body0": [str(x) for x in joint.GetBody0Rel().GetTargets()],
        "body1": [str(x) for x in joint.GetBody1Rel().GetTargets()],
    }


env = None
failed = False

try:
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1

    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    u = env.unwrapped
    robot = u.scene["robot"]

    stage = get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage unavailable")

    joint_names = list(robot.data.joint_names)
    if len(joint_names) != 37:
        raise RuntimeError(f"expected 37 joints, got {len(joint_names)}")

    joint_index = {name: i for i, name in enumerate(joint_names)}

    pairs = []
    for name in joint_names:
        if not name.startswith("left_"):
            continue
        other = counterpart(name)
        if other is None or other not in joint_index:
            raise RuntimeError(f"missing counterpart for {name}: {other}")
        pairs.append((name, joint_index[name], other, joint_index[other]))

    if len(pairs) != 18:
        raise RuntimeError(f"expected 18 L/R pairs, got {len(pairs)}")

    default_q = robot.data.default_joint_pos[0].detach().cpu().numpy()

    terms = getattr(u.action_manager, "_terms", None)
    if not isinstance(terms, dict) or "joint_pos" not in terms:
        raise RuntimeError("live action term 'joint_pos' not found")

    action_term = terms["joint_pos"]
    action_scale = getattr(action_term, "_scale", None)
    action_offset = getattr(action_term, "_offset", None)

    p()
    p("=" * 154)
    p("v4-B G1 JOINT MIRROR GEOMETRY AUDIT")
    p("=" * 154)
    p(f"Task              : {TASK}")
    p(f"device            : {u.device}")
    p(f"joint count       : {len(joint_names)}")
    p(f"L/R pairs         : {len(pairs)}")
    p(f"observation space : {u.observation_space}")
    p(f"action space      : {u.action_space}")
    p()

    p("-" * 154)
    p("1) ACTION PARAMETERIZATION")
    p("-" * 154)

    for name, value in (("_scale", action_scale), ("_offset", action_offset)):
        if value is None:
            p(f"{name}: None")
        elif hasattr(value, "detach"):
            arr = value.detach().cpu().reshape(-1)
            p(f"{name}: shape={tuple(value.shape)}")
            if arr.numel() <= 40:
                p(f"{name}: {arr.tolist()}")
        else:
            p(f"{name}: {value}")

    p()
    p("-" * 154)
    p("2) OBSERVATION TERM ORDER")
    p("-" * 154)
    active_terms = u.observation_manager.active_terms["policy"]
    for i, term_name in enumerate(active_terms):
        p(f"policy term[{i}] = {term_name}")

    p("Expected frozen v4-B layout:")
    p("  [  0:  3] base_lin_vel")
    p("  [  3:  6] base_ang_vel")
    p("  [  6:  9] projected_gravity")
    p("  [  9: 12] velocity_commands")
    p("  [ 12: 49] joint_pos")
    p("  [ 49: 86] joint_vel")
    p("  [ 86:123] actions")
    p()

    p("-" * 154)
    p("3) LEFT/RIGHT REVOLUTE-JOINT USD GEOMETRY")
    p("-" * 154)

    for left, li, right, ri in pairs:
        lm = get_joint_metadata(stage, left)
        rm = get_joint_metadata(stage, right)

        p()
        p(f"PAIR: {left} [{li:02d}]  <->  {right} [{ri:02d}]")
        p(
            f"default q : L={float(default_q[li]):+.6f}  "
            f"R={float(default_q[ri]):+.6f}"
        )
        p(
            f"limits    : L=[{lm['lower']},{lm['upper']}]  "
            f"R=[{rm['lower']},{rm['upper']}]"
        )
        p(f"axis      : L={fmt(lm['axis'])}  R={fmt(rm['axis'])}")
        p(f"L prim    : {lm['prim_path']}")
        p(f"R prim    : {rm['prim_path']}")
        p(f"body0     : L={lm['body0']}  R={rm['body0']}")
        p(f"body1     : L={lm['body1']}  R={rm['body1']}")
        p(f"localPos0 : L={fmt(lm['local_pos0'])}  R={fmt(rm['local_pos0'])}")
        p(f"localRot0 : L={fmt(lm['local_rot0'])}  R={fmt(rm['local_rot0'])}")
        p(f"localPos1 : L={fmt(lm['local_pos1'])}  R={fmt(rm['local_pos1'])}")
        p(f"localRot1 : L={fmt(lm['local_rot1'])}  R={fmt(rm['local_rot1'])}")

    p()
    p("-" * 154)
    p("SUMMARY")
    p("-" * 154)
    p(f"joint count                 : {len(joint_names)}")
    p(f"left/right pairs            : {len(pairs)}")
    p("USD RevoluteJoint metadata : PASS for all pairs")
    p()
    p(
        "This audit intentionally does NOT infer mirror signs yet. "
        "It only reports the loaded joint geometry needed to derive them safely."
    )
    p()
    p("V4B_G1_JOINT_MIRROR_GEOMETRY_AUDIT_PASS")
    p("=" * 154)

except Exception:
    failed = True
    p()
    p("=" * 154)
    p("V4B_G1_JOINT_MIRROR_GEOMETRY_AUDIT_FAIL")
    p("=" * 154)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()

finally:
    if env is not None:
        try:
            env.close()
        except Exception:
            traceback.print_exc(file=sys.stdout)

    try:
        simulation_app.close()
    except Exception:
        traceback.print_exc(file=sys.stdout)

if failed:
    raise SystemExit(1)
