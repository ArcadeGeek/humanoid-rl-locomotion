from __future__ import annotations

import sys
import traceback

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import humanoid_rl_lab.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from omni.usd import get_context  # noqa: E402
from pxr import UsdPhysics  # noqa: E402


TASK = "Humanoid-RL-Lab-G1-Flat-V4B-Direct-Yaw-v0"
OBS_DIM = 123
ACT_DIM = 37

# Reflection y -> -y:
# polar vector     : [ x, -y,  z]
# axial vector     : [-x,  y, -z]
AXIS_SIGN = {
    "X": -1.0,
    "Y": +1.0,
    "Z": -1.0,
}


def p(*args):
    print(*args, flush=True)


def counterpart(name: str) -> str | None:
    if name.startswith("left_"):
        return name.replace("left_", "right_", 1)
    if name.startswith("right_"):
        return name.replace("right_", "left_", 1)
    return None


def find_revolute_joint(stage, joint_name: str):
    matches = [
        prim
        for prim in stage.Traverse()
        if prim.GetName() == joint_name and prim.IsA(UsdPhysics.RevoluteJoint)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{joint_name}: expected exactly one RevoluteJoint prim, found {len(matches)}"
        )
    return UsdPhysics.RevoluteJoint(matches[0])


def axis_name(joint: UsdPhysics.RevoluteJoint) -> str:
    raw = joint.GetAxisAttr().Get()
    value = str(raw)
    if value not in AXIS_SIGN:
        raise RuntimeError(f"unsupported revolute axis {raw!r}")
    return value


def mirror_joint_tensor(
    x: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    return x.index_select(-1, permutation) * signs


def mirror_policy_obs(
    obs: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    if obs.shape[-1] != OBS_DIM:
        raise RuntimeError(
            f"expected obs last dim {OBS_DIM}, got {obs.shape[-1]}"
        )

    out = obs.clone()

    # polar base linear velocity
    out[..., 0:3] = obs[..., 0:3] * obs.new_tensor([+1.0, -1.0, +1.0])
    # axial base angular velocity
    out[..., 3:6] = obs[..., 3:6] * obs.new_tensor([-1.0, +1.0, -1.0])
    # polar projected gravity
    out[..., 6:9] = obs[..., 6:9] * obs.new_tensor([+1.0, -1.0, +1.0])
    # command [vx, vy, wz]
    out[..., 9:12] = obs[..., 9:12] * obs.new_tensor([+1.0, -1.0, -1.0])

    out[..., 12:49] = mirror_joint_tensor(obs[..., 12:49], permutation, signs)
    out[..., 49:86] = mirror_joint_tensor(obs[..., 49:86], permutation, signs)
    out[..., 86:123] = mirror_joint_tensor(obs[..., 86:123], permutation, signs)

    return out


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
    if len(joint_names) != ACT_DIM:
        raise RuntimeError(
            f"expected {ACT_DIM} joints, got {len(joint_names)}"
        )

    joint_index = {name: i for i, name in enumerate(joint_names)}

    permutation = list(range(ACT_DIM))
    signs = [None] * ACT_DIM
    axes = [None] * ACT_DIM

    for name, index in joint_index.items():
        joint = find_revolute_joint(stage, name)
        axis = axis_name(joint)
        axes[index] = axis
        signs[index] = AXIS_SIGN[axis]

        other = counterpart(name)
        if other is not None:
            if other not in joint_index:
                raise RuntimeError(
                    f"missing counterpart for {name}: {other}"
                )
            permutation[index] = joint_index[other]
        else:
            # Central unpaired joint(s), e.g. torso_joint, mirror to themselves.
            permutation[index] = index

    if any(value is None for value in signs):
        raise RuntimeError("failed to derive complete sign vector")
    if any(value is None for value in axes):
        raise RuntimeError("failed to derive complete axis vector")

    perm_t = torch.tensor(
        permutation,
        dtype=torch.long,
        device=u.device,
    )
    sign_t = torch.tensor(
        signs,
        dtype=torch.float32,
        device=u.device,
    )

    # Involution structure.
    perm_twice = perm_t.index_select(0, perm_t)
    identity = torch.arange(ACT_DIM, device=u.device)

    if not torch.equal(perm_twice, identity):
        raise RuntimeError(
            f"permutation is not an involution: {perm_twice.tolist()}"
        )

    sign_twice = sign_t * sign_t.index_select(0, perm_t)
    if not torch.allclose(
        sign_twice,
        torch.ones_like(sign_twice),
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"sign map is not an involution: {sign_twice.tolist()}"
        )

    # Runtime default pose and action offsets.
    default_q = robot.data.default_joint_pos[0].detach().to(u.device)

    action_terms = getattr(u.action_manager, "_terms", None)
    if not isinstance(action_terms, dict) or "joint_pos" not in action_terms:
        raise RuntimeError("live action term 'joint_pos' not found")

    action_term = action_terms["joint_pos"]
    scale = getattr(action_term, "_scale", None)
    offset = getattr(action_term, "_offset", None)

    if float(scale) != 0.5:
        raise RuntimeError(f"unexpected action scale: {scale!r}")

    if offset is None or tuple(offset.shape) != (1, ACT_DIM):
        raise RuntimeError(
            f"unexpected action offset shape: {getattr(offset, 'shape', None)}"
        )

    offset_vec = offset[0].detach().to(u.device)

    mirrored_default = mirror_joint_tensor(
        default_q.unsqueeze(0),
        perm_t,
        sign_t,
    )[0]

    mirrored_offset = mirror_joint_tensor(
        offset_vec.unsqueeze(0),
        perm_t,
        sign_t,
    )[0]

    default_err = float(
        torch.max(torch.abs(mirrored_default - default_q)).item()
    )
    offset_err = float(
        torch.max(torch.abs(mirrored_offset - offset_vec)).item()
    )

    if default_err > 1.0e-6:
        raise RuntimeError(
            f"default pose is not mirror invariant: max error={default_err}"
        )

    if offset_err > 1.0e-6:
        raise RuntimeError(
            f"action offset is not mirror invariant: max error={offset_err}"
        )

    # Joint-limit consistency using live soft limits (radians).
    limits = robot.data.soft_joint_pos_limits[0].detach().to(u.device)
    limit_checks = []

    for i, name in enumerate(joint_names):
        j = permutation[i]
        s = signs[i]

        lo_i = float(limits[i, 0].item())
        hi_i = float(limits[i, 1].item())
        lo_j = float(limits[j, 0].item())
        hi_j = float(limits[j, 1].item())

        if s > 0:
            expected_lo = lo_j
            expected_hi = hi_j
        else:
            expected_lo = -hi_j
            expected_hi = -lo_j

        err = max(
            abs(lo_i - expected_lo),
            abs(hi_i - expected_hi),
        )

        limit_checks.append((name, err))

    max_limit_err = max(err for _, err in limit_checks)

    if max_limit_err > 2.0e-5:
        bad = [
            (name, err)
            for name, err in limit_checks
            if err > 2.0e-5
        ]
        raise RuntimeError(
            f"joint-limit mirror mismatch: {bad}"
        )

    # Pure algebraic involution tests.
    generator = torch.Generator(device=u.device)
    generator.manual_seed(123456)

    random_actions = torch.randn(
        (32, ACT_DIM),
        generator=generator,
        device=u.device,
    )
    random_obs = torch.randn(
        (32, OBS_DIM),
        generator=generator,
        device=u.device,
    )

    mirrored_actions = mirror_joint_tensor(
        random_actions,
        perm_t,
        sign_t,
    )
    twice_actions = mirror_joint_tensor(
        mirrored_actions,
        perm_t,
        sign_t,
    )

    action_involution_err = float(
        torch.max(torch.abs(twice_actions - random_actions)).item()
    )

    mirrored_obs = mirror_policy_obs(
        random_obs,
        perm_t,
        sign_t,
    )
    twice_obs = mirror_policy_obs(
        mirrored_obs,
        perm_t,
        sign_t,
    )

    obs_involution_err = float(
        torch.max(torch.abs(twice_obs - random_obs)).item()
    )

    if action_involution_err != 0.0:
        raise RuntimeError(
            f"action mirror involution failed: {action_involution_err}"
        )

    if obs_involution_err != 0.0:
        raise RuntimeError(
            f"obs mirror involution failed: {obs_involution_err}"
        )

    # Action-target consistency:
    # target = default_offset + scale * action.
    # Mirroring normalized action must equal mirroring the resulting joint target.
    target = offset_vec.unsqueeze(0) + 0.5 * random_actions
    mirrored_target_direct = mirror_joint_tensor(
        target,
        perm_t,
        sign_t,
    )
    mirrored_target_via_action = (
        offset_vec.unsqueeze(0)
        + 0.5 * mirrored_actions
    )

    target_err = float(
        torch.max(
            torch.abs(
                mirrored_target_direct
                - mirrored_target_via_action
            )
        ).item()
    )

    if target_err > 1.0e-6:
        raise RuntimeError(
            f"action-target mirror consistency failed: {target_err}"
        )

    p()
    p("=" * 150)
    p("v4-B G1 MIRROR TRANSFORM VALIDATION")
    p("=" * 150)
    p(f"Task                 : {TASK}")
    p(f"Obs / action dim     : {OBS_DIM} / {ACT_DIM}")
    p("Reflection           : y -> -y")
    p("Joint sign rule      : X -> -, Y -> +, Z -> -")
    p()

    p("-" * 150)
    p("DERIVED JOINT PERMUTATION + SIGNS")
    p("-" * 150)
    p("idx | joint name                     | axis | sign | source idx/name after mirror")

    for i, name in enumerate(joint_names):
        j = permutation[i]
        p(
            f"{i:02d}  | {name:<30s} | "
            f"{axes[i]:>4s} | {int(signs[i]):+d}   | "
            f"{j:02d} {joint_names[j]}"
        )

    p()
    torso_index = joint_index.get("torso_joint")
    if torso_index is None:
        raise RuntimeError("torso_joint not found")

    p("-" * 150)
    p("TORSO CLOSURE")
    p("-" * 150)
    p(
        f"torso index / axis / sign : "
        f"{torso_index} / {axes[torso_index]} / {int(signs[torso_index]):+d}"
    )
    p(
        f"torso default / offset    : "
        f"{float(default_q[torso_index]):+.6f} / "
        f"{float(offset_vec[torso_index]):+.6f}"
    )

    p()
    p("-" * 150)
    p("VALIDATION SUMMARY")
    p("-" * 150)
    p(f"permutation involution              : PASS")
    p(f"sign involution                     : PASS")
    p(f"default-pose mirror max error        : {default_err:.3e}")
    p(f"action-offset mirror max error       : {offset_err:.3e}")
    p(f"joint-limit mirror max error         : {max_limit_err:.3e}")
    p(f"action double-mirror max error       : {action_involution_err:.3e}")
    p(f"obs123 double-mirror max error       : {obs_involution_err:.3e}")
    p(f"action-target mirror max error       : {target_err:.3e}")
    p()
    p("Observation transform:")
    p("  base_lin_vel       [x,y,z] -> [ x,-y, z]")
    p("  base_ang_vel       [x,y,z] -> [-x, y,-z]")
    p("  projected_gravity  [x,y,z] -> [ x,-y, z]")
    p("  command [vx,vy,wz]         -> [vx,-vy,-wz]")
    p("  joint_pos / joint_vel / actions use derived permutation + signs")
    p()
    p("V4B_G1_MIRROR_TRANSFORM_VALIDATION_PASS")
    p("=" * 150)

except Exception:
    failed = True
    p()
    p("=" * 150)
    p("V4B_G1_MIRROR_TRANSFORM_VALIDATION_FAIL")
    p("=" * 150)
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
