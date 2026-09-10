from __future__ import annotations

import sys
import traceback

import gymnasium as gym
import torch
from tensordict import TensorDict

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import humanoid_rl_lab.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402


TASK = "Humanoid-RL-Lab-G1-Flat-V4N-Symmetry-Augmentation-v0"
NUM_ENVS = 4
OBS_DIM = 123
ACT_DIM = 37

EXPECTED_PERMUTATION = (
    1, 0, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17,
    20, 19, 22, 21, 26, 27, 28, 23, 24, 25, 32, 33, 34, 29, 30, 31, 36, 35,
)

EXPECTED_SIGNS = (
    +1.0, +1.0, -1.0, -1.0, -1.0, +1.0, +1.0, -1.0, -1.0, -1.0,
    -1.0, +1.0, +1.0, -1.0, -1.0, +1.0, +1.0, +1.0, +1.0, -1.0,
    -1.0, -1.0, -1.0, -1.0, -1.0, +1.0, -1.0, -1.0, +1.0, -1.0,
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
)


def p(*args):
    print(*args, flush=True)


def independent_mirror_joint_data(x: torch.Tensor) -> torch.Tensor:
    permutation = torch.tensor(
        EXPECTED_PERMUTATION,
        dtype=torch.long,
        device=x.device,
    )
    signs = x.new_tensor(EXPECTED_SIGNS)
    return x.index_select(-1, permutation) * signs


def independent_mirror_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.shape[-1] != OBS_DIM:
        raise RuntimeError(
            f"independent mirror expected obs dim {OBS_DIM}, got {obs.shape[-1]}"
        )

    out = obs.clone()
    out[..., 0:3] = obs[..., 0:3] * obs.new_tensor((+1.0, -1.0, +1.0))
    out[..., 3:6] = obs[..., 3:6] * obs.new_tensor((-1.0, +1.0, -1.0))
    out[..., 6:9] = obs[..., 6:9] * obs.new_tensor((+1.0, -1.0, +1.0))
    out[..., 9:12] = obs[..., 9:12] * obs.new_tensor((+1.0, -1.0, -1.0))
    out[..., 12:49] = independent_mirror_joint_data(obs[..., 12:49])
    out[..., 49:86] = independent_mirror_joint_data(obs[..., 49:86])
    out[..., 86:123] = independent_mirror_joint_data(obs[..., 86:123])
    return out


env = None
failed = False

try:
    env_cfg = load_cfg_from_registry(
        TASK,
        "env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = NUM_ENVS

    agent_cfg = load_cfg_from_registry(
        TASK,
        "rsl_rl_cfg_entry_point",
    )

    symmetry_cfg = agent_cfg.algorithm.symmetry_cfg

    if symmetry_cfg is None:
        raise RuntimeError("resolved symmetry_cfg is None")
    if symmetry_cfg.use_data_augmentation is not True:
        raise RuntimeError("use_data_augmentation is not True")
    if symmetry_cfg.use_mirror_loss is not False:
        raise RuntimeError("use_mirror_loss is not False")
    if float(symmetry_cfg.mirror_loss_coeff) != 0.0:
        raise RuntimeError(
            f"mirror_loss_coeff != 0.0: {symmetry_cfg.mirror_loss_coeff}"
        )

    symmetry_func = symmetry_cfg.data_augmentation_func

    if getattr(symmetry_func, "__name__", None) != "compute_g1_left_right_symmetric_states":
        raise RuntimeError(
            f"unexpected symmetry function: {symmetry_func}"
        )

    env = gym.make(
        TASK,
        cfg=env_cfg,
        render_mode=None,
    )
    u = env.unwrapped

    device = torch.device(u.device)

    if tuple(u.observation_space["policy"].shape)[-1] != OBS_DIM:
        raise RuntimeError(
            f"unexpected policy observation dim: {u.observation_space}"
        )

    if tuple(u.action_space.shape)[-1] != ACT_DIM:
        raise RuntimeError(
            f"unexpected action dim: {u.action_space}"
        )

    p()
    p("=" * 136)
    p("v4-N RUNTIME SYMMETRY INTEGRATION SMOKE")
    p("=" * 136)
    p(f"Task                    : {TASK}")
    p(f"device                  : {device}")
    p(f"num envs                : {NUM_ENVS}")
    p(f"policy obs dim          : {OBS_DIM}")
    p(f"action dim              : {ACT_DIM}")
    p(f"symmetry function       : {symmetry_func.__module__}.{symmetry_func.__name__}")
    p(f"use_data_augmentation   : {symmetry_cfg.use_data_augmentation}")
    p(f"use_mirror_loss         : {symmetry_cfg.use_mirror_loss}")
    p(f"mirror_loss_coeff        : {symmetry_cfg.mirror_loss_coeff}")
    p()

    # Synthetic inputs isolate the symmetry transform from policy/simulator state.
    # Runtime env is still real, so the function's layout guards execute against the loaded G1 task.
    generator = torch.Generator(device=device)
    generator.manual_seed(20260904)

    obs_tensor = torch.randn(
        (NUM_ENVS, OBS_DIM),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    actions = torch.randn(
        (NUM_ENVS, ACT_DIM),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    obs_td = TensorDict(
        {"policy": obs_tensor.clone()},
        batch_size=[NUM_ENVS],
        device=device,
    )

    obs_aug, actions_aug = symmetry_func(
        env,
        obs=obs_td,
        actions=actions,
    )

    if obs_aug is None or actions_aug is None:
        raise RuntimeError("symmetry function returned None unexpectedly")

    aug_policy = obs_aug["policy"]

    if tuple(aug_policy.shape) != (NUM_ENVS * 2, OBS_DIM):
        raise RuntimeError(
            f"unexpected augmented obs shape: {tuple(aug_policy.shape)}"
        )
    if tuple(actions_aug.shape) != (NUM_ENVS * 2, ACT_DIM):
        raise RuntimeError(
            f"unexpected augmented action shape: {tuple(actions_aug.shape)}"
        )

    # Original half must be exact.
    original_obs_err = float(
        torch.max(
            torch.abs(
                aug_policy[:NUM_ENVS]
                - obs_tensor
            )
        ).item()
    )
    original_action_err = float(
        torch.max(
            torch.abs(
                actions_aug[:NUM_ENVS]
                - actions
            )
        ).item()
    )

    if original_obs_err != 0.0:
        raise RuntimeError(
            f"original obs half changed: max error={original_obs_err}"
        )
    if original_action_err != 0.0:
        raise RuntimeError(
            f"original action half changed: max error={original_action_err}"
        )

    expected_obs_mirror = independent_mirror_obs(obs_tensor)
    expected_action_mirror = independent_mirror_joint_data(actions)

    mirrored_obs_err = float(
        torch.max(
            torch.abs(
                aug_policy[NUM_ENVS:]
                - expected_obs_mirror
            )
        ).item()
    )
    mirrored_action_err = float(
        torch.max(
            torch.abs(
                actions_aug[NUM_ENVS:]
                - expected_action_mirror
            )
        ).item()
    )

    if mirrored_obs_err != 0.0:
        raise RuntimeError(
            f"runtime mirrored obs differs from independent transform: {mirrored_obs_err}"
        )
    if mirrored_action_err != 0.0:
        raise RuntimeError(
            f"runtime mirrored actions differ from independent transform: {mirrored_action_err}"
        )

    # Independent involution closure.
    twice_obs = independent_mirror_obs(expected_obs_mirror)
    twice_actions = independent_mirror_joint_data(expected_action_mirror)

    obs_involution_err = float(
        torch.max(torch.abs(twice_obs - obs_tensor)).item()
    )
    action_involution_err = float(
        torch.max(torch.abs(twice_actions - actions)).item()
    )

    if obs_involution_err != 0.0:
        raise RuntimeError(
            f"obs involution failed: {obs_involution_err}"
        )
    if action_involution_err != 0.0:
        raise RuntimeError(
            f"action involution failed: {action_involution_err}"
        )

    # Exercise None branches required by native RSL-RL symmetry API.
    obs_only, no_actions = symmetry_func(
        env,
        obs=obs_td,
        actions=None,
    )
    no_obs, actions_only = symmetry_func(
        env,
        obs=None,
        actions=actions,
    )

    if obs_only is None or no_actions is not None:
        raise RuntimeError("obs-only symmetry branch failed")
    if no_obs is not None or actions_only is None:
        raise RuntimeError("action-only symmetry branch failed")

    if tuple(obs_only["policy"].shape) != (NUM_ENVS * 2, OBS_DIM):
        raise RuntimeError(
            f"obs-only branch wrong shape: {tuple(obs_only['policy'].shape)}"
        )
    if tuple(actions_only.shape) != (NUM_ENVS * 2, ACT_DIM):
        raise RuntimeError(
            f"action-only branch wrong shape: {tuple(actions_only.shape)}"
        )

    p("-" * 136)
    p("RUNTIME RESULTS")
    p("-" * 136)
    p(f"augmented obs shape                : {tuple(aug_policy.shape)}")
    p(f"augmented action shape             : {tuple(actions_aug.shape)}")
    p(f"original obs max error             : {original_obs_err:.3e}")
    p(f"original action max error          : {original_action_err:.3e}")
    p(f"independent mirrored obs max error : {mirrored_obs_err:.3e}")
    p(f"independent mirrored act max error : {mirrored_action_err:.3e}")
    p(f"obs double-mirror max error        : {obs_involution_err:.3e}")
    p(f"act double-mirror max error        : {action_involution_err:.3e}")
    p("obs-only API branch                : PASS")
    p("action-only API branch             : PASS")
    p()
    p("V4N_RUNTIME_SYMMETRY_SMOKE_PASS")
    p("=" * 136)

except Exception:
    failed = True
    p()
    p("=" * 136)
    p("V4N_RUNTIME_SYMMETRY_SMOKE_FAIL")
    p("=" * 136)
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
