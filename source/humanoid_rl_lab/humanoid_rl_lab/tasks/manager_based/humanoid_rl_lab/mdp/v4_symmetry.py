"""Native RSL-RL left/right symmetry augmentation for the project Unitree G1 task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


__all__ = ["compute_g1_left_right_symmetric_states"]


_EXPECTED_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "torso_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_elbow_roll_joint",
    "right_elbow_roll_joint",
    "left_five_joint",
    "left_three_joint",
    "left_zero_joint",
    "right_five_joint",
    "right_three_joint",
    "right_zero_joint",
    "left_six_joint",
    "left_four_joint",
    "left_one_joint",
    "right_six_joint",
    "right_four_joint",
    "right_one_joint",
    "left_two_joint",
    "right_two_joint",
)

_JOINT_PERMUTATION = (
    1, 0, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17,
    20, 19, 22, 21, 26, 27, 28, 23, 24, 25, 32, 33, 34, 29, 30, 31, 36, 35,
)

_JOINT_SIGNS = (
    +1.0, +1.0, -1.0, -1.0, -1.0, +1.0, +1.0, -1.0, -1.0, -1.0,
    -1.0, +1.0, +1.0, -1.0, -1.0, +1.0, +1.0, +1.0, +1.0, -1.0,
    -1.0, -1.0, -1.0, -1.0, -1.0, +1.0, -1.0, -1.0, +1.0, -1.0,
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
)


def _validate_runtime_layout(env: ManagerBasedRLEnv) -> None:
    joint_names = tuple(env.scene["robot"].data.joint_names)
    if joint_names != _EXPECTED_JOINT_NAMES:
        raise RuntimeError(
            "G1 symmetry augmentation joint-order mismatch.\n"
            f"Expected: {_EXPECTED_JOINT_NAMES}\n"
            f"Actual  : {joint_names}"
        )

    policy_terms = tuple(env.observation_manager.active_terms["policy"])
    expected_terms = (
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
    )
    if policy_terms != expected_terms:
        raise RuntimeError(
            "G1 symmetry augmentation policy-observation layout mismatch.\n"
            f"Expected: {expected_terms}\n"
            f"Actual  : {policy_terms}"
        )


def _mirror_joint_data(data: torch.Tensor) -> torch.Tensor:
    if data.shape[-1] != 37:
        raise RuntimeError(
            f"G1 symmetry expected joint/action dimension 37, got {data.shape[-1]}"
        )

    permutation = torch.tensor(
        _JOINT_PERMUTATION,
        dtype=torch.long,
        device=data.device,
    )
    signs = data.new_tensor(_JOINT_SIGNS)
    return data.index_select(-1, permutation) * signs


def _mirror_policy_observation(obs: torch.Tensor) -> torch.Tensor:
    if obs.shape[-1] != 123:
        raise RuntimeError(
            f"G1 symmetry expected policy observation dimension 123, got {obs.shape[-1]}"
        )

    mirrored = obs.clone()

    mirrored[..., 0:3] = obs[..., 0:3] * obs.new_tensor((+1.0, -1.0, +1.0))
    mirrored[..., 3:6] = obs[..., 3:6] * obs.new_tensor((-1.0, +1.0, -1.0))
    mirrored[..., 6:9] = obs[..., 6:9] * obs.new_tensor((+1.0, -1.0, +1.0))
    mirrored[..., 9:12] = obs[..., 9:12] * obs.new_tensor((+1.0, -1.0, -1.0))

    mirrored[..., 12:49] = _mirror_joint_data(obs[..., 12:49])
    mirrored[..., 49:86] = _mirror_joint_data(obs[..., 49:86])
    mirrored[..., 86:123] = _mirror_joint_data(obs[..., 86:123])

    return mirrored


@torch.no_grad()
def compute_g1_left_right_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Return original + left/right mirrored observations/actions for native RSL-RL augmentation."""

    _validate_runtime_layout(env.unwrapped)

    if obs is not None:
        if "policy" not in obs.keys():
            raise RuntimeError(
                "G1 symmetry augmentation expected TensorDict key 'policy'"
            )

        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"]
        obs_aug["policy"][batch_size:] = _mirror_policy_observation(obs["policy"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.empty(
            (batch_size * 2, actions.shape[1]),
            dtype=actions.dtype,
            device=actions.device,
        )
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = _mirror_joint_data(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug
