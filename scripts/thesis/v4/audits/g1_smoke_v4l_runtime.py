from __future__ import annotations

import argparse
import math


TASK_ID = "Humanoid-RL-Lab-G1-Flat-V4L-Path-Pose-Precision-v0"
REWARD_NAME = "geometric_command_path_pose_bounded"

NUM_ENVS = 4
SEED = 42

EXPECTED_STEP_DT = 0.02
EXPECTED_WEIGHT = -0.15
EXPECTED_SIGMA_CROSS_M = 0.20
EXPECTED_SIGMA_HEADING_RAD = math.radians(5.0)

LATERAL_TEST_OFFSET_M = 0.10
HEADING_TEST_OFFSET_RAD = math.radians(5.0)


def fail(message: str) -> None:
    raise RuntimeError(message)


parser = argparse.ArgumentParser(
    description="Runtime smoke test for v4-L path-pose reward."
)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

raw_env = None

try:
    import gymnasium as gym
    import torch

    import humanoid_rl_lab.tasks  # noqa: F401
    from isaaclab.managers import ManagerTermBase
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=NUM_ENVS,
    )
    env_cfg.seed = SEED

    raw_env = gym.make(
        TASK_ID,
        cfg=env_cfg,
    )
    env = raw_env.unwrapped

    if env.num_envs != NUM_ENVS:
        fail(
            f"unexpected num_envs: {env.num_envs}"
        )

    if abs(
        float(env.step_dt)
        - EXPECTED_STEP_DT
    ) > 1.0e-12:
        fail(
            f"unexpected step_dt: {env.step_dt}"
        )

    if REWARD_NAME not in env.reward_manager.active_terms:
        fail(
            f"reward term not active: {REWARD_NAME}"
        )

    reward_cfg = env.reward_manager.get_term_cfg(
        REWARD_NAME
    )
    reward_term = reward_cfg.func

    if not isinstance(
        reward_term,
        ManagerTermBase,
    ):
        fail(
            "reward term is not ManagerTermBase: "
            f"{type(reward_term)}"
        )

    if (
        reward_term.__class__.__name__
        != REWARD_NAME
    ):
        fail(
            "unexpected reward class: "
            f"{reward_term.__class__.__name__}"
        )

    if abs(
        float(reward_cfg.weight)
        - EXPECTED_WEIGHT
    ) > 1.0e-12:
        fail(
            "unexpected reward weight: "
            f"{reward_cfg.weight}"
        )

    if abs(
        float(
            reward_cfg.params[
                "sigma_cross_m"
            ]
        )
        - EXPECTED_SIGMA_CROSS_M
    ) > 1.0e-12:
        fail(
            "unexpected sigma_cross_m: "
            f"{reward_cfg.params['sigma_cross_m']}"
        )

    if abs(
        float(
            reward_cfg.params[
                "sigma_heading_rad"
            ]
        )
        - EXPECTED_SIGMA_HEADING_RAD
    ) > 1.0e-12:
        fail(
            "unexpected sigma_heading_rad: "
            f"{reward_cfg.params['sigma_heading_rad']}"
        )

    expected_state_shapes = {
        "_anchor_pos_xy": (NUM_ENVS, 2),
        "_anchor_heading": (NUM_ENVS,),
        "_stored_command": (NUM_ENVS, 3),
        "_elapsed_s": (NUM_ENVS,),
        "_was_active": (NUM_ENVS,),
    }

    for name, expected_shape in expected_state_shapes.items():
        if not hasattr(
            reward_term,
            name,
        ):
            fail(
                f"reward term missing runtime state {name}"
            )

        tensor = getattr(
            reward_term,
            name,
        )

        if tuple(tensor.shape) != expected_shape:
            fail(
                f"unexpected {name} shape: "
                f"{tuple(tensor.shape)} "
                f"expected {expected_shape}"
            )

        if tensor.device != torch.device(env.device):
            fail(
                f"{name} device mismatch: "
                f"{tensor.device} vs {env.device}"
            )

    # ----------------------------------------------------------------------------------------------
    # Test 1: RewardManager selective reset propagation.
    # ----------------------------------------------------------------------------------------------
    reward_term._anchor_pos_xy.fill_(1.0)
    reward_term._anchor_heading.fill_(1.0)
    reward_term._stored_command.fill_(1.0)
    reward_term._elapsed_s.fill_(1.0)
    reward_term._was_active.fill_(True)

    reset_ids = torch.tensor(
        [0, 2],
        device=env.device,
        dtype=torch.long,
    )
    keep_ids = torch.tensor(
        [1, 3],
        device=env.device,
        dtype=torch.long,
    )

    env.reward_manager.reset(
        env_ids=reset_ids
    )

    for name in (
        "_anchor_pos_xy",
        "_anchor_heading",
        "_stored_command",
        "_elapsed_s",
    ):
        tensor = getattr(
            reward_term,
            name,
        )

        if not torch.allclose(
            tensor[reset_ids],
            torch.zeros_like(
                tensor[reset_ids]
            ),
            rtol=0.0,
            atol=0.0,
        ):
            fail(
                f"selective reset did not zero {name}"
            )

        if not torch.allclose(
            tensor[keep_ids],
            torch.ones_like(
                tensor[keep_ids]
            ),
            rtol=0.0,
            atol=0.0,
        ):
            fail(
                f"selective reset altered kept {name}"
            )

    if torch.any(
        reward_term._was_active[
            reset_ids
        ]
    ):
        fail(
            "selective reset did not clear requested _was_active"
        )

    if not torch.all(
        reward_term._was_active[
            keep_ids
        ]
    ):
        fail(
            "selective reset altered kept _was_active"
        )

    reward_term.reset()

    # Physical environment reset before direct reward evaluations.
    raw_env.reset(
        seed=SEED
    )

    command = env.command_manager.get_command(
        "base_velocity"
    )

    if (
        command.shape[0] != NUM_ENVS
        or command.shape[1] < 3
    ):
        fail(
            "unexpected command-buffer shape: "
            f"{tuple(command.shape)}"
        )

    asset_cfg = reward_cfg.params[
        "asset_cfg"
    ]
    asset = env.scene[
        asset_cfg.name
    ]

    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w

    if hasattr(
        root_pos_w,
        "torch",
    ):
        root_pos_w = root_pos_w.torch

    if hasattr(
        root_quat_w,
        "torch",
    ):
        root_quat_w = root_quat_w.torch

    pos_xy = root_pos_w[:, :2]
    heading = reward_term._heading_from_wxyz(
        root_quat_w
    )

    command.zero_()
    command[:, 0] = 0.40

    # ----------------------------------------------------------------------------------------------
    # Test 2: exact straight path + exact heading => zero cost.
    # ----------------------------------------------------------------------------------------------
    reward_term._anchor_pos_xy[:] = pos_xy
    reward_term._anchor_heading[:] = heading
    reward_term._stored_command[:] = command[:, :3]
    reward_term._elapsed_s.zero_()
    reward_term._was_active.fill_(True)

    zero_cost = reward_term(
        env,
        **reward_cfg.params,
    )

    if zero_cost.shape != (NUM_ENVS,):
        fail(
            "unexpected zero-cost shape: "
            f"{tuple(zero_cost.shape)}"
        )

    if not torch.isfinite(
        zero_cost
    ).all():
        fail(
            f"non-finite zero cost: {zero_cost}"
        )

    if not torch.allclose(
        zero_cost,
        torch.zeros_like(
            zero_cost
        ),
        rtol=0.0,
        atol=1.0e-7,
    ):
        fail(
            "straight/on-heading state is not zero cost: "
            f"{zero_cost.tolist()}"
        )

    # ----------------------------------------------------------------------------------------------
    # Test 3: +10 cm lateral offset, zero heading error.
    # Manipulate reward reference state only; simulator state is unchanged.
    # ----------------------------------------------------------------------------------------------
    lateral_axis = torch.stack(
        (
            -torch.sin(heading),
            torch.cos(heading),
        ),
        dim=1,
    )

    reward_term._anchor_pos_xy[:] = (
        pos_xy
        - LATERAL_TEST_OFFSET_M
        * lateral_axis
    )
    reward_term._anchor_heading[:] = heading
    reward_term._stored_command[:] = command[:, :3]
    reward_term._elapsed_s.zero_()
    reward_term._was_active.fill_(True)

    lateral_cost = reward_term(
        env,
        **reward_cfg.params,
    )

    expected_lateral_cost = (
        1.0
        - math.exp(
            -(
                LATERAL_TEST_OFFSET_M
                / EXPECTED_SIGMA_CROSS_M
            )
            ** 2
        )
    )

    expected_lateral_tensor = torch.full(
        (NUM_ENVS,),
        expected_lateral_cost,
        device=env.device,
        dtype=lateral_cost.dtype,
    )

    if not torch.allclose(
        lateral_cost,
        expected_lateral_tensor,
        rtol=0.0,
        atol=2.0e-6,
    ):
        fail(
            "10 cm lateral-only cost mismatch: "
            f"actual={lateral_cost.tolist()} "
            f"expected={expected_lateral_cost:.8f}"
        )

    # ----------------------------------------------------------------------------------------------
    # Test 4: zero cross-track, +5 deg heading error => 1-exp(-1).
    # ----------------------------------------------------------------------------------------------
    anchor_heading_for_heading_test = (
        heading
        - HEADING_TEST_OFFSET_RAD
    )

    reward_term._anchor_pos_xy[:] = pos_xy
    reward_term._anchor_heading[:] = (
        anchor_heading_for_heading_test
    )
    reward_term._stored_command[:] = command[:, :3]
    reward_term._elapsed_s.zero_()
    reward_term._was_active.fill_(True)

    heading_cost = reward_term(
        env,
        **reward_cfg.params,
    )

    expected_heading_cost = (
        1.0
        - math.exp(-1.0)
    )

    expected_heading_tensor = torch.full(
        (NUM_ENVS,),
        expected_heading_cost,
        device=env.device,
        dtype=heading_cost.dtype,
    )

    if not torch.allclose(
        heading_cost,
        expected_heading_tensor,
        rtol=0.0,
        atol=2.0e-6,
    ):
        fail(
            "5 deg heading-only cost mismatch: "
            f"actual={heading_cost.tolist()} "
            f"expected={expected_heading_cost:.8f}"
        )

    # ----------------------------------------------------------------------------------------------
    # Test 5: +10 cm cross-track AND +5 deg heading error => 1-exp(-(0.5^2 + 1^2)).
    # Reference anchor position is offset along the lateral axis associated with anchor heading.
    # ----------------------------------------------------------------------------------------------
    combined_lateral_axis = torch.stack(
        (
            -torch.sin(
                anchor_heading_for_heading_test
            ),
            torch.cos(
                anchor_heading_for_heading_test
            ),
        ),
        dim=1,
    )

    reward_term._anchor_pos_xy[:] = (
        pos_xy
        - LATERAL_TEST_OFFSET_M
        * combined_lateral_axis
    )
    reward_term._anchor_heading[:] = (
        anchor_heading_for_heading_test
    )
    reward_term._stored_command[:] = command[:, :3]
    reward_term._elapsed_s.zero_()
    reward_term._was_active.fill_(True)

    combined_cost = reward_term(
        env,
        **reward_cfg.params,
    )

    expected_combined_cost = (
        1.0
        - math.exp(
            -(
                (
                    LATERAL_TEST_OFFSET_M
                    / EXPECTED_SIGMA_CROSS_M
                )
                ** 2
                + (
                    HEADING_TEST_OFFSET_RAD
                    / EXPECTED_SIGMA_HEADING_RAD
                )
                ** 2
            )
        )
    )

    expected_combined_tensor = torch.full(
        (NUM_ENVS,),
        expected_combined_cost,
        device=env.device,
        dtype=combined_cost.dtype,
    )

    if not torch.allclose(
        combined_cost,
        expected_combined_tensor,
        rtol=0.0,
        atol=3.0e-6,
    ):
        fail(
            "combined cross+heading cost mismatch: "
            f"actual={combined_cost.tolist()} "
            f"expected={expected_combined_cost:.8f}"
        )

    # ----------------------------------------------------------------------------------------------
    # Test 6: command change/resample => immediate re-anchor and zero cost.
    # ----------------------------------------------------------------------------------------------
    reward_term._anchor_pos_xy.zero_()
    reward_term._anchor_heading.zero_()
    reward_term._stored_command.zero_()
    reward_term._stored_command[:, 0] = 0.40
    reward_term._elapsed_s.fill_(3.0)
    reward_term._was_active.fill_(True)

    command.zero_()
    command[:, 0] = 0.50

    resample_cost = reward_term(
        env,
        **reward_cfg.params,
    )

    if not torch.allclose(
        resample_cost,
        torch.zeros_like(
            resample_cost
        ),
        rtol=0.0,
        atol=1.0e-7,
    ):
        fail(
            "command-change did not re-anchor to zero cost: "
            f"{resample_cost.tolist()}"
        )

    if not torch.allclose(
        reward_term._stored_command,
        command[:, :3],
        rtol=0.0,
        atol=0.0,
    ):
        fail(
            "new command not stored on segment start"
        )

    expected_elapsed = torch.full(
        (NUM_ENVS,),
        float(env.step_dt),
        device=env.device,
        dtype=reward_term._elapsed_s.dtype,
    )

    if not torch.allclose(
        reward_term._elapsed_s,
        expected_elapsed,
        rtol=0.0,
        atol=1.0e-7,
    ):
        fail(
            "elapsed time mismatch after re-anchor: "
            f"{reward_term._elapsed_s.tolist()}"
        )

    # ----------------------------------------------------------------------------------------------
    # Test 7: inactive gate => zero cost; re-entry => re-anchor.
    # ----------------------------------------------------------------------------------------------
    command.zero_()

    inactive_cost = reward_term(
        env,
        **reward_cfg.params,
    )

    if not torch.allclose(
        inactive_cost,
        torch.zeros_like(
            inactive_cost
        ),
        rtol=0.0,
        atol=0.0,
    ):
        fail(
            f"inactive gate returned non-zero cost: {inactive_cost}"
        )

    if torch.any(
        reward_term._was_active
    ):
        fail(
            "inactive gate did not clear _was_active"
        )

    if not torch.allclose(
        reward_term._elapsed_s,
        torch.zeros_like(
            reward_term._elapsed_s
        ),
        rtol=0.0,
        atol=0.0,
    ):
        fail(
            "inactive gate did not clear elapsed time"
        )

    command[:, 0] = 0.40

    reentry_cost = reward_term(
        env,
        **reward_cfg.params,
    )

    if not torch.allclose(
        reentry_cost,
        torch.zeros_like(
            reentry_cost
        ),
        rtol=0.0,
        atol=1.0e-7,
    ):
        fail(
            "gate re-entry did not re-anchor to zero cost: "
            f"{reentry_cost.tolist()}"
        )

    # ----------------------------------------------------------------------------------------------
    # Test 8: normal real env.step path.
    # ----------------------------------------------------------------------------------------------
    reward_term.reset()
    command.zero_()
    command[:, 0] = 0.40

    action_dim = int(
        env.action_manager.total_action_dim
    )

    actions = torch.zeros(
        (NUM_ENVS, action_dim),
        device=env.device,
        dtype=torch.float32,
    )

    for _ in range(5):
        raw_env.step(actions)

    for name in (
        "_anchor_pos_xy",
        "_anchor_heading",
        "_stored_command",
        "_elapsed_s",
    ):
        tensor = getattr(
            reward_term,
            name,
        )

        if not torch.isfinite(
            tensor
        ).all():
            fail(
                f"{name} became non-finite in normal env.step"
            )

    runtime_elapsed_max = float(
        torch.max(
            reward_term._elapsed_s
        ).item()
    )

    print("=" * 116)
    print("v4-L RUNTIME SMOKE")
    print("=" * 116)
    print(f"Task ID                         : {TASK_ID}")
    print(f"num_envs                        : {env.num_envs}")
    print(f"device                          : {env.device}")
    print(f"step_dt                         : {float(env.step_dt):.5f} s")
    print(f"action dim                      : {action_dim}")
    print("reward active                   : YES")
    print(f"reward class                    : {reward_term.__class__.__name__}")
    print(f"reward weight                   : {float(reward_cfg.weight):.5f}")
    print(
        "sigma_cross_m                   : "
        f"{float(reward_cfg.params['sigma_cross_m']):.5f}"
    )
    print(
        "sigma_heading_deg               : "
        f"{math.degrees(float(reward_cfg.params['sigma_heading_rad'])):.5f}"
    )
    print(f"anchor_pos_xy shape             : {tuple(reward_term._anchor_pos_xy.shape)}")
    print(f"stored_command shape            : {tuple(reward_term._stored_command.shape)}")
    print("RewardManager selective reset   : PASS")
    print("straight/on-heading             : cost 0 — PASS")
    print(
        f"+{LATERAL_TEST_OFFSET_M:.2f} m lateral only            : "
        f"cost {float(lateral_cost.mean().item()):.8f} — PASS"
    )
    print(
        "+5 deg heading only             : "
        f"cost {float(heading_cost.mean().item()):.8f} — PASS"
    )
    print(
        "+0.10 m + 5 deg combined        : "
        f"cost {float(combined_cost.mean().item()):.8f} — PASS"
    )
    print("active command resample         : re-anchor — PASS")
    print("inactive gate                   : cost 0 / state transition — PASS")
    print("gate re-entry                   : re-anchor — PASS")
    print("normal env.step executions      : 5")
    print(f"post-step elapsed_s max         : {runtime_elapsed_max:.5f} s")
    print()
    print("V4L_RUNTIME_SMOKE_PASS")
    print("=" * 116)

finally:
    if raw_env is not None:
        raw_env.close()

    simulation_app.close()
