from __future__ import annotations

import math

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

# Launch Isaac Sim before importing task modules that require simulation bindings.
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import humanoid_rl_lab.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402


TASK = "Humanoid-RL-Lab-G1-Flat-V4M-LowFreq-Local-Residual-v0"
REWARD_NAME = "low_frequency_command_residual_vy_wz_bounded"

EXPECTED_WEIGHT = -0.10
EXPECTED_TAU_S = 1.50
EXPECTED_SIGMA_VY = 0.030
EXPECTED_SIGMA_WZ = 0.025
EXPECTED_ALPHA = 1.0 - math.exp(-0.02 / 1.50)


def assert_close(
    actual: float,
    expected: float,
    label: str,
    atol: float = 1.0e-8,
) -> None:
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=atol,
    ):
        raise RuntimeError(
            f"{label}: expected {expected}, got {actual}"
        )


def get_reward_term(env):
    reward_manager = env.unwrapped.reward_manager

    # Prefer public manager configuration lists if available.
    term_names = getattr(
        reward_manager,
        "active_terms",
        None,
    )

    if term_names is not None and REWARD_NAME not in term_names:
        raise RuntimeError(
            f"{REWARD_NAME} is not active in RewardManager; "
            f"active_terms={term_names}"
        )

    cfg = env.unwrapped.cfg.rewards
    if not hasattr(cfg, REWARD_NAME):
        raise RuntimeError(
            f"resolved env rewards missing {REWARD_NAME}"
        )

    cfg_term = getattr(cfg, REWARD_NAME)

    # Isaac Lab manager terms are instantiated internally. Across versions,
    # the live objects are stored in _terms or _term_cfgs/_class_term_cfgs.
    candidates = []

    terms = getattr(
        reward_manager,
        "_terms",
        None,
    )
    if isinstance(terms, dict):
        candidates.append(
            terms.get(REWARD_NAME)
        )
    elif isinstance(terms, (list, tuple)):
        names = getattr(
            reward_manager,
            "_term_names",
            None,
        )
        if isinstance(names, (list, tuple)) and REWARD_NAME in names:
            candidates.append(
                terms[names.index(REWARD_NAME)]
            )

    class_term_cfgs = getattr(
        reward_manager,
        "_class_term_cfgs",
        None,
    )
    if isinstance(class_term_cfgs, list):
        for item in class_term_cfgs:
            if not isinstance(item, tuple):
                continue
            for element in item:
                if (
                    type(element).__name__
                    == REWARD_NAME
                ):
                    candidates.append(
                        element
                    )

    term_cfgs = getattr(
        reward_manager,
        "_term_cfgs",
        None,
    )
    term_names_private = getattr(
        reward_manager,
        "_term_names",
        None,
    )

    if (
        isinstance(term_cfgs, (list, tuple))
        and isinstance(term_names_private, (list, tuple))
        and REWARD_NAME in term_names_private
    ):
        index = term_names_private.index(
            REWARD_NAME
        )
        if index < len(term_cfgs):
            possible_cfg = term_cfgs[index]
            possible_func = getattr(
                possible_cfg,
                "func",
                None,
            )
            if (
                type(possible_func).__name__
                == REWARD_NAME
            ):
                candidates.append(
                    possible_func
                )

    for candidate in candidates:
        if (
            candidate is not None
            and type(candidate).__name__
            == REWARD_NAME
        ):
            return candidate, cfg_term

    raise RuntimeError(
        "could not locate live v4-M reward class term "
        f"{REWARD_NAME} in RewardManager"
    )


env = None

try:
    env_cfg = load_cfg_from_registry(
        TASK,
        "env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = 4

    env = gym.make(
        TASK,
        cfg=env_cfg,
        render_mode=None,
    )

    unwrapped = env.unwrapped

    if int(unwrapped.num_envs) != 4:
        raise RuntimeError(
            f"expected 4 envs, got {unwrapped.num_envs}"
        )

    assert_close(
        unwrapped.step_dt,
        0.02,
        "env step_dt",
        atol=1.0e-10,
    )

    action_space = env.action_space
    action_shape = getattr(
        action_space,
        "shape",
        None,
    )

    if action_shape is None:
        raise RuntimeError(
            f"action space has no shape: {action_space}"
        )

    action_dim = int(action_shape[-1])

    if action_dim != 37:
        raise RuntimeError(
            f"expected action dim 37, got {action_dim}"
        )

    reward_term, cfg_term = get_reward_term(
        env
    )

    if type(reward_term).__name__ != REWARD_NAME:
        raise RuntimeError(
            "unexpected live reward term class: "
            f"{type(reward_term).__name__}"
        )

    assert_close(
        cfg_term.weight,
        EXPECTED_WEIGHT,
        "reward weight",
    )

    params = cfg_term.params

    assert_close(
        params["tau_s"],
        EXPECTED_TAU_S,
        "tau_s",
    )
    assert_close(
        params["sigma_vy_mps"],
        EXPECTED_SIGMA_VY,
        "sigma_vy_mps",
    )
    assert_close(
        params["sigma_wz_radps"],
        EXPECTED_SIGMA_WZ,
        "sigma_wz_radps",
    )

    assert_close(
        reward_term._alpha,
        EXPECTED_ALPHA,
        "runtime EMA alpha",
        atol=1.0e-10,
    )

    for name in (
        "_ema_vy_residual",
        "_ema_wz_residual",
        "_stored_command",
        "_was_active",
    ):
        if not hasattr(
            reward_term,
            name,
        ):
            raise RuntimeError(
                f"live reward term missing state {name}"
            )

    if tuple(
        reward_term._ema_vy_residual.shape
    ) != (4,):
        raise RuntimeError(
            "unexpected ema vy shape: "
            f"{tuple(reward_term._ema_vy_residual.shape)}"
        )

    if tuple(
        reward_term._ema_wz_residual.shape
    ) != (4,):
        raise RuntimeError(
            "unexpected ema wz shape: "
            f"{tuple(reward_term._ema_wz_residual.shape)}"
        )

    if tuple(
        reward_term._stored_command.shape
    ) != (4, 3):
        raise RuntimeError(
            "unexpected stored command shape: "
            f"{tuple(reward_term._stored_command.shape)}"
        )

    if tuple(
        reward_term._was_active.shape
    ) != (4,):
        raise RuntimeError(
            "unexpected active mask shape: "
            f"{tuple(reward_term._was_active.shape)}"
        )

    # Reward-state-only selective reset test. No simulator state injection.
    reward_term._ema_vy_residual[:] = torch.tensor(
        [0.01, 0.02, 0.03, 0.04],
        device=reward_term._ema_vy_residual.device,
        dtype=reward_term._ema_vy_residual.dtype,
    )
    reward_term._ema_wz_residual[:] = torch.tensor(
        [-0.01, -0.02, -0.03, -0.04],
        device=reward_term._ema_wz_residual.device,
        dtype=reward_term._ema_wz_residual.dtype,
    )
    reward_term._stored_command[:] = 1.0
    reward_term._was_active[:] = True

    reset_ids = torch.tensor(
        [1, 3],
        device=reward_term._ema_vy_residual.device,
        dtype=torch.long,
    )

    reward_term.reset(
        reset_ids
    )

    if not torch.allclose(
        reward_term._ema_vy_residual,
        torch.tensor(
            [0.01, 0.0, 0.03, 0.0],
            device=reward_term._ema_vy_residual.device,
            dtype=reward_term._ema_vy_residual.dtype,
        ),
        atol=1.0e-8,
        rtol=0.0,
    ):
        raise RuntimeError(
            "selective reset failed for ema vy"
        )

    if not torch.allclose(
        reward_term._ema_wz_residual,
        torch.tensor(
            [-0.01, 0.0, -0.03, 0.0],
            device=reward_term._ema_wz_residual.device,
            dtype=reward_term._ema_wz_residual.dtype,
        ),
        atol=1.0e-8,
        rtol=0.0,
    ):
        raise RuntimeError(
            "selective reset failed for ema wz"
        )

    if not torch.all(
        reward_term._stored_command[
            reset_ids
        ]
        == 0.0
    ):
        raise RuntimeError(
            "selective reset failed for stored command"
        )

    if torch.any(
        reward_term._was_active[
            reset_ids
        ]
    ):
        raise RuntimeError(
            "selective reset failed for active mask"
        )

    # Clear all term state before stepping the real environment.
    reward_term.reset()

    obs, info = env.reset()

    action_device = getattr(
        unwrapped,
        "device",
        "cuda:0",
    )

    actions = torch.zeros(
        (4, action_dim),
        device=action_device,
        dtype=torch.float32,
    )

    observed_costs = []

    for _ in range(5):
        step_result = env.step(
            actions
        )

        if len(step_result) != 5:
            raise RuntimeError(
                "unexpected Gymnasium step result length: "
                f"{len(step_result)}"
            )

        _, reward, terminated, truncated, info = step_result

        if not torch.all(
            torch.isfinite(
                reward
            )
        ):
            raise RuntimeError(
                "non-finite total reward during smoke step"
            )

        if not torch.all(
            torch.isfinite(
                reward_term._ema_vy_residual
            )
        ):
            raise RuntimeError(
                "non-finite EMA vy state"
            )

        if not torch.all(
            torch.isfinite(
                reward_term._ema_wz_residual
            )
        ):
            raise RuntimeError(
                "non-finite EMA wz state"
            )

        # Compute the bounded cost from the live internal state only.
        exponent = (
            torch.square(
                reward_term._ema_vy_residual
                / EXPECTED_SIGMA_VY
            )
            + torch.square(
                reward_term._ema_wz_residual
                / EXPECTED_SIGMA_WZ
            )
        )
        cost = 1.0 - torch.exp(
            -exponent
        )

        if torch.any(
            cost < -1.0e-7
        ) or torch.any(
            cost > 1.0 + 1.0e-7
        ):
            raise RuntimeError(
                "v4-M reconstructed bounded cost left [0,1]"
            )

        observed_costs.append(
            float(
                torch.max(
                    cost
                ).item()
            )
        )

    print("=" * 120)
    print("v4-M RUNTIME SMOKE")
    print("=" * 120)
    print(f"Task                      : {TASK}")
    print(f"num_envs                  : {unwrapped.num_envs}")
    print(f"device                    : {unwrapped.device}")
    print(f"step_dt                   : {unwrapped.step_dt}")
    print(f"action_dim                : {action_dim}")
    print(f"reward class              : {type(reward_term).__name__}")
    print(f"reward weight             : {cfg_term.weight}")
    print(f"tau_s                     : {params['tau_s']}")
    print(f"sigma_vy_mps              : {params['sigma_vy_mps']}")
    print(f"sigma_wz_radps            : {params['sigma_wz_radps']}")
    print(f"runtime alpha             : {reward_term._alpha:.12f}")
    print(
        "EMA state shapes          : "
        f"{tuple(reward_term._ema_vy_residual.shape)} / "
        f"{tuple(reward_term._ema_wz_residual.shape)}"
    )
    print(
        "stored command shape      : "
        f"{tuple(reward_term._stored_command.shape)}"
    )
    print("selective reward reset    : PASS")
    print("5 real env steps          : PASS")
    print("finite EMA states         : PASS")
    print("bounded reconstructed cost: PASS")
    print(
        "max reconstructed costs   : "
        + ", ".join(
            f"{value:.6f}"
            for value in observed_costs
        )
    )
    print()
    print("V4M_RUNTIME_SMOKE_PASS")
    print("=" * 120)

finally:
    if env is not None:
        env.close()

    simulation_app.close()
