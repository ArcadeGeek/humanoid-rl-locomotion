from __future__ import annotations

import math

import torch

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.mdp.v4_rewards import (  # noqa: E402
    low_frequency_command_residual_vy_wz_bounded,
)


class FakeTensorHolder:
    def __init__(self, tensor):
        self.root_lin_vel_b = tensor.clone()
        self.root_ang_vel_b = tensor.clone()


class FakeAsset:
    def __init__(self, lin_vel_b, ang_vel_b):
        self.data = type("Data", (), {})()
        self.data.root_lin_vel_b = lin_vel_b
        self.data.root_ang_vel_b = ang_vel_b


class FakeScene:
    def __init__(self, asset):
        self._asset = asset

    def __getitem__(self, name):
        if name != "robot":
            raise KeyError(name)
        return self._asset


class FakeCommandManager:
    def __init__(self, command):
        self.command = command

    def get_command(self, name):
        if name != "base_velocity":
            raise KeyError(name)
        return self.command


class FakeEnv:
    def __init__(
        self,
        command,
        lin_vel_b,
        ang_vel_b,
        device,
    ):
        self.num_envs = int(command.shape[0])
        self.device = device
        self.step_dt = 0.02
        self.command_manager = FakeCommandManager(command)
        self.scene = FakeScene(
            FakeAsset(
                lin_vel_b,
                ang_vel_b,
            )
        )


class FakeCfg:
    def __init__(self):
        self.params = {
            "command_name": "base_velocity",
            "tau_s": 1.50,
            "sigma_vy_mps": 0.030,
            "sigma_wz_radps": 0.025,
            "min_forward_command": 0.10,
            "lateral_command_threshold": 0.10,
            "yaw_command_threshold": 0.20,
            "command_change_threshold": 1.0e-6,
            "asset_cfg": type("AssetCfg", (), {"name": "robot"})(),
        }


def assert_close(
    actual,
    expected,
    label,
    atol=1.0e-7,
):
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=atol,
    ):
        raise RuntimeError(
            f"{label}: expected {expected}, got {actual}"
        )


device = "cuda:0" if torch.cuda.is_available() else "cpu"

command = torch.tensor(
    [
        [0.40, 0.00, 0.00],   # active, perfect tracking
        [0.40, 0.00, 0.00],   # active, vy residual
        [0.40, 0.00, 0.00],   # active, wz residual
        [0.40, 0.05, -0.10],  # active, command-consistent nonzero vy/wz
        [0.05, 0.00, 0.00],   # inactive: vx too small
        [0.40, 0.15, 0.00],   # inactive: |vy cmd| too large
        [0.40, 0.00, 0.25],   # inactive: |wz cmd| too large
    ],
    device=device,
    dtype=torch.float32,
)

lin_vel_b = torch.zeros(
    (7, 3),
    device=device,
    dtype=torch.float32,
)
ang_vel_b = torch.zeros(
    (7, 3),
    device=device,
    dtype=torch.float32,
)

# Perfect tracking, zero residual.
lin_vel_b[0, 1] = 0.00
ang_vel_b[0, 2] = 0.00

# vy residual = +0.03 m/s.
lin_vel_b[1, 1] = 0.03
ang_vel_b[1, 2] = 0.00

# wz residual = +0.025 rad/s.
lin_vel_b[2, 1] = 0.00
ang_vel_b[2, 2] = 0.025

# Non-zero command, but perfect command-consistent tracking.
lin_vel_b[3, 1] = 0.05
ang_vel_b[3, 2] = -0.10

# Inactive cases intentionally have arbitrary residuals.
lin_vel_b[4:, 1] = 0.50
ang_vel_b[4:, 2] = 0.50


env = FakeEnv(
    command,
    lin_vel_b,
    ang_vel_b,
    device,
)
cfg = FakeCfg()

term = low_frequency_command_residual_vy_wz_bounded(
    cfg,
    env,
)

alpha = 1.0 - math.exp(-0.02 / 1.50)
assert_close(
    term._alpha,
    alpha,
    "runtime alpha",
    atol=1.0e-10,
)

kwargs = dict(cfg.params)

# First call starts a new active segment, so EMA is updated once from zero.
cost1 = term(
    env,
    **kwargs,
)

expected_vy_ema = alpha * 0.03
expected_wz_ema = alpha * 0.025

expected_vy_cost = 1.0 - math.exp(
    -((expected_vy_ema / 0.03) ** 2)
)
expected_wz_cost = 1.0 - math.exp(
    -((expected_wz_ema / 0.025) ** 2)
)

assert_close(
    cost1[0].item(),
    0.0,
    "perfect zero-command tracking cost",
)
assert_close(
    cost1[1].item(),
    expected_vy_cost,
    "single-step vy residual cost",
)
assert_close(
    cost1[2].item(),
    expected_wz_cost,
    "single-step wz residual cost",
)
assert_close(
    cost1[3].item(),
    0.0,
    "nonzero command-consistent tracking cost",
)

for index in (4, 5, 6):
    assert_close(
        cost1[index].item(),
        0.0,
        f"inactive gate cost env {index}",
    )

# Keep the same command and residuals for 500 calls.
# The two active residual cases should converge to cost ~= 1-exp(-1).
cost_last = cost1
for _ in range(499):
    cost_last = term(
        env,
        **kwargs,
    )

target_one_sigma_cost = 1.0 - math.exp(-1.0)

assert_close(
    cost_last[1].item(),
    target_one_sigma_cost,
    "500-step vy one-sigma cost",
    atol=1.0e-3,
)
assert_close(
    cost_last[2].item(),
    target_one_sigma_cost,
    "500-step wz one-sigma cost",
    atol=1.0e-3,
)

assert_close(
    cost_last[0].item(),
    0.0,
    "500-step perfect tracking cost",
)
assert_close(
    cost_last[3].item(),
    0.0,
    "500-step nonzero command-consistent cost",
)

# Command change while remaining inside gate must reset EMA segment state.
env.command_manager.command[1] = torch.tensor(
    [0.40, 0.02, 0.00],
    device=device,
)

# Make body vy exactly match the new command.
env.scene["robot"].data.root_lin_vel_b[1, 1] = 0.02

cost_after_command_change = term(
    env,
    **kwargs,
)

assert_close(
    term._ema_vy_residual[1].item(),
    0.0,
    "EMA reset after active command change",
    atol=1.0e-7,
)
assert_close(
    cost_after_command_change[1].item(),
    0.0,
    "cost after command-consistent resample",
    atol=1.0e-7,
)

# Gate exit must zero state.
env.command_manager.command[2, 0] = 0.05

cost_after_gate_exit = term(
    env,
    **kwargs,
)

assert_close(
    term._ema_wz_residual[2].item(),
    0.0,
    "EMA zero after gate exit",
)
assert_close(
    cost_after_gate_exit[2].item(),
    0.0,
    "cost zero after gate exit",
)

print("=" * 118)
print("v4-M ACTIVE-BRANCH REWARD SMOKE")
print("=" * 118)
print(f"device                              : {device}")
print(f"alpha                               : {term._alpha:.12f}")
print(f"single-step vy residual cost        : {cost1[1].item():.9f}")
print(f"single-step wz residual cost        : {cost1[2].item():.9f}")
print(f"500-step one-sigma target           : {target_one_sigma_cost:.9f}")
print(f"500-step vy residual cost           : {cost_last[1].item():.9f}")
print(f"500-step wz residual cost           : {cost_last[2].item():.9f}")
print("perfect zero-command tracking       : PASS")
print("command-consistent nonzero tracking : PASS")
print("vy active branch                    : PASS")
print("wz active branch                    : PASS")
print("inactive gate branches              : PASS")
print("command-change reset                : PASS")
print("gate-exit reset                     : PASS")
print("bounded EMA convergence             : PASS")
print()
print("V4M_ACTIVE_BRANCH_REWARD_SMOKE_PASS")
print("=" * 118)

simulation_app.close()
