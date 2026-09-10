from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import math
import statistics
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from importlib import metadata
from pathlib import Path
from typing import Any

parser = argparse.ArgumentParser(
    description=(
        "Functional runtime smoke of the frozen v2-G/v4-N router with provisional hysteresis thresholds "
        "and frozen 0.36 s v4-N zero-command braking dwell. Threshold calibration is explicitly out of scope."
    )
)
parser.add_argument("--seed", type=int, default=104)
parser.add_argument("--output_dir", type=str, default=None)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import humanoid_rl_lab.tasks  # noqa: F401
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.upright_yaw_direct_wide_yaw_flat_orientation_ppo_cfg import (
    UprightYawDirectWideYawFlatOrientationPPORunnerCfg,
)
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.v4n_ppo_cfg import (
    V4NSymmetryAugmentationPPORunnerCfg,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner


STAND_TASK = "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-v0"

STAND_CHECKPOINT = Path(
    r"C:\humanoid_rl_lab\checkpoints\local_g1_yaw_direct_wide_yaw_flat_orientation_v2g\model_600.pt"
)
STAND_CHECKPOINT_SHA256 = "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3"

LOCO_CHECKPOINT = Path(
    r"C:\humanoid_rl_lab\logs\rsl_rl\humanoid_rl_lab_g1_v4n_symmetry_augmentation_scratch"
    r"\2026-09-04_13-58-13_v4n_genuine_scratch_seed42_official1500\model_1499.pt"
)
LOCO_CHECKPOINT_SHA256 = "367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342"

EXPECTED_OBS_DIM = 123
EXPECTED_ACTION_DIM = 37
EXPECTED_STEP_DT = 0.02

# FROZEN handoff primitive from paired development validation.
BRAKE_S = 0.36
BRAKE_STEPS = 18

# PROVISIONAL engineering-only thresholds.
# This smoke validates integration/invariants, NOT threshold quality.
ENTER_LOCO_SPEED = 0.20
EXIT_LOCO_SPEED = 0.10

# Deterministic request schedule, chosen only to exercise every router transition.
STAND_PRE_S = 3.00
LOCO_MAIN_S = 8.00
HYSTERESIS_BAND_HOLD_S = 1.00
CANCEL_BRAKE_REQUEST_S = 0.10
LOCO_RESUME_S = 2.00
FINAL_STAND_S = 5.00

LOCO_REQUEST = 0.40
BAND_REQUEST = 0.15
ZERO_REQUEST = 0.00

CONTACT_FORCE_THRESHOLD_N = 20.0

PASS_MARKER = "V2G_V4N_ROUTER_HYSTERESIS_RUNTIME_SMOKE_PASS"
FAIL_MARKER = "V2G_V4N_ROUTER_HYSTERESIS_RUNTIME_SMOKE_FAIL"


class RouterState(Enum):
    STAND = auto()
    LOCO = auto()
    BRAKE = auto()


@dataclass(frozen=True)
class RouterOutput:
    state: RouterState
    active_policy: str
    policy_command_vx: float
    brake_index: int


class Router:
    def __init__(
        self,
        enter_loco_speed: float,
        exit_loco_speed: float,
        brake_steps: int,
    ) -> None:
        if not (0.0 <= exit_loco_speed < enter_loco_speed):
            raise ValueError("require 0 <= EXIT_LOCO_SPEED < ENTER_LOCO_SPEED")
        if brake_steps <= 0:
            raise ValueError("brake_steps must be positive")
        self.enter = float(enter_loco_speed)
        self.exit = float(exit_loco_speed)
        self.brake_steps = int(brake_steps)
        self.state = RouterState.STAND
        self._brake_followup_count = 0

    def update(self, requested_vx: float) -> RouterOutput:
        speed = abs(float(requested_vx))

        if self.state is RouterState.STAND:
            if speed >= self.enter:
                self.state = RouterState.LOCO

        elif self.state is RouterState.LOCO:
            if speed <= self.exit:
                self.state = RouterState.BRAKE
                self._brake_followup_count = 0

        elif self.state is RouterState.BRAKE:
            if speed >= self.enter:
                self.state = RouterState.LOCO
                self._brake_followup_count = 0
            else:
                self._brake_followup_count += 1
                if self._brake_followup_count >= self.brake_steps:
                    self.state = RouterState.STAND
                    self._brake_followup_count = 0
        else:
            raise RuntimeError(f"unknown router state {self.state}")

        if self.state is RouterState.STAND:
            return RouterOutput(self.state, "v2-G", 0.0, 0)
        if self.state is RouterState.LOCO:
            return RouterOutput(self.state, "v4-N", float(requested_vx), 0)

        # BRAKE: keep v4-N, override its command to zero.
        # brake_index=1 on the entry control step, then 2..18 on follow-up BRAKE steps.
        return RouterOutput(
            self.state,
            "v4-N",
            0.0,
            self._brake_followup_count + 1,
        )


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def torch_value(value: Any) -> torch.Tensor:
    if hasattr(value, "torch"):
        value = value.torch
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value


def extract_policy_observation_tensor(obs: Any) -> torch.Tensor:
    candidate = None
    if torch.is_tensor(obs) or hasattr(obs, "torch"):
        candidate = obs
    else:
        try:
            candidate = obs["policy"]
        except Exception:
            get_fn = getattr(obs, "get", None)
            if callable(get_fn):
                try:
                    candidate = get_fn("policy")
                except Exception:
                    candidate = None

    if candidate is None:
        fail(f"cannot resolve policy observation from {type(obs).__name__}")

    tensor = torch_value(candidate)
    if tensor.ndim != 2 or tensor.shape[0] != 1:
        fail(f"unexpected policy observation shape={tuple(tensor.shape)}")
    return tensor


def validate_checkpoint(path: Path, expected_sha: str, label: str) -> str:
    if not path.is_file():
        fail(f"{label} checkpoint missing: {path}")
    actual = sha256(path)
    if actual != expected_sha:
        fail(f"{label} SHA mismatch: {actual} != {expected_sha}")
    return actual


def write_command(base_env: Any, vx: float) -> None:
    command_buffer = base_env.command_manager.get_command("base_velocity")
    with torch.no_grad():
        command_buffer[:, 0] = float(vx)
        command_buffer[:, 1] = 0.0
        command_buffer[:, 2] = 0.0


def projected_gravity_tilt_deg(robot: Any) -> float:
    g_b = torch_value(robot.data.projected_gravity_b)[0]
    c = torch.clamp(-g_b[2], -1.0, 1.0)
    return float(torch.rad2deg(torch.acos(c)).detach().cpu().item())


def action_rms(action: torch.Tensor) -> float:
    a = action.detach().float()
    return float(torch.sqrt(torch.mean(a * a)).cpu().item())


def reset_policy_after_done(policy: Any, runner: Any, dones: torch.Tensor) -> None:
    if hasattr(policy, "reset") and callable(policy.reset):
        policy.reset(dones)
        return
    alg = getattr(runner, "alg", None)
    for name in ("policy", "actor_critic"):
        network = getattr(alg, name, None)
        if network is not None and hasattr(network, "reset") and callable(network.reset):
            network.reset(dones)
            return


def requested_vx_for_step(step: int, dt: float) -> tuple[float, str]:
    s0 = int(round(STAND_PRE_S / dt))
    s1 = s0 + int(round(LOCO_MAIN_S / dt))
    s2 = s1 + int(round(HYSTERESIS_BAND_HOLD_S / dt))
    s3 = s2 + int(round(CANCEL_BRAKE_REQUEST_S / dt))
    s4 = s3 + int(round(LOCO_RESUME_S / dt))

    if step < s0:
        return ZERO_REQUEST, "REQ_STAND_PRE"
    if step < s1:
        return LOCO_REQUEST, "REQ_LOCO_MAIN"
    if step < s2:
        return BAND_REQUEST, "REQ_HYSTERESIS_BAND"
    if step < s3:
        return ZERO_REQUEST, "REQ_BRAKE_CANCEL_WINDOW"
    if step < s4:
        return LOCO_REQUEST, "REQ_LOCO_RESUME"
    return ZERO_REQUEST, "REQ_FINAL_STOP"


def tail_stats(rows: list[dict[str, Any]], seconds: float) -> dict[str, float]:
    n = max(1, int(round(seconds / EXPECTED_STEP_DT)))
    tail = rows[-n:]
    return {
        "samples": float(len(tail)),
        "mean_vxy_mps": statistics.fmean(r["root_vxy_mps"] for r in tail),
        "max_vxy_mps": max(r["root_vxy_mps"] for r in tail),
        "mean_tilt_deg": statistics.fmean(r["root_tilt_deg"] for r in tail),
        "max_tilt_deg": max(r["root_tilt_deg"] for r in tail),
        "mean_abs_wz_radps": statistics.fmean(abs(r["root_wz_radps"]) for r in tail),
        "double_support_pct": (
            100.0 * statistics.fmean(r["double_support"] for r in tail)
            if all(r["double_support"] >= 0 for r in tail)
            else float("nan")
        ),
    }


def main() -> None:
    if BRAKE_STEPS != int(round(BRAKE_S / EXPECTED_STEP_DT)):
        fail("frozen dwell guardrail failed: 0.36 s must equal 18 control steps")

    stand_sha = validate_checkpoint(
        STAND_CHECKPOINT, STAND_CHECKPOINT_SHA256, "v2-G STAND"
    )
    loco_sha = validate_checkpoint(
        LOCO_CHECKPOINT, LOCO_CHECKPOINT_SHA256, "v4-N LOCOMOTION"
    )

    installed_version = metadata.version("rsl-rl-lib")

    env_cfg = parse_env_cfg(
        STAND_TASK,
        device=args_cli.device,
        num_envs=1,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args_cli.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Engineering-only startup conditioning, already used for paired handoff validation.
    baseline_root_velocity_range = dict(
        env_cfg.events.reset_base.params["velocity_range"]
    )
    zero_root_velocity_range = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_base.params["velocity_range"] = zero_root_velocity_range

    total_requested_s = (
        STAND_PRE_S
        + LOCO_MAIN_S
        + HYSTERESIS_BAND_HOLD_S
        + CANCEL_BRAKE_REQUEST_S
        + LOCO_RESUME_S
        + BRAKE_S
        + FINAL_STAND_S
    )
    env_cfg.episode_length_s = max(
        float(env_cfg.episode_length_s),
        total_requested_s + 10.0,
    )

    if hasattr(env_cfg.observations.policy, "enable_corruption"):
        env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg, "events"):
        if hasattr(env_cfg.events, "base_external_force_torque"):
            env_cfg.events.base_external_force_torque = None
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None

    command_cfg = env_cfg.commands.base_velocity
    command_cfg.heading_command = False
    if hasattr(command_cfg, "rel_heading_envs"):
        command_cfg.rel_heading_envs = 0.0

    stand_agent_cfg = UprightYawDirectWideYawFlatOrientationPPORunnerCfg()
    loco_agent_cfg = V4NSymmetryAugmentationPPORunnerCfg()
    if args_cli.device is not None:
        stand_agent_cfg.device = args_cli.device
        loco_agent_cfg.device = args_cli.device

    stand_agent_cfg = handle_deprecated_rsl_rl_cfg(stand_agent_cfg, installed_version)
    loco_agent_cfg = handle_deprecated_rsl_rl_cfg(loco_agent_cfg, installed_version)

    if stand_agent_cfg.class_name != "OnPolicyRunner":
        fail(f"unexpected v2-G runner class: {stand_agent_cfg.class_name}")
    if loco_agent_cfg.class_name != "OnPolicyRunner":
        fail(f"unexpected v4-N runner class: {loco_agent_cfg.class_name}")
    if stand_agent_cfg.clip_actions != loco_agent_cfg.clip_actions:
        fail("v2-G/v4-N clip_actions mismatch")

    env = None
    try:
        raw_env = gym.make(STAND_TASK, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=stand_agent_cfg.clip_actions)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        dt = float(base_env.step_dt)
        if not math.isclose(dt, EXPECTED_STEP_DT, rel_tol=0.0, abs_tol=1e-12):
            fail(f"step_dt={dt}; expected {EXPECTED_STEP_DT}")
        if int(getattr(base_env.action_manager, "total_action_dim", -1)) != EXPECTED_ACTION_DIM:
            fail("unexpected action dimension")

        stand_runner = OnPolicyRunner(
            env,
            stand_agent_cfg.to_dict(),
            log_dir=None,
            device=stand_agent_cfg.device,
        )
        stand_runner.load(str(STAND_CHECKPOINT))
        stand_policy = stand_runner.get_inference_policy(device=base_env.device)

        loco_runner = OnPolicyRunner(
            env,
            loco_agent_cfg.to_dict(),
            log_dir=None,
            device=loco_agent_cfg.device,
        )
        loco_runner.load(str(LOCO_CHECKPOINT))
        loco_policy = loco_runner.get_inference_policy(device=base_env.device)

        obs = env.get_observations()
        policy_obs = extract_policy_observation_tensor(obs)
        if int(policy_obs.shape[1]) != EXPECTED_OBS_DIM:
            fail(f"unexpected observation dimension {policy_obs.shape[1]}")

        # Verify zero-root-velocity startup realization.
        root_lin_w = torch_value(robot.data.root_lin_vel_w)[0]
        root_ang_w = torch_value(robot.data.root_ang_vel_w)[0]
        if float(torch.max(torch.abs(root_lin_w)).cpu().item()) > 1e-6:
            fail(f"nonzero startup root linear velocity: {root_lin_w}")
        if float(torch.max(torch.abs(root_ang_w)).cpu().item()) > 1e-6:
            fail(f"nonzero startup root angular velocity: {root_ang_w}")

        router = Router(
            enter_loco_speed=ENTER_LOCO_SPEED,
            exit_loco_speed=EXIT_LOCO_SPEED,
            brake_steps=BRAKE_STEPS,
        )

        pre_steps = int(round(STAND_PRE_S / dt))
        loco_steps = int(round(LOCO_MAIN_S / dt))
        band_steps = int(round(HYSTERESIS_BAND_HOLD_S / dt))
        cancel_steps = int(round(CANCEL_BRAKE_REQUEST_S / dt))
        resume_steps = int(round(LOCO_RESUME_S / dt))
        final_stand_steps = int(round(FINAL_STAND_S / dt))

        # After the schedule reaches final zero request, the router itself inserts 18 BRAKE steps.
        total_steps = (
            pre_steps
            + loco_steps
            + band_steps
            + cancel_steps
            + resume_steps
            + BRAKE_STEPS
            + final_stand_steps
        )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args_cli.output_dir:
            result_dir = Path(args_cli.output_dir).resolve()
            result_dir.mkdir(parents=True, exist_ok=False)
        else:
            result_dir = (
                Path(r"C:\humanoid_rl_lab\logs\thesis\router")
                / f"{timestamp}_v2g_v4n_router_runtime_smoke_seed{args_cli.seed}"
            )
            result_dir.mkdir(parents=True, exist_ok=False)

        csv_path = result_dir / "trajectory.csv"
        summary_path = result_dir / "summary.txt"

        contact_sensor = None
        left_contact_idx = None
        right_contact_idx = None
        try:
            contact_sensor = base_env.scene.sensors["contact_forces"]
            contact_names = list(contact_sensor.body_names)
            left_name = "left_ankle_roll_link"
            right_name = "right_ankle_roll_link"
            if left_name in contact_names and right_name in contact_names:
                left_contact_idx = contact_names.index(left_name)
                right_contact_idx = contact_names.index(right_name)
            else:
                contact_sensor = None
        except Exception:
            contact_sensor = None

        print("=" * 124)
        print("v2-G / v4-N ROUTER + HYSTERESIS RUNTIME SMOKE")
        print("=" * 124)
        print(f"seed                     : {args_cli.seed}")
        print("scientific role          : FUNCTIONAL INTEGRATION ONLY")
        print("threshold calibration    : OUT OF SCOPE")
        print(f"ENTER_LOCO_SPEED         : {ENTER_LOCO_SPEED:.3f} m/s [PROVISIONAL]")
        print(f"EXIT_LOCO_SPEED          : {EXIT_LOCO_SPEED:.3f} m/s [PROVISIONAL]")
        print(f"brake dwell              : {BRAKE_S:.2f} s = {BRAKE_STEPS} steps [FROZEN]")
        print("blending                 : ABSENT")
        print("startup root velocities  : ZERO [engineering-only conditioning]")
        print(f"startup baseline range   : {baseline_root_velocity_range}")
        print(f"startup test range       : {zero_root_velocity_range}")
        print(f"step_dt                  : {dt:.12f}")
        print(f"policy_obs_shape         : {tuple(policy_obs.shape)}")
        print(f"action_dim               : {base_env.action_manager.total_action_dim}")
        print(f"contact instrumentation  : {'AVAILABLE' if contact_sensor is not None else 'UNAVAILABLE'}")
        print("")
        print("REQUEST SCHEDULE")
        print(f"  {STAND_PRE_S:5.2f}s : vx=0.00  -> expect STAND")
        print(f"  {LOCO_MAIN_S:5.2f}s : vx=0.40  -> expect LOCO")
        print(f"  {HYSTERESIS_BAND_HOLD_S:5.2f}s : vx=0.15  -> expect LOCO hold inside band")
        print(f"  {CANCEL_BRAKE_REQUEST_S:5.2f}s : vx=0.00  -> enter BRAKE, but do not finish")
        print(f"  {LOCO_RESUME_S:5.2f}s : vx=0.40  -> expect BRAKE cancellation -> LOCO")
        print(f"  final      : vx=0.00  -> expect BRAKE 18 steps -> STAND for {FINAL_STAND_S:.2f}s")
        print(f"result_dir               : {result_dir}")
        print("=" * 124)

        rows: list[dict[str, Any]] = []
        transitions: list[tuple[str, str, int, float, str]] = []
        completed_brake_lengths: list[int] = []
        canceled_brake_lengths: list[int] = []
        active_brake_len = 0
        terminated = False
        terminated_step = -1
        terminated_state = ""

        previous_state = router.state

        for step in range(total_steps):
            requested_vx, request_phase = requested_vx_for_step(step, dt)

            old_state = router.state
            route = router.update(requested_vx)
            new_state = route.state

            if new_state is RouterState.BRAKE:
                active_brake_len += 1
            elif old_state is RouterState.BRAKE and new_state is not RouterState.BRAKE:
                if new_state is RouterState.LOCO:
                    canceled_brake_lengths.append(active_brake_len)
                elif new_state is RouterState.STAND:
                    completed_brake_lengths.append(active_brake_len)
                active_brake_len = 0

            if new_state is not old_state:
                transitions.append(
                    (old_state.name, new_state.name, step, step * dt, request_phase)
                )
                print(
                    f"[ROUTER] step={step:04d} t={step * dt:7.3f}s "
                    f"{old_state.name}->{new_state.name} "
                    f"request_vx={requested_vx:.3f} "
                    f"policy={route.active_policy} cmd_vx={route.policy_command_vx:.3f}",
                    flush=True,
                )

            # Runtime invariants.
            if new_state is RouterState.STAND:
                if route.active_policy != "v2-G" or abs(route.policy_command_vx) > 1e-12:
                    fail("STAND invariant violated")
            elif new_state is RouterState.LOCO:
                if route.active_policy != "v4-N":
                    fail("LOCO policy invariant violated")
                if abs(route.policy_command_vx - requested_vx) > 1e-12:
                    fail("LOCO command-forwarding invariant violated")
            elif new_state is RouterState.BRAKE:
                if route.active_policy != "v4-N" or abs(route.policy_command_vx) > 1e-12:
                    fail("BRAKE invariant violated")

            write_command(base_env, route.policy_command_vx)
            obs = env.get_observations()

            if route.active_policy == "v2-G":
                active_policy = stand_policy
                active_runner = stand_runner
            elif route.active_policy == "v4-N":
                active_policy = loco_policy
                active_runner = loco_runner
            else:
                fail(f"unknown active policy {route.active_policy}")

            with torch.inference_mode():
                actions = active_policy(obs)

            if not torch.isfinite(actions).all():
                fail(f"non-finite action at step {step}")

            with torch.inference_mode():
                obs, _, dones, _ = env.step(actions)

            reset_policy_after_done(active_policy, active_runner, dones)

            root_pos = torch_value(robot.data.root_pos_w)[0]
            root_lin_b = torch_value(robot.data.root_lin_vel_b)[0]
            root_ang_b = torch_value(robot.data.root_ang_vel_b)[0]

            double_support = -1.0
            if (
                contact_sensor is not None
                and left_contact_idx is not None
                and right_contact_idx is not None
            ):
                try:
                    net_forces = torch_value(contact_sensor.data.net_forces_w)[0]
                    lf = abs(float(net_forces[left_contact_idx, 2].detach().cpu().item()))
                    rf = abs(float(net_forces[right_contact_idx, 2].detach().cpu().item()))
                    double_support = 1.0 if (
                        lf >= CONTACT_FORCE_THRESHOLD_N and
                        rf >= CONTACT_FORCE_THRESHOLD_N
                    ) else 0.0
                except Exception:
                    contact_sensor = None
                    double_support = -1.0

            row = {
                "step": step,
                "time_s": (step + 1) * dt,
                "request_phase": request_phase,
                "requested_vx_mps": requested_vx,
                "router_state": new_state.name,
                "active_policy": route.active_policy,
                "policy_command_vx_mps": route.policy_command_vx,
                "brake_index": route.brake_index,
                "root_x_m": float(root_pos[0].detach().cpu().item()),
                "root_y_m": float(root_pos[1].detach().cpu().item()),
                "root_z_m": float(root_pos[2].detach().cpu().item()),
                "root_vxy_mps": math.hypot(
                    float(root_lin_b[0].detach().cpu().item()),
                    float(root_lin_b[1].detach().cpu().item()),
                ),
                "root_wz_radps": float(root_ang_b[2].detach().cpu().item()),
                "root_tilt_deg": projected_gravity_tilt_deg(robot),
                "action_rms": action_rms(actions),
                "double_support": double_support,
                "done": int(bool(torch.any(dones).item())),
            }
            rows.append(row)

            if bool(torch.any(dones).item()):
                terminated = True
                terminated_step = step
                terminated_state = new_state.name
                print(
                    f"[ERROR] termination step={step} t={(step + 1) * dt:.3f}s "
                    f"state={new_state.name} request_phase={request_phase}",
                    flush=True,
                )
                break

            previous_state = new_state

        with csv_path.open("w", newline="", encoding="utf-8") as fobj:
            writer = csv.DictWriter(fobj, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        if terminated:
            fail(
                f"router runtime smoke terminated: step={terminated_step}, state={terminated_state}"
            )

        if len(rows) != total_steps:
            fail(f"incomplete trajectory {len(rows)} vs expected {total_steps}")

        expected_transitions = [
            ("STAND", "LOCO"),
            ("LOCO", "BRAKE"),
            ("BRAKE", "LOCO"),
            ("LOCO", "BRAKE"),
            ("BRAKE", "STAND"),
        ]
        observed_pairs = [(a, b) for a, b, _, _, _ in transitions]
        if observed_pairs != expected_transitions:
            fail(
                "unexpected router transition sequence: "
                f"{observed_pairs} != {expected_transitions}"
            )

        if canceled_brake_lengths != [cancel_steps]:
            fail(
                f"unexpected canceled BRAKE length {canceled_brake_lengths}; "
                f"expected [{cancel_steps}]"
            )

        if completed_brake_lengths != [BRAKE_STEPS]:
            fail(
                f"completed BRAKE length {completed_brake_lengths}; "
                f"expected [{BRAKE_STEPS}]"
            )

        # Hysteresis band must remain LOCO for every band-hold request step.
        band_rows = [r for r in rows if r["request_phase"] == "REQ_HYSTERESIS_BAND"]
        if len(band_rows) != band_steps:
            fail("unexpected hysteresis-band row count")
        if any(r["router_state"] != "LOCO" for r in band_rows):
            fail("router chatter/exit occurred inside hysteresis band")

        # Final requested stop should end in a real STAND segment.
        final_stand_rows = [r for r in rows if r["router_state"] == "STAND"][-final_stand_steps:]
        if len(final_stand_rows) != final_stand_steps:
            fail("final STAND duration shorter than requested")
        if any(r["active_policy"] != "v2-G" for r in final_stand_rows):
            fail("final STAND contains non-v2-G policy")

        final2 = tail_stats(rows, 2.0)

        script_path = Path(__file__).resolve()
        script_hash = sha256(script_path)

        summary_lines = [
            "v2-G / v4-N ROUTER + HYSTERESIS RUNTIME SMOKE",
            f"timestamp={timestamp}",
            f"seed={args_cli.seed}",
            f"script={script_path}",
            f"script_sha256={script_hash}",
            f"stand_checkpoint_sha256={stand_sha}",
            f"loco_checkpoint_sha256={loco_sha}",
            f"step_dt={dt}",
            f"enter_loco_speed={ENTER_LOCO_SPEED}",
            f"exit_loco_speed={EXIT_LOCO_SPEED}",
            "threshold_status=PROVISIONAL_ENGINEERING_ONLY_NOT_CALIBRATED",
            f"brake_s={BRAKE_S}",
            f"brake_steps={BRAKE_STEPS}",
            "dwell_status=FROZEN_ENGINEERING_HANDOFF_PRIMITIVE",
            f"startup_root_velocity_baseline={baseline_root_velocity_range}",
            f"startup_root_velocity_test={zero_root_velocity_range}",
            "startup_conditioning_role=engineering-only; isolates router/handoff from randomized v2-G startup robustness",
            f"transition_sequence={observed_pairs}",
            f"canceled_brake_lengths={canceled_brake_lengths}",
            f"completed_brake_lengths={completed_brake_lengths}",
            f"final2_mean_vxy_mps={final2['mean_vxy_mps']}",
            f"final2_max_vxy_mps={final2['max_vxy_mps']}",
            f"final2_mean_tilt_deg={final2['mean_tilt_deg']}",
            f"final2_max_tilt_deg={final2['max_tilt_deg']}",
            f"final2_mean_abs_wz_radps={final2['mean_abs_wz_radps']}",
            f"final2_double_support_pct={final2['double_support_pct']}",
            "termination=NO",
            "scientific_role=FUNCTIONAL_INTEGRATION_ONLY",
            "threshold_calibration=OUT_OF_SCOPE",
            "blending=ABSENT",
            PASS_MARKER,
        ]
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        print()
        print("-" * 124)
        print("RUNTIME ROUTER VALIDATION")
        print("-" * 124)
        print(f"transition sequence       : {observed_pairs}")
        print(f"canceled BRAKE lengths    : {canceled_brake_lengths} steps")
        print(f"completed BRAKE lengths   : {completed_brake_lengths} steps")
        print(f"hysteresis-band rows      : {len(band_rows)} / {band_steps}, all LOCO")
        print(f"final STAND rows          : {len(final_stand_rows)} / {final_stand_steps}, all v2-G")
        print("")
        print("FINAL 2 s")
        print(f"mean_vxy_mps              : {final2['mean_vxy_mps']:.9f}")
        print(f"max_vxy_mps               : {final2['max_vxy_mps']:.9f}")
        print(f"mean_tilt_deg             : {final2['mean_tilt_deg']:.9f}")
        print(f"max_tilt_deg              : {final2['max_tilt_deg']:.9f}")
        print(f"mean_abs_wz_radps         : {final2['mean_abs_wz_radps']:.9f}")
        print(f"double_support_pct        : {final2['double_support_pct']:.9f}")
        print("")
        print(f"trajectory CSV            : {csv_path}")
        print(f"summary                   : {summary_path}")
        print(f"script SHA256             : {script_hash}")
        print("termination               : NO")
        print("ROUTER LOGIC              : RUNTIME FUNCTIONAL")
        print("THRESHOLD STATUS          : PROVISIONAL / NOT CALIBRATED")
        print("DWELL STATUS              : FROZEN 0.36 s")
        print("BLENDING                  : ABSENT")
        print(PASS_MARKER)
        print("-" * 124)

    finally:
        if env is not None:
            env.close()
        gc.collect()


try:
    main()
except BaseException as exc:
    print(FAIL_MARKER, flush=True)
    print(f"EXCEPTION_TYPE={type(exc).__name__}", flush=True)
    print(f"EXCEPTION_MESSAGE={exc}", flush=True)
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
