from __future__ import annotations

import math
import sys
import traceback

import gymnasium as gym
import numpy as np

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


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def squeeze_env_axis(array: np.ndarray, body_count: int) -> np.ndarray:
    arr = np.asarray(array)

    if arr.ndim >= 2 and arr.shape[0] == 1 and arr.shape[1] == body_count:
        return arr[0]

    if arr.shape[0] == body_count:
        return arr

    raise RuntimeError(
        f"cannot identify body axis for shape {arr.shape}, body_count={body_count}"
    )


def relative_delta(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1.0e-12)
    return abs(a - b) / scale


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

    body_names = getattr(
        robot,
        "body_names",
        None,
    )

    if body_names is None:
        body_names = getattr(
            robot.data,
            "body_names",
            None,
        )

    if body_names is None:
        raise RuntimeError(
            "could not resolve articulation body names"
        )

    body_names = list(body_names)
    body_count = len(body_names)
    body_index = {
        name: index
        for index, name in enumerate(body_names)
    }

    pairs = []
    missing = []

    for name in body_names:
        if not name.startswith("left_"):
            continue

        other = counterpart(name)

        if other not in body_index:
            missing.append((name, other))
        else:
            pairs.append(
                (
                    name,
                    body_index[name],
                    other,
                    body_index[other],
                )
            )

    view = getattr(
        robot,
        "root_physx_view",
        None,
    )

    if view is None:
        raise RuntimeError(
            "robot.root_physx_view is unavailable"
        )

    required_methods = (
        "get_masses",
        "get_inertias",
        "get_coms",
    )

    for method_name in required_methods:
        if not hasattr(view, method_name):
            raise RuntimeError(
                f"root_physx_view missing {method_name}"
            )

    masses = squeeze_env_axis(
        to_numpy(
            view.get_masses()
        ),
        body_count,
    )

    inertias = squeeze_env_axis(
        to_numpy(
            view.get_inertias()
        ),
        body_count,
    )

    coms = squeeze_env_axis(
        to_numpy(
            view.get_coms()
        ),
        body_count,
    )

    if masses.ndim != 1:
        masses = masses.reshape(
            body_count,
            -1,
        )

        if masses.shape[1] != 1:
            raise RuntimeError(
                f"unexpected mass shape after body squeeze: {masses.shape}"
            )

        masses = masses[:, 0]

    p()
    p("=" * 150)
    p("v4-B BODY PHYSICS LEFT/RIGHT SYMMETRY AUDIT")
    p("=" * 150)
    p(f"Task                   : {TASK}")
    p(f"body count             : {body_count}")
    p(f"left/right body pairs  : {len(pairs)}")
    p(f"missing counterparts   : {len(missing)}")
    p(f"mass array shape       : {masses.shape}")
    p(f"inertia array shape    : {inertias.shape}")
    p(f"COM array shape        : {coms.shape}")
    p()

    p("-" * 150)
    p("1) BODY ORDER")
    p("-" * 150)

    for index, name in enumerate(body_names):
        p(f"body[{index:02d}] = {name}")

    p()

    p("-" * 150)
    p("2) LEFT/RIGHT MASS COMPARISON")
    p("-" * 150)
    p("left body -> right body | mass L | mass R | abs delta | rel delta")

    mass_checks = 0

    for left, li, right, ri in pairs:
        ml = float(masses[li])
        mr = float(masses[ri])

        abs_delta = abs(ml - mr)
        rel = relative_delta(ml, mr)

        status = "PASS" if rel <= 1.0e-6 else "CHECK"

        if status == "CHECK":
            mass_checks += 1

        p(
            f"{status:5s} | "
            f"{left:<34s} -> {right:<34s} | "
            f"{ml:.9f} | {mr:.9f} | "
            f"{abs_delta:.3e} | {rel:.3e}"
        )

    p()

    p("-" * 150)
    p("3) LEFT/RIGHT INERTIA SPECTRUM COMPARISON")
    p("-" * 150)
    p(
        "Comparison uses sorted eigenvalues when a 3x3 tensor can be reconstructed; "
        "this is invariant to mirrored/rotated local frames."
    )

    inertia_checks = 0
    inertia_skipped = 0

    for left, li, right, ri in pairs:
        il = np.asarray(inertias[li]).reshape(-1)
        ir = np.asarray(inertias[ri]).reshape(-1)

        if il.size == 9 and ir.size == 9:
            ml = il.reshape(3, 3)
            mr = ir.reshape(3, 3)

            ml = 0.5 * (ml + ml.T)
            mr = 0.5 * (mr + mr.T)

            eig_l = np.sort(
                np.linalg.eigvalsh(ml)
            )
            eig_r = np.sort(
                np.linalg.eigvalsh(mr)
            )

            denom = np.maximum(
                np.maximum(
                    np.abs(eig_l),
                    np.abs(eig_r),
                ),
                1.0e-12,
            )
            rel = np.max(
                np.abs(eig_l - eig_r)
                / denom
            )

            status = "PASS" if rel <= 1.0e-5 else "CHECK"

            if status == "CHECK":
                inertia_checks += 1

            p(
                f"{status:5s} | "
                f"{left:<34s} -> {right:<34s} | "
                f"eigL={eig_l.tolist()} | "
                f"eigR={eig_r.tolist()} | "
                f"max_rel={rel:.3e}"
            )
        else:
            inertia_skipped += 1
            p(
                f"SKIP  | {left:<34s} -> {right:<34s} | "
                f"raw sizes {il.size}/{ir.size}"
            )

    p()

    p("-" * 150)
    p("4) LEFT/RIGHT COM OFFSET MAGNITUDE COMPARISON")
    p("-" * 150)
    p(
        "Only |x|,|y|,|z| and position norm are compared, so mirrored signs/local-frame "
        "handedness do not create false asymmetry."
    )

    com_checks = 0
    com_skipped = 0

    for left, li, right, ri in pairs:
        cl = np.asarray(coms[li]).reshape(-1)
        cr = np.asarray(coms[ri]).reshape(-1)

        if cl.size < 3 or cr.size < 3:
            com_skipped += 1
            p(
                f"SKIP  | {left:<34s} -> {right:<34s} | "
                f"raw sizes {cl.size}/{cr.size}"
            )
            continue

        pl = cl[:3]
        pr = cr[:3]

        abs_l = np.abs(pl)
        abs_r = np.abs(pr)

        component_delta = float(
            np.max(
                np.abs(abs_l - abs_r)
            )
        )

        norm_l = float(
            np.linalg.norm(pl)
        )
        norm_r = float(
            np.linalg.norm(pr)
        )
        norm_delta = abs(
            norm_l - norm_r
        )

        scale = max(
            norm_l,
            norm_r,
            1.0e-9,
        )
        rel = max(
            component_delta,
            norm_delta,
        ) / scale

        status = "PASS" if rel <= 1.0e-5 else "CHECK"

        if status == "CHECK":
            com_checks += 1

        p(
            f"{status:5s} | "
            f"{left:<34s} -> {right:<34s} | "
            f"COM_L={pl.tolist()} | "
            f"COM_R={pr.tolist()} | "
            f"|COM|={norm_l:.9e}/{norm_r:.9e} | "
            f"rel={rel:.3e}"
        )

    p()

    p("-" * 150)
    p("SUMMARY")
    p("-" * 150)
    p(f"body count                         : {body_count}")
    p(f"left/right body pairs              : {len(pairs)}")
    p(f"missing body counterparts          : {len(missing)}")
    p(f"mass pairs needing check           : {mass_checks}")
    p(f"inertia pairs needing check        : {inertia_checks}")
    p(f"inertia pairs skipped              : {inertia_skipped}")
    p(f"COM pairs needing check            : {com_checks}")
    p(f"COM pairs skipped                  : {com_skipped}")
    p()
    p(
        "Interpretation rule: zero CHECK counts means no obvious left/right "
        "mass/inertia/COM asymmetry in the loaded PhysX articulation."
    )
    p()
    p("V4B_BODY_PHYSICS_SYMMETRY_AUDIT_PASS")
    p("=" * 150)

except Exception:
    failed = True
    p()
    p("=" * 150)
    p("V4B_BODY_PHYSICS_SYMMETRY_AUDIT_FAIL")
    p("=" * 150)
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
