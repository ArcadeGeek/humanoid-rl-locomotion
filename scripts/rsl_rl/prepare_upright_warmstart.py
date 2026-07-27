"""Prepare a conservative RSL-RL warm-start checkpoint.

The actor, critic, and iteration counter are retained. Adam moments are cleared
and every optimizer parameter group receives the requested learning rate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.source.resolve() == args.destination.resolve():
        raise ValueError("Source and destination must be different files.")

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)

    required_keys = {
        "actor_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "iter",
    }
    missing = required_keys.difference(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint is missing keys: {sorted(missing)}")

    optimizer_state = checkpoint["optimizer_state_dict"]
    if not isinstance(optimizer_state, dict):
        raise TypeError("optimizer_state_dict is not a dictionary.")
    if "state" not in optimizer_state or "param_groups" not in optimizer_state:
        raise KeyError("optimizer_state_dict lacks state or param_groups.")
    if not optimizer_state["param_groups"]:
        raise ValueError("optimizer_state_dict contains no parameter groups.")

    state_entries_before = len(optimizer_state["state"])
    optimizer_state["state"] = {}

    for group in optimizer_state["param_groups"]:
        group["lr"] = args.learning_rate
        if "initial_lr" in group:
            group["initial_lr"] = args.learning_rate

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.destination)

    verified = torch.load(args.destination, map_location="cpu", weights_only=False)
    verified_optimizer = verified["optimizer_state_dict"]
    verified_lrs = [float(group["lr"]) for group in verified_optimizer["param_groups"]]

    if verified_optimizer["state"]:
        raise RuntimeError("Optimizer state was not cleared.")
    if any(abs(lr - args.learning_rate) > 1.0e-12 for lr in verified_lrs):
        raise RuntimeError(f"Unexpected optimizer learning rates: {verified_lrs}")

    print("UPRIGHT_WARMSTART_OK")
    print(f"SOURCE={args.source.resolve()}")
    print(f"DESTINATION={args.destination.resolve()}")
    print(f"ITERATION={verified['iter']}")
    print(f"OPTIMIZER_STATE_ENTRIES_BEFORE={state_entries_before}")
    print("OPTIMIZER_STATE_ENTRIES_AFTER=0")
    print(f"OPTIMIZER_LRS={verified_lrs}")
    print(f"SIZE_BYTES={args.destination.stat().st_size}")


if __name__ == "__main__":
    main()
