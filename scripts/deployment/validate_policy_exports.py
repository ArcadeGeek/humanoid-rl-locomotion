"""Validate exported Unitree G1 locomotion policies.

Checks artifact hashes, TorchScript and ONNX interfaces, numerical parity,
finite outputs, and inference latency. The RSL-RL checkpoint is hashed but
is not numerically executed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch


EXPECTED_INPUT_DIM = 123
EXPECTED_OUTPUT_DIM = 37
DEFAULT_WARMUP_RUNS = 100
DEFAULT_BENCHMARK_RUNS = 1000
PARITY_MAX_ABS_TOL = 1.0e-5
LATENCY_P99_LIMIT_MS = 20.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def latency_stats(values_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values_ms),
        "p50_ms": percentile(values_ms, 50),
        "p95_ms": percentile(values_ms, 95),
        "p99_ms": percentile(values_ms, 99),
        "max_ms": max(values_ms),
        "jitter_std_ms": statistics.pstdev(values_ms),
    }


def build_test_inputs() -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    inputs = [
        np.zeros((1, EXPECTED_INPUT_DIM), dtype=np.float32),
        np.linspace(-1.0, 1.0, EXPECTED_INPUT_DIM, dtype=np.float32).reshape(1, -1),
        np.linspace(1.0, -1.0, EXPECTED_INPUT_DIM, dtype=np.float32).reshape(1, -1),
    ]
    for scale in (0.01, 0.1, 0.5, 1.0, 2.0):
        inputs.append(
            rng.normal(0.0, scale, size=(1, EXPECTED_INPUT_DIM)).astype(np.float32)
        )
    return inputs


def synchronize_torch(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    default_artifact_dir = repo_root / "checkpoints" / "local_g1_upright_v1"

    parser = argparse.ArgumentParser(
        description="Validate Unitree G1 checkpoint, TorchScript, and ONNX artifacts."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=default_artifact_dir,
        help="Directory containing model_1010.pt, policy.pt, policy.onnx and policy.onnx.data.",
    )
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--benchmark-runs", type=int, default=DEFAULT_BENCHMARK_RUNS)
    parser.add_argument(
        "--torch-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for the machine-readable validation report.",
    )
    args = parser.parse_args()

    if args.warmup_runs < 1 or args.benchmark_runs < 1:
        raise ValueError("warmup-runs and benchmark-runs must both be positive.")

    artifact_dir = args.artifact_dir.resolve()
    paths = {
        "checkpoint": artifact_dir / "model_1010.pt",
        "torchscript": artifact_dir / "policy.pt",
        "onnx": artifact_dir / "policy.onnx",
        "onnx_external_data": artifact_dir / "policy.onnx.data",
    }

    report: dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "expected_interface": {
            "input_dim": EXPECTED_INPUT_DIM,
            "output_dim": EXPECTED_OUTPUT_DIM,
        },
        "artifacts": {},
        "checks": {},
        "parity": {},
        "latency": {},
        "pass": False,
    }
    failures: list[str] = []

    print("=== ARTIFACTS ===")
    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"Missing artifact: {path}")
            print(f"[FAIL] {name}: missing ({path})")
            continue
        file_info = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        report["artifacts"][name] = file_info
        print(f"[PASS] {name}: {file_info['sha256']} ({file_info['size_bytes']} bytes)")

    if failures:
        report["failures"] = failures
        print("\nRESULT: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    if args.torch_device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(args.torch_device)

    if torch_device.type == "cuda" and not torch.cuda.is_available():
        failures.append("Torch CUDA was requested but torch.cuda.is_available() is False.")

    print("\n=== TORCHSCRIPT ===")
    torch_policy = torch.jit.load(str(paths["torchscript"]), map_location=torch_device)
    torch_policy.eval()
    zero_torch = torch.zeros(
        (1, EXPECTED_INPUT_DIM), dtype=torch.float32, device=torch_device
    )
    with torch.inference_mode():
        zero_torch_output = torch_policy(zero_torch)
    torch_shape = tuple(int(value) for value in zero_torch_output.shape)
    torch_finite = bool(torch.isfinite(zero_torch_output).all().item())
    report["checks"]["torchscript"] = {
        "device": str(torch_device),
        "input_shape": [1, EXPECTED_INPUT_DIM],
        "output_shape": list(torch_shape),
        "finite": torch_finite,
    }
    print(f"Device: {torch_device}")
    print(f"Output shape: {torch_shape}")
    print(f"Finite: {torch_finite}")
    if torch_shape != (1, EXPECTED_OUTPUT_DIM):
        failures.append(f"TorchScript output shape is {torch_shape}, expected (1, 37).")
    if not torch_finite:
        failures.append("TorchScript produced NaN or Inf.")

    print("\n=== ONNX ===")
    onnx_model = onnx.load(str(paths["onnx"]), load_external_data=True)
    onnx.checker.check_model(onnx_model)
    available_providers = ort.get_available_providers()
    requested_providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available_providers
    ]
    if not requested_providers:
        failures.append("No supported ONNX Runtime provider is available.")
        requested_providers = ["CPUExecutionProvider"]

    ort_session = ort.InferenceSession(
        str(paths["onnx"]),
        providers=requested_providers,
    )
    ort_input = ort_session.get_inputs()[0]
    ort_output = ort_session.get_outputs()[0]
    ort_zero_output = ort_session.run(
        [ort_output.name],
        {ort_input.name: np.zeros((1, EXPECTED_INPUT_DIM), dtype=np.float32)},
    )[0]
    ort_shape = tuple(int(value) for value in ort_zero_output.shape)
    ort_finite = bool(np.isfinite(ort_zero_output).all())
    report["checks"]["onnx"] = {
        "checker": "pass",
        "available_providers": available_providers,
        "active_providers": ort_session.get_providers(),
        "input_name": ort_input.name,
        "input_shape": ort_input.shape,
        "input_type": ort_input.type,
        "output_name": ort_output.name,
        "output_shape": ort_output.shape,
        "output_type": ort_output.type,
        "runtime_output_shape": list(ort_shape),
        "finite": ort_finite,
    }
    print("ONNX checker: PASS")
    print(f"Active providers: {ort_session.get_providers()}")
    print(f"Input: {ort_input.name} {ort_input.shape} {ort_input.type}")
    print(f"Output: {ort_output.name} {ort_output.shape} {ort_output.type}")
    print(f"Runtime output shape: {ort_shape}")
    print(f"Finite: {ort_finite}")

    if ort_shape != (1, EXPECTED_OUTPUT_DIM):
        failures.append(f"ONNX output shape is {ort_shape}, expected (1, 37).")
    if not ort_finite:
        failures.append("ONNX Runtime produced NaN or Inf.")
    if "CUDAExecutionProvider" not in ort_session.get_providers():
        failures.append("CUDAExecutionProvider is not active.")

    print("\n=== TORCHSCRIPT / ONNX PARITY ===")
    all_differences: list[np.ndarray] = []
    test_inputs = build_test_inputs()
    with torch.inference_mode():
        for index, input_array in enumerate(test_inputs):
            torch_input = torch.from_numpy(input_array).to(torch_device)
            torch_output = torch_policy(torch_input).detach().cpu().numpy()
            onnx_output = ort_session.run(
                [ort_output.name],
                {ort_input.name: input_array},
            )[0]

            if not np.isfinite(torch_output).all():
                failures.append(f"TorchScript produced NaN/Inf for parity input {index}.")
            if not np.isfinite(onnx_output).all():
                failures.append(f"ONNX produced NaN/Inf for parity input {index}.")

            all_differences.append(
                np.abs(
                    torch_output.astype(np.float64)
                    - onnx_output.astype(np.float64)
                )
            )

    stacked_diff = np.concatenate([item.reshape(-1) for item in all_differences])
    max_abs = float(stacked_diff.max())
    mean_abs = float(stacked_diff.mean())
    rmse = float(math.sqrt(np.mean(np.square(stacked_diff))))
    report["parity"] = {
        "num_test_inputs": len(test_inputs),
        "num_compared_values": int(stacked_diff.size),
        "max_abs_difference": max_abs,
        "mean_abs_difference": mean_abs,
        "rmse": rmse,
        "max_abs_tolerance": PARITY_MAX_ABS_TOL,
        "pass": max_abs <= PARITY_MAX_ABS_TOL,
    }
    print(f"Test inputs: {len(test_inputs)}")
    print(f"Compared values: {stacked_diff.size}")
    print(f"Max abs difference: {max_abs:.10e}")
    print(f"Mean abs difference: {mean_abs:.10e}")
    print(f"RMSE: {rmse:.10e}")
    if max_abs > PARITY_MAX_ABS_TOL:
        failures.append(
            f"Parity max abs difference {max_abs:.3e} exceeds "
            f"{PARITY_MAX_ABS_TOL:.3e}."
        )

    print("\n=== LATENCY ===")
    benchmark_input_np = test_inputs[3]
    benchmark_input_torch = torch.from_numpy(benchmark_input_np).to(torch_device)

    with torch.inference_mode():
        for _ in range(args.warmup_runs):
            torch_policy(benchmark_input_torch)
        synchronize_torch(torch_device)

        torch_times_ms: list[float] = []
        for _ in range(args.benchmark_runs):
            synchronize_torch(torch_device)
            start = time.perf_counter_ns()
            torch_policy(benchmark_input_torch)
            synchronize_torch(torch_device)
            torch_times_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)

    for _ in range(args.warmup_runs):
        ort_session.run(
            [ort_output.name],
            {ort_input.name: benchmark_input_np},
        )

    onnx_times_ms: list[float] = []
    for _ in range(args.benchmark_runs):
        start = time.perf_counter_ns()
        ort_session.run(
            [ort_output.name],
            {ort_input.name: benchmark_input_np},
        )
        onnx_times_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)

    torch_latency = latency_stats(torch_times_ms)
    onnx_latency = latency_stats(onnx_times_ms)
    report["latency"] = {
        "warmup_runs": args.warmup_runs,
        "benchmark_runs": args.benchmark_runs,
        "p99_limit_ms": LATENCY_P99_LIMIT_MS,
        "torchscript": torch_latency,
        "onnx": onnx_latency,
    }

    print(
        "TorchScript: "
        f"mean={torch_latency['mean_ms']:.4f} ms, "
        f"p95={torch_latency['p95_ms']:.4f} ms, "
        f"p99={torch_latency['p99_ms']:.4f} ms, "
        f"max={torch_latency['max_ms']:.4f} ms"
    )
    print(
        "ONNX: "
        f"mean={onnx_latency['mean_ms']:.4f} ms, "
        f"p95={onnx_latency['p95_ms']:.4f} ms, "
        f"p99={onnx_latency['p99_ms']:.4f} ms, "
        f"max={onnx_latency['max_ms']:.4f} ms"
    )

    if torch_latency["p99_ms"] >= LATENCY_P99_LIMIT_MS:
        failures.append(
            f"TorchScript p99 latency is {torch_latency['p99_ms']:.3f} ms, "
            f"not below {LATENCY_P99_LIMIT_MS:.1f} ms."
        )
    if onnx_latency["p99_ms"] >= LATENCY_P99_LIMIT_MS:
        failures.append(
            f"ONNX p99 latency is {onnx_latency['p99_ms']:.3f} ms, "
            f"not below {LATENCY_P99_LIMIT_MS:.1f} ms."
        )

    report["pass"] = not failures
    report["failures"] = failures

    if args.json_out is not None:
        json_path = args.json_out.resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\nReport written to: {json_path}")

    print(f"\nRESULT: {'PASS' if report['pass'] else 'FAIL'}")
    if failures:
        for failure in failures:
            print(f"- {failure}")

    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
