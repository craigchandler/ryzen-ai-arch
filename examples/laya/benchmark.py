#!/usr/bin/env python3
"""
Benchmark the tested fixed-shape, logits-only Laya model on CPU and Ryzen AI NPU.

The benchmark measures steady-state inference only:

- model compilation and session creation are excluded from timings;
- CPU and NPU sessions remain alive for the complete benchmark;
- NPU output is checked for finite values;
- every measured NPU result is compared with the CPU reference;
- batch latency, per-decision latency and throughput are reported.

On the tested Ryzen AI 1.8 Arch/CachyOS environment, the raw Vitis cache was
not reliably reusable from a second Python process. Use --fresh-cache when
reproducing the published results so compilation and benchmarking occur in
the same process.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


SEQ_LEN = 512
MAX_OPTIONS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model/laya-logits.onnx"),
    )

    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("model/tokenizer"),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("vai_ep_config.json"),
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
    )

    parser.add_argument(
        "--cache-key",
        default=None,
        help=(
            "Vitis cache key. If omitted, a separate key is generated "
            "from the selected batch size."
        ),
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        choices=[1, 8, 16],
    )

    parser.add_argument(
        "--warmups",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--fresh-cache",
        action="store_true",
        help="Delete only the selected benchmark cache before compilation.",
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help=(
            "Optional result JSON path. "
            "Defaults to benchmark-results-batch-N.json."
        ),
    )

    return parser.parse_args()


def build_single_input(
    tokenizer_path: Path,
) -> tuple[dict[str, np.ndarray], list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )

    question = "Which action should be taken first?"

    state = (
        "A production web service has stopped accepting requests. "
        "Monitoring shows that the database disk is completely full."
    )

    options = [
        "Restart the web server",
        "Free space on the database disk",
        "Wait and see if the service recovers",
    ]

    ids = [tokenizer.cls_token_id]

    ids.extend(
        tokenizer(
            f"choice question: {question}",
            add_special_tokens=False,
        )["input_ids"]
    )

    ids.append(tokenizer.sep_token_id)

    markers: list[int] = []

    for option in options:
        markers.append(len(ids))

        ids.append(tokenizer.mask_token_id)

        ids.extend(
            tokenizer(
                " " + option,
                add_special_tokens=False,
            )["input_ids"][:48]
        )

    ids.append(tokenizer.sep_token_id)

    room = SEQ_LEN - len(ids) - 1

    if room <= 0:
        raise RuntimeError("No sequence space remains for the state text")

    ids.extend(
        tokenizer(
            state,
            add_special_tokens=False,
        )["input_ids"][:room]
    )

    ids.append(tokenizer.sep_token_id)

    real_length = len(ids)

    if real_length > SEQ_LEN:
        raise RuntimeError(
            f"Input length {real_length} exceeds fixed sequence length "
            f"{SEQ_LEN}"
        )

    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else 0
    )

    ids += [pad_id] * (SEQ_LEN - len(ids))

    attention = (
        [1] * real_length
        + [0] * (SEQ_LEN - real_length)
    )

    marker_pos = (
        markers
        + [0] * (MAX_OPTIONS - len(markers))
    )

    marker_mask = (
        [True] * len(markers)
        + [False] * (MAX_OPTIONS - len(markers))
    )

    inputs = {
        "input_ids": np.asarray(
            [ids],
            dtype=np.int64,
        ),
        "attention_mask": np.asarray(
            [attention],
            dtype=np.int64,
        ),
        "marker_pos": np.asarray(
            [marker_pos],
            dtype=np.int64,
        ),
        "marker_mask": np.asarray(
            [marker_mask],
            dtype=np.bool_,
        ),
        "qtype": np.asarray(
            [0],
            dtype=np.int64,
        ),
    }

    return inputs, options


def make_batch(
    single: dict[str, np.ndarray],
    batch_size: int,
) -> dict[str, np.ndarray]:
    return {
        name: np.repeat(
            value,
            batch_size,
            axis=0,
        )
        for name, value in single.items()
    }


def make_session_options(
    batch_size: int,
) -> ort.SessionOptions:
    options = ort.SessionOptions()

    options.add_free_dimension_override_by_name(
        "batch",
        batch_size,
    )

    options.add_free_dimension_override_by_name(
        "seq",
        SEQ_LEN,
    )

    options.add_free_dimension_override_by_name(
        "options",
        MAX_OPTIONS,
    )

    return options


def make_cpu_session(
    model: Path,
    batch_size: int,
) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(model),
        sess_options=make_session_options(batch_size),
        providers=[
            "CPUExecutionProvider",
        ],
    )


def make_npu_session(
    args: argparse.Namespace,
    cache_key: str,
) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(args.model),
        sess_options=make_session_options(args.batch),
        providers=[
            "VitisAIExecutionProvider",
            "CPUExecutionProvider",
        ],
        provider_options=[
            {
                "config_file": str(
                    args.config.resolve()
                ),
                "cache_dir": str(
                    args.cache_dir.resolve()
                ),
                "cache_key": cache_key,
                "enable_cache_file_io_in_mem": "0",
            },
            {},
        ],
    )


def collect_stats(
    values: list[float],
) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "p95_ms": float(
            np.percentile(
                values,
                95,
            )
        ),
    }


def warm_up(
    session: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
    iterations: int,
    reference: np.ndarray | None = None,
    tolerance: float = 1e-3,
) -> None:
    for index in range(iterations):
        output = session.run(
            None,
            inputs,
        )[0]

        if not np.all(
            np.isfinite(output)
        ):
            raise RuntimeError(
                "Non-finite output during warm-up "
                f"{index + 1}"
            )

        if reference is not None:
            diff = float(
                np.max(
                    np.abs(
                        reference - output
                    )
                )
            )

            if diff > tolerance:
                raise RuntimeError(
                    "CPU/NPU mismatch during warm-up "
                    f"{index + 1}: {diff} > {tolerance}"
                )


def timed_runs(
    session: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
    iterations: int,
    reference: np.ndarray | None = None,
    tolerance: float = 1e-3,
) -> tuple[list[float], list[float]]:
    times_ms: list[float] = []
    differences: list[float] = []

    for index in range(iterations):
        start = time.perf_counter_ns()

        output = session.run(
            None,
            inputs,
        )[0]

        elapsed_ms = (
            time.perf_counter_ns() - start
        ) / 1_000_000

        times_ms.append(
            elapsed_ms
        )

        if not np.all(
            np.isfinite(output)
        ):
            raise RuntimeError(
                "Non-finite output on measured run "
                f"{index + 1}"
            )

        if reference is not None:
            diff = float(
                np.max(
                    np.abs(
                        reference - output
                    )
                )
            )

            differences.append(
                diff
            )

            if diff > tolerance:
                raise RuntimeError(
                    "CPU/NPU mismatch on measured run "
                    f"{index + 1}: {diff} > {tolerance}"
                )

    return times_ms, differences


def print_stats(
    name: str,
    values: list[float],
    batch_size: int,
) -> tuple[
    dict[str, float],
    float,
    float,
]:
    result = collect_stats(
        values
    )

    median_per_decision = (
        result["median_ms"]
        / batch_size
    )

    throughput = (
        batch_size
        * 1000.0
        / result["median_ms"]
    )

    print(f"\n{name}")
    print(f"  Runs:             {len(values)}")
    print(f"  Batch size:       {batch_size}")
    print(f"  Mean batch:       {result['mean_ms']:.2f} ms")
    print(f"  Median batch:     {result['median_ms']:.2f} ms")
    print(f"  Min batch:        {result['min_ms']:.2f} ms")
    print(f"  Max batch:        {result['max_ms']:.2f} ms")
    print(f"  P95 batch:        {result['p95_ms']:.2f} ms")
    print(
        "  Median/decision:  "
        f"{median_per_decision:.2f} ms"
    )
    print(
        "  Throughput:       "
        f"{throughput:.2f} decisions/s"
    )

    return (
        result,
        throughput,
        median_per_decision,
    )


def main() -> None:
    args = parse_args()

    if (
        "VitisAIExecutionProvider"
        not in ort.get_available_providers()
    ):
        raise SystemExit(
            "VitisAIExecutionProvider is not available. "
            "Activate the Ryzen AI environment and source "
            "/opt/xilinx/xrt/setup.sh first."
        )

    cache_key = (
        args.cache_key
        or (
            f"laya-logits-b{args.batch}"
            f"-s{SEQ_LEN}"
            f"-k{MAX_OPTIONS}"
            "-benchmark"
        )
    )

    selected_cache = (
        args.cache_dir
        / cache_key
    )

    if (
        args.fresh_cache
        and selected_cache.exists()
    ):
        print(
            "Removing benchmark cache:",
            selected_cache,
        )

        shutil.rmtree(
            selected_cache
        )

    args.cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    single_inputs, options = (
        build_single_input(
            args.tokenizer
        )
    )

    inputs = make_batch(
        single_inputs,
        args.batch,
    )

    print(
        "Laya CPU vs NPU benchmark"
    )
    print(
        "========================="
    )
    print(f"Batch:     {args.batch}")
    print(f"Model:     {args.model}")
    print(f"Cache key: {cache_key}")
    print(f"Warm-ups:  {args.warmups}")
    print(f"Runs:      {args.iterations}")
    print(
        "Input:     "
        f"{inputs['input_ids'].shape}"
    )

    print("\nCreating CPU session...")

    cpu = make_cpu_session(
        args.model,
        args.batch,
    )

    print("Creating NPU session...")
    print(
        "Compilation/session creation is "
        "NOT included in benchmark timings."
    )

    npu = make_npu_session(
        args,
        cache_key,
    )

    print(
        "NPU providers:",
        npu.get_providers(),
    )

    reference = cpu.run(
        None,
        inputs,
    )[0]

    if not np.all(
        np.isfinite(reference)
    ):
        raise RuntimeError(
            "CPU reference contains "
            "non-finite values"
        )

    print(
        f"\nCPU warm-up "
        f"({args.warmups})..."
    )

    warm_up(
        cpu,
        inputs,
        args.warmups,
    )

    print(
        f"NPU warm-up "
        f"({args.warmups})..."
    )

    warm_up(
        npu,
        inputs,
        args.warmups,
        reference=reference,
        tolerance=args.tolerance,
    )

    print(
        f"\nBenchmarking CPU "
        f"({args.iterations})..."
    )

    cpu_times, _ = timed_runs(
        cpu,
        inputs,
        args.iterations,
    )

    print(
        f"Benchmarking NPU "
        f"({args.iterations})..."
    )

    npu_times, differences = (
        timed_runs(
            npu,
            inputs,
            args.iterations,
            reference=reference,
            tolerance=args.tolerance,
        )
    )

    (
        cpu_stats,
        cpu_throughput,
        cpu_per_decision,
    ) = print_stats(
        "CPU",
        cpu_times,
        args.batch,
    )

    (
        npu_stats,
        npu_throughput,
        npu_per_decision,
    ) = print_stats(
        "NPU",
        npu_times,
        args.batch,
    )

    speedup = (
        cpu_stats["median_ms"]
        / npu_stats["median_ms"]
    )

    throughput_gain = (
        (
            npu_throughput
            / cpu_throughput
        )
        - 1.0
    ) * 100.0

    max_diff = (
        max(differences)
        if differences
        else 0.0
    )

    k = len(options)

    cpu_logits = (
        reference[0, :k]
    )

    final_npu = npu.run(
        None,
        inputs,
    )[0]

    npu_logits = (
        final_npu[0, :k]
    )

    final_diff = float(
        np.max(
            np.abs(
                reference - final_npu
            )
        )
    )

    max_diff = max(
        max_diff,
        final_diff,
    )

    print("\nComparison")
    print(
        "  NPU speedup:      "
        f" {speedup:.3f}x"
    )
    print(
        "  Throughput gain:  "
        f" {throughput_gain:.1f}%"
    )
    print(
        "  CPU throughput:   "
        f" {cpu_throughput:.2f} decisions/s"
    )
    print(
        "  NPU throughput:   "
        f" {npu_throughput:.2f} decisions/s"
    )

    print("\nCorrectness")
    print(
        "  Maximum output difference: "
        f"{max_diff:.8f}"
    )
    print(
        "  Finite NPU output: True"
    )

    print(
        "\nCPU logits:",
        cpu_logits,
    )

    print(
        "NPU logits:",
        npu_logits,
    )

    print(
        "\nRaw CPU timings (ms):"
    )
    print(
        ", ".join(
            f"{value:.2f}"
            for value in cpu_times
        )
    )

    print(
        "\nRaw NPU timings (ms):"
    )
    print(
        ", ".join(
            f"{value:.2f}"
            for value in npu_times
        )
    )

    result = {
        "benchmark": {
            "batch_size": args.batch,
            "sequence_length": SEQ_LEN,
            "max_options": MAX_OPTIONS,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "cache_key": cache_key,
        },
        "cpu": {
            **cpu_stats,
            "median_ms_per_decision":
                cpu_per_decision,
            "decisions_per_second":
                cpu_throughput,
            "raw_ms":
                cpu_times,
        },
        "npu": {
            **npu_stats,
            "median_ms_per_decision":
                npu_per_decision,
            "decisions_per_second":
                npu_throughput,
            "raw_ms":
                npu_times,
        },
        "comparison": {
            "speedup_x":
                speedup,
            "throughput_gain_percent":
                throughput_gain,
        },
        "correctness": {
            "finite_npu_output":
                True,
            "maximum_output_difference":
                max_diff,
            "tolerance":
                args.tolerance,
            "cpu_logits":
                cpu_logits.tolist(),
            "npu_logits":
                npu_logits.tolist(),
        },
    }

    output_path = (
        args.json_output
        or Path(
            "benchmark-results-"
            f"batch-{args.batch}.json"
        )
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\nResults JSON:",
        output_path,
    )


if __name__ == "__main__":
    main()