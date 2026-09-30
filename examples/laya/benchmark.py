#!/usr/bin/env python3
"""Benchmark the tested fixed-shape, logits-only Laya model on CPU and Ryzen AI NPU.

This intentionally validates correctness as well as latency. It is designed for
AMD's Ryzen AI Python environment where VitisAIExecutionProvider is available.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

SEQ_LEN = 512
MAX_OPTIONS = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("model/laya-logits.onnx"))
    p.add_argument("--tokenizer", type=Path, default=Path("model/tokenizer"))
    p.add_argument("--config", type=Path, default=Path("vai_ep_config.json"))
    p.add_argument("--cache-dir", type=Path, default=Path("cache"))
    p.add_argument("--cache-key", default="laya-logits-b1-s512-k20")
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--atol", type=float, default=1e-3)
    return p.parse_args()


def build_inputs(tokenizer_path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

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
            tokenizer(" " + option, add_special_tokens=False)["input_ids"][:48]
        )

    ids.append(tokenizer.sep_token_id)
    room = SEQ_LEN - len(ids) - 1
    ids.extend(tokenizer(state, add_special_tokens=False)["input_ids"][:room])
    ids.append(tokenizer.sep_token_id)

    real_length = len(ids)
    if real_length > SEQ_LEN:
        raise RuntimeError(f"Input length {real_length} exceeds fixed sequence length")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ids += [pad_id] * (SEQ_LEN - len(ids))

    attention = [1] * real_length + [0] * (SEQ_LEN - real_length)
    marker_pos = markers + [0] * (MAX_OPTIONS - len(markers))
    marker_mask = [True] * len(markers) + [False] * (MAX_OPTIONS - len(markers))

    inputs = {
        "input_ids": np.asarray([ids], dtype=np.int64),
        "attention_mask": np.asarray([attention], dtype=np.int64),
        "marker_pos": np.asarray([marker_pos], dtype=np.int64),
        "marker_mask": np.asarray([marker_mask], dtype=np.bool_),
        "qtype": np.asarray([0], dtype=np.int64),
    }
    return inputs, options


def make_cpu_session(model: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])


def make_npu_session(args: argparse.Namespace) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.add_free_dimension_override_by_name("batch", 1)
    so.add_free_dimension_override_by_name("seq", SEQ_LEN)
    so.add_free_dimension_override_by_name("options", MAX_OPTIONS)

    return ort.InferenceSession(
        str(args.model),
        sess_options=so,
        providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        provider_options=[
            {
                "config_file": str(args.config.resolve()),
                "cache_dir": str(args.cache_dir.resolve()),
                "cache_key": args.cache_key,
                "enable_cache_file_io_in_mem": "0",
            },
            {},
        ],
    )


def timed_runs(
    session: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
    warmups: int,
    iterations: int,
) -> tuple[np.ndarray, list[float]]:
    for _ in range(warmups):
        session.run(None, inputs)

    times_ms: list[float] = []
    last = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        last = session.run(None, inputs)[0]
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    assert last is not None
    return last, times_ms


def stats(name: str, values: list[float]) -> None:
    print(f"{name}:")
    print(f"  iterations : {len(values)}")
    print(f"  mean       : {statistics.fmean(values):.2f} ms")
    print(f"  median     : {statistics.median(values):.2f} ms")
    print(f"  min        : {min(values):.2f} ms")
    print(f"  max        : {max(values):.2f} ms")


def main() -> None:
    args = parse_args()

    if "VitisAIExecutionProvider" not in ort.get_available_providers():
        raise SystemExit(
            "VitisAIExecutionProvider is not available. Activate the Ryzen AI "
            "environment and source /opt/xilinx/xrt/setup.sh first."
        )

    inputs, options = build_inputs(args.tokenizer)

    print("Creating CPU session...")
    cpu = make_cpu_session(args.model)
    print("Creating NPU session...")
    npu = make_npu_session(args)
    print("NPU providers:", npu.get_providers())

    cpu_output, cpu_times = timed_runs(cpu, inputs, args.warmups, args.iterations)
    npu_output, npu_times = timed_runs(npu, inputs, args.warmups, args.iterations)

    k = len(options)
    cpu_logits = cpu_output[0, :k]
    npu_logits = npu_output[0, :k]

    if not np.all(np.isfinite(npu_logits)):
        raise SystemExit(f"NPU returned non-finite logits: {npu_logits}")

    if not np.allclose(cpu_logits, npu_logits, rtol=args.rtol, atol=args.atol):
        diff = np.max(np.abs(cpu_logits - npu_logits))
        raise SystemExit(
            f"NPU output does not match CPU within tolerance; max abs diff={diff}"
        )

    cpu_probs = np.exp(cpu_logits - cpu_logits.max())
    cpu_probs /= cpu_probs.sum()
    npu_probs = np.exp(npu_logits - npu_logits.max())
    npu_probs /= npu_probs.sum()

    print("\nDecision probabilities:")
    for option, cp, np_ in zip(options, cpu_probs, npu_probs):
        print(f"  CPU {cp:8.4f} | NPU {np_:8.4f} | {option}")

    print("\nCorrectness:")
    print("  CPU logits:", cpu_logits)
    print("  NPU logits:", npu_logits)
    print("  max abs diff:", float(np.max(np.abs(cpu_logits - npu_logits))))
    print("  finite NPU output: True")

    print()
    stats("CPU", cpu_times)
    print()
    stats("NPU", npu_times)

    cpu_median = statistics.median(cpu_times)
    npu_median = statistics.median(npu_times)
    if cpu_median > 0:
        reduction = (1.0 - npu_median / cpu_median) * 100.0
        print(f"\nMedian latency change: {reduction:.2f}%")


if __name__ == "__main__":
    main()
