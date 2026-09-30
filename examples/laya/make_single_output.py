#!/usr/bin/env python3
"""Create a logits-only Laya ONNX graph without loading external weights into RAM.

Keep the generated ONNX file in the same directory as laya.onnx.data so the
external-data reference remains valid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Source laya.onnx")
    parser.add_argument("destination", type=Path, help="Output logits-only ONNX")
    args = parser.parse_args()

    model = onnx.load(args.source, load_external_data=False)
    outputs = [output.name for output in model.graph.output]

    if "logits" not in outputs:
        raise SystemExit(f"Expected a 'logits' output; found: {outputs}")

    logits = next(output for output in model.graph.output if output.name == "logits")
    del model.graph.output[:]
    model.graph.output.append(logits)

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, args.destination)

    print(f"Source outputs: {outputs}")
    print("New outputs: ['logits']")
    print(f"Saved: {args.destination}")
    print("Keep this file beside the original external weight file (laya.onnx.data).")


if __name__ == "__main__":
    main()
