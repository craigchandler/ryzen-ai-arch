# Laya on the Ryzen AI XDNA2 NPU

This document records the real-world model validation used after the base Arch/CachyOS Ryzen AI stack was working.

## Model

The test uses [`receptron/laya-onnx`](https://huggingface.co/receptron/laya-onnx), an ONNX export of [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya).

The published model card describes an English 421M-parameter ModernBERT-based model with these inputs:

```text
input_ids       [B,L] int64
attention_mask  [B,L] int64
marker_pos      [B,K] int64
marker_mask     [B,K] bool
qtype           [B]   int64
```

and outputs:

```text
logits      [B,K] float32
act_probs   [B,2] float32
```

The bundle contains `laya.onnx`, `laya.onnx.data`, `laya_config.json` and tokenizer assets.

## Download

With the Ryzen AI Python environment active:

```bash
pip install huggingface_hub transformers
mkdir -p ~/Source/laya-npu
cd ~/Source/laya-npu

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="receptron/laya-onnx",
    local_dir="model",
)
PY
```

Do not replace AMD's ONNX Runtime package with a stock PyPI `onnxruntime` package; the Ryzen AI environment needs AMD's `VitisAIExecutionProvider` build.

## Confirm CPU loading first

```bash
python - <<'PY'
import onnxruntime as ort

s = ort.InferenceSession(
    "model/laya.onnx",
    providers=["CPUExecutionProvider"],
)

print("Providers:", s.get_providers())
print("Inputs:")
for x in s.get_inputs():
    print(" ", x.name, x.shape, x.type)

print("Outputs:")
for x in s.get_outputs():
    print(" ", x.name, x.shape, x.type)
PY
```

The original model loaded successfully on CPU.

## Dynamic-shape compile failure

The public ONNX graph uses symbolic `batch`, `seq` and `options` dimensions. A direct Vitis AI compile loaded the graph but crashed during compilation after seeing dynamic shapes.

For the tested NPU build, ONNX Runtime free-dimension overrides were used:

```python
so = ort.SessionOptions()
so.add_free_dimension_override_by_name("batch", 1)
so.add_free_dimension_override_by_name("seq", 512)
so.add_free_dimension_override_by_name("options", 20)
```

Vitis then saw:

```text
input_ids       [1x512]
attention_mask  [1x512]
marker_pos      [1x20]
marker_mask     [1x20]
qtype           [1]

logits          [1x20]
act_probs       [1x2]
```

and progressed through graph partitioning and compilation.

The fixed `options=20` shape does not mean every request needs 20 real choices. Unused entries are padded and disabled through `marker_mask`.

## Multi-output failure

The original graph compiled and initialized through `VitisAIExecutionProvider`, but both outputs were `NaN` during the tested inference.

CPU reference:

```text
CPU logits: [-1.7870014  3.3337939 -1.7052455]
CPU act_probs: [1. 0.]
```

NPU with the original two-output graph:

```text
NPU logits: [nan nan nan]
NPU act_probs: [nan nan]
```

This resembles AMD RyzenAI-SW issue #369, which documents incorrect or NaN results for some multi-output ONNX graphs with `VitisAIExecutionProvider`:

<https://github.com/amd/RyzenAI-SW/issues/369>

That issue does not establish that every multi-output model is affected. It is recorded here because the observed Laya failure mode is consistent with it.

## Single-output workaround

Create a graph that exposes only `logits` while retaining the same external weights:

```bash
python examples/laya/make_single_output.py \
  model/laya.onnx \
  model/laya-logits.onnx
```

The generated graph should remain in the same directory as `laya.onnx.data` so the external-data reference remains valid.

Use a separate Vitis cache key for the modified graph, for example:

```text
laya-logits-b1-s512-k20
```

## Verified partitioning

For the final single-output model, Vitis reported:

```text
Number of operators in the model: 1611
Number of operators supported by VAIML: 1598 (99.193%)
GOPs of the model: 420.556
GOPs supported by VAIML: 420.546 (99.998%)
Number of subgraphs supported by VAIML: 4
```

The execution-provider report counted:

```text
1461 VAIML
119 CPU
```

The largest identified VAIML subgraph contained:

```text
1460 operators
391.888 GOPs
93.183% of model compute
```

The low-cost CPU nodes do not materially change the main result: essentially all measured model GOPs were considered VAIML-supported.

## Verified inference

A test choice question used this state:

```text
A production web service has stopped accepting requests.
Monitoring shows that the database disk is completely full.
```

Choices:

```text
Restart the web server
Free space on the database disk
Wait and see if the service recovers
```

CPU probabilities:

```text
0.0059  Restart the web server
0.9877  Free space on the database disk
0.0064  Wait and see if the service recovers
```

Single-output NPU probabilities:

```text
0.0059  Restart the web server
0.9877  Free space on the database disk
0.0064  Wait and see if the service recovers
```

Raw logits:

```text
CPU: [-1.7870014  3.3337939 -1.7052455]
NPU: [-1.7870014  3.3337939 -1.7052455]
Max observed difference: 0.0
Finite NPU output: True
```

The Vitis execution-provider report independently confirmed VAIML assignment, so the matching result was not simply an unobserved all-CPU fallback.

## Benchmarking

Do not publish a speedup based on a single timing. Use the supplied benchmark script:

```bash
python examples/laya/benchmark.py \
  --model model/laya-logits.onnx \
  --tokenizer model/tokenizer \
  --config vai_ep_config.json \
  --cache-dir cache \
  --cache-key laya-logits-b1-s512-k20
```

The default run performs 5 warm-up iterations and 20 measured iterations on each provider and checks every NPU result for finite values and agreement with the CPU reference.
