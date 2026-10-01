# Laya on the Ryzen AI XDNA2 NPU

This document records the real-world model validation used after the base Arch/CachyOS Ryzen AI stack was working.

It covers model preparation, graph partitioning, correctness validation, performance benchmarking and the compiled-cache behaviour observed on the tested Ryzen AI 1.8 Linux stack.

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

Laya is a structured decision model rather than a generative LLM. Its outputs can be used for tasks such as ranking choices, scoring alternatives and yes/no decision support.

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

The public ONNX graph uses symbolic `batch`, `seq` and `options` dimensions.

A direct Vitis AI compile loaded the graph but failed during compilation when presented with the dynamic shapes.

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

For later throughput tests the `batch` override was changed to 8 and 16 while retaining sequence length 512 and 20 option slots.

## Multi-output failure

The original graph compiled and initialized through `VitisAIExecutionProvider`, but both outputs were `NaN` during the tested inference.

CPU reference:

```text
CPU logits:
[-1.7870014  3.3337939 -1.7052455]

CPU act_probs:
[1. 0.]
```

NPU with the original two-output graph:

```text
NPU logits:
[nan nan nan]

NPU act_probs:
[nan nan]
```

This resembles AMD RyzenAI-SW issue #369, which documents incorrect or NaN results for **some** multi-output ONNX graphs with `VitisAIExecutionProvider`:

<https://github.com/amd/RyzenAI-SW/issues/369>

That issue specifically documents working multi-output counterexamples, so this should not be interpreted as a blanket statement that Vitis AI cannot execute multi-output graphs.

## Single-output workaround

Create a graph that exposes only `logits` while retaining the same external weights:

```bash
python examples/laya/make_single_output.py \
  model/laya.onnx \
  model/laya-logits.onnx
```

The generated graph should remain in the same directory as `laya.onnx.data` so the external-data reference remains valid.

Use a separate Vitis cache key for the modified graph.

For example:

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

High VAIML coverage does not, however, guarantee an end-to-end latency advantage. Runtime scheduling, transfers, remaining CPU work and the model's tensor shapes still affect total execution time.

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
CPU:
[-1.7870014  3.3337939 -1.7052455]

NPU:
[-1.7870014  3.3337939 -1.7052455]

Max observed difference:
0.00000000

Finite NPU output:
True
```

The Vitis execution-provider report independently confirmed VAIML assignment, so the matching result was not simply an unobserved all-CPU fallback.

## Benchmark methodology

The benchmark uses [`examples/laya/benchmark.py`](../examples/laya/benchmark.py).

For each provider it performs:

```text
5 warm-up iterations
20 measured iterations
```

Compilation and session creation are excluded from timing.

Every measured NPU output is checked for:

```text
finite output
CPU/NPU agreement within tolerance
```

The batch tests repeat the same logical decision within the batch. That is intentional: these tests measure model compute throughput, not variation in decision quality.

Because compiled Vitis cache reload was unreliable on the tested system, the published tests use a fresh cache and perform compilation and inference within the same Python process.

## Batch 1 results

Tested shape:

```text
batch   = 1
seq     = 512
options = 20
```

CPU:

```text
Runs    : 20
Mean    : 1192.48 ms
Median  : 1196.04 ms
Min     : 1087.21 ms
Max     : 1291.13 ms
P95     : 1245.68 ms
```

NPU:

```text
Runs    : 20
Mean    : 1165.34 ms
Median  : 1156.77 ms
Min     : 1121.18 ms
Max     : 1228.78 ms
P95     : 1210.85 ms
```

Comparison:

```text
Median speedup           : 1.034x
Median latency reduction : 3.3%
```

Correctness:

```text
Maximum output difference: 0.00000000
```

The NPU was slightly faster in this run, but the difference is too small to describe as a substantial performance advantage.

## Batch 8 results

Tested shape:

```text
batch   = 8
seq     = 512
options = 20
```

CPU:

```text
Runs             : 20
Mean batch       : 8964.12 ms
Median batch     : 8951.86 ms
Min batch        : 8822.35 ms
Max batch        : 9136.58 ms
P95 batch        : 9103.74 ms
Median/decision  : 1118.98 ms
Throughput       : 0.89 decisions/s
```

NPU:

```text
Runs             : 20
Mean batch       : 9369.88 ms
Median batch     : 8993.51 ms
Min batch        : 8672.90 ms
Max batch        : 11278.05 ms
P95 batch        : 10996.21 ms
Median/decision  : 1124.19 ms
Throughput       : 0.89 decisions/s
```

Comparison:

```text
NPU speedup      : 0.995x
Throughput gain  : -0.5%
```

Correctness:

```text
Maximum output difference: 0.00000000
```

Batch 8 therefore produced effectively identical CPU and NPU throughput.

## Batch 16 results

Tested shape:

```text
batch   = 16
seq     = 512
options = 20
```

CPU:

```text
Runs             : 20
Mean batch       : 21683.78 ms
Median batch     : 21700.34 ms
Min batch        : 21227.84 ms
Max batch        : 22060.57 ms
P95 batch        : 21914.13 ms
Median/decision  : 1356.27 ms
Throughput       : 0.74 decisions/s
```

NPU:

```text
Runs             : 20
Mean batch       : 21740.49 ms
Median batch     : 21752.76 ms
Min batch        : 21279.26 ms
Max batch        : 22169.35 ms
P95 batch        : 22010.95 ms
Median/decision  : 1359.55 ms
Throughput       : 0.74 decisions/s
```

Comparison:

```text
NPU speedup      : 0.998x
Throughput gain  : -0.2%
```

Correctness:

```text
Maximum output difference: 0.00000000
```

Batch 16 again produced CPU/NPU performance parity.

It also reduced throughput for both providers relative to batch 8.

## Benchmark summary

| Batch | CPU median / decision | NPU median / decision | CPU throughput | NPU throughput | NPU speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 1196.04 ms | 1156.77 ms | ~0.84/s | ~0.86/s | 1.034x |
| 8 | 1118.98 ms | 1124.19 ms | 0.89/s | 0.89/s | 0.995x |
| 16 | 1356.27 ms | 1359.55 ms | 0.74/s | 0.74/s | 0.998x |

The results show that Laya is **not a workload where the XDNA2 NPU materially outperforms the Ryzen AI 9 HX 470 CPU**.

That is still a useful result:

- the NPU executes the model correctly;
- virtually all model GOPs are VAIML-supported;
- NPU and CPU logits match exactly in the measured tests;
- batching does not reveal a hidden NPU throughput advantage;
- batch 8 is slightly more efficient per decision than batch 1;
- batch 16 is worse for both providers.

This benchmark should therefore be described as a functional and compatibility validation rather than a demonstration of large NPU acceleration.

A separate host-resource benchmark would be useful to determine whether NPU execution reduces CPU utilisation sufficiently to be valuable while another CPU-heavy workload is running.

## Raw Vitis cache reload limitation

The NPU session compiled and executed successfully in the Python process that created it.

However, attempting to start another process using the same cache produced:

```text
Failed to open file:
.../vaiml_par_0/partition-info.json
```

followed by:

```text
Failed to parse JSON ...
attempting to parse an empty input
```

and:

```text
HW context creation unsuccessful,
check for XRT version or recompile model
with latest VAIML version
```

The failure was reproduced without rebooting.

The environment remained unchanged:

```text
Kernel             : 7.2.8-1-cachyos
amdxdna            : 7.2.8-1-cachyos
XRT                 : 2.26.0
XDNA plugin         : 2.26.0
NPU firmware        : 1.1.2.64
```

This rules out a kernel/XRT version change as the explanation for that specific failure.

AMD issue #324 records a similar class of disk-cache/load behaviour:

<https://github.com/amd/RyzenAI-SW/issues/324>

The practical workaround for these benchmarks is therefore:

```text
create fresh cache
        ↓
compile NPU session
        ↓
keep session alive
        ↓
warm up
        ↓
benchmark
```

For a persistent application, a long-running process could similarly keep the compiled ONNX Runtime session alive and service multiple inference requests without recompiling between requests.

A service restart would still require compilation until reliable cross-process persistence is available for this tested configuration.

## EP Context Cache experiment

An ONNX Runtime EP Context Cache was also tested as a possible persistent artifact.

The session used context configuration equivalent to:

```python
so.add_session_config_entry("ep.context_enable", "1")
so.add_session_config_entry(
    "ep.context_file_path",
    "laya-logits-b1-s512-k20-context.onnx",
)
so.add_session_config_entry("ep.context_embed_mode", "1")
```

After substantial compilation work, ONNX Runtime reported:

```text
Unable to compile any nodes.
ONNX Runtime will not generate a compiled model.
Either the session EPs do not support compilation
or the model is already compiled.
```

The expected context ONNX file was not created.

This records only the behaviour of this model and tested Ryzen AI 1.8 Linux configuration. It should not be interpreted as a general statement about EP Context Cache support for other execution providers, models or platforms.

## Reproducing the benchmarks

Batch 1:

```bash
python examples/laya/benchmark.py \
  --batch 1 \
  --fresh-cache
```

Batch 8:

```bash
python examples/laya/benchmark.py \
  --batch 8 \
  --fresh-cache
```

Batch 16:

```bash
python examples/laya/benchmark.py \
  --batch 16 \
  --fresh-cache
```

The script automatically uses separate cache keys for each batch size.