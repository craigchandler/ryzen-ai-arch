# AMD Ryzen AI / XDNA2 on Arch Linux and CachyOS

A reproducible setup for running AMD Ryzen AI / XDNA2 NPU workloads on Arch Linux and CachyOS while retaining the Linux kernel's **in-tree `amdxdna` driver**.

This repository documents a tested stack using XRT, the XDNA userspace plugin, Ryzen AI Software 1.8, ONNX Runtime with `VitisAIExecutionProvider`, and a real-world Laya inference workload.

> **Last tested:** 1 October 2026  
> **Status:** Working on the hardware/software combination below. Treat the versions as a known-good set rather than assuming arbitrary future kernel/XRT/Ryzen AI combinations are interchangeable.

## Tested system

### Hardware

```text
CPU/NPU : AMD Ryzen AI 9 HX 470 w/ Radeon 890M
NPU     : AMD XDNA2
Device  : NPU Gorgon Point 1
AIE     : aie2p, 6x8
RAM     : 32 GB
```

### Software

```text
Distribution       : CachyOS / Arch Linux
Kernel             : 7.2.8-1-cachyos
amdxdna            : in-tree kernel driver
XRT                 : 2.26.0
XDNA userspace      : 2.26.0
NPU firmware        : 1.1.2.64
Ryzen AI Software   : 1.8
ONNX Runtime        : 1.27.0
Python              : 3.12.14 via uv
Execution provider  : VitisAIExecutionProvider
```

## Why this setup is different

AMD's current [`xdna-driver`](https://github.com/amd/xdna-driver) repository supports Arch Linux and supplies Arch packaging for XRT, the XDNA driver and the userspace plugin. Its standard Arch path packages the external/DKMS `amdxdna` driver as a dependency of the plugin.

On this CachyOS system the in-tree kernel driver already detected and operated the NPU correctly. Replacing it added an unnecessary kernel/compiler dependency, so this setup instead:

1. keeps the in-tree `amdxdna` driver;
2. builds XRT;
3. builds the XDNA project with `-nokmod`;
4. packages only the XDNA userspace plugin.

AMD's official Ryzen AI Linux instructions currently target Ubuntu and Python 3.12. The Ryzen AI portions of this guide are an Arch adaptation of that environment:

<https://ryzenai.docs.amd.com/en/latest/linux.html>

## Repository layout

```text
README.md
pkgbuild/
  PKGBUILD-xrt-plugin-amdxdna-intree
docs/
  laya.md
  troubleshooting.md
examples/
  laya/
    make_single_output.py
    benchmark.py
scripts/
  activate-ryzen-ai.sh
```

## 1. Verify that the kernel already supports the NPU

Before building anything:

```bash
uname -r
lspci -nnk | grep -A4 -i -E '17f0|neural|signal processing'
lsmod | grep amdxdna
ls -l /dev/accel 2>/dev/null
journalctl -b -k | grep -i amdxdna
```

The tested machine showed:

```text
Kernel driver in use: amdxdna
/dev/accel/accel0
firmware amdnpu/17f0_10/npu_7.sbin
```

If `/dev/accel/accel0` exists and `amdxdna` is loaded, do not assume you need another kernel driver.

### Check for a kernel/header mismatch

During this setup, the running kernel was initially `7.2.7` while the installed kernel and headers were `7.2.8`. That made `modinfo amdxdna` misleading until the machine was rebooted into the matching kernel.

Check before debugging driver builds:

```bash
uname -r
pacman -Q linux-cachyos linux-cachyos-headers
```

## 2. Configure memlock

AMD's Arch instructions require sufficient locked memory for NPU access and recommend configuring it through `limits.d`.

```bash
sudo mkdir -p /etc/security/limits.d
sudo tee /etc/security/limits.d/99-amdxdna.conf >/dev/null <<'LIMITS'
* soft memlock unlimited
* hard memlock unlimited
LIMITS
```

Log out and back in, or reboot, then verify:

```bash
ulimit -Sl
ulimit -Hl
```

Both should report `unlimited`.

## 3. Clone XDNA and build XRT

AMD's repository uses submodules, including XRT:

```bash
git clone https://github.com/amd/xdna-driver.git
cd xdna-driver
git submodule update --init --recursive
```

Build XRT:

```bash
cd xrt/build
./build.sh -npu -opt
```

### XRT Arch PKGBUILD version mismatch encountered

The tested XRT build generated:

```text
xrt_202620.2.26.0_--base.tar.gz
xrt_202620.2.26.0_--npu.tar.gz
```

while the supplied PKGBUILDs still contained `pkgver=202620.2.25.0`.

The double `--` in the generated filenames was not the problem; the stale `pkgver` was.

For that source revision, the fix was:

```bash
cd arch
sed -i 's/pkgver=202620\.2\.25\.0/pkgver=202620.2.26.0/' \
  PKGBUILD-xrt-base PKGBUILD-xrt-npu
```

Then:

```bash
makepkg -C -p PKGBUILD-xrt-base
sudo pacman -U ./xrt-base-202620.2.26.0-1-x86_64.pkg.tar.zst

makepkg -C -p PKGBUILD-xrt-npu
sudo pacman -U ./xrt-npu-202620.2.26.0-1-x86_64.pkg.tar.zst
```

Do not copy these exact version substitutions blindly for a newer XRT checkout; inspect the generated archive version first.

## 4. Build only the XDNA userspace plugin

A normal full XDNA build tried to compile `amdxdna.ko`. On this system CachyOS had built the kernel with Clang 22.1.8, while the XDNA build invoked GCC 16.2.1. GCC rejected Clang-specific kernel flags.

Forcing Clang globally made the kernel module build but caused the userspace shim to fail because new Clang warnings were promoted to errors.

Neither was necessary: keep the working in-tree driver and use AMD's `-nokmod` build mode.

```bash
cd ~/Source/Personal/xdna-driver/build
./build.sh -clean

unset CC
unset CXX
unset LLVM
unset LLVM_IAS

./build.sh -release -nokmod
```

The tested build produced:

```text
Release/xrt_plugin.2.26.0_-x86_64-amdxdna.tar.gz
```

containing userspace components such as `libvxdna.so` and `libxrt_driver_xdna.so`, without a replacement kernel module.

## 5. Package the userspace plugin

Use [`pkgbuild/PKGBUILD-xrt-plugin-amdxdna-intree`](pkgbuild/PKGBUILD-xrt-plugin-amdxdna-intree).

From `xdna-driver/build/arch`:

```bash
makepkg -C -p /path/to/ryzen-ai-arch/pkgbuild/PKGBUILD-xrt-plugin-amdxdna-intree
```

Or copy the PKGBUILD into that directory first and run:

```bash
makepkg -C -p PKGBUILD-xrt-plugin-amdxdna-intree
sudo pacman -U ./xrt-plugin-amdxdna-intree-2.26.0-1-x86_64.pkg.tar.zst
```

## 6. Validate XRT and the NPU

```bash
source /opt/xilinx/xrt/setup.sh
xrt-smi examine
xrt-smi validate
```

The tested machine reported:

```text
XRT Version          : 2.26.0
amdxdna Version      : 7.2.8-1-cachyos
NPU Firmware Version : 1.1.2.64
Device               : NPU Gorgon Point 1
Architecture         : aie2p
Topology             : 6x8
```

Validation results:

```text
GEMM       : 51.0 TOPS   PASSED
Latency    : 57.0 us     PASSED
Throughput : 94922 ops/s PASSED
```

Some `xrt-smi examine --report all` telemetry fields were `N/A` on this particular in-tree-driver/userspace combination, while execution and validation still worked.

## 7. Install Python 3.12 without replacing Arch Python

Ryzen AI Software 1.8's Linux instructions use Python 3.12. Arch on the tested machine was already on Python 3.14, so Python 3.12 was installed with `uv`:

```bash
uv python install 3.12
uv python find 3.12
```

Expose it to AMD's installer:

```bash
PY312="$(uv python find 3.12)"
export PATH="$(dirname "$PY312"):$PATH"
python3.12 --version
```

Tested version:

```text
Python 3.12.14
```

## 8. Install Ryzen AI Software 1.8

Install Arch equivalents of the Ubuntu prerequisites checked by the installer:

```bash
sudo pacman -S --needed linux-api-headers zip
```

Then from the extracted Ryzen AI 1.8 directory:

```bash
PY312="$(uv python find 3.12)"
export PATH="$(dirname "$PY312"):$PATH"

./install_ryzen_ai.sh \
  -a yes \
  -p ~/Source/ryzen_ai/venv
```

Activate it:

```bash
source ~/Source/ryzen_ai/venv/bin/activate
source /opt/xilinx/xrt/setup.sh
```

## 9. Arch compatibility fixes required on the tested system

### Executable-stack metadata

Importing AMD's ONNX Runtime initially failed with:

```text
cannot enable executable stack as shared object requires: Invalid argument
```

The affected object was `onnxruntime_pybind11_state.so`.

```bash
sudo pacman -S --needed patchelf

SO=~/Source/ryzen_ai/venv/lib/python3.12/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.so
patchelf --print-execstack "$SO"
patchelf --clear-execstack "$SO"
patchelf --print-execstack "$SO"
```

The final state should be `execstack: -`.

### `libncurses.so.6`

The Vitis AI EP expected `libncurses.so.6`, while Arch supplied the ABI-compatible wide-character runtime as `/usr/lib/libncursesw.so.6`.

Keep the compatibility link private rather than modifying `/usr/lib`:

```bash
mkdir -p ~/Source/ryzen_ai/compat-lib
ln -sf /usr/lib/libncursesw.so.6 \
  ~/Source/ryzen_ai/compat-lib/libncurses.so.6
```

### `libpython3.12.so.1.0`

FlexML also expected the Python 3.12 shared library on a normal runtime library path. With `uv`, it is under the managed Python installation.

```bash
PYBASE="$(python -c 'import sys; print(sys.base_prefix)')"
export LD_LIBRARY_PATH="$HOME/Source/ryzen_ai/compat-lib:$PYBASE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Verify:

```bash
ldd ~/Source/ryzen_ai/venv/lib/python3.12/site-packages/flexmlrt/lib/libflexmlrt.so \
  | grep -E 'libpython|not found'
```

The provided [`scripts/activate-ryzen-ai.sh`](scripts/activate-ryzen-ai.sh) wraps the activation steps used by this setup.

## 10. Verify the Ryzen AI runtime

```bash
python - <<'PY'
import onnxruntime as ort
print(ort.__version__)
print(ort.get_available_providers())
PY
```

The tested environment returned:

```text
1.27.0
['VitisAIExecutionProvider', 'CPUExecutionProvider']
```

AMD's bundled quicktest then passed:

```bash
cd ~/Source/ryzen_ai/venv/quicktest
python quicktest.py
```

A successful quicktest proves an actual model can compile and execute through the NPU path, not merely that the PCI device is visible.

## 11. Real-world validation: Laya

The real-world test used the public [`receptron/laya-onnx`](https://huggingface.co/receptron/laya-onnx) export of [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya), a 421M-parameter ModernBERT-based decision model.

Key results on the final single-output graph:

```text
Operators in model           : 1611
Operators supported by VAIML : 1598 (99.193%)
Model GOPs                   : 420.556
GOPs supported by VAIML      : 420.546 (99.998%)
VAIML subgraphs              : 4
EP report                    : 1461 VAIML / 119 CPU
```

The published two-output graph (`logits`, `act_probs`) compiled but returned `NaN` on this tested stack.

AMD issue #369 documents incorrect or NaN output for some multi-output ONNX graphs under `VitisAIExecutionProvider`:

<https://github.com/amd/RyzenAI-SW/issues/369>

The issue does **not** show that every multi-output graph is affected.

A single-output graph containing only `logits` produced correct finite results matching the CPU reference for the test input.

See [`docs/laya.md`](docs/laya.md) for the complete model-specific workflow.

## 12. Laya benchmark results

The single-output Laya model was benchmarked with:

- 5 warm-up iterations per provider;
- 20 measured iterations per provider;
- fixed sequence length `512`;
- 20 option slots;
- CPU and NPU sessions kept alive for the complete test;
- compilation/session creation excluded from measured latency;
- every NPU result checked for finite values and agreement with the CPU reference.

### Batch 1

```text
CPU mean             : 1192.48 ms
CPU median           : 1196.04 ms
CPU P95              : 1245.68 ms

NPU mean             : 1165.34 ms
NPU median           : 1156.77 ms
NPU P95              : 1210.85 ms

Median NPU speedup   : 1.034x
Median reduction     : 3.3%
Maximum output diff  : 0.00000000
```

### Batch 8

```text
CPU median batch     : 8951.86 ms
CPU median/decision  : 1118.98 ms
CPU throughput       : 0.89 decisions/s

NPU median batch     : 8993.51 ms
NPU median/decision  : 1124.19 ms
NPU throughput       : 0.89 decisions/s

NPU speedup          : 0.995x
Throughput gain      : -0.5%
Maximum output diff  : 0.00000000
```

### Batch 16

```text
CPU median batch     : 21700.34 ms
CPU median/decision  : 1356.27 ms
CPU throughput       : 0.74 decisions/s

NPU median batch     : 21752.76 ms
NPU median/decision  : 1359.55 ms
NPU throughput       : 0.74 decisions/s

NPU speedup          : 0.998x
Throughput gain      : -0.2%
Maximum output diff  : 0.00000000
```

### Interpretation

For this model on this Ryzen AI 9 HX 470 system, the XDNA2 NPU does **not** provide a meaningful end-to-end latency or throughput advantage over CPU execution.

Batch 1 showed a small 3.3% median latency advantage for the NPU. At batch sizes 8 and 16, CPU and NPU throughput were effectively identical.

Batch 16 was slower per decision than batch 8 on both providers, so increasing the batch beyond 8 did not improve this workload.

This does not imply that the NPU is generally no faster than the CPU. Accelerator performance is workload-dependent. The result here is specifically that **Laya is a useful functional NPU validation workload, but not a performance showcase for XDNA2 on this CPU**.

A remaining useful comparison would be host CPU utilisation under sustained CPU-provider versus NPU-provider inference. Similar latency could still be operationally useful if NPU execution leaves substantially more CPU capacity available to other workloads.

## 13. Compiled cache persistence limitation

The raw Vitis cache used during the Laya tests could not be reliably reloaded by a second Python process.

After a successful compile and inference, a later process failed with:

```text
Failed to open file:
.../vaiml_par_0/partition-info.json

Failed to parse JSON ...
attempting to parse an empty input

HW context creation unsuccessful
```

The failure was reproduced without rebooting.

At the time of the failure the environment remained:

```text
Kernel             : 7.2.8-1-cachyos
amdxdna            : 7.2.8-1-cachyos
XRT                 : 2.26.0
XDNA userspace      : 2.26.0
NPU firmware        : 1.1.2.64
```

so a version change did not explain the failed reload.

AMD issue #324 describes a similar class of disk-cache/load problem when using `enable_cache_file_io_in_mem=0`:

<https://github.com/amd/RyzenAI-SW/issues/324>

For reproducible benchmarking on this setup, a fresh cache is compiled and the benchmark is performed in the **same Python process** that creates the NPU session.

An ONNX Runtime EP Context Cache experiment was also attempted. After compilation work, ONNX Runtime reported:

```text
Unable to compile any nodes.
ONNX Runtime will not generate a compiled model.
Either the session EPs do not support compilation
or the model is already compiled.
```

No reusable EP-context model was generated in this tested configuration.

This is documented as an observed limitation of the tested stack, not as a general statement that Vitis AI caching or EP Context Cache cannot work on other models or supported platforms.

## Benchmarking

Use [`examples/laya/benchmark.py`](examples/laya/benchmark.py).

For the tested environment, use a fresh cache so compilation and benchmarking happen within the same process:

```bash
python examples/laya/benchmark.py \
  --batch 1 \
  --fresh-cache
```

Batch throughput tests:

```bash
python examples/laya/benchmark.py \
  --batch 8 \
  --fresh-cache

python examples/laya/benchmark.py \
  --batch 16 \
  --fresh-cache
```

The script defaults to 5 warm-ups and 20 measured iterations and validates every NPU output against the CPU reference.

## Troubleshooting

All failure modes encountered during the setup are collected in [`docs/troubleshooting.md`](docs/troubleshooting.md).

## Result

The final tested path is:

```text
Laya / ONNX
      ↓
ONNX Runtime 1.27.0
      ↓
VitisAIExecutionProvider
      ↓
Ryzen AI Software 1.8
      ↓
XRT 2.26.0
      ↓
XDNA userspace plugin 2.26.0
      ↓
Linux in-tree amdxdna
      ↓
AMD XDNA2 NPU
```

The important outcome is not merely that Linux detects the NPU: a real transformer-derived ONNX model was compiled, substantially offloaded to VAIML, executed successfully, numerically validated against CPU execution, and benchmarked while retaining the CachyOS in-tree `amdxdna` driver.

For Laya specifically, the benchmark shows CPU/NPU performance parity rather than a substantial NPU speedup.

## References

- AMD XDNA Linux driver: <https://github.com/amd/xdna-driver>
- AMD Ryzen AI Software documentation: <https://ryzenai.docs.amd.com/en/latest/>
- AMD Ryzen AI Linux installation: <https://ryzenai.docs.amd.com/en/latest/linux.html>
- AMD RyzenAI-SW examples: <https://github.com/amd/RyzenAI-SW>
- AMD multi-output Vitis AI issue #369: <https://github.com/amd/RyzenAI-SW/issues/369>
- AMD cache issue #324: <https://github.com/amd/RyzenAI-SW/issues/324>
- Laya: <https://huggingface.co/convaiinnovations/laya>
- Laya ONNX export: <https://huggingface.co/receptron/laya-onnx>