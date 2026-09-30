# AMD Ryzen AI / XDNA2 on Arch Linux and CachyOS

This guide documents a working AMD Ryzen AI NPU stack on CachyOS/Arch Linux using the Linux kernel's **in-tree `amdxdna` driver**, XRT, the XDNA userspace plugin, and AMD Ryzen AI SDK.

It also includes a real-world validation using **Laya**, a 421M-parameter ModernBERT-based decision model.

The important distinction from AMD's standard Arch packaging is that this setup **does not replace the working in-tree kernel driver with AMD's DKMS driver**. Only the userspace XDNA/XRT components are installed.

## Tested system

Hardware:

```text
CPU/NPU : AMD Ryzen AI 9 HX 470 w/ Radeon 890M
NPU     : AMD XDNA2
Device  : NPU Gorgon Point 1
AIE     : aie2p, 6x8
RAM     : 32 GB
```

Software:

```text
Distribution       : CachyOS / Arch Linux
Kernel             : 7.2.8-1-cachyos
amdxdna             : in-tree kernel driver
XRT                 : 2.26.0
XDNA userspace      : 2.26.0
NPU firmware        : 1.1.2.64
Ryzen AI SDK        : 1.8
ONNX Runtime        : 1.27.0
Python              : 3.12.14 via uv
Execution provider  : VitisAIExecutionProvider
```

AMD's current `xdna-driver` project explicitly lists Arch Linux as supported and documents XRT, the XDNA plugin and the required memlock configuration. Its normal packaging path installs AMD's external driver as well as the plugin; this guide deliberately retains the distro kernel driver instead.

---

## 1. Verify that the kernel already supports the NPU

Before building anything:

```bash
uname -r

lspci -nnk | grep -A4 -i -E '17f0|neural|signal processing'

lsmod | grep amdxdna

ls -l /dev/accel

journalctl -b -k | grep -i amdxdna
```

On the tested machine:

```text
Kernel driver in use: amdxdna
/dev/accel/accel0

amdxdna 0000:c5:00.1:
firmware amdnpu/17f0_10/npu_7.sbin

Initialized amdxdna_accel_driver
```

If `/dev/accel/accel0` exists and `amdxdna` is loaded, do not assume you need another kernel driver.

### Kernel upgrade caveat

During setup the running kernel was initially `7.2.7`, while the installed kernel and headers were `7.2.8`.

That caused confusing results such as:

```text
modinfo amdxdna
```

failing even though the driver was loaded.

After rebooting into the matching kernel:

```text
7.2.8-1-cachyos
```

`modinfo amdxdna` worked normally.

Always check:

```bash
uname -r
pacman -Q linux-cachyos linux-cachyos-headers
```

before debugging driver build problems.

---

## 2. Configure memlock

AMD requires sufficient locked memory for NPU buffer allocation. Their Arch instructions recommend unlimited memlock.

Create:

```bash
sudo mkdir -p /etc/security/limits.d

sudo tee /etc/security/limits.d/99-amdxdna.conf >/dev/null <<'EOF'
* soft memlock unlimited
* hard memlock unlimited
EOF
```

Log out and back in, or reboot.

Verify:

```bash
ulimit -Sl
ulimit -Hl
```

Both should report:

```text
unlimited
```

---

## 3. Build XRT

Clone AMD's XDNA driver repository including its XRT submodule.

From the XRT build directory:

```bash
cd ~/Source/Personal/xdna-driver/xrt/build

./build.sh -npu -opt
```

### Arch PKGBUILD version mismatch

In this build, XRT generated:

```text
xrt_202620.2.26.0_--base.tar.gz
xrt_202620.2.26.0_--npu.tar.gz
```

but the supplied Arch PKGBUILDs still contained:

```text
pkgver=202620.2.25.0
```

The double `--` in the generated filename is intentional. The issue was the stale package version.

Update the PKGBUILDs:

```bash
cd arch

sed -i 's/pkgver=202620\.2\.25\.0/pkgver=202620.2.26.0/' \
  PKGBUILD-xrt-base \
  PKGBUILD-xrt-npu
```

Build and install:

```bash
makepkg -C -p PKGBUILD-xrt-base

sudo pacman -U \
  ./xrt-base-202620.2.26.0-1-x86_64.pkg.tar.zst
```

Then:

```bash
makepkg -C -p PKGBUILD-xrt-npu

sudo pacman -U \
  ./xrt-npu-202620.2.26.0-1-x86_64.pkg.tar.zst
```

Verify:

```bash
pacman -Q xrt-base xrt-npu
ls /opt/xilinx/xrt/setup.sh
```

---

## 4. Build only the XDNA userspace plugin

This was the most important Arch/CachyOS-specific part of the setup.

A normal full XDNA build attempted to build `amdxdna.ko`.

CachyOS had built the kernel with:

```text
Clang 22.1.8
```

while the XDNA build invoked:

```text
GCC 16.2.1
```

GCC then failed on Clang-specific kernel flags including:

```text
-mstack-alignment=8
-mretpoline-external-thunk
-fexperimental-late-parse-attributes
-fsplit-lto-unit
-mllvm
```

Forcing Clang globally fixed the kernel-module build, but then caused the userspace XDNA shim to fail because Clang warnings were promoted to errors with `-Werror`.

Neither approach was necessary.

AMD's build supports `-nokmod`, allowing us to keep the working kernel driver and build only userspace:

```bash
cd ~/Source/Personal/xdna-driver/build

./build.sh -clean

unset CC
unset CXX
unset LLVM
unset LLVM_IAS

./build.sh -release -nokmod
```

The resulting archive was:

```text
Release/xrt_plugin.2.26.0_-x86_64-amdxdna.tar.gz
```

It contained userspace components such as:

```text
/opt/xilinx/xrt/lib/libvxdna.so
/opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2
/opt/xilinx/xrt/share/amdxdna/bins/...
```

and no replacement kernel module.

---

## 5. Package the userspace plugin for Arch

AMD's standard Arch plugin package expects its `amdxdna-driver` package and installation hooks.

For an in-tree-driver setup, create a userspace-only package.

Create:

```text
PKGBUILD-xrt-plugin-amdxdna-intree
```

with:

```bash
pkgname=xrt-plugin-amdxdna-intree
pkgver=2.26.0
pkgrel=1
pkgdesc="AMD XDNA XRT userspace plugin using the in-tree Linux amdxdna driver"
arch=('x86_64')
url="https://github.com/amd/xdna-driver/"
license=('Apache-2.0')

depends=('xrt-base' 'xrt-npu')

provides=("xrt-plugin-amdxdna=${pkgver}")
conflicts=('xrt-plugin-amdxdna')

options=('!debug' '!strip')

package() {
    local xdna_build_dir="${XDNA_BUILD_DIR:-$startdir/../Release}"
    local tarball="${xdna_build_dir}/xrt_plugin.${pkgver}_-${CARCH}-amdxdna.tar.gz"

    if [[ ! -f "$tarball" ]]; then
        error "XDNA plugin tarball not found: $tarball"
        return 1
    fi

    msg2 "Extracting $tarball"

    tar -xzf "$tarball" -C "$pkgdir"
}
```

Build:

```bash
makepkg -C -p PKGBUILD-xrt-plugin-amdxdna-intree
```

Install:

```bash
sudo pacman -U \
  ./xrt-plugin-amdxdna-intree-2.26.0-1-x86_64.pkg.tar.zst
```

---

## 6. Validate XRT and the NPU

Load the XRT environment:

```bash
source /opt/xilinx/xrt/setup.sh
```

Then:

```bash
xrt-smi examine
```

The tested system reported:

```text
XRT
 Version              : 2.26.0
 amdxdna Version      : 7.2.8-1-cachyos
 NPU Firmware Version : 1.1.2.64

Device(s) Present
[0000:c5:00.1] NPU Gorgon Point 1 aie2p 6x8
```

Now validate:

```bash
xrt-smi validate
```

Results on this machine:

```text
Test 1 gemm:
  TOPS: 51.0
  PASSED

Test 2 latency:
  Average latency: 57.0 us
  PASSED

Test 3 throughput:
  Average throughput: 94922.0 ops/s
  PASSED
```

At this point the kernel/XRT/XDNA stack is operational.

### Telemetry

On this particular in-tree-driver setup:

```bash
xrt-smi examine --report all
```

reported N/A for some telemetry such as power, load and temperature.

Execution nevertheless worked correctly.

AMD notes that newer XRT userspace can request ioctls not present in an older in-tree `amdxdna`, causing telemetry or array-query functionality to be unavailable while core execution still works.

Do not assume this particular in-tree/XRT combination will remain compatible indefinitely. Pin and record known-working versions.

---

# Ryzen AI SDK

## 7. Install Python 3.12 without changing Arch's system Python

Ryzen AI 1.8 expects Python 3.12.

The tested Arch installation was already on Python 3.14, so Python 3.12 was installed using `uv` rather than replacing the system interpreter:

```bash
uv python install 3.12

uv python find 3.12
```

Result:

```text
~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12
```

Expose it temporarily:

```bash
PY312="$(uv python find 3.12)"
export PATH="$(dirname "$PY312"):$PATH"
```

Verify:

```bash
python3.12 --version
```

Tested version:

```text
Python 3.12.14
```

---

## 8. Install Ryzen AI SDK 1.8

The AMD installer contains some Ubuntu-specific `dpkg` checks.

Install the Arch equivalents:

```bash
sudo pacman -S --needed linux-api-headers zip
```

Then:

```bash
cd ~/Source/ryzen_ai

PY312="$(uv python find 3.12)"
export PATH="$(dirname "$PY312"):$PATH"

./install_ryzen_ai.sh \
  -a yes \
  -p ~/Source/ryzen_ai/venv
```

Activate:

```bash
source ~/Source/ryzen_ai/venv/bin/activate
source /opt/xilinx/xrt/setup.sh
```

---

## 9. Fix executable-stack metadata

Initially:

```python
import onnxruntime
```

failed with:

```text
cannot enable executable stack as shared object requires: Invalid argument
```

The affected library was:

```text
onnxruntime/capi/onnxruntime_pybind11_state.so
```

Install `patchelf`:

```bash
sudo pacman -S --needed patchelf
```

Check:

```bash
SO=~/Source/ryzen_ai/venv/lib/python3.12/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.so

patchelf --print-execstack "$SO"
```

It reported:

```text
execstack: X
```

Clear it:

```bash
patchelf --clear-execstack "$SO"
```

Verify:

```bash
patchelf --print-execstack "$SO"
```

Expected:

```text
execstack: -
```

ONNX Runtime then loaded correctly:

```python
import onnxruntime as ort

print(ort.__version__)
print(ort.get_available_providers())
```

Result:

```text
1.27.0
['VitisAIExecutionProvider', 'CPUExecutionProvider']
```

---

## 10. Arch compatibility libraries

Two more Ubuntu assumptions appeared while running real models.

### `libncurses.so.6`

AMD's Vitis AI binary expected:

```text
libncurses.so.6
```

Arch provided:

```text
/usr/lib/libncursesw.so.6
```

Rather than modifying `/usr/lib`, create an isolated compatibility directory:

```bash
mkdir -p ~/Source/ryzen_ai/compat-lib

ln -sf /usr/lib/libncursesw.so.6 \
  ~/Source/ryzen_ai/compat-lib/libncurses.so.6
```

### `libpython3.12.so.1.0`

The FlexML runtime also expected the shared Python library on the system library path.

The `uv` Python contained:

```text
~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/lib/libpython3.12.so.1.0
```

Set:

```bash
PYBASE="$(python -c 'import sys; print(sys.base_prefix)')"

export LD_LIBRARY_PATH="$HOME/Source/ryzen_ai/compat-lib:$PYBASE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Verify:

```bash
ldd \
  ~/Source/ryzen_ai/venv/lib/python3.12/site-packages/flexmlrt/lib/libflexmlrt.so \
  | grep -E 'libpython|not found'
```

No dependency should report `not found`.

These environment settings are good candidates for a small activation script rather than placing them globally in the shell environment.

---

## 11. Run AMD quicktest

Run AMD's bundled Ryzen AI quicktest.

A successful run verifies:

```text
Python
  ↓
ONNX Runtime
  ↓
VitisAIExecutionProvider
  ↓
XRT
  ↓
XDNA userspace
  ↓
amdxdna
  ↓
NPU
```

The quicktest passed on the tested system.

At this point the Ryzen AI platform installation was considered complete.

---

# Real-world test: Laya

## 12. Model

Laya is a 421M-parameter ModernBERT-based decision model.

The published ONNX export has these inputs:

```text
input_ids       [B,L] int64
attention_mask  [B,L] int64
marker_pos      [B,K] int64
marker_mask     [B,K] bool
qtype           [B]   int64
```

and outputs:

```text
logits      [B,K]
act_probs   [B,2]
```

The public ONNX bundle contains the FP32 graph and external weights.

---

## 13. Dynamic shapes do not compile cleanly

The original graph used dynamic dimensions:

```text
batch
seq
options
```

Vitis loaded the graph but crashed while compiling the dynamic version.

For the first NPU build, the dimensions were fixed through ONNX Runtime:

```python
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

logits           [1x20]
act_probs        [1x2]
```

and successfully compiled the model.

---

## 14. Vitis partitioning result

For the final single-output Laya graph:

```text
Number of operators in model:
1611

Operators supported by VAIML:
1598
99.193%

GOPs in model:
420.556

GOPs supported by VAIML:
420.546
99.998%

VAIML subgraphs:
4
```

The execution-provider report contained:

```text
1461 VAIML
119 CPU
```

The most computationally expensive subgraph alone contained:

```text
1460 operators
391.888 GOPs
93.183% of total model compute
```

So although some graph operations remain on CPU, essentially all significant model compute is NPU-capable.

---

# Laya multi-output issue

## 15. Original graph produced NaNs

The original Laya ONNX model has two outputs:

```text
logits
act_probs
```

It compiled and executed through `VitisAIExecutionProvider`, but both outputs contained `NaN`.

The CPU version of the same graph was correct.

For a test question:

```text
A production web service has stopped accepting requests.
Monitoring shows that the database disk is completely full.
```

with choices:

```text
Restart the web server
Free space on the database disk
Wait and see if the service recovers
```

CPU produced:

```text
0.0059  Restart the web server
0.9877  Free space on the database disk
0.0064  Wait and see if the service recovers
```

while the NPU graph returned:

```text
NaN
NaN
NaN
```

This behaviour is consistent with an open AMD Ryzen AI issue documenting incorrect or NaN results for some multi-output ONNX graphs under Vitis AI EP. The issue specifically reports cases where changing only the number of declared outputs turns correct single-output execution into incorrect multi-output execution.

The issue does **not** claim every multi-output model is broken; AMD models exist which work correctly with multiple outputs.

---

## 16. Single-output workaround

A second ONNX graph was created containing only:

```text
logits
```

The external model weights were unchanged.

That graph compiled successfully using a separate cache key:

```text
laya-logits-b1-s512-k20
```

The same test then produced:

```text
CPU:
0.0059  Restart the web server
0.9877  Free space on the database disk
0.0064  Wait and see if the service recovers

NPU:
0.0059  Restart the web server
0.9877  Free space on the database disk
0.0064  Wait and see if the service recovers
```

Raw logits:

```text
CPU:
[-1.7870014, 3.3337939, -1.7052455]

NPU:
[-1.7870014, 3.3337939, -1.7052455]
```

Maximum observed difference:

```text
0.0
```

and:

```text
Finite NPU output: True
```

The Vitis execution report independently confirmed substantial VAIML assignment, so this was not simply CPU fallback.

---

# Current status

The following has been verified on this machine:

- Linux in-tree `amdxdna` detects the XDNA2 NPU.
- XRT 2.26 communicates with it.
- AMD's userspace XDNA plugin works without replacing the kernel driver.
- `xrt-smi validate` passes.
- Ryzen AI SDK 1.8 runs on CachyOS/Arch with compatibility fixes.
- AMD quicktest passes.
- `VitisAIExecutionProvider` is available.
- Laya compiles for the NPU with fixed dimensions.
- 99.998% of Laya's measured GOPs are VAIML-supported.
- Actual Laya NPU inference produces correct finite results using the single-output graph.
- The original two-output graph produces invalid values on this tested stack.

## Benchmark status

A single observed inference showed lower NPU latency than CPU, but that figure is intentionally not published as a benchmark.

A proper benchmark should use:

- several warm-up iterations;
- at least 20 measured CPU iterations;
- at least 20 measured NPU iterations;
- median and mean latency;
- output validation on every iteration.

Until that test is performed, this guide makes no general performance claim.

---

# Troubleshooting summary

| Symptom | Cause in this setup | Resolution |
|---|---|---|
| `modinfo amdxdna` fails | Running kernel older than installed modules | Reboot into matching kernel |
| XRT Arch package looks for wrong tarball version | Stale `pkgver` | Match PKGBUILD version to generated XRT |
| Kernel module build fails with GCC flags | CachyOS kernel built with Clang | Do not rebuild kernel driver; use `-nokmod` |
| Userspace build fails under forced Clang | Warnings promoted to errors | Build userspace normally with GCC |
| `xrt-smi` finds zero devices | XDNA userspace shim absent | Install userspace plugin |
| memlock allocation errors | Locked-memory limit | Set memlock unlimited |
| ONNX Runtime refuses executable stack | ELF executable-stack metadata | `patchelf --clear-execstack` |
| Vitis EP cannot find `libncurses.so.6` | Ubuntu/Arch ncurses naming difference | Private compatibility symlink |
| FlexML cannot find `libpython3.12.so.1.0` | Python 3.12 supplied by `uv` | Add `$PYBASE/lib` to `LD_LIBRARY_PATH` |
| Dynamic Laya compile crashes | Dynamic dimensions | Override `batch`, `seq`, `options` |
| Laya returns `NaN` on NPU | Two-output graph on tested Vitis AI stack | Use single-output graph |
| Some XRT telemetry is N/A | Userspace/in-tree ioctl difference | Core execution still works; pin versions |

---

# Why retain the in-tree driver?

This approach made sense because the kernel already contained a functioning `amdxdna` driver for the hardware.

Replacing it would have introduced:

- DKMS lifecycle management;
- another kernel/compiler compatibility dependency;
- replacement firmware;
- increased risk during kernel upgrades.

The compromise is that newer userspace packages may eventually expect driver APIs absent from a distro's in-tree driver.

For that reason this guide should be treated as a **known-working versioned configuration**, not a claim that arbitrary future XRT and kernel versions can be mixed safely.

---

# Result

A Ryzen AI XDNA2 NPU can be used for real ONNX inference on CachyOS/Arch Linux without replacing the distro's working in-tree `amdxdna` driver.

The final tested path is:

```text
Laya / ONNX
      ↓
ONNX Runtime 1.27.0
      ↓
VitisAIExecutionProvider
      ↓
Ryzen AI SDK 1.8
      ↓
XRT 2.26.0
      ↓
XDNA userspace plugin 2.26.0
      ↓
Linux in-tree amdxdna
      ↓
AMD XDNA2 NPU
```

The process is not yet a simple `pacman -S` installation, but it is reproducible — and, importantly, it runs an actual transformer-derived model rather than merely detecting the accelerator.
