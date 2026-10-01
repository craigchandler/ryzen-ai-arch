# Troubleshooting

These are the concrete failure modes encountered while bringing up and validating the tested Arch/CachyOS Ryzen AI stack.

## `modinfo amdxdna` fails while the driver is loaded

**Observed cause:** the system was still running kernel 7.2.7 after packages for kernel 7.2.8 had been installed.

Check:

```bash
uname -r
pacman -Q linux-cachyos linux-cachyos-headers
```

Reboot into the matching installed kernel before diagnosing missing modules.

## XRT Arch PKGBUILD looks for a nonexistent tarball

The build generated version `202620.2.26.0`, while the checked-in PKGBUILDs still expected `202620.2.25.0`.

Inspect the actual generated files first:

```bash
ls -lh ../Release 2>/dev/null
ls -lh ./*.tar.gz 2>/dev/null
```

Update the `pkgver` to match that source/build revision.

The double `--` in names such as:

```text
xrt_..._--base.tar.gz
```

was intentional in the tested build.

## Full XDNA kernel-module build fails with GCC

The tested CachyOS kernel had been built with Clang 22.1.8, while the XDNA build invoked GCC 16.2.1.

GCC then rejected Clang-specific kernel build flags.

If the in-tree `amdxdna` driver already works, avoid rebuilding the module:

```bash
./build.sh -release -nokmod
```

## Forcing Clang makes userspace fail with `-Werror`

Forcing:

```text
CC=clang
CXX=clang++
LLVM=1
LLVM_IAS=1
```

made the kernel module compile, but newer Clang warnings in userspace were promoted to errors.

The clean solution for the tested machine was to retain the in-tree kernel driver and build only userspace with:

```bash
./build.sh -release -nokmod
```

## `xrt-smi examine` shows zero devices

If the kernel sees:

```text
/dev/accel/accel0
```

but XRT shows no device, verify that the XDNA userspace shim is installed under:

```text
/opt/xilinx/xrt/lib
```

The custom package in this repository installs the userspace plugin without replacing the kernel driver.

## NPU allocation/memlock errors

Configure:

```bash
sudo tee /etc/security/limits.d/99-amdxdna.conf >/dev/null <<'LIMITS'
* soft memlock unlimited
* hard memlock unlimited
LIMITS
```

Then log out/in or reboot and confirm:

```bash
ulimit -Sl
ulimit -Hl
```

Both should report:

```text
unlimited
```

## ONNX Runtime: `cannot enable executable stack`

Check the affected AMD ONNX Runtime object:

```bash
SO=~/Source/ryzen_ai/venv/lib/python3.12/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.so
patchelf --print-execstack "$SO"
```

If it reports an executable stack requirement:

```bash
patchelf --clear-execstack "$SO"
```

Then retry the import.

## Vitis AI EP: `libncurses.so.6: cannot open shared object file`

Arch's current ncurses package provides:

```text
libncursesw.so.6
```

while the tested AMD binary expected the Ubuntu-style:

```text
libncurses.so.6
```

SONAME.

Keep the compatibility link local to the Ryzen AI environment:

```bash
mkdir -p ~/Source/ryzen_ai/compat-lib

ln -sf /usr/lib/libncursesw.so.6 \
  ~/Source/ryzen_ai/compat-lib/libncurses.so.6

export LD_LIBRARY_PATH="$HOME/Source/ryzen_ai/compat-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Do not add a replacement system-wide file under `/usr/lib` for this workaround.

## FlexML: `libpython3.12.so.1.0` not found

When Python 3.12 is supplied by `uv`, the shared runtime library lives inside the managed Python installation.

```bash
PYBASE="$(python -c 'import sys; print(sys.base_prefix)')"

ls -l "$PYBASE/lib/libpython3.12.so.1.0"

export LD_LIBRARY_PATH="$HOME/Source/ryzen_ai/compat-lib:$PYBASE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Verify:

```bash
ldd ~/Source/ryzen_ai/venv/lib/python3.12/site-packages/flexmlrt/lib/libflexmlrt.so \
  | grep -E 'libpython|not found'
```

## Laya dynamic-shape compile crashes

The published Laya ONNX graph uses symbolic:

```text
batch
seq
options
```

dimensions.

For the tested model compile:

```python
so.add_free_dimension_override_by_name("batch", 1)
so.add_free_dimension_override_by_name("seq", 512)
so.add_free_dimension_override_by_name("options", 20)
```

This allowed Vitis to see fixed shapes and compile the graph.

For batch throughput testing, the `batch` override was changed to 8 or 16.

## Laya compiles but returns `NaN`

The original Laya ONNX graph declares two outputs:

```text
logits
act_probs
```

On the tested Ryzen AI 1.8 stack, the two-output graph returned `NaN` on NPU while the CPU reference was correct.

A single-output graph containing only `logits` returned correct finite NPU results.

See:

- `docs/laya.md`
- <https://github.com/amd/RyzenAI-SW/issues/369>

AMD issue #369 documents failures for some multi-output graphs, but also contains working multi-output examples. Do not generalise this issue to every multi-output ONNX graph.

## Laya compiled cache cannot be loaded by another Python process

A fresh compile and inference succeeded.

Starting another Python process with the same cache then failed with:

```text
Failed to open file:
.../vaiml_par_0/partition-info.json
```

and:

```text
Failed to parse JSON from file ...
attempting to parse an empty input
```

followed by:

```text
HW context creation unsuccessful,
check for XRT version or recompile model
with latest VAIML version
```

The failure was reproduced **without rebooting**.

The tested versions remained:

```text
Kernel             : 7.2.8-1-cachyos
amdxdna            : 7.2.8-1-cachyos
XRT                 : 2.26.0
XDNA userspace      : 2.26.0
NPU firmware        : 1.1.2.64
```

so the failure was not caused by a post-reboot version change in that test.

AMD issue #324 describes a similar class of disk-cache/load problem:

<https://github.com/amd/RyzenAI-SW/issues/324>

### Benchmark workaround

Use a fresh cache and perform the benchmark in the same Python process that creates the NPU session:

```bash
python examples/laya/benchmark.py \
  --batch 1 \
  --fresh-cache
```

The same approach applies to batch 8 and batch 16.

This requires recompilation when the cache is deliberately refreshed.

For a long-running application, keeping the ONNX Runtime NPU session alive avoids recompilation for individual inference requests.

## EP Context Cache model is not generated

An ONNX Runtime EP Context Cache was tested as an alternative persistent artifact.

The session requested:

```text
ep.context_enable = 1
ep.context_embed_mode = 1
```

and supplied an output context path.

After compilation work, ONNX Runtime reported:

```text
Unable to compile any nodes.
ONNX Runtime will not generate a compiled model.
Either the session EPs do not support compilation
or the model is already compiled.
```

The expected context model was not generated.

This is the observed behaviour for the tested Laya/Ryzen AI 1.8 configuration only.

It does not establish that EP Context Cache is unavailable for all Vitis AI models or platforms.

## Laya NPU is not significantly faster than CPU

This is not necessarily a configuration failure.

The measured results on the Ryzen AI 9 HX 470 were:

```text
Batch 1:
CPU median  : 1196.04 ms
NPU median  : 1156.77 ms
NPU speedup : 1.034x

Batch 8:
CPU         : 0.89 decisions/s
NPU         : 0.89 decisions/s
NPU speedup : 0.995x

Batch 16:
CPU         : 0.74 decisions/s
NPU         : 0.74 decisions/s
NPU speedup : 0.998x
```

At the same time, the model showed:

```text
99.998% of GOPs supported by VAIML
```

and NPU output exactly matched CPU output in the measured tests.

High accelerator coverage does not guarantee an end-to-end speedup.

Possible contributors include:

- execution-provider overhead;
- host/NPU scheduling;
- memory movement;
- remaining CPU nodes;
- tensor shapes;
- workload architecture;
- a strong host CPU.

For this specific model, CPU and NPU performance should be treated as effectively equivalent.

## Some `xrt-smi` telemetry is `N/A`

On the tested in-tree-driver/XRT combination, some power/load/temperature fields were unavailable even though:

```bash
xrt-smi validate
```

passed and real model execution worked.

Treat this as a userspace/driver-version compatibility signal rather than assuming failed model execution.

Pin a known-working stack rather than assuming arbitrary future kernel, XRT and Ryzen AI versions will remain interchangeable.