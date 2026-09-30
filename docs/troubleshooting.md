# Troubleshooting

These are the concrete failure modes encountered while bringing up the tested Arch/CachyOS Ryzen AI stack.

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

Update the `pkgver` to match that source/build revision. The double `--` in names such as `xrt_..._--base.tar.gz` was intentional in the tested build.

## Full XDNA kernel-module build fails with GCC

The tested CachyOS kernel had been built with Clang 22.1.8, while the XDNA build invoked GCC 16.2.1. GCC then rejected Clang-specific kernel build flags.

If the in-tree `amdxdna` driver already works, avoid rebuilding the module:

```bash
./build.sh -release -nokmod
```

## Forcing Clang makes userspace fail with `-Werror`

Forcing `CC=clang CXX=clang++ LLVM=1 LLVM_IAS=1` made the kernel module compile, but newer Clang warnings in userspace were promoted to errors.

Again, the clean solution for the tested machine was to retain the in-tree kernel driver and build only userspace with `-nokmod`.

## `xrt-smi examine` shows zero devices

If the kernel sees `/dev/accel/accel0` but XRT shows no device, verify that the XDNA userspace shim is installed under `/opt/xilinx/xrt/lib`.

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

Arch's current ncurses package provides `libncursesw.so.6`, while the tested AMD binary expected the Ubuntu-style `libncurses.so.6` SONAME.

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

The published Laya ONNX graph uses symbolic `batch`, `seq` and `options` dimensions.

For the tested model compile:

```python
so.add_free_dimension_override_by_name("batch", 1)
so.add_free_dimension_override_by_name("seq", 512)
so.add_free_dimension_override_by_name("options", 20)
```

This allowed Vitis to see fixed shapes and compile the graph.

## Laya compiles but returns `NaN`

The original Laya ONNX graph declares two outputs: `logits` and `act_probs`.

On the tested Ryzen AI 1.8 stack, the two-output graph returned `NaN` on NPU while the CPU reference was correct.

A single-output graph containing only `logits` returned correct finite NPU results.

See:

- `docs/laya.md`
- <https://github.com/amd/RyzenAI-SW/issues/369>

## Some `xrt-smi` telemetry is `N/A`

On the tested in-tree-driver/XRT combination, some power/load/temperature fields were unavailable even though:

```bash
xrt-smi validate
```

passed and real model execution worked.

Treat this as a userspace/driver-version compatibility signal and pin a known-working stack rather than assuming all future versions will remain compatible.
