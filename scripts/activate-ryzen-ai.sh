#!/usr/bin/env bash
# Source this file from an interactive shell:
#   source scripts/activate-ryzen-ai.sh

set -e

RYZEN_AI_ROOT="${RYZEN_AI_ROOT:-$HOME/Source/ryzen_ai}"
RYZEN_AI_VENV="${RYZEN_AI_VENV:-$RYZEN_AI_ROOT/venv}"

if [[ ! -f "$RYZEN_AI_VENV/bin/activate" ]]; then
    echo "Ryzen AI venv not found: $RYZEN_AI_VENV" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$RYZEN_AI_VENV/bin/activate"

if [[ -f /opt/xilinx/xrt/setup.sh ]]; then
    # shellcheck disable=SC1091
    source /opt/xilinx/xrt/setup.sh
else
    echo "XRT setup not found: /opt/xilinx/xrt/setup.sh" >&2
    return 1 2>/dev/null || exit 1
fi

PYBASE="$(python -c 'import sys; print(sys.base_prefix)')"
COMPAT_LIB="$RYZEN_AI_ROOT/compat-lib"

export LD_LIBRARY_PATH="$COMPAT_LIB:$PYBASE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python - <<'PY'
import onnxruntime as ort
print("ONNX Runtime:", ort.__version__)
print("Providers:", ort.get_available_providers())
PY
