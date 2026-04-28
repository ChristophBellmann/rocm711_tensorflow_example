#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
ROCM_PREFIX="${ROCM_PREFIX:-/opt/rocm}"
WHEEL_DIR="${WHEEL_DIR:-${ROCM_PREFIX}/wheels/tensorflow_rocm_custom}"
TENSORFLOW_WHEEL="${TENSORFLOW_WHEEL:-}"

usage() {
  cat <<'USAGE'
Usage: ./setup_rocm_venv.sh [options]

Creates/updates a Python venv for the installed custom ROCm TensorFlow stack.

Options:
  --venv <dir>              Venv directory (default: .venv)
  --rocm-prefix <dir>       Installed ROCm prefix (default: /opt/rocm)
  --tensorflow-wheel <path> Explicit TensorFlow wheel (default: tensorflow-current.whl or newest tensorflow-*.whl)
  -h, --help                Show help
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_DIR="${2:-}"
      shift 2
      ;;
    --rocm-prefix)
      ROCM_PREFIX="${2:-}"
      WHEEL_DIR="${ROCM_PREFIX}/wheels/tensorflow_rocm_custom"
      shift 2
      ;;
    --tensorflow-wheel)
      TENSORFLOW_WHEEL="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown arg: $1"
      ;;
  esac
done

command -v python3 >/dev/null 2>&1 || die "python3 not found"
[[ -d "${ROCM_PREFIX}" ]] || die "ROCm prefix not found: ${ROCM_PREFIX}"

if [[ -z "${TENSORFLOW_WHEEL}" ]]; then
  if [[ -f "${WHEEL_DIR}/tensorflow-current.whl" ]]; then
    TENSORFLOW_WHEEL="${WHEEL_DIR}/tensorflow-current.whl"
  else
    TENSORFLOW_WHEEL="$(ls -1t "${WHEEL_DIR}"/tensorflow-*.whl 2>/dev/null | head -n 1 || true)"
  fi
fi

[[ -n "${TENSORFLOW_WHEEL}" ]] || die "No TensorFlow wheel found in ${WHEEL_DIR}"
[[ -f "${TENSORFLOW_WHEEL}" ]] || die "TensorFlow wheel not found: ${TENSORFLOW_WHEEL}"
if [[ -L "${TENSORFLOW_WHEEL}" ]]; then
  TENSORFLOW_WHEEL="$(readlink -f "${TENSORFLOW_WHEEL}")"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

venv_python="${VENV_DIR}/bin/python"
"${venv_python}" -m pip install -U pip setuptools wheel
"${venv_python}" -m pip install --upgrade --force-reinstall "${TENSORFLOW_WHEEL}"

activate_script="${VENV_DIR}/bin/activate_rocm_tensorflow.sh"
python_wrapper="${VENV_DIR}/bin/python-rocm"

cat >"${activate_script}" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="\$(cd -- "\$(dirname -- "\${BASH_SOURCE[0]}")/.." && pwd)"
export ROCM_PATH="${ROCM_PREFIX}"
export HIP_PATH="\${HIP_PATH:-\$ROCM_PATH}"
export HSA_PATH="\${HSA_PATH:-\$ROCM_PATH}"
export PATH="\$ROCM_PATH/bin:\$ROCM_PATH/llvm/bin:\$VENV_DIR/bin:\${PATH:-}"
export LD_LIBRARY_PATH="\$ROCM_PATH/lib:\$ROCM_PATH/lib64:\$ROCM_PATH/lib/host-math/lib:\$ROCM_PATH/lib/rocm_sysdeps/lib:\$ROCM_PATH/lib/llvm/lib:\$ROCM_PATH/llvm/lib:\${LD_LIBRARY_PATH:-}"
export TF_ROCM_DISABLE_HIPBLASLT="\${TF_ROCM_DISABLE_HIPBLASLT:-1}"
export TF_ROCM_USE_HIPBLASLT="\${TF_ROCM_USE_HIPBLASLT:-0}"
export TF_ROCM_DISABLE_HIPBLASLT_INIT="\${TF_ROCM_DISABLE_HIPBLASLT_INIT:-1}"
export PYTHONUNBUFFERED="\${PYTHONUNBUFFERED:-1}"
export PYTHONFAULTHANDLER="\${PYTHONFAULTHANDLER:-1}"

# shellcheck source=/dev/null
source "\$VENV_DIR/bin/activate"
SCRIPT
chmod +x "${activate_script}"

cat >"${python_wrapper}" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/activate_rocm_tensorflow.sh"
exec "${VIRTUAL_ENV}/bin/python" "$@"
SCRIPT
chmod +x "${python_wrapper}"

echo "Installed ${TENSORFLOW_WHEEL}"
echo "Activate with:"
echo "  source \"${activate_script}\""
