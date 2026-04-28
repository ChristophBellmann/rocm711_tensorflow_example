# rocm711_tensorflow_example

Minimal example project that installs your custom ROCm 7.11 TensorFlow wheel from:

- `/opt/rocm/wheels/tensorflow_rocm_custom/`

This prevents accidentally pulling a different TensorFlow build from the internet.

## Prereqs

- `/opt/rocm` installed and working (`/opt/rocm/bin/rocminfo`)
- The custom TensorFlow wheel exists:
  - `/opt/rocm/wheels/tensorflow_rocm_custom/tensorflow-current.whl`

## Setup

```bash
cd /path/to/ML-Lab/examples/rocm711_tensorflow_example
./setup_rocm_venv.sh

. .venv/bin/activate_rocm_tensorflow.sh
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Notes:
- `setup_rocm_venv.sh` installs **TensorFlow** from the local wheel under `/opt/rocm/wheels/tensorflow_rocm_custom/`.
- `requirements.txt` contains only project-local Python dependencies.
- Small dependencies are still allowed to come from PyPI.
- This project intentionally does **not** support an offline mode.

## Run

```bash
. .venv/bin/activate_rocm_tensorflow.sh
python run.py
```

Expected:

- `tensorflow` imports
- TensorFlow reports a ROCm build
- a GPU is visible
- a sustained matmul runs on GPU and prints timing + TFLOPS estimate

Defaults:
- matmul size: `4096`
- dtype: `float16`
- benchmark time: `5s`

Override example:

```bash
TF_MATMUL_MNK=2048 TF_DTYPE=float32 TF_BENCH_S=3 python run.py
```

## Notes

- The activation script exposes ROCm runtime libraries from `/opt/rocm`; `run.py` still re-execs once if needed.
- It also sets:
  - `TF_ROCM_DISABLE_HIPBLASLT=1`
  - `TF_ROCM_USE_HIPBLASLT=0`
  - `TF_ROCM_DISABLE_HIPBLASLT_INIT=1`
- Those settings are intentional for this custom gfx1031 ROCm/TensorFlow combination.
