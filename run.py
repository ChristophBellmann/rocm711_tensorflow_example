#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
import time


def _fmt_s(x: float) -> str:
    if x < 1e-3:
        return f"{x * 1e6:.0f}us"
    if x < 1.0:
        return f"{x * 1e3:.1f}ms"
    return f"{x:.3f}s"


def _maybe_reexec_with_rocm_env() -> None:
    """
    Ensure ROCm runtime libraries are visible from process start.

    LD_LIBRARY_PATH changes must be present before the interpreter loads
    TensorFlow and its dependent DSOs, so the example re-execs once if needed.
    """
    if os.environ.get("ROCM711_EXAMPLE_REEXEC", "") == "1":
        return

    rocm = os.environ.get("ROCM_PATH", "").strip() or "/opt/rocm"
    env = dict(os.environ)
    env["ROCM_PATH"] = rocm
    env.setdefault("HIP_PATH", rocm)
    env.setdefault("HSA_PATH", rocm)

    path = env.get("PATH", "")
    want_path = [f"{rocm}/bin", f"{rocm}/llvm/bin"]
    if not all(p in path.split(":") for p in want_path):
        env["PATH"] = ":".join(want_path + ([path] if path else []))

    ld = env.get("LD_LIBRARY_PATH", "")
    want_ld = [
        f"{rocm}/lib",
        f"{rocm}/lib64",
        f"{rocm}/lib/host-math/lib",
        f"{rocm}/lib/rocm_sysdeps/lib",
        f"{rocm}/lib/llvm/lib",
        f"{rocm}/llvm/lib",
    ]
    if not all(p in ld.split(":") for p in want_ld):
        env["LD_LIBRARY_PATH"] = ":".join(want_ld + ([ld] if ld else []))

    env.setdefault("TF_ROCM_DISABLE_HIPBLASLT", "1")
    env.setdefault("TF_ROCM_USE_HIPBLASLT", "0")
    env.setdefault("TF_ROCM_DISABLE_HIPBLASLT_INIT", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")

    changed = any(
        env.get(k) != os.environ.get(k)
        for k in (
            "ROCM_PATH",
            "HIP_PATH",
            "HSA_PATH",
            "PATH",
            "LD_LIBRARY_PATH",
            "TF_ROCM_DISABLE_HIPBLASLT",
            "TF_ROCM_USE_HIPBLASLT",
            "TF_ROCM_DISABLE_HIPBLASLT_INIT",
            "PYTHONUNBUFFERED",
            "PYTHONFAULTHANDLER",
        )
    )
    if changed:
        env["ROCM711_EXAMPLE_REEXEC"] = "1"
        os.execvpe(sys.executable, [sys.executable, __file__] + sys.argv[1:], env)


def _find_loaded_lib(base: str) -> str:
    pat = re.compile(re.escape(base) + r"(\..*)?$")
    try:
        with open("/proc/self/maps", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                p = parts[5]
                if not p.startswith("/") or ".so" not in p:
                    continue
                if pat.match(os.path.basename(p)):
                    return p
    except OSError:
        return ""
    return ""


def _dtype_from_env(tf):
    name = (os.environ.get("TF_DTYPE", "") or "float16").strip().lower()
    if name in {"float16", "fp16", "half"}:
        return tf.float16, "float16"
    if name in {"float32", "fp32"}:
        return tf.float32, "float32"
    raise SystemExit(f"Unsupported TF_DTYPE={name!r}; use float16/fp16 or float32/fp32")


def main() -> int:
    _maybe_reexec_with_rocm_env()

    try:
        import tensorflow as tf  # type: ignore
    except Exception as e:
        print("FAIL: import tensorflow")
        print(f"  {e!r}")
        return 1

    dtype, dtype_name = _dtype_from_env(tf)
    target_s = float(os.environ.get("TF_BENCH_S", "5") or "5")
    n = int(float(os.environ.get("TF_MATMUL_MNK", "4096") or "4096"))

    print("== TensorFlow ROCm smoke ==")
    print(f"tf.__version__               : {getattr(tf, '__version__', '')}")
    print(f"tf.test.is_built_with_rocm() : {bool(tf.test.is_built_with_rocm())}")
    print(f"ROCM_PATH                    : {os.environ.get('ROCM_PATH', '')}")
    print(f"TF_ROCM_DISABLE_HIPBLASLT    : {os.environ.get('TF_ROCM_DISABLE_HIPBLASLT', '')}")
    print(f"TF_ROCM_USE_HIPBLASLT        : {os.environ.get('TF_ROCM_USE_HIPBLASLT', '')}")
    print(f"TF_ROCM_DISABLE_HIPBLASLT_INIT: {os.environ.get('TF_ROCM_DISABLE_HIPBLASLT_INIT', '')}")

    gpus = tf.config.list_physical_devices("GPU")
    print(f"gpu_count                    : {len(gpus)}")
    if not gpus:
        print("FAIL: no GPU visible to TensorFlow.")
        print("Hints:")
        print("- Check: /opt/rocm/bin/rocminfo")
        print("- Check: the venv installed the local wheel from /opt/rocm/wheels/tensorflow_rocm_custom/")
        print("- Check: this process sees ROCm libs via LD_LIBRARY_PATH")
        return 2

    hip_lib = _find_loaded_lib("libamdhip64.so")
    hsa_lib = _find_loaded_lib("libhsa-runtime64.so")
    hipblaslt_lib = _find_loaded_lib("libhipblaslt.so")
    if hip_lib:
        print(f"loaded libamdhip64.so        : {hip_lib}")
    if hsa_lib:
        print(f"loaded libhsa-runtime64.so   : {hsa_lib}")
    if hipblaslt_lib:
        print(f"loaded libhipblaslt.so       : {hipblaslt_lib}")

    with tf.device("/GPU:0"):
        a = tf.random.uniform((n, n), dtype=dtype)
        b = tf.random.uniform((n, n), dtype=dtype)
        c = tf.matmul(a, b)
        _ = float(tf.reduce_sum(c).numpy())

    start = time.perf_counter()
    iters = 0
    with tf.device("/GPU:0"):
        while (time.perf_counter() - start) < target_s:
            c = tf.matmul(a, b)
            _ = float(tf.reduce_sum(c).numpy())
            iters += 1
    wall = time.perf_counter() - start

    flops = iters * (2.0 * (n**3))
    tflops = flops / wall / 1e12 if wall > 0 else 0.0

    print("")
    print("== Matmul benchmark (sustained) ==")
    print(f"shape       : {n}x{n}x{n}")
    print(f"dtype       : {dtype_name}")
    print(f"iters       : {iters}")
    print(f"wall        : {_fmt_s(wall)}")
    print(f"TFLOPS(est) : {tflops:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
