"""
Sets up this project for GPU-accelerated OCR benchmarking on a machine with an
NVIDIA GPU. Run once after cloning, before `testing/benchmark.py --devices cpu,cuda`.

    uv run python scripts/setup_cuda_benchmark.py

The base project dependencies (installed via `uv sync`) are deliberately CPU-only
so the project still installs cleanly on machines without an NVIDIA GPU (for
example, Apple Silicon). onnxruntime-gpu and paddlepaddle-gpu are separate,
GPU-only wheels that can't be declared as normal dependencies without breaking
that CPU-only install elsewhere, so this script detects the local CUDA version
and installs the matching packages on top of the base `uv sync`.

Caveat: onnxruntime and paddlepaddle (the CPU builds) stay declared in
pyproject.toml/uv.lock. If you ever run `uv sync` again after this script
(for example after a `git pull`), uv will reconcile the environment back to
the locked CPU builds and silently undo this script's GPU packages — just
re-run this script afterwards to restore them.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
PADDLE_VERSION = "3.3.1"

# CUDA channels each library actually publishes wheels for. A detected CUDA
# version is rounded down to the nearest channel below it, since a newer
# NVIDIA driver is backward compatible with wheels built for an older CUDA
# toolkit. Update these if torch/paddle drop an old channel or add a new one.
TORCH_CUDA_CHANNELS = ["12.6", "12.4", "12.1", "11.8"]
PADDLE_CUDA_CHANNELS = ["12.6", "11.8"]


def run(command, check=True):
    print(f"$ {' '.join(command)}")
    return subprocess.run(command, cwd=PROJECT_FOLDER, check=check)


def run_python(code):
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_FOLDER,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()


def detect_cuda_version():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    match = re.search(r"CUDA Version:\s*([\d.]+)", result.stdout)
    return float(match.group(1)) if match else None


def pick_channel(cuda_version, available_channels):
    for channel in available_channels:
        if cuda_version >= float(channel):
            return channel
    return None


def sync_base_dependencies():
    print("\n=== Installing base project dependencies (uv sync) ===")
    run(["uv", "sync"])


def ensure_torch_sees_cuda(cuda_version):
    print("\n=== Checking PyTorch CUDA support ===")
    ok, stdout, stderr = run_python("import torch; print(torch.cuda.is_available())")

    if ok and stdout == "True":
        print("torch already sees the GPU, nothing to do.")
        return

    print("torch does not see the GPU yet — installing a CUDA-matched build.")
    channel = pick_channel(cuda_version, TORCH_CUDA_CHANNELS)

    if channel is None:
        print(
            f"No known torch wheel channel supports CUDA {cuda_version}; "
            "install torch manually from https://pytorch.org/get-started/locally/"
        )
        return

    index_url = f"https://download.pytorch.org/whl/cu{channel.replace('.', '')}"
    run(["uv", "pip", "install", "torch", "--index-url", index_url])


def install_onnxruntime_gpu():
    print("\n=== Installing onnxruntime-gpu (for RapidOCR on CUDA) ===")
    run(["uv", "pip", "uninstall", "onnxruntime"], check=False)
    run(["uv", "pip", "install", "onnxruntime-gpu"])


def install_paddlepaddle_gpu(cuda_version):
    print("\n=== Installing paddlepaddle-gpu (for PaddleOCR on CUDA) ===")
    channel = pick_channel(cuda_version, PADDLE_CUDA_CHANNELS)

    if channel is None:
        print(
            f"No known paddlepaddle-gpu channel supports CUDA {cuda_version}; "
            "pick one manually from https://www.paddlepaddle.org.cn/en/install/quick"
        )
        return

    run(["uv", "pip", "uninstall", "paddlepaddle"], check=False)
    index_url = f"https://www.paddlepaddle.org.cn/packages/stable/cu{channel.replace('.', '')}/"

    result = run(
        ["uv", "pip", "install", f"paddlepaddle-gpu=={PADDLE_VERSION}", "-i", index_url],
        check=False,
    )

    if result.returncode != 0:
        print(f"paddlepaddle-gpu=={PADDLE_VERSION} isn't available on that channel yet, trying latest.")
        run(["uv", "pip", "install", "paddlepaddle-gpu", "-i", index_url])


def verify_setup():
    print("\n=== Verifying GPU support ===")

    _, torch_output, _ = run_python(
        "import torch; print(torch.cuda.is_available(), "
        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    )
    print(f"torch.cuda.is_available(): {torch_output}")

    _, paddle_output, _ = run_python("import paddle; print(paddle.is_compiled_with_cuda())")
    print(f"paddle.is_compiled_with_cuda(): {paddle_output}")

    _, models_output, _ = run_python(
        "from Backend.processors.image_processor import get_available_ocr_models; "
        "print(get_available_ocr_models())"
    )
    print(f"available OCR models: {models_output}")

    print(
        "\nIf everything above looks right, run:\n"
        "  uv run python testing/benchmark.py batch all --devices cpu,cuda\n"
        "\nNote: if you ever run `uv sync` again later (e.g. after `git pull`), it will\n"
        "reset onnxruntime/paddlepaddle back to their CPU builds — just re-run this\n"
        "script afterwards to restore GPU support."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda-version",
        type=float,
        default=None,
        help="Override auto-detected CUDA version (from `nvidia-smi`) if detection fails.",
    )
    arguments = parser.parse_args()

    cuda_version = arguments.cuda_version or detect_cuda_version()

    if cuda_version is None:
        print(
            "Could not detect an NVIDIA GPU (`nvidia-smi` not found or failed).\n"
            "Install/update the NVIDIA driver first, or pass --cuda-version manually."
        )
        sys.exit(1)

    print(f"Detected CUDA version: {cuda_version}")

    sync_base_dependencies()
    ensure_torch_sees_cuda(cuda_version)
    install_onnxruntime_gpu()
    install_paddlepaddle_gpu(cuda_version)
    verify_setup()


if __name__ == "__main__":
    main()
