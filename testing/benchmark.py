import argparse
import contextlib
import io
import json
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import psutil

testing_folder = Path(__file__).resolve().parent
project_folder = testing_folder.parent

project_folder_text = str(project_folder)

if project_folder_text in sys.path:
    sys.path.remove(project_folder_text)

sys.path.insert(0, project_folder_text)

from Backend.processors.image_processor import (
    get_available_ocr_models,
    get_doctr_predictor,
    get_easyocr_reader,
    get_paddle_ocr,
    get_rapidocr_reader,
    check_tesseract_available,
    release_ocr_model,
)
from testing.reporting import outputs_folder
from testing.testing import (
    aadhaar_ground_truth_folder,
    aadhaar_images_folder,
    compare_with_ground_truth,
    invoice_ground_truth_folder,
    invoice_images_folder,
    list_document_images,
    load_ground_truth_for_image,
    pan_ground_truth_folder,
    pan_images_folder,
    passbook_ground_truth_folder,
    passbook_images_folder,
    process_aadhaar_image_with_model,
    process_invoice_image_with_model,
    process_pan_image_with_model,
    process_passbook_image_with_model,
)

benchmarks_outputs_folder = outputs_folder / "benchmarks"

ALL_MODELS = ["easyocr", "tesseract", "paddleocr", "rapidocr", "doctr"]

DOCUMENT_PROCESSORS = {
    "aadhaar": (aadhaar_images_folder, aadhaar_ground_truth_folder, process_aadhaar_image_with_model),
    "pan": (pan_images_folder, pan_ground_truth_folder, process_pan_image_with_model),
    "passbook": (passbook_images_folder, passbook_ground_truth_folder, process_passbook_image_with_model),
    "invoice": (invoice_images_folder, invoice_ground_truth_folder, process_invoice_image_with_model),
}

TABLE_COLUMNS = [
    "model",
    "device",
    "status",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "throughput_img_s",
    "peak_mem_mb",
    "gpu_mem_mb",
    "gpu_util_pct",
    "accuracy_pct",
    "error_rate_pct",
    "speedup_vs_cpu",
]

CONCURRENCY_TABLE_COLUMNS = [
    "model",
    "device",
    "concurrency",
    "status",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "throughput_req_s",
    "peak_mem_mb",
    "gpu_mem_mb",
    "gpu_util_pct",
    "error_rate_pct",
]


def mps_is_available():
    try:
        import torch

        return torch.backends.mps.is_available()
    except Exception:
        return False


def cuda_is_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def device_is_supported(model_name, device):
    if device == "cpu":
        return True, ""

    if model_name == "tesseract":
        return False, "tesseract has no GPU execution path"

    if device == "mps":
        if model_name == "paddleocr":
            return False, "paddlepaddle has no MPS/Metal backend on Apple Silicon (CPU/CUDA/XPU/NPU only)"
        if not mps_is_available():
            return False, "MPS backend not available on this machine"
        return True, ""

    if device == "cuda":
        if not cuda_is_available():
            return False, "CUDA backend not available on this machine"
        return True, ""

    return False, f"Unknown device: {device}"


def load_model(model_name, device):
    if model_name == "easyocr":
        return get_easyocr_reader(device)
    if model_name == "doctr":
        return get_doctr_predictor(device)
    if model_name == "rapidocr":
        return get_rapidocr_reader(device)
    if model_name == "paddleocr":
        return get_paddle_ocr("en", device)
    if model_name == "tesseract":
        check_tesseract_available()
        return None
    raise ValueError(f"Unknown model: {model_name}")


def get_rapidocr_active_providers(reader):
    try:
        return reader.text_det.session.session.get_providers()
    except Exception:
        return None


def detect_inconclusive(model_name, device, loaded_model, warmup_output):
    if device == "cpu":
        return False, ""

    if model_name == "easyocr":
        actual_device = str(getattr(loaded_model, "device", ""))
        if actual_device != device:
            return True, f"EasyOCR resolved to device={actual_device!r} instead of {device!r}"
        return False, ""

    if model_name == "doctr":
        actual_device_type = next(loaded_model.parameters()).device.type
        if actual_device_type != device:
            return True, f"docTR resolved to device={actual_device_type!r} instead of {device!r}"
        return False, ""

    if model_name == "rapidocr":
        if device == "mps" and "E5RT" in warmup_output:
            return True, (
                "CoreML execution provider registered but the ONNX graph fell back to CPU "
                "per-node (dynamic input shapes unsupported by CoreML's MIL compiler); "
                "reported numbers may not reflect real GPU execution"
            )
        if device == "cuda":
            providers = get_rapidocr_active_providers(loaded_model)
            if providers is None:
                return True, (
                    "could not verify active onnxruntime providers (introspection failed) — "
                    "treating as unverified rather than assuming CUDA is active"
                )
            if "CUDAExecutionProvider" not in providers:
                return True, (
                    "CUDAExecutionProvider not active in the onnxruntime session — check the console "
                    "output for a 'No registered plugin EP device found for CUDAExecutionProvider' "
                    "warning, which means it silently fell back to CPU for this model"
                )
        return False, ""

    return False, ""


class GpuMonitor:
    # Shared per (model, device) benchmark run — holds the NVML handle (cuda) or
    # nothing (mps has no utilization API, only a memory counter sampled directly
    # from torch). Avoids re-initializing NVML on every single image.
    def __init__(self, device):
        self.device = device
        self.error = None
        self._pynvml = None
        self._handle = None

        if device != "cuda":
            return

        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as error:
            self.error = f"pynvml unavailable, GPU metrics will be skipped: {error}"

    def sample(self):
        if self.device == "cuda" and self._handle is not None:
            memory_used_bytes = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used
            utilization_percent = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
            return memory_used_bytes, utilization_percent

        if self.device == "mps":
            import torch

            return torch.mps.current_allocated_memory(), None

        return None, None

    def close(self):
        if self._handle is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


class GpuSampler:
    def __init__(self, monitor, interval_seconds):
        self._monitor = monitor
        self._interval = interval_seconds
        self.peak_memory_bytes = 0
        self.utilization_samples = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                memory_bytes, utilization_percent = self._monitor.sample()
                if memory_bytes is not None:
                    self.peak_memory_bytes = max(self.peak_memory_bytes, memory_bytes)
                if utilization_percent is not None:
                    self.utilization_samples.append(utilization_percent)
            except Exception:
                pass
            self._stop_event.wait(self._interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop_event.set()
        self._thread.join(timeout=1)


class RssSampler:
    def __init__(self, process, interval_seconds):
        self._process = process
        self._interval = interval_seconds
        self.peak = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self.peak = max(self.peak, self._process.memory_info().rss)
            except Exception:
                pass
            self._stop_event.wait(self._interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop_event.set()
        self._thread.join(timeout=1)


def compute_latency_statistics(samples_seconds):
    sample_count = len(samples_seconds)

    if sample_count == 0:
        return {
            "p50_seconds": None,
            "p95_seconds": None,
            "p99_seconds": None,
            "mean_seconds": None,
            "stdev_seconds": None,
            "n_samples": 0,
            "percentile_confidence": "no samples",
        }

    sorted_samples = sorted(samples_seconds)
    mean = statistics.mean(sorted_samples)
    stdev = statistics.pstdev(sorted_samples) if sample_count > 1 else 0.0
    p50 = statistics.median(sorted_samples)

    if sample_count >= 20:
        quantiles = statistics.quantiles(sorted_samples, n=100, method="inclusive")
        p95 = quantiles[94]
        p99 = quantiles[98]
        confidence = "ok"
    else:
        p95 = sorted_samples[min(sample_count - 1, round(0.95 * (sample_count - 1)))]
        p99 = sorted_samples[min(sample_count - 1, round(0.99 * (sample_count - 1)))]
        confidence = f"low (n={sample_count})"

    return {
        "p50_seconds": round(p50, 4),
        "p95_seconds": round(p95, 4),
        "p99_seconds": round(p99, 4),
        "mean_seconds": round(mean, 4),
        "stdev_seconds": round(stdev, 4),
        "n_samples": sample_count,
        "percentile_confidence": confidence,
    }


def resolve_sample_images(all_images, sample_size, seed):
    if not all_images:
        return []

    if not sample_size or sample_size >= len(all_images):
        return all_images

    return sorted(random.Random(seed).sample(all_images, sample_size))


def run_model_device_benchmark(
    model_name,
    device,
    process_fn,
    sample_images,
    ground_truth_folder,
    repeats,
    warmup,
    memory_sampling,
    memory_sampling_interval_seconds,
    compare_accuracy,
):
    supported, reason = device_is_supported(model_name, device)

    if not supported:
        return {"model": model_name, "device": device, "status": "unsupported", "reason": reason}

    try:
        load_started_at = time.perf_counter()
        loaded_model = load_model(model_name, device)
        model_load_seconds = round(time.perf_counter() - load_started_at, 4)
    except Exception as error:
        return {"model": model_name, "device": device, "status": "error", "reason": str(error)}

    warmup_output = io.StringIO()

    try:
        with contextlib.redirect_stdout(warmup_output), contextlib.redirect_stderr(warmup_output):
            for _ in range(max(warmup, 0)):
                process_fn(sample_images[0], model_name, device=device)
    except Exception as error:
        return {
            "model": model_name,
            "device": device,
            "status": "error",
            "reason": f"warmup failed: {error}",
            "model_load_seconds": model_load_seconds,
        }

    is_inconclusive, inconclusive_reason = detect_inconclusive(
        model_name, device, loaded_model, warmup_output.getvalue()
    )

    process = psutil.Process()
    process.cpu_percent(interval=None)

    gpu_monitor = GpuMonitor(device) if device != "cpu" else None

    latency_samples = []
    rss_deltas_mb = []
    rss_peaks_mb = []
    cpu_percents = []
    accuracy_samples = []
    gpu_peaks_mb = []
    gpu_utilization_percents = []

    image_errors = []
    attempted_count = 0

    for _ in range(repeats):
        for image_path in sample_images:
            attempted_count += 1

            try:
                rss_before = process.memory_info().rss
                sampler = (
                    RssSampler(process, memory_sampling_interval_seconds)
                    if memory_sampling == "threaded"
                    else None
                )
                gpu_sampler = (
                    GpuSampler(gpu_monitor, memory_sampling_interval_seconds)
                    if gpu_monitor is not None and gpu_monitor.error is None and memory_sampling == "threaded"
                    else None
                )

                started_at = time.perf_counter()
                with contextlib.ExitStack() as stack:
                    if sampler:
                        stack.enter_context(sampler)
                    if gpu_sampler:
                        stack.enter_context(gpu_sampler)
                    result = process_fn(image_path, model_name, device=device)
                elapsed_seconds = time.perf_counter() - started_at

                rss_after = process.memory_info().rss

                latency_samples.append(elapsed_seconds)
                rss_deltas_mb.append((rss_after - rss_before) / 1e6)
                peak_rss = max(rss_before, rss_after, sampler.peak if sampler else 0)
                rss_peaks_mb.append(peak_rss / 1e6)
                cpu_percents.append(process.cpu_percent(interval=None))

                if gpu_sampler:
                    gpu_peaks_mb.append(gpu_sampler.peak_memory_bytes / 1e6)
                    if gpu_sampler.utilization_samples:
                        gpu_utilization_percents.append(statistics.mean(gpu_sampler.utilization_samples))

                if compare_accuracy:
                    _, ground_truth = load_ground_truth_for_image(image_path, ground_truth_folder)
                    if ground_truth:
                        comparison = compare_with_ground_truth(result["extracted_data"], ground_truth)
                        accuracy_samples.append(comparison["accuracy_percent"])
            except Exception as error:
                # One bad image (e.g. a deliberately degraded test sample) shouldn't
                # throw away every other successful measurement in this run.
                message = f"{image_path.name}: {error}"
                image_errors.append(message)
                print(f"  [{model_name}/{device}] image failed, continuing: {message}", file=sys.stderr)

    if gpu_monitor is not None:
        gpu_monitor.close()

    if not latency_samples:
        return {
            "model": model_name,
            "device": device,
            "status": "error",
            "reason": f"all {attempted_count} attempts failed: {image_errors[0] if image_errors else 'unknown error'}",
            "model_load_seconds": model_load_seconds,
        }

    latency_stats = compute_latency_statistics(latency_samples)
    total_elapsed_seconds = sum(latency_samples)
    throughput = (
        round(len(latency_samples) / total_elapsed_seconds, 3) if total_elapsed_seconds > 0 else None
    )

    gpu_metrics = None
    if device != "cpu":
        gpu_metrics = {
            "peak_memory_mb_mean": round(statistics.mean(gpu_peaks_mb), 2) if gpu_peaks_mb else None,
            "utilization_percent_mean": (
                round(statistics.mean(gpu_utilization_percents), 2) if gpu_utilization_percents else None
            ),
            "source": "pynvml" if device == "cuda" else "torch.mps (memory only, no utilization API)",
            "unavailable_reason": gpu_monitor.error if gpu_monitor is not None else None,
        }

    return {
        "model": model_name,
        "device": device,
        "status": "inconclusive" if is_inconclusive else "ok",
        "reason": inconclusive_reason,
        "model_load_seconds": model_load_seconds,
        "latency": latency_stats,
        "throughput_images_per_second": throughput,
        "memory": {
            "rss_delta_mb_mean": round(statistics.mean(rss_deltas_mb), 2) if rss_deltas_mb else None,
            "rss_peak_mb_mean": round(statistics.mean(rss_peaks_mb), 2) if rss_peaks_mb else None,
            "sampling": memory_sampling,
        },
        "gpu": gpu_metrics,
        "cpu_percent_mean": round(statistics.mean(cpu_percents), 2) if cpu_percents else None,
        "accuracy_percent_mean": round(statistics.mean(accuracy_samples), 2) if accuracy_samples else None,
        "sample_count": len(sample_images),
        "images_attempted": attempted_count,
        "images_failed": len(image_errors),
        "error_rate_percent": round(len(image_errors) / attempted_count * 100, 2) if attempted_count else 0,
        "image_errors": image_errors[:5],
    }


def run_concurrent_load_benchmark(
    model_name,
    device,
    process_fn,
    sample_images,
    concurrency_levels,
    requests_per_level,
    memory_sampling,
    memory_sampling_interval_seconds,
):
    # A single already-loaded model instance is shared across worker threads —
    # this mirrors a typical GPU-serving setup (one model behind a server
    # handling concurrent requests), not N independent processes each holding
    # their own model copy, which would multiply VRAM use by concurrency level.
    load_model(model_name, device)
    process_fn(sample_images[0], model_name, device=device)

    gpu_monitor = GpuMonitor(device) if device != "cpu" else None
    process = psutil.Process()
    results = []

    for concurrency in concurrency_levels:
        images_for_run = [sample_images[i % len(sample_images)] for i in range(requests_per_level)]
        latency_samples = []
        task_errors = []

        process.cpu_percent(interval=None)
        sampler = RssSampler(process, memory_sampling_interval_seconds) if memory_sampling == "threaded" else None
        gpu_sampler = (
            GpuSampler(gpu_monitor, memory_sampling_interval_seconds)
            if gpu_monitor is not None and gpu_monitor.error is None and memory_sampling == "threaded"
            else None
        )

        def run_one(image_path):
            started_at = time.perf_counter()
            process_fn(image_path, model_name, device=device)
            return time.perf_counter() - started_at

        wall_started_at = time.perf_counter()
        with contextlib.ExitStack() as stack:
            if sampler:
                stack.enter_context(sampler)
            if gpu_sampler:
                stack.enter_context(gpu_sampler)
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(run_one, image_path) for image_path in images_for_run]
                for future in as_completed(futures):
                    try:
                        latency_samples.append(future.result())
                    except Exception as error:
                        task_errors.append(str(error))
        wall_elapsed_seconds = time.perf_counter() - wall_started_at
        cpu_percent = process.cpu_percent(interval=None)

        latency_stats = compute_latency_statistics(latency_samples)
        throughput = (
            round(len(latency_samples) / wall_elapsed_seconds, 3) if wall_elapsed_seconds > 0 else None
        )

        gpu_metrics = None
        if device != "cpu":
            gpu_metrics = {
                "peak_memory_mb": round(gpu_sampler.peak_memory_bytes / 1e6, 2) if gpu_sampler else None,
                "utilization_percent_mean": (
                    round(statistics.mean(gpu_sampler.utilization_samples), 2)
                    if gpu_sampler and gpu_sampler.utilization_samples
                    else None
                ),
                "unavailable_reason": gpu_monitor.error if gpu_monitor is not None else None,
            }

        results.append(
            {
                "model": model_name,
                "device": device,
                "concurrency": concurrency,
                "status": "ok" if latency_samples else "error",
                "requests_attempted": requests_per_level,
                "requests_failed": len(task_errors),
                "error_rate_percent": round(len(task_errors) / requests_per_level * 100, 2),
                "wall_clock_seconds": round(wall_elapsed_seconds, 4),
                "throughput_requests_per_second": throughput,
                "latency_under_load": latency_stats,
                "peak_rss_mb": round(sampler.peak / 1e6, 2) if sampler else None,
                "gpu": gpu_metrics,
                "cpu_percent_mean": cpu_percent,
            }
        )

    if gpu_monitor is not None:
        gpu_monitor.close()

    return results


def run_benchmark_for_document(
    document_name,
    models,
    devices,
    sample_size,
    seed,
    repeats,
    warmup,
    memory_sampling,
    memory_sampling_interval_ms,
    compare_accuracy,
):
    images_folder, ground_truth_folder, process_fn = DOCUMENT_PROCESSORS[document_name]
    all_images = list_document_images(images_folder)
    sample_images = resolve_sample_images(all_images, sample_size, seed)

    if not sample_images:
        print(f"No test images found for {document_name} in {images_folder}", file=sys.stderr)
        return [], sample_images

    results = []

    for model_name in models:
        for device in devices:
            print(
                f"[{document_name}] {model_name} on {device}: "
                f"{len(sample_images)} images x {repeats} repeats",
                file=sys.stderr,
            )
            results.append(
                run_model_device_benchmark(
                    model_name,
                    device,
                    process_fn,
                    sample_images,
                    ground_truth_folder,
                    repeats,
                    warmup,
                    memory_sampling,
                    memory_sampling_interval_ms / 1000,
                    compare_accuracy,
                )
            )
            # Release this model's memory before loading the next one — without
            # this, every model tested stays resident for the rest of the
            # process, and memory accumulates across the whole batch run.
            release_ocr_model(model_name, device)

    return results, sample_images


def run_concurrency_benchmark_for_document(
    document_name,
    sequential_results,
    sample_images,
    concurrency_levels,
    concurrent_requests,
    memory_sampling,
    memory_sampling_interval_ms,
):
    _, _, process_fn = DOCUMENT_PROCESSORS[document_name]
    testable = [result for result in sequential_results if result.get("status") in {"ok", "inconclusive"}]
    concurrent_results = []

    for result in testable:
        model_name, device = result["model"], result["device"]
        print(
            f"[{document_name}] concurrency test: {model_name} on {device} "
            f"(levels={concurrency_levels}, {concurrent_requests} requests/level)",
            file=sys.stderr,
        )
        try:
            concurrent_results.extend(
                run_concurrent_load_benchmark(
                    model_name,
                    device,
                    process_fn,
                    sample_images,
                    concurrency_levels,
                    concurrent_requests,
                    memory_sampling,
                    memory_sampling_interval_ms / 1000,
                )
            )
        except Exception as error:
            print(f"  concurrency test failed for {model_name}/{device}: {error}", file=sys.stderr)
        finally:
            release_ocr_model(model_name, device)

    return concurrent_results


def add_speedup_vs_cpu(results):
    cpu_p50_by_model = {
        result["model"]: result["latency"]["p50_seconds"]
        for result in results
        if result["device"] == "cpu" and result.get("status") == "ok" and result.get("latency")
    }

    for result in results:
        if result["device"] == "cpu" or result.get("status") != "ok" or not result.get("latency"):
            continue

        cpu_p50 = cpu_p50_by_model.get(result["model"])
        gpu_p50 = result["latency"]["p50_seconds"]

        if cpu_p50 and gpu_p50:
            result["speedup_vs_cpu"] = round(cpu_p50 / gpu_p50, 2)

    return results


def _format_ms(seconds):
    return "-" if seconds is None else f"{seconds * 1000:.0f}"


def _format_number(value, suffix=""):
    return "-" if value is None else f"{value}{suffix}"


def result_to_row(result):
    latency = result.get("latency") or {}
    memory = result.get("memory") or {}
    gpu = result.get("gpu") or {}

    return {
        "model": result["model"],
        "device": result["device"],
        "status": result["status"],
        "p50_ms": _format_ms(latency.get("p50_seconds")),
        "p95_ms": _format_ms(latency.get("p95_seconds")),
        "p99_ms": _format_ms(latency.get("p99_seconds")),
        "throughput_img_s": _format_number(result.get("throughput_images_per_second")),
        "peak_mem_mb": _format_number(memory.get("rss_peak_mb_mean")),
        "gpu_mem_mb": _format_number(gpu.get("peak_memory_mb_mean")),
        "gpu_util_pct": _format_number(gpu.get("utilization_percent_mean")),
        "accuracy_pct": _format_number(result.get("accuracy_percent_mean")),
        "error_rate_pct": _format_number(result.get("error_rate_percent")),
        "speedup_vs_cpu": _format_number(result.get("speedup_vs_cpu"), suffix="x"),
    }


def render_console_table(results):
    rows = [result_to_row(result) for result in results]

    if not rows:
        return "(no results)"

    widths = {
        column: max(len(column), *(len(row[column]) for row in rows)) for column in TABLE_COLUMNS
    }
    header_line = "  ".join(column.ljust(widths[column]) for column in TABLE_COLUMNS)
    separator = "  ".join("-" * widths[column] for column in TABLE_COLUMNS)
    body_lines = [
        "  ".join(row[column].ljust(widths[column]) for column in TABLE_COLUMNS) for row in rows
    ]

    return "\n".join([header_line, separator, *body_lines])


def render_markdown_table(results):
    rows = [result_to_row(result) for result in results]
    header_line = "| " + " | ".join(TABLE_COLUMNS) + " |"
    separator_line = "| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |"
    body_lines = ["| " + " | ".join(row[column] for column in TABLE_COLUMNS) + " |" for row in rows]

    return "\n".join([header_line, separator_line, *body_lines])


def concurrency_result_to_row(result):
    latency = result.get("latency_under_load") or {}
    gpu = result.get("gpu") or {}

    return {
        "model": result["model"],
        "device": result["device"],
        "concurrency": str(result["concurrency"]),
        "status": result["status"],
        "p50_ms": _format_ms(latency.get("p50_seconds")),
        "p95_ms": _format_ms(latency.get("p95_seconds")),
        "p99_ms": _format_ms(latency.get("p99_seconds")),
        "throughput_req_s": _format_number(result.get("throughput_requests_per_second")),
        "peak_mem_mb": _format_number(result.get("peak_rss_mb")),
        "gpu_mem_mb": _format_number(gpu.get("peak_memory_mb")),
        "gpu_util_pct": _format_number(gpu.get("utilization_percent_mean")),
        "error_rate_pct": _format_number(result.get("error_rate_percent")),
    }


def render_concurrency_console_table(results):
    rows = [concurrency_result_to_row(result) for result in results]

    if not rows:
        return "(no results)"

    widths = {
        column: max(len(column), *(len(row[column]) for row in rows))
        for column in CONCURRENCY_TABLE_COLUMNS
    }
    header_line = "  ".join(column.ljust(widths[column]) for column in CONCURRENCY_TABLE_COLUMNS)
    separator = "  ".join("-" * widths[column] for column in CONCURRENCY_TABLE_COLUMNS)
    body_lines = [
        "  ".join(row[column].ljust(widths[column]) for column in CONCURRENCY_TABLE_COLUMNS)
        for row in rows
    ]

    return "\n".join([header_line, separator, *body_lines])


def render_concurrency_markdown_table(results):
    rows = [concurrency_result_to_row(result) for result in results]
    header_line = "| " + " | ".join(CONCURRENCY_TABLE_COLUMNS) + " |"
    separator_line = "| " + " | ".join("---" for _ in CONCURRENCY_TABLE_COLUMNS) + " |"
    body_lines = [
        "| " + " | ".join(row[column] for column in CONCURRENCY_TABLE_COLUMNS) + " |" for row in rows
    ]

    return "\n".join([header_line, separator_line, *body_lines])


def machine_info():
    info = {"mps_available": mps_is_available(), "cuda_available": cuda_is_available()}

    try:
        import torch

        info["torch_version"] = torch.__version__
    except Exception:
        info["torch_version"] = None

    return info


def write_json_report(report, output_folder, run_label):
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{report['document']}_{run_label}.json"

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2)

    return output_path


def write_markdown_report(results, output_folder, document_name, run_label):
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{document_name}_{run_label}.md"
    output_path.write_text(render_markdown_table(results), encoding="utf-8")
    return output_path


def write_concurrency_markdown_report(results, output_folder, document_name, run_label):
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{document_name}_{run_label}_concurrency.md"
    output_path.write_text(render_concurrency_markdown_table(results), encoding="utf-8")
    return output_path


def build_parser():
    parser = argparse.ArgumentParser(prog="python testing/benchmark.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    batch_parser = subparsers.add_parser(
        "batch", help="Benchmark OCR models' latency, memory, throughput, and accuracy."
    )
    batch_parser.add_argument("document", choices=[*DOCUMENT_PROCESSORS.keys(), "all"])
    batch_parser.add_argument("--models", default=",".join(ALL_MODELS))
    batch_parser.add_argument("--devices", default="cpu")
    batch_parser.add_argument("--sample-size", type=int, default=None)
    batch_parser.add_argument("--repeats", type=int, default=3)
    batch_parser.add_argument("--warmup", type=int, default=1)
    batch_parser.add_argument("--seed", type=int, default=42)
    batch_parser.add_argument("--memory-sampling", choices=["off", "threaded"], default="threaded")
    batch_parser.add_argument("--memory-sampling-interval-ms", type=int, default=25)
    batch_parser.add_argument("--output-format", choices=["json", "markdown", "both"], default="both")
    batch_parser.add_argument(
        "--compare-accuracy", dest="compare_accuracy", action="store_true", default=True
    )
    batch_parser.add_argument("--no-compare-accuracy", dest="compare_accuracy", action="store_false")
    batch_parser.add_argument(
        "--test-concurrency",
        action="store_true",
        default=False,
        help=(
            "Also load-test each model+device that succeeded above with concurrent "
            "requests against one shared model instance (not one instance per worker)."
        ),
    )
    batch_parser.add_argument("--concurrency-levels", default="1,2,4,8")
    batch_parser.add_argument("--concurrent-requests", type=int, default=16)

    return parser


def run_batch_command(arguments):
    requested_models = [model.strip() for model in arguments.models.split(",") if model.strip()]
    requested_devices = [device.strip() for device in arguments.devices.split(",") if device.strip()]

    available_models = get_available_ocr_models()
    models = [model for model in requested_models if model in available_models]
    skipped_models = sorted(set(requested_models) - set(models))

    if skipped_models:
        print(f"Skipping models not installed on this machine: {skipped_models}", file=sys.stderr)

    documents = list(DOCUMENT_PROCESSORS.keys()) if arguments.document == "all" else [arguments.document]
    run_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for document_name in documents:
        results, sample_images = run_benchmark_for_document(
            document_name,
            models,
            requested_devices,
            arguments.sample_size,
            arguments.seed,
            arguments.repeats,
            arguments.warmup,
            arguments.memory_sampling,
            arguments.memory_sampling_interval_ms,
            arguments.compare_accuracy,
        )

        if not results:
            continue

        results = add_speedup_vs_cpu(results)

        report = {
            "document": document_name,
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sample_size": len(sample_images),
            "repeats": arguments.repeats,
            "warmup": arguments.warmup,
            "seed": arguments.seed,
            "machine": machine_info(),
            "results": results,
        }

        concurrent_results = []

        if arguments.test_concurrency:
            concurrency_levels = [
                int(level.strip()) for level in arguments.concurrency_levels.split(",") if level.strip()
            ]
            concurrent_results = run_concurrency_benchmark_for_document(
                document_name,
                results,
                sample_images,
                concurrency_levels,
                arguments.concurrent_requests,
                arguments.memory_sampling,
                arguments.memory_sampling_interval_ms,
            )
            report["concurrency_levels_tested"] = concurrency_levels
            report["concurrent_requests_per_level"] = arguments.concurrent_requests
            report["concurrent_results"] = concurrent_results

        if arguments.output_format in {"json", "both"}:
            json_path = write_json_report(report, benchmarks_outputs_folder, run_label)
            print(f"Wrote {json_path}")

        if arguments.output_format in {"markdown", "both"}:
            markdown_path = write_markdown_report(
                results, benchmarks_outputs_folder, document_name, run_label
            )
            print(f"Wrote {markdown_path}")

            if concurrent_results:
                concurrency_markdown_path = write_concurrency_markdown_report(
                    concurrent_results, benchmarks_outputs_folder, document_name, run_label
                )
                print(f"Wrote {concurrency_markdown_path}")

        print(f"\n=== {document_name} ===")
        print(render_console_table(results))
        print()

        if concurrent_results:
            print(f"=== {document_name} (concurrent load) ===")
            print(render_concurrency_console_table(concurrent_results))
            print()


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command == "batch":
        run_batch_command(arguments)


if __name__ == "__main__":
    main()
