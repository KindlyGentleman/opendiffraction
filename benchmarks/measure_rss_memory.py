"""Measure and verify peak RSS memory during single- and multi-worker dataset generation."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import psutil

sys.path.insert(0, os.path.abspath("."))

from opendiffraction.generator.pipeline import generate_dataset


def profile_worker_memory(num_workers: int, total_samples: int = 12) -> dict[str, float]:
    """Measure peak RSS during dataset generation."""
    process = psutil.Process()
    peak_rss = 0
    stop_monitoring = threading.Event()

    def monitor() -> None:
        nonlocal peak_rss
        while not stop_monitoring.is_set():
            try:
                rss = process.memory_info().rss
                if rss > peak_rss:
                    peak_rss = rss
            except Exception:
                pass
            time.sleep(0.01)

    with tempfile.TemporaryDirectory() as tmp_dir:
        t_mon = threading.Thread(target=monitor, daemon=True)
        t_mon.start()

        t0 = time.time()
        res = generate_dataset(
            output_dir=tmp_dir,
            total_samples=total_samples,
            shard_size=6,
            profiles_mode="12",
            num_workers=num_workers,
            resume=False,
            overwrite=True,
        )
        elapsed = time.time() - t0

        stop_monitoring.set()
        t_mon.join(timeout=1.0)

    peak_rss_mb = peak_rss / (1024 * 1024)
    peak_rss_gib = peak_rss / (1024 * 1024 * 1024)

    return {
        "num_workers": num_workers,
        "total_samples": total_samples,
        "elapsed_seconds": float(elapsed),
        "peak_rss_mb": float(peak_rss_mb),
        "peak_rss_gib": float(peak_rss_gib),
        "throughput_sps": float(total_samples / max(elapsed, 1e-4)),
        "memory_bounded_ok": peak_rss_gib < 4.0,
    }


def run_benchmark() -> dict[str, Any]:
    print("Profiling CPU Peak RSS memory across worker concurrency levels...")
    results = {}
    for w in [1, 2, 4]:
        print(f"Running with num_workers={w}...")
        metrics = profile_worker_memory(num_workers=w, total_samples=12)
        print(f"  Workers: {w} -> Peak RSS: {metrics['peak_rss_mb']:.1f} MB, Time: {metrics['elapsed_seconds']:.2f}s")
        results[f"workers_{w}"] = metrics

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "rss_memory_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved memory benchmark summary to {out_path}")
    return results


if __name__ == "__main__":
    run_benchmark()
