"""Unified evaluation runner for OpenDiffraction classical benchmark baselines."""

from __future__ import annotations

import argparse
import json
import numpy as np

from opendiffraction.objects.specimens import make_virus_particle, make_core_shell_nanoparticle
from opendiffraction.physics.forward import SimulationConfig, simulate_diffraction
from benchmarks.phase_retrieval import evaluate_phase_retrieval
from benchmarks.denoising import evaluate_denoising
from benchmarks.hit_detection import classify_hit
from benchmarks.missing_pixel_recovery import evaluate_inpainting


def run_benchmark_suite(seed: int = 42) -> dict[str, dict[str, float]]:
    """Execute all CPU baseline benchmarks and return structured metrics."""
    rng = np.random.default_rng(seed)
    results = {}

    print("[Benchmark] Running B1: Phase Retrieval (HIO/ER)...")
    obj_pr = make_virus_particle(64, rng=rng)
    cfg_pr = SimulationConfig(photon_budget=5e6, coherence_length=float("inf"), include_beam_stop=False)
    sim_pr = simulate_diffraction(obj_pr, cfg_pr, rng=rng)
    results["B1_phase_retrieval"] = evaluate_phase_retrieval(sim_pr.intensity_measured, obj_pr, seed=seed)

    print("[Benchmark] Running B2: Denoising...")
    obj_dn = make_core_shell_nanoparticle(64, rng=rng)
    cfg_dn = SimulationConfig(photon_budget=1e6)
    sim_dn = simulate_diffraction(obj_dn, cfg_dn, rng=rng)
    results["B2_denoising"] = evaluate_denoising(sim_dn.intensity_measured, sim_dn.intensity_expected)

    print("[Benchmark] Running B3: Missing Pixel Inpainting...")
    cfg_in = SimulationConfig(photon_budget=3e6, include_beam_stop=True, include_module_gaps=True)
    sim_in = simulate_diffraction(obj_dn, cfg_in, rng=rng)
    results["B3_inpainting"] = evaluate_inpainting(sim_in.intensity_measured, sim_in.valid_mask, sim_in.intensity_ideal)

    print("[Benchmark] Running B4: Hit Classification...")
    hit_res = classify_hit(sim_in.intensity_measured)
    results["B4_hit_detection"] = {
        "is_hit": 1.0 if hit_res["is_hit"] else 0.0,
        "energy": hit_res["total_energy"],
        "peak": hit_res["peak_intensity"],
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OpenDiffraction scientific baselines.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic evaluation seed")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    results = run_benchmark_suite(seed=args.seed)
    print("\n--- Benchmark Baseline Summary ---")
    print(json.dumps(results, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
