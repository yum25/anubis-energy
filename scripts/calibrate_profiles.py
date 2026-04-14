# calibrate_gpu_profiles.py
#
# Derives GPU profile constants for gpu_profiles.py from empirical sources.
#
# Run this script once to produce the fitted polynomial coefficients and
# representative energy-per-token figures for H100 and B200. Copy the
# printed output into gpu_profiles.py.
#
# Usage:
#   python calibrate_gpu_profiles.py
#
# No GPU or network access required — raw data is embedded below.
# Requires only: numpy
#
# ── Sources ───────────────────────────────────────────────────────────────────
#
#   H100 / B200 energy and throughput:
#     ML.Energy Benchmark v3.0 dataset (ml-energy/benchmark-v3, Hugging Face Hub)
#     Model:  meta-llama/Llama-3.1-70B-Instruct
#     Task:   lm-arena-chat
#     Retrieved via mlenergy-data toolkit (pip install mlenergy-data)
#     Citation: Chung et al., NeurIPS Datasets & Benchmarks 2025
#               arXiv:2505.06371
#
#   V100 energy and throughput:
#     Samsi et al. 2023, "From Words to Watts" (arXiv:2310.03003)
#     Model:  LLaMA 7B
#     GPU:    Single NVIDIA V100 32GB
#     Config: batch size 64, max generation length 256 (Table II)
#     Power (~220W) read from Figure 3 (log-scale bar chart).
#     Throughput (~400 tok/s) read from Figure 2.
#     NOTE: Figure 3 uses a log scale; visual reading introduces uncertainty
#     of roughly ±20%. J/tok = power_W / throughput_tok_s = 220/400 = 0.55.
#     This is an approximation, not a directly stated figure in the paper.
#
#   TAPAS thermal coefficients (EQ1, EQ2):
#     Stojkovic et al. 2025, "TAPAS" (ASPLOS 2025, doi:10.1145/3676641.3716025)
#     EQ1 coefficients read from Figure 5 (inlet temp vs outside temp/DC load).
#     EQ2 coefficients read from Figure 7 (GPU temp vs inlet temp/GPU load).
#     MAE < 1C reported by authors for EQ2 on A100 hardware.
#     Applied uniformly to V100/H100/B200 — per-GPU differences not captured.
#
#   GPU TDP and throttle temperatures:
#     NVIDIA product spec sheets.
#     V100: 250W TDP, throttle ~83C
#     H100: 700W TDP, throttle ~83C
#     B200: 1000W TDP, throttle ~85C
#
#   Thermal time constant (THERMAL_TAU_S = 45s):
#     Estimated from TAPAS Figure 6 (GPU temperature response over time).
#     Not directly stated in the paper. Treat as a modeling assumption.
#
#   Prefill energy multiplier (~2-3x decode):
#     Qualitative finding from TokenPowerBench (arXiv:2512.03024), which
#     partitions energy by prefill/decode phase. No exact ratio for 7B/70B
#     at these batch sizes; 2.5x is used as a midpoint estimate.
#
# ── Batch size restriction for polynomial fitting ─────────────────────────────
#
#   Llama-3.1-70B-Instruct requires tensor parallelism across multiple GPUs.
#   At large batch sizes the recovered power (J/tok * tok/s) reflects
#   multi-GPU system power, not single-GPU draw. We restrict fitting to
#   batch sizes 8-32, which:
#     (a) keeps recovered power within plausible single-GPU range
#     (b) matches the simulator's max_num_seqs sweep of 1-32
#
#   With only 3 fit points (batch=8,16,32) a degree-2 polynomial fits
#   exactly by construction (3 points = 3 coefficients, R²=1.0).
#   This is not a validated model — it is a plausible curve through
#   empirically grounded anchor points. Noted as such in output.

from __future__ import annotations
import numpy as np

# ── Raw data ──────────────────────────────────────────────────────────────────

# H100 and B200: from ML.Energy benchmark-v3, Llama-3.1-70B-Instruct,
# lm-arena-chat task. Fields: batch size, energy per token (J/tok),
# output throughput (tok/s).
MLENERGY_DATA: dict[str, list[dict]] = {
    "H100": [
        {"batch": 8, "ept": 3.7587, "tps": 469.7},
        {"batch": 16, "ept": 2.0910, "tps": 862.6},
        {"batch": 32, "ept": 1.2069, "tps": 1486.7},
        {"batch": 64, "ept": 0.7972, "tps": 2510.5},
        {"batch": 96, "ept": 0.6478, "tps": 3244.3},
        {"batch": 128, "ept": 0.5634, "tps": 3876.3},
        {"batch": 192, "ept": 0.5140, "tps": 4461.2},
        {"batch": 256, "ept": 0.4790, "tps": 4915.1},
        {"batch": 384, "ept": 0.4406, "tps": 5318.8},
        {"batch": 512, "ept": 0.4210, "tps": 5921.3},
        {"batch": 768, "ept": 0.3896, "tps": 6507.5},
        {"batch": 1024, "ept": 0.3731, "tps": 6827.2},
    ],
    "B200": [
        {"batch": 8, "ept": 3.5693, "tps": 475.7},
        {"batch": 16, "ept": 1.9027, "tps": 910.8},
        {"batch": 32, "ept": 1.0915, "tps": 1589.6},
        {"batch": 64, "ept": 0.6971, "tps": 2597.8},
        {"batch": 128, "ept": 0.4810, "tps": 3977.5},
        {"batch": 256, "ept": 0.3614, "tps": 5372.4},
        {"batch": 512, "ept": 0.2708, "tps": 7134.0},
        {"batch": 1024, "ept": 0.2267, "tps": 8551.4},
        {"batch": 1536, "ept": 0.2210, "tps": 8682.2},
        {"batch": 2048, "ept": 0.1887, "tps": 10107.8},
    ],
}

# V100: derived from Samsi et al. 2023, LLaMA 7B, single V100.
# Power and throughput read from Figures 2 and 3 (log-scale approximation).
# Only one operating point available; polynomial cannot be fitted from
# this data alone — see note in output.
V100_SAMSI = {
    "batch": 64,
    "power_w": 220.0,  # approx from Figure 3, log-scale bar
    "tps": 400.0,  # approx from Figure 2
}

# GPU spec sheet constants
GPU_SPECS = {
    "V100": {"tdp_w": 250, "throttle_temp_c": 83.0, "max_freq_mhz": 1530},
    "H100": {"tdp_w": 700, "throttle_temp_c": 83.0, "max_freq_mhz": 1980},
    "B200": {"tdp_w": 1000, "throttle_temp_c": 85.0, "max_freq_mhz": 2050},
}

# Polynomial fitting range (see module docstring for rationale)
SIM_BATCH_MIN = 8
SIM_BATCH_MAX = 32

# Representative decode batch for the profile constant.
# Using the smallest fit batch (8) as it is closest to single-request serving.
REP_BATCH = 8

# Prefill multiplier — modeling assumption, see source note above.
PREFILL_MULTIPLIER = 2.5

# ── TAPAS thermal constants (same for all GPUs) ───────────────────────────────
# These are printed once and apply to all profiles.
TAPAS_EQ1 = {"eq1_a": 0.60, "eq1_b": 0.02, "eq1_c": 8.0}
TAPAS_EQ2 = {"eq2_a": 1.05, "eq2_b": 0.22, "eq2_c": 3.5}
THERMAL_TAU_S = 45.0


# ── Helpers ───────────────────────────────────────────────────────────────────


def fit_power_polynomial(
    rows: list[dict],
    batch_min: int,
    batch_max: int,
    tdp_w: float,
    gpu_name: str,
) -> tuple[float, float, float]:
    """
    Fit a degree-2 polynomial P = a*load² + b*load + c to recovered power
    values within [batch_min, batch_max].

    Load proxy: load_pct = (batch / max_fit_batch) * 100
    Power recovery: power_W = ept_J_tok * tps_tok_s

    Returns (a, b, c) coefficients.
    """
    fit_rows = [r for r in rows if batch_min <= r["batch"] <= batch_max]
    if len(fit_rows) < 2:
        raise ValueError(
            f"{gpu_name}: fewer than 2 data points in batch range "
            f"[{batch_min}, {batch_max}]. Cannot fit polynomial."
        )

    batches = np.array([r["batch"] for r in fit_rows])
    power_w = np.array([r["ept"] * r["tps"] for r in fit_rows])
    max_batch = batches.max()
    load_pct = (batches / max_batch) * 100.0

    coeffs = np.polyfit(load_pct, power_w, deg=2)
    a, b, c = coeffs

    print(
        f"  Fit points ({len(fit_rows)} — degree-2 fits exactly, R²=1.0 by construction):"
    )
    for batch, load, pw in zip(batches, load_pct, power_w):
        print(f"    batch={batch:4d}  load={load:5.1f}%  power={pw:.1f}W")

    p_idle = float(np.polyval(coeffs, 5))
    p_full = float(np.polyval(coeffs, 100))
    print(f"  Extrapolated P(5%  load) = {p_idle:.1f}W  (idle estimate)")
    print(f"  Extrapolated P(100% load) = {p_full:.1f}W  (TDP = {tdp_w}W)")
    if p_full > tdp_w * 1.15:
        print("  NOTE: P(100%) exceeds TDP by >15% — multi-GPU power likely included.")
        print(
            "        Divide by num_gpus if known, or accept as conservative estimate."
        )
    if p_idle < 0:
        print(
            "  WARNING: P(5%) is negative — polynomial extrapolates poorly at low load."
        )
        print("           Consider using max(P(load), idle_floor) in the simulator.")

    return float(a), float(b), float(c)


def ept_for_batch(rows: list[dict], batch: int) -> float:
    """Return ept for an exact batch size, or interpolate between neighbors."""
    exact = [r for r in rows if r["batch"] == batch]
    if exact:
        return exact[0]["ept"]
    # Linear interpolation between surrounding points
    lower = [r for r in rows if r["batch"] < batch]
    upper = [r for r in rows if r["batch"] > batch]
    if not lower or not upper:
        raise ValueError(f"Cannot interpolate ept for batch={batch}")
    lo = max(lower, key=lambda r: r["batch"])
    hi = min(upper, key=lambda r: r["batch"])
    t = (batch - lo["batch"]) / (hi["batch"] - lo["batch"])
    return lo["ept"] + t * (hi["ept"] - lo["ept"])


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    separator = "=" * 65

    print("'''")
    print(separator)
    print("GPU Profile Calibration")
    print("Produces constants for gpu_profiles.py")
    print(separator)

    # ── Shared TAPAS thermal constants ────────────────────────────────────────
    print("\n── Shared thermal constants (all GPUs) ──────────────────────────")
    print("Source: TAPAS Stojkovic et al. ASPLOS 2025")
    print("        EQ1 from Figure 5, EQ2 from Figure 7 (A100 measurements)")
    print("        Applied uniformly — per-GPU differences not captured")
    print(f"  eq1_a = {TAPAS_EQ1['eq1_a']}  # inlet vs outside temp slope")
    print(f"  eq1_b = {TAPAS_EQ1['eq1_b']}  # inlet vs DC load slope")
    print(f"  eq1_c = {TAPAS_EQ1['eq1_c']}  # inlet temp intercept")
    print(f"  eq2_a = {TAPAS_EQ2['eq2_a']}  # GPU temp vs inlet slope")
    print(f"  eq2_b = {TAPAS_EQ2['eq2_b']}  # GPU temp vs load slope (per %)")
    print(f"  eq2_c = {TAPAS_EQ2['eq2_c']}  # GPU temp intercept")
    print(
        f"  thermal_tau_s = {THERMAL_TAU_S}  # RC time constant (estimated from Fig 6)"
    )

    results = {}

    # ── H100 and B200 ─────────────────────────────────────────────────────────
    for gpu, rows in MLENERGY_DATA.items():
        print(f"\n{separator}")
        print(f"  {gpu}")
        print(separator)
        print("Source: ML.Energy benchmark-v3 (HF Hub: ml-energy/benchmark-v3)")
        print("        Llama-3.1-70B-Instruct, lm-arena-chat, Chung et al. 2025")
        specs = GPU_SPECS[gpu]

        # Full data table
        print("\n  Full dataset (batch, J/tok, tok/s, recovered power):")
        print(f"  {'batch':>6} | {'J/tok':>7} | {'tok/s':>7} | {'power_W':>8}")
        print(f"  {'-' * 6}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 8}")
        for r in rows:
            pw = r["ept"] * r["tps"]
            print(
                f"  {r['batch']:>6} | {r['ept']:>7.4f} | {r['tps']:>7.1f} | {pw:>8.1f}"
            )

        # Polynomial fit on restricted range
        print("\n  Power polynomial fit (batch {SIM_BATCH_MIN}–{SIM_BATCH_MAX}):")
        print("  NOTE: restricted to simulator's max_num_seqs range.")
        print("        3 points → degree-2 polynomial is exact, not validated.")
        a, b, c = fit_power_polynomial(
            rows, SIM_BATCH_MIN, SIM_BATCH_MAX, specs["tdp_w"], gpu
        )

        # Representative ept
        rep_ept = ept_for_batch(rows, REP_BATCH)
        prefill_ept = rep_ept * PREFILL_MULTIPLIER
        ept_min = min(r["ept"] for r in rows)
        ept_max = max(r["ept"] for r in rows)

        print("\n  Energy per token:")
        print(f"    Representative decode ept (batch={REP_BATCH}): {rep_ept:.4f} J/tok")
        print(
            f"    Prefill ept ({PREFILL_MULTIPLIER}x decode, modeling assumption): "
            f"{prefill_ept:.4f} J/tok"
        )
        print(f"    Full range: {ept_min:.4f} – {ept_max:.4f} J/tok")

        results[gpu] = {
            "power_a": a,
            "power_b": b,
            "power_c": c,
            "energy_per_token_decode_j": rep_ept,
            "energy_per_token_prefill_j": prefill_ept,
        }

    # ── V100 ──────────────────────────────────────────────────────────────────
    print(f"\n{separator}")
    print("  V100")
    print(separator)
    print("Source: Samsi et al. 2023, arXiv:2310.03003")
    print("        LLaMA 7B, single V100 32GB, batch=64, max_gen_len=256")
    print("        Power (~220W) read from Figure 3 (log-scale bar, ±~20%)")
    print("        Throughput (~400 tok/s) read from Figure 2")
    print("        J/tok = 220W / 400 tok/s = 0.55 J/tok (derived, not stated)")
    print()
    print("  Only one operating point available from this source.")
    print("  A degree-2 polynomial cannot be fitted from a single point.")
    print("  Power polynomial for V100 is manually calibrated to two anchors:")
    specs_v100 = GPU_SPECS["V100"]
    print("    Idle (~5% load):  ~80W")
    print(f"    Full load (100%): ~{specs_v100['tdp_w']}W (TDP)")
    print("  Coefficients chosen so P(5)≈80W and P(100)≈250W.")
    print("  This is an estimate with no empirical validation.")

    # Manual calibration for V100
    # Solve: a*5^2 + b*5 + c = 80
    #        a*100^2 + b*100 + c = 250
    # Two equations, three unknowns — fix a=0.005 (small quadratic term)
    # and solve for b and c:
    #   25a + 5b + c = 80   →  5b + c = 80 - 0.125 = 79.875
    #   10000a + 100b + c = 250 → 100b + c = 250 - 50 = 200
    # Subtract: 95b = 120.125 → b = 1.264
    #           c = 79.875 - 5*1.264 = 73.555
    v100_a = 0.005
    v100_b = 1.264
    v100_c = 73.555

    v100_ept_decode = V100_SAMSI["power_w"] / V100_SAMSI["tps"]
    v100_ept_prefill = v100_ept_decode * PREFILL_MULTIPLIER

    p_idle_v100 = v100_a * 25 + v100_b * 5 + v100_c
    p_full_v100 = v100_a * 10000 + v100_b * 100 + v100_c
    print("\n  Manual polynomial verification:")
    print(f"    P(5%  load) = {p_idle_v100:.1f}W")
    print(f"    P(100% load) = {p_full_v100:.1f}W  (TDP = {specs_v100['tdp_w']}W)")
    print("\n  Energy per token:")
    print(f"    Decode ept (220W / 400 tok/s): {v100_ept_decode:.4f} J/tok")
    print(f"    Prefill ept ({PREFILL_MULTIPLIER}x): {v100_ept_prefill:.4f} J/tok")

    results["V100"] = {
        "power_a": v100_a,
        "power_b": v100_b,
        "power_c": v100_c,
        "energy_per_token_decode_j": v100_ept_decode,
        "energy_per_token_prefill_j": v100_ept_prefill,
    }

    # ── Final profile output ──────────────────────────────────────────────────
    print(f"\n{separator}")
    print("  FINAL GPU_PROFILES DICTIONARY")
    print("  Copy this into gpu_profiles.py")
    print(separator)
    print("'''")

    print("\nGPU_PROFILES = {")
    for gpu in ["V100", "H100", "B200"]:
        specs = GPU_SPECS[gpu]
        r = results[gpu]
        print(f'    "{gpu}": {{')
        print("        # Hardware — NVIDIA spec sheet")
        print(f'        "TDP_W":           {specs["tdp_w"]},')
        print(f'        "THROTTLE_TEMP_C": {specs["throttle_temp_c"]},')
        print(f'        "MAX_FREQ_MHZ":    {specs["max_freq_mhz"]},')
        print()
        if gpu == "V100":
            src = (
                "# Samsi et al. 2023 arXiv:2310.03003, LLaMA 7B single V100\n"
                f"        # Approximated from Figures 2 & 3 (log-scale); "
                f"J/tok = 220W/400toks = {r['energy_per_token_decode_j']:.4f}"
            )
        else:
            src = (
                "# ML.Energy benchmark-v3 (Chung et al. NeurIPS 2025)\n"
                f"        # Llama-3.1-70B-Instruct, lm-arena-chat, batch={REP_BATCH}"
            )
        print(f"        # Energy per token — {src}")
        print(
            f'        "ENERGY_PER_TOKEN_DECODE_J":  {r["energy_per_token_decode_j"]:.4f},'
        )
        print(
            f'        "ENERGY_PER_TOKEN_PREFILL_J": {r["energy_per_token_prefill_j"]:.4f},  '
            f"# {PREFILL_MULTIPLIER}x decode (TokenPowerBench arXiv:2512.03024, modeling assumption)"
        )
        print()
        if gpu == "V100":
            poly_src = "# Manually calibrated to TDP anchors (single data point from Samsi et al.)"
        else:
            poly_src = (
                f"# Fitted from ML.Energy data, batch {SIM_BATCH_MIN}-{SIM_BATCH_MAX}\n"
                f"        # 3-point exact fit (degree-2); plausible curve, not validated"
            )
        print(f"        # Power polynomial P = a*load² + b*load + c  — {poly_src}")
        print(f'        "POW_A": {r["power_a"]:.6f},')
        print(f'        "POW_B": {r["power_b"]:.6f},')
        print(f'        "POW_C": {r["power_c"]:.6f},')
        print()
        print("        # TAPAS thermal model — Stojkovic et al. ASPLOS 2025")
        print("        # EQ1 (inlet temp): Figure 5.  EQ2 (GPU temp): Figure 7.")
        print("        # Measured on A100; applied uniformly as approximation.")
        print(f'        "EQ1_A": {TAPAS_EQ1["eq1_a"]},')
        print(f'        "EQ1_B": {TAPAS_EQ1["eq1_b"]},')
        print(f'        "EQ1_C": {TAPAS_EQ1["eq1_c"]},')
        print(f'        "EQ2_A": {TAPAS_EQ2["eq2_a"]},')
        print(f'        "EQ2_B": {TAPAS_EQ2["eq2_b"]},')
        print(f'        "EQ2_C": {TAPAS_EQ2["eq2_c"]},')
        print()
        print("        # Thermal time constant — estimated from TAPAS Figure 6.")
        print("        # Not stated in paper; treat as modeling assumption.")
        print(f'        "THERMAL_TAU_S": {THERMAL_TAU_S},')
        print("    },")
    print("}")


if __name__ == "__main__":
    main()
