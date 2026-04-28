# generate_sim_static_profile.py
#
# Generates a synthetic static_profile.json for H100 or B200 (or both)
# using the same physics constants as SimulatedGpuDataSource, without
# requiring real hardware or a running vLLM server.
#
# The synthetic profile is structurally identical to the JSON produced by
# run_static_profiling.py, so build_static_models.py can consume it unchanged
# (modulo the MAX_FREQ_MHZ normalization constant — see note below).
#
# IMPORTANT — normalization constant in build_static_models.py:
#   The real V100 sweep used GPU_FREQS topped at 1410 MHz, so that file
#   hardcodes _MAX_FREQ_MHZ = 1377 (the actual top of its grid).
#   H100 and B200 have different max clocks. The generated profile embeds
#   the correct per-GPU max freq in each record, and this script prints
#   a reminder to patch build_static_models.py before fitting.
#
# Usage:
#   # Both GPUs (default)
#   python generate_sim_static_profile.py
#
#   # Single GPU
#   python generate_sim_static_profile.py --gpu H100
#   python generate_sim_static_profile.py --gpu B200
#
#   # Custom output directory
#   python generate_sim_static_profile.py --out-dir static_profiling/sim_profiles
#
# Output:
#   static_profile_H100.json   (or B200, or both)
#
# No GPU, network, or vLLM access required. Requires only: numpy (for
#   the polynomial evaluation, though math.fsum would also work).

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# ── GPU profiles (mirrors profilers/profiles.py exactly) ─────────────────────
# Copied here so this script can run standalone without importing from profilers/.
GPU_PROFILES: dict[str, dict] = {
    "H100": {
        "TDP_W": 700,
        "THROTTLE_TEMP_C": 83.0,
        "MAX_FREQ_MHZ": 1980,
        "ENERGY_PER_TOKEN_DECODE_J": 3.7587,
        "ENERGY_PER_TOKEN_PREFILL_J": 9.3968,
        # POW_* fitted from ML.Energy data, all 12 batch sizes (full range).
        # Overdetermined least-squares (degree-2); reflects multi-GPU system power.
        "POW_A": -0.150237,
        "POW_B": 21.620341,
        "POW_C": 1832.885504,
        "EQ1_A": 0.6,
        "EQ1_B": 0.02,
        "EQ1_C": 8.0,
        "EQ2_A": 1.05,
        "EQ2_B": 0.22,
        "EQ2_C": 3.5,
        "THERMAL_TAU_S": 45.0,
    },
    "B200": {
        "TDP_W": 1000,
        "THROTTLE_TEMP_C": 85.0,
        "MAX_FREQ_MHZ": 2050,
        "ENERGY_PER_TOKEN_DECODE_J": 3.5693,
        "ENERGY_PER_TOKEN_PREFILL_J": 8.9232,
        # POW_* fitted from ML.Energy data, all 10 batch sizes (full range).
        # Overdetermined least-squares (degree-2); reflects multi-GPU system power.
        "POW_A": -0.061269,
        "POW_B": 7.151743,
        "POW_C": 1772.585383,
        "EQ1_A": 0.6,
        "EQ1_B": 0.02,
        "EQ1_C": 8.0,
        "EQ2_A": 1.05,
        "EQ2_B": 0.22,
        "EQ2_C": 3.5,
        "THERMAL_TAU_S": 45.0,
    },
}

# ── Sweep grid — mirrors run_static_profiling.py ──────────────────────────────
# Frequencies chosen as round fractions of each GPU's max clock so the
# load% grid is comparable across GPUs (25 / 50 / 75 / 100% of max).
# The static_scheduler normalises by the max freq in the sweep, so we
# use the same 4-point grid structure as the V100 sweep.
GPU_FREQS_FRACTION = [0.25, 0.50, 0.75, 1.00]  # fraction of MAX_FREQ_MHZ

MAX_SEQS = [1, 8, 16, 32]

PHASES = {
    "prefill": {"input_mean": 1000.0, "output_mean": 50.0},
    "decode": {"input_mean": 50.0, "output_mean": 500.0},
}

# ── Simulator constants (from SimulatedGpuDataSource) ─────────────────────────
BASE_LOAD_PER_SEQ = 2.5  # % GPU utilisation per concurrent sequence
MEASUREMENT_DUR_S = 60.0  # synthetic "steady-state window" duration

# Standard ambient used when computing the offline profile
T_OUTSIDE_C = 22.0
DC_LOAD_PCT = 40.0

# SLO threshold from build_static_models.py
SLO_P99_MS = 5000.0


# ── Physics helpers ───────────────────────────────────────────────────────────


def inlet_temp(p: dict, t_outside: float, dc_load: float) -> float:
    """TAPAS Eq 1."""
    return p["EQ1_A"] * t_outside + p["EQ1_B"] * dc_load + p["EQ1_C"]


def gpu_temp_steady(p: dict, t_inlet: float, load_pct: float) -> float:
    """TAPAS Eq 2 — steady-state GPU temperature."""
    return p["EQ2_A"] * t_inlet + p["EQ2_B"] * load_pct + p["EQ2_C"]


def gpu_power(p: dict, load_pct: float) -> float:
    """TAPAS power polynomial (Watts)."""
    return p["POW_A"] * load_pct**2 + p["POW_B"] * load_pct + p["POW_C"]


def freq_factor(freq_mhz: int, max_freq_mhz: int) -> float:
    """Same exponent as SimulatedGpuDataSource.next_observation()."""
    return (freq_mhz / max_freq_mhz) ** 0.7


def load_pct(max_num_seqs: int, ff: float) -> float:
    """GPU utilisation % — mirrors simulator formula (no load_multiplier)."""
    return min(max_num_seqs * BASE_LOAD_PER_SEQ * ff, 100.0)


def toks_per_s(phase: str, max_num_seqs: int, ff: float) -> float:
    """Token throughput — mirrors simulator formula (no noise, no multiplier)."""
    if phase == "prefill":
        return max_num_seqs * 15 * ff
    else:
        return max_num_seqs * 40 * ff


def p99_latency_ms(phase: str, max_num_seqs: int, tps: float) -> float:
    """
    Estimate P99 end-to-end latency.

    Approach: model each request as waiting behind (max_num_seqs - 1) others
    in a batch.  Each token takes 1/tps seconds; a response of output_mean
    tokens costs output_mean / tps seconds.

    prefill:  output_mean = 50  tokens  → latency dominated by input processing
    decode:   output_mean = 500 tokens  → latency dominated by generation

    We use the phase's output_mean as the response length, then add a
    head-of-line blocking term proportional to batch depth.
    This is a rough approximation — real P99 depends on request arrival
    distribution, but it gives a plausible ordering across configs.
    """
    output_mean = PHASES[phase]["output_mean"]
    if tps <= 0:
        return float("inf")
    # Mean latency per request at this throughput
    mean_lat_ms = (output_mean / tps) * 1000.0
    # P99 heuristic: ~3x mean under moderate load (empirically conservative)
    hol_factor = 1.0 + 2.0 * min(max_num_seqs / 32.0, 1.0)
    return mean_lat_ms * hol_factor


# ── Profile generation ────────────────────────────────────────────────────────


def generate_profile(
    gpu_name: str,
    t_outside_c: float = T_OUTSIDE_C,
    dc_load_pct: float = DC_LOAD_PCT,
) -> list[dict]:
    """
    Sweep all (freq, max_seqs, phase) combinations for `gpu_name` and return
    a list of records in the same schema as run_static_profiling.py.
    """
    p = GPU_PROFILES[gpu_name]
    max_freq = p["MAX_FREQ_MHZ"]

    # Pre-compute ambient baseline (no events, standard DC conditions)
    t_inlet = inlet_temp(p, t_outside_c, dc_load_pct)

    records = []

    for frac in GPU_FREQS_FRACTION:
        freq_mhz = round(max_freq * frac)
        ff = freq_factor(freq_mhz, max_freq)

        for max_seqs in MAX_SEQS:
            lp = load_pct(max_seqs, ff)

            # Steady-state temperature at this config
            temp_c = gpu_temp_steady(p, t_inlet, lp)
            # Hard cap at throttle threshold (mirrors simulator)
            temp_c = min(temp_c, p["THROTTLE_TEMP_C"])

            # Power and energy
            power_w = max(gpu_power(p, lp), 0.0)
            steady_energy_j = power_w * MEASUREMENT_DUR_S

            for phase, phase_cfg in PHASES.items():
                tps = toks_per_s(phase, max_seqs, ff)

                # Energy per token at steady state (no thermal penalty —
                # this is the "offline" profile, measured at baseline temp)
                ept = (power_w / tps) if tps > 0 else None

                p99_ms = p99_latency_ms(phase, max_seqs, tps)
                p50_ms = p99_ms / 3.0  # rough P50 ≈ P99/3

                run_id = f"{gpu_name.lower()}-sim_{freq_mhz}_{max_seqs}_{phase}"

                records.append(
                    {
                        "run_id": run_id,
                        "gpu_freq_mhz": freq_mhz,
                        "max_num_seqs": max_seqs,
                        "phase": phase,
                        "input_mean": phase_cfg["input_mean"],
                        "output_mean": phase_cfg["output_mean"],
                        # Energy
                        "steady_state_energy_j": round(steady_energy_j, 3),
                        "steady_state_energy_per_token_j": round(ept, 6)
                        if ept
                        else None,
                        "steady_state_duration_s": MEASUREMENT_DUR_S,
                        # Latency (synthetic estimates)
                        "mean_e2el_ms": round(p99_ms / 1.5, 1),
                        "p50_e2el_ms": round(p50_ms, 1),
                        "p99_e2el_ms": round(p99_ms, 1),
                        "mean_ttft_ms": round(p99_ms * 0.1, 1),  # TTFT << E2EL
                        "p99_ttft_ms": round(p99_ms * 0.2, 1),
                        # Throughput
                        "request_throughput": round(tps / phase_cfg["output_mean"], 4),
                        "output_throughput": round(tps, 2),
                        # Temperature
                        "avg_gpu_temp_c": round(temp_c, 2),
                        "temp_before_c": round(t_inlet + 5.0, 2),  # approx idle start
                        "temp_after_c": round(temp_c, 2),
                        # Metadata
                        "_source": f"synthetic — {gpu_name} sim profile (t_outside={t_outside_c:.1f}C)",
                        "_profile_t_outside_c": t_outside_c,
                        "_gpu_model": gpu_name,
                        "_max_freq_mhz_for_normalization": max_freq,
                    }
                )

    return records


# ── Normalization patch reminder ───────────────────────────────────────────────


def print_patch_reminder(gpu_name: str) -> None:
    max_freq = GPU_PROFILES[gpu_name]["MAX_FREQ_MHZ"]
    top_freq = round(max_freq * GPU_FREQS_FRACTION[-1])
    print(
        f"\n  [IMPORTANT] build_static_models.py patch needed for {gpu_name}:\n"
        f"    Change  _MAX_FREQ_MHZ = 1377\n"
        f"    To      _MAX_FREQ_MHZ = {top_freq}   # top of {gpu_name} sweep\n"
        f"    (or pass it as a parameter if you've refactored the script)\n"
        f"    The static_scheduler.py _MAX_FREQ_MHZ must match too."
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


# -- CLI ----------------------------------------------------------------------


# -- CLI ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic static profile for H100 or B200 and write\n"
        "to static_profiling/static_profile.json -- the path\n"
        "build_static_models.py reads by default. Run once per GPU model\n"
        "before running build_static_models.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gpu",
        choices=["H100", "B200"],
        required=True,
        help="GPU model to generate a profile for. Must be specified explicitly.",
    )
    parser.add_argument(
        "--out-dir",
        default="static_profiling",
        help="Directory to write static_profile.json (default: static_profiling/).",
    )
    parser.add_argument(
        "--t-outside-c",
        type=float,
        default=22.0,
        dest="t_outside_c",
        help=(
            "Ambient outside temperature (°C) assumed during offline profiling. "
            "Lower this relative to the experiment runtime ambient to simulate a "
            "profile measured on a cool day being used in a warmer datacenter. "
            "Default: 22.0°C."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu = args.gpu
    records = generate_profile(gpu, t_outside_c=args.t_outside_c)

    # Always write to static_profile.json so build_static_models.py picks it
    # up without any path changes. Whichever GPU was generated last is the
    # active profile for the next build_static_models.py run.
    out_path = out_dir / "static_profile.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    within_slo = sum(
        1
        for r in records
        if r["p99_e2el_ms"] is not None and r["p99_e2el_ms"] < SLO_P99_MS
    )
    freqs = sorted(set(r["gpu_freq_mhz"] for r in records))
    energy_vals = [r["steady_state_energy_j"] for r in records]
    temp_vals = [r["avg_gpu_temp_c"] for r in records]

    print(f"\n{'=' * 60}")
    print(f"  {gpu} synthetic profile")
    print(f"{'=' * 60}")
    print(
        f"  Records:         {len(records)}  ({len(freqs)} freqs x {len(MAX_SEQS)} batch sizes x 2 phases)"
    )
    print(f"  Freq grid (MHz): {freqs}")
    print(
        f"  Within SLO:      {within_slo}/{len(records)} configs (P99 < {SLO_P99_MS:.0f} ms)"
    )
    print(
        f"  Energy range:    {min(energy_vals):.1f} - {max(energy_vals):.1f} J  (over {MEASUREMENT_DUR_S:.0f}s window)"
    )
    print(f"  Temp range:      {min(temp_vals):.1f} - {max(temp_vals):.1f} C")
    print(f"  Profiling temp:  {args.t_outside_c:.1f} C outside ambient")
    print(f"  Written to:      {out_path}")
    print_patch_reminder(gpu)

    print(
        "\nNext steps:\n"
        "  1. Apply the _MAX_FREQ_MHZ patch above to:\n"
        "       static_profiling/build_static_models.py\n"
        "       schedulers/static_scheduler.py\n"
        "  2. Run build_static_models.py (no path changes needed):\n"
        "       python static_profiling/build_static_models.py\n"
        "  3. The resulting static_pareto.json / .pkl files are ready for the\n"
        "     static scheduler in sim mode.\n"
    )


if __name__ == "__main__":
    main()
