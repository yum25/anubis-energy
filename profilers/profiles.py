# profiles.py
# Hardware profiles for the GPU simulator.
#
# Sources:
#   V100:  Samsi et al. 2023 (arXiv:2310.03003) — direct measurement of
#          LLaMA inference on V100 32GB. ~3-4 J/tok for 65B model;
#          ~0.5 J/tok for 7B model based on approximations with provided figures.
#          V100 TDP: 250W (Table I of Samsi et al.)
#
#   H100:  ML.Energy benchmark paper (Chung et al. 2025, arXiv:2505.06371)
#          Figure 10/11: LLMs draw ~30-40% of H100 TDP (700W) during decode,
#          so ~210-280W. Paper notes A100 and H100 have comparable energy/tok
#          for memory-bound LLM decode. H100 TDP: 700W.
#          ML.Energy v3.0 blog: B200 achieves ~35% median energy reduction
#          over H100, used to back-derive H100 figures.
#
#   B200:  ML.Energy Leaderboard v3.0 (ml-energy/benchmark-v3 on HF Hub,
#          accessed via ml.energy/data). B200 TDP: 1000W.
#          Blog post reports B200 wins 88% of H100 comparisons at median
#          35% energy reduction for LLMs.
#
#   Thermal coefficients (EQ1, EQ2):
#          TAPAS Stojkovic et al. ASPLOS 2025, Equations 1-2 and Figures 5-7.
#          Coefficients are approximated from the regression curves shown;
#          exact values were not published.
#
#   Thermal time constant (THERMAL_TAU_S):
#          Estimated from TAPAS Figure 6 (GPU temperature response over time).
#          Not directly stated in the paper — treat as a modeling assumption.

GPU_PROFILES = {
    "V100": {
        # Hardware
        "TDP_W": 250,
        "THROTTLE_TEMP_C": 83.0,
        "MAX_FREQ_MHZ": 1530,
        # Energy per token (decode phase, Llama-2-7B equivalent)
        # Source: Samsi et al. 2023, scaled from 65B to 7B by parameter ratio
        # 65B ~0.35 J/tok on V100 at batch=64 → 7B ~0.035-0.05 J/tok
        # Using 0.04 as midpoint estimate
        "ENERGY_PER_TOKEN_DECODE_J": 0.04,
        # Prefill is compute-bound vs decode memory-bound; empirically ~2-3x
        # more energy per output token equivalent (TokenPowerBench 2025)
        "ENERGY_PER_TOKEN_PREFILL_J": 0.10,
        # Power model: P = a*load^2 + b*load + c (Watts)
        # Calibrated to V100 TDP=250W at full load, idle ~60W
        "POWER_A": 0.005,
        "POWER_B": 1.50,
        "POWER_C": 60.0,
        # TAPAS Eq 1: T_inlet = a*T_outside + b*dc_load + c
        "EQ1_A": 0.60,
        "EQ1_B": 0.02,
        "EQ1_C": 8.0,
        # TAPAS Eq 2: T_GPU = a*T_inlet + b*gpu_load + c
        # V100 runs slightly hotter than A100 at same load
        "EQ2_A": 1.05,
        "EQ2_B": 0.22,
        "EQ2_C": 3.5,
        # Thermal time constant (seconds to reach new steady state)
        # Estimated from TAPAS Fig 6 — treat as assumption
        "THERMAL_TAU_S": 45.0,
    },
    "H100": {
        # Hardware
        "TDP_W": 700,
        "THROTTLE_TEMP_C": 83.0,
        "MAX_FREQ_MHZ": 1980,
        # Energy per token
        # Source: ML.Energy paper Fig 10/11 — LLMs draw ~30-40% H100 TDP
        # during decode. At 700W TDP, ~210-280W power draw.
        # For Llama-2-7B at typical batch, ML.Energy notes comparable
        # energy/tok to A100 for memory-bound decode.
        # Using 0.012 J/tok as estimate (more efficient than V100 per token
        # due to higher memory bandwidth: 3.35 TB/s vs 0.9 TB/s)
        "ENERGY_PER_TOKEN_DECODE_J": 0.012,
        "ENERGY_PER_TOKEN_PREFILL_J": 0.030,
        # Power model: calibrated to H100 TDP=700W at full load, idle ~150W
        "POWER_A": 0.015,
        "POWER_B": 4.50,
        "POWER_C": 150.0,
        # TAPAS thermal equations — same structure, H100 has better cooling
        # headroom so slightly lower intercept
        "EQ1_A": 0.60,
        "EQ1_B": 0.02,
        "EQ1_C": 8.0,
        "EQ2_A": 1.03,
        "EQ2_B": 0.20,
        "EQ2_C": 3.0,
        "THERMAL_TAU_S": 40.0,
    },
    "B200": {
        # Hardware
        "TDP_W": 1000,
        "THROTTLE_TEMP_C": 85.0,
        "MAX_FREQ_MHZ": 2050,
        # Energy per token
        # Source: ML.Energy v3.0 blog — B200 achieves median 35% energy
        # reduction vs H100 across LLM tasks. 0.012 * 0.65 ≈ 0.008 J/tok
        "ENERGY_PER_TOKEN_DECODE_J": 0.008,
        "ENERGY_PER_TOKEN_PREFILL_J": 0.020,
        # Power model: calibrated to B200 TDP=1000W at full load, idle ~200W
        "POWER_A": 0.020,
        "POWER_B": 6.50,
        "POWER_C": 200.0,
        "EQ1_A": 0.60,
        "EQ1_B": 0.02,
        "EQ1_C": 8.0,
        "EQ2_A": 1.02,
        "EQ2_B": 0.18,
        "EQ2_C": 2.5,
        "THERMAL_TAU_S": 35.0,
    },
}
