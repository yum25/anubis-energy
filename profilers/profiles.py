"""
=================================================================
GPU Profile Calibration
Produces constants for gpu_profiles.py
=================================================================

── Shared thermal constants (all GPUs) ──────────────────────────
Source: TAPAS Stojkovic et al. ASPLOS 2025
        EQ1 from Figure 5, EQ2 from Figure 7 (A100 measurements)
        Applied uniformly — per-GPU differences not captured
  eq1_a = 0.6  # inlet vs outside temp slope
  eq1_b = 0.02  # inlet vs DC load slope
  eq1_c = 8.0  # inlet temp intercept
  eq2_a = 1.05  # GPU temp vs inlet slope
  eq2_b = 0.22  # GPU temp vs load slope (per %)
  eq2_c = 3.5  # GPU temp intercept
  thermal_tau_s = 45.0  # RC time constant (estimated from Fig 6)

=================================================================
  H100
=================================================================
Source: ML.Energy benchmark-v3 (HF Hub: ml-energy/benchmark-v3)
        Llama-3.1-70B-Instruct, lm-arena-chat, Chung et al. 2025
        Full batch range used for polynomial fit (multi-GPU system power).

  Full dataset (batch, J/tok, tok/s, recovered power):
   batch |   J/tok |   tok/s |  power_W
  -------+---------+---------+---------
       8 |  3.7587 |   469.7 |   1765.5
      16 |  2.0910 |   862.6 |   1803.7
      32 |  1.2069 |  1486.7 |   1794.3
      64 |  0.7972 |  2510.5 |   2001.4
      96 |  0.6478 |  3244.3 |   2101.7
     128 |  0.5634 |  3876.3 |   2183.9
     192 |  0.5140 |  4461.2 |   2293.1
     256 |  0.4790 |  4915.1 |   2354.3
     384 |  0.4406 |  5318.8 |   2343.5
     512 |  0.4210 |  5921.3 |   2492.9
     768 |  0.3896 |  6507.5 |   2535.3
    1024 |  0.3731 |  6827.2 |   2547.2

  Power polynomial fit (all 12 batch sizes):
  Fit points (12 — overdetermined least-squares, R²=0.9179):
   batch |  load% |  power_W | fitted_W |  resid_W
  -------+--------+----------+----------+---------
       8 |    0.8 |   1765.5 |   1849.7 |    -84.2
      16 |    1.6 |   1803.7 |   1866.3 |    -62.6
      32 |    3.1 |   1794.3 |   1899.0 |   -104.7
      64 |    6.2 |   2001.4 |   1962.1 |    +39.2
      96 |    9.4 |   2101.7 |   2022.4 |    +79.3
     128 |   12.5 |   2183.9 |   2079.7 |   +104.2
     192 |   18.8 |   2293.1 |   2185.4 |   +107.6
     256 |   25.0 |   2354.3 |   2279.5 |    +74.8
     384 |   37.5 |   2343.5 |   2432.4 |    -88.9
     512 |   50.0 |   2492.9 |   2538.3 |    -45.4
     768 |   75.0 |   2535.3 |   2609.3 |    -74.0
    1024 |  100.0 |   2547.2 |   2492.6 |    +54.7
  Extrapolated P(5%  load) = 1937.2W  (idle estimate)
  Extrapolated P(100% load) = 2492.6W  (TDP = 700W)
  NOTE: P(100%) exceeds TDP — multi-GPU system power reflected in data.
        Consistent throughout sim mode; not representative of single GPU.

  Energy per token:
    Representative decode ept (batch=8): 3.7587 J/tok
    Prefill ept (2.5x decode, modeling assumption): 9.3968 J/tok
    Full range: 0.3731 – 3.7587 J/tok

=================================================================
  B200
=================================================================
Source: ML.Energy benchmark-v3 (HF Hub: ml-energy/benchmark-v3)
        Llama-3.1-70B-Instruct, lm-arena-chat, Chung et al. 2025
        Full batch range used for polynomial fit (multi-GPU system power).

  Full dataset (batch, J/tok, tok/s, recovered power):
   batch |   J/tok |   tok/s |  power_W
  -------+---------+---------+---------
       8 |  3.5693 |   475.7 |   1697.9
      16 |  1.9027 |   910.8 |   1733.0
      32 |  1.0915 |  1589.6 |   1735.0
      64 |  0.6971 |  2597.8 |   1810.9
     128 |  0.4810 |  3977.5 |   1913.2
     256 |  0.3614 |  5372.4 |   1941.6
     512 |  0.2708 |  7134.0 |   1931.9
    1024 |  0.2267 |  8551.4 |   1938.6
    1536 |  0.2210 |  8682.2 |   1918.8
    2048 |  0.1887 | 10107.8 |   1907.3

  Power polynomial fit (all 10 batch sizes):
  Fit points (10 — overdetermined least-squares, R²=0.6156):
   batch |  load% |  power_W | fitted_W |  resid_W
  -------+--------+----------+----------+---------
       8 |    0.4 |   1697.9 |   1775.4 |    -77.5
      16 |    0.8 |   1733.0 |   1778.1 |    -45.2
      32 |    1.6 |   1735.0 |   1783.6 |    -48.6
      64 |    3.1 |   1810.9 |   1794.3 |    +16.6
     128 |    6.2 |   1913.2 |   1814.9 |    +98.3
     256 |   12.5 |   1941.6 |   1852.4 |    +89.2
     512 |   25.0 |   1931.9 |   1913.1 |    +18.8
    1024 |   50.0 |   1938.6 |   1977.0 |    -38.4
    1536 |   75.0 |   1918.8 |   1964.3 |    -45.6
    2048 |  100.0 |   1907.3 |   1875.1 |    +32.3
  Extrapolated P(5%  load) = 1806.8W  (idle estimate)
  Extrapolated P(100% load) = 1875.1W  (TDP = 1000W)
  NOTE: P(100%) exceeds TDP — multi-GPU system power reflected in data.
        Consistent throughout sim mode; not representative of single GPU.

  Energy per token:
    Representative decode ept (batch=8): 3.5693 J/tok
    Prefill ept (2.5x decode, modeling assumption): 8.9232 J/tok
    Full range: 0.1887 – 3.5693 J/tok

=================================================================
  V100
=================================================================
Source: Samsi et al. 2023, arXiv:2310.03003
        LLaMA 7B, single V100 32GB, batch=64, max_gen_len=256
        Power (~220W) read from Figure 3 (log-scale bar, ±~20%)
        Throughput (~400 tok/s) read from Figure 2
        J/tok = 220W / 400 tok/s = 0.55 J/tok (derived, not stated)

  Only one operating point available from this source.
  A degree-2 polynomial cannot be fitted from a single point.
  Power polynomial for V100 is manually calibrated to two anchors:
    Idle (~5% load):  ~80W
    Full load (100%): ~250W (TDP)
  Coefficients chosen so P(5)≈80W and P(100)≈250W.
  This is an estimate with no empirical validation.

  Manual polynomial verification:
    P(5%  load) = 80.0W
    P(100% load) = 250.0W  (TDP = 250W)

  Energy per token:
    Decode ept (220W / 400 tok/s): 0.5500 J/tok
    Prefill ept (2.5x): 1.3750 J/tok

=================================================================
  FINAL GPU_PROFILES DICTIONARY
  Copy this into gpu_profiles.py
=================================================================
"""

GPU_PROFILES = {
    "V100": {
        # Hardware — NVIDIA spec sheet
        "TDP_W": 250,
        "THROTTLE_TEMP_C": 83.0,
        "MAX_FREQ_MHZ": 1530,
        # Energy per token — # Samsi et al. 2023 arXiv:2310.03003, LLaMA 7B single V100
        # Approximated from Figures 2 & 3 (log-scale); J/tok = 220W/400toks = 0.5500
        "ENERGY_PER_TOKEN_DECODE_J": 0.5500,
        "ENERGY_PER_TOKEN_PREFILL_J": 1.3750,  # 2.5x decode (TokenPowerBench arXiv:2512.03024, modeling assumption)
        # Power polynomial P = a*load² + b*load + c  — # Manually calibrated to TDP anchors (single data point from Samsi et al.)
        "POW_A": 0.005000,
        "POW_B": 1.264000,
        "POW_C": 73.555000,
        # TAPAS thermal model — Stojkovic et al. ASPLOS 2025
        # EQ1 (inlet temp): Figure 5.  EQ2 (GPU temp): Figure 7.
        # Measured on A100; applied uniformly as approximation.
        "EQ1_A": 0.6,
        "EQ1_B": 0.02,
        "EQ1_C": 8.0,
        "EQ2_A": 1.05,
        "EQ2_B": 0.22,
        "EQ2_C": 3.5,
        # Thermal time constant — estimated from TAPAS Figure 6.
        # Not stated in paper; treat as modeling assumption.
        "THERMAL_TAU_S": 45.0,
    },
    "H100": {
        # Hardware — NVIDIA spec sheet
        "TDP_W": 700,
        "THROTTLE_TEMP_C": 83.0,
        "MAX_FREQ_MHZ": 1980,
        # Energy per token — # ML.Energy benchmark-v3 (Chung et al. NeurIPS 2025)
        # Llama-3.1-70B-Instruct, lm-arena-chat, batch=8
        "ENERGY_PER_TOKEN_DECODE_J": 3.7587,
        "ENERGY_PER_TOKEN_PREFILL_J": 9.3968,  # 2.5x decode (TokenPowerBench arXiv:2512.03024, modeling assumption)
        # Power polynomial P = a*load² + b*load + c  — # Fitted from ML.Energy data, all 12 batch sizes (full range)
        # Overdetermined least-squares (degree-2); multi-GPU system power
        "POW_A": -0.150237,
        "POW_B": 21.620341,
        "POW_C": 1832.885504,
        # TAPAS thermal model — Stojkovic et al. ASPLOS 2025
        # EQ1 (inlet temp): Figure 5.  EQ2 (GPU temp): Figure 7.
        # Measured on A100; applied uniformly as approximation.
        "EQ1_A": 0.6,
        "EQ1_B": 0.02,
        "EQ1_C": 8.0,
        "EQ2_A": 1.05,
        "EQ2_B": 0.22,
        "EQ2_C": 3.5,
        # Thermal time constant — estimated from TAPAS Figure 6.
        # Not stated in paper; treat as modeling assumption.
        "THERMAL_TAU_S": 45.0,
    },
    "B200": {
        # Hardware — NVIDIA spec sheet
        "TDP_W": 1000,
        "THROTTLE_TEMP_C": 85.0,
        "MAX_FREQ_MHZ": 2050,
        # Energy per token — # ML.Energy benchmark-v3 (Chung et al. NeurIPS 2025)
        # Llama-3.1-70B-Instruct, lm-arena-chat, batch=8
        "ENERGY_PER_TOKEN_DECODE_J": 3.5693,
        "ENERGY_PER_TOKEN_PREFILL_J": 8.9232,  # 2.5x decode (TokenPowerBench arXiv:2512.03024, modeling assumption)
        # Power polynomial P = a*load² + b*load + c  — # Fitted from ML.Energy data, all 10 batch sizes (full range)
        # Overdetermined least-squares (degree-2); multi-GPU system power
        "POW_A": -0.061269,
        "POW_B": 7.151743,
        "POW_C": 1772.585383,
        # TAPAS thermal model — Stojkovic et al. ASPLOS 2025
        # EQ1 (inlet temp): Figure 5.  EQ2 (GPU temp): Figure 7.
        # Measured on A100; applied uniformly as approximation.
        "EQ1_A": 0.6,
        "EQ1_B": 0.02,
        "EQ1_C": 8.0,
        "EQ2_A": 1.05,
        "EQ2_B": 0.22,
        "EQ2_C": 3.5,
        # Thermal time constant — estimated from TAPAS Figure 6.
        # Not stated in paper; treat as modeling assumption.
        "THERMAL_TAU_S": 45.0,
    },
}
