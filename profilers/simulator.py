# simulator.py
#
# A self-contained GPU state simulator grounded in the TAPAS thermal/power
# models and ML.Energy benchmark energy-per-token figures.
#
# DESIGN GOAL: schedulers (static or runtime) never import this module.
# Instead, both schedulers depend only on the GpuObservation dataclass and
# the DataSource protocol.  In production you swap the simulator for a
# RealGpuDataSource that reads from Zeus.  In simulation you use
# SimulatedGpuDataSource.  The schedulers are never touched.
#
# TAPAS equations implemented here:
#   Eq 1  T_inlet,s   = f(T_outside, Load_DC)          (linear regression)
#   Eq 2  T_GPU,s,g   = f(T_inlet,s, Load_GPU,g)       (linear regression, MAE < 1°C)
#   Power model       = polynomial f(Load_GPU)          (per TAPAS §2.2)
#
# Energy-per-token baseline: ML.Energy benchmark paper, Llama-2-7B on A100
#   ~0.003 J/token at steady state, prefill ~2x decode per token.
#
# References:
#   [TAPAS]  Stojkovic et al., ASPLOS 2025  (equations 1-4, figures 6-7)
#   [MLENERGY] Chung et al., arXiv 2505.06371 (Table 2, Llama-2-7B A100)

from __future__ import annotations

import time
import math
import threading
from dataclasses import dataclass

from interface import GpuObservation, DataSource
from profiles import GPU_PROFILES

# ── TAPAS-grounded physics constants ──────────────────────────────────────────
# Calibrated for V100 (our CloudLab hardware) rather than A100, since that is
# what we will actually run on.  Coefficients are derived from:
#   - TAPAS Fig 7 linear regression slopes (T_GPU vs T_inlet and Load_GPU)
#   - V100 TDP: 300W; A100 TDP: 400W → scale energy figures by 300/400 = 0.75
#   - ML.Energy Llama-2-7B A100 baseline ~0.003 J/tok → V100 ~0.004 J/tok
#     (V100 is ~25-30% slower/less efficient than A100 for transformer workloads)


class _TapasPhysics:
    """
    Implements the TAPAS thermal and power regression models.
    All equations are named to match the paper.
    """

    # ── Eq 1 coefficients: T_inlet = a*T_outside + b*Load_DC + c ──────────────
    # From TAPAS Fig 5: ~linear relationship, slope ~0.6 vs outside temp
    # Load_DC contribution is smaller (~0.02 per percentage point of DC load)
    EQ1_A = 0.60  # T_outside coefficient
    EQ1_B = 0.02  # datacenter load coefficient (load as % 0-100)
    EQ1_C = 8.0  # intercept (cooling system base offset)

    # ── Eq 2 coefficients: T_GPU = a*T_inlet + b*Load_GPU + c ─────────────────
    # Measured on A100 servers, MAE < 1C.
    # Applied to V100/H100/B200 as an approximation —
    # per-GPU thermal differences are not captured.
    EQ2_A = 1.05  # T_inlet coefficient (slightly above 1 due to heat recirculation)
    EQ2_B = 0.22  # GPU load coefficient (°C per % load point)
    EQ2_C = 3.5  # intercept (GPU self-heating above inlet)

    # From TAPAS Fig 6: GPU temperature responds to load changes over ~45s
    # (RC circuit analogy: tau = thermal_mass / thermal_resistance)
    THERMAL_TAU_S = 45.0

    def __init__(self, gpu_model):
        self.POW_A = GPU_PROFILES[gpu_model]["POW_A"]
        self.POW_B = GPU_PROFILES[gpu_model]["POW_B"]
        self.POW_C = GPU_PROFILES[gpu_model]["POW_C"]

        self.THROTTLE_TEMP_C = GPU_PROFILES[gpu_model]["THROTTLE_TEMP_C"]

    def inlet_temp(self, t_outside: float, dc_load_pct: float) -> float:
        """TAPAS Equation 1."""
        return self.EQ1_A * t_outside + self.EQ1_B * dc_load_pct + self.EQ1_C

    def gpu_temp_steady(self, t_inlet: float, gpu_load_pct: float) -> float:
        """TAPAS Equation 2 — steady-state GPU temperature."""
        return self.EQ2_A * t_inlet + self.EQ2_B * gpu_load_pct + self.EQ2_C

    def gpu_power(self, gpu_load_pct: float) -> float:
        """TAPAS power polynomial model (Watts)."""
        return self.POW_A * gpu_load_pct**2 + self.POW_B * gpu_load_pct + self.POW_C

    @classmethod
    def transient_temp(
        cls,
        current_temp: float,
        target_temp: float,
        elapsed_s: float,
    ) -> float:
        """
        RC-circuit thermal model for temperature transient.
        T(t) = T_target + (T_current - T_target) * exp(-t / tau)
        This is what makes the simulation non-circular:
        the static profile only knows steady-state values;
        the runtime profiler sees the transient path.
        """
        alpha = math.exp(-elapsed_s / cls.THERMAL_TAU_S)
        return target_temp + (current_temp - target_temp) * alpha


# ── Workload model ─────────────────────────────────────────────────────────────


@dataclass
class WorkloadStep:
    """One step in a simulation scenario."""

    duration_s: float
    request_rate: float  # req/s
    phase: str  # "prefill" or "decode"
    max_num_seqs: int
    gpu_freq_mhz: int
    label: str = ""  # human-readable label for logging


# ── Simulated data source ──────────────────────────────────────────────────────


class SimulatedGpuDataSource(DataSource):
    """
    Physics-grounded GPU simulator.
    Implements the DataSource protocol — drop-in replaceable with
    RealGpuDataSource when running on actual CloudLab hardware.

    State that carries between calls (making it non-trivial for static profiler):
      - _gpu_temp_c: current GPU temperature (follows RC transient model)
      - _t_outside: slowly drifting ambient temperature
      - _dc_load_pct: datacenter background load
      - _active_events: injected disturbances and their remaining duration
    """

    BASE_LOAD_PER_SEQ = 2.5  # % GPU utilisation per concurrent sequence

    def __init__(
        self,
        gpu_indices: list[int] = [0, 1],
        t_outside_c: float = 22.0,
        dc_load_pct: float = 40.0,
        gpu_model: str = "V100",
        seed: int = 42,
    ):
        import random

        # Set baseline energy figures based on gpu models. It's important to note that
        # these are not exact figures, but adapted/approximate figures based on incomplete
        # data.

        self.ENERGY_PER_TOKEN_DECODE_J = GPU_PROFILES[gpu_model][
            "ENERGY_PER_TOKEN_DECODE_J"
        ]
        self.ENERGY_PER_TOKEN_PREFILL_J = GPU_PROFILES[gpu_model][
            "ENERGY_PER_TOKEN_PREFILL_J"
        ]
        self.MAX_FREQ_MHZ = GPU_PROFILES[gpu_model]["MAX_FREQ_MHZ"]

        self._TapasPhysics = _TapasPhysics(gpu_model)

        self._rng = random.Random(seed)

        self.gpu_indices = gpu_indices
        self._t_outside = t_outside_c
        self._dc_load = dc_load_pct

        # Thermal state per GPU — starts at idle temperature
        idle_inlet = self._TapasPhysics.inlet_temp(t_outside_c, dc_load_pct)
        idle_temp = self._TapasPhysics.gpu_temp_steady(idle_inlet, 5.0)
        self._gpu_temp = {str(i): idle_temp for i in gpu_indices}

        # Current serving config (scheduler calls set_config to change these)
        self._max_num_seqs = 8
        self._gpu_freq_mhz = 1410
        self._phase = "decode"
        self._request_rate = 2.0

        # Accumulated energy per GPU across the whole run
        self._total_energy_j = {str(i): 0.0 for i in gpu_indices}

        # Active injected events: list of {"type": str, "remaining_s": float, ...}
        self._events: list[dict] = []
        self._lock = threading.Lock()

        # Simulation clock
        self._sim_time = 0.0

    # ── DataSource protocol ────────────────────────────────────────────────────

    def set_config(self, max_num_seqs: int, gpu_freq_mhz: int) -> None:
        """Scheduler calls this to apply a new configuration."""
        with self._lock:
            self._max_num_seqs = max_num_seqs
            self._gpu_freq_mhz = gpu_freq_mhz

    def inject_event(self, event: str, **kwargs) -> None:
        """
        Inject a disturbance. No-op when swapped for real hardware.
        Supported:
            thermal_spike    delta_c=15.0
            load_spike       duration_s=60.0, multiplier=2.0
            cooling_failure  duration_s=120.0
        """
        with self._lock:
            if event == "thermal_spike":
                delta = kwargs.get("delta_c", 15.0)
                for key in self._gpu_temp:
                    self._gpu_temp[key] += delta
            elif event in ("load_spike", "cooling_failure"):
                self._events.append(
                    {
                        "type": event,
                        "remaining_s": kwargs.get("duration_s", 60.0),
                        **kwargs,
                    }
                )

    def next_observation(self, interval_s: float) -> GpuObservation:
        """
        Advance the simulation by interval_s and return the resulting state.
        All physics updates happen here.
        """
        with self._lock:
            self._sim_time += interval_s
            wall_time = time.time()

            # Step 1: advance active events
            load_multiplier = 1.0
            cooling_degraded = False
            next_events = []
            for ev in self._events:
                ev["remaining_s"] -= interval_s
                if ev["remaining_s"] > 0:
                    next_events.append(ev)
                    if ev["type"] == "load_spike":
                        load_multiplier = ev.get("multiplier", 2.0)
                    elif ev["type"] == "cooling_failure":
                        cooling_degraded = True
            self._events = next_events

            # Step 2: compute GPU utilisation from serving config
            freq_factor = (self._gpu_freq_mhz / self.MAX_FREQ_MHZ) ** 0.7
            raw_load_pct = (
                min(
                    self._max_num_seqs * self.BASE_LOAD_PER_SEQ * freq_factor,
                    100.0,
                )
                * load_multiplier
            )

            # Step 3: TAPAS Eq 1 — inlet temperature
            # Cooling failure raises effective outside temperature
            effective_outside = self._t_outside + (10.0 if cooling_degraded else 0.0)
            t_inlet = self._TapasPhysics.inlet_temp(effective_outside, self._dc_load)

            # Step 4: TAPAS Eq 2 — steady-state GPU temperature target
            t_gpu_steady = self._TapasPhysics.gpu_temp_steady(t_inlet, raw_load_pct)

            # Step 5: RC transient model — actual temperature moves toward target
            # Each GPU gets small per-unit noise (TAPAS Fig 8: up to 10°C spread)
            gpu_temps = {}
            gpu_powers = {}
            gpu_energies = {}
            throttled = False

            for key in [str(i) for i in self.gpu_indices]:
                unit_noise_target = self._rng.gauss(0, 1.5)
                new_temp = _TapasPhysics.transient_temp(
                    current_temp=self._gpu_temp[key],
                    target_temp=t_gpu_steady + unit_noise_target,
                    elapsed_s=interval_s,
                )
                # Hard cap at throttle threshold
                if new_temp >= self._TapasPhysics.THROTTLE_TEMP_C:
                    new_temp = self._TapasPhysics.THROTTLE_TEMP_C
                    throttled = True

                self._gpu_temp[key] = new_temp
                gpu_temps[key] = round(new_temp, 2)

                # Step 6: TAPAS power model
                power_w = self._TapasPhysics.gpu_power(raw_load_pct)
                power_w += self._rng.gauss(0, 3.0)  # measurement noise
                power_w = max(power_w, 0.0)
                gpu_powers[key] = round(power_w, 1)

                # Step 7: energy = power × time
                energy_j = power_w * interval_s
                self._total_energy_j[key] += energy_j
                gpu_energies[key] = round(energy_j, 3)

            # Step 8: tokens generated and energy-per-token
            # Phase affects both throughput and energy cost per token
            if self._phase == "prefill":
                base_ept = self.ENERGY_PER_TOKEN_PREFILL_J
                # Prefill throughput: lower token rate (compute-bound)
                toks_per_s = self._max_num_seqs * 15 * freq_factor * load_multiplier
            else:
                base_ept = self.ENERGY_PER_TOKEN_DECODE_J
                # Decode throughput: higher token rate (memory-bound, batching helps)
                toks_per_s = self._max_num_seqs * 40 * freq_factor

            tokens = max(
                0,
                int(
                    toks_per_s * interval_s
                    + self._rng.gauss(0, toks_per_s * 0.05 * interval_s)
                ),
            )

            # Energy-per-token scales with thermal state — hotter = less efficient
            # This is the key non-circular signal: static profile doesn't know
            # current temp; runtime profiler does
            avg_temp = sum(gpu_temps.values()) / len(gpu_temps)
            thermal_penalty = 1.0 + max(0.0, (avg_temp - 65.0) * 0.003)
            ept = base_ept * thermal_penalty if tokens > 0 else None

            return GpuObservation(
                timestamp=wall_time,
                sim_time=self._sim_time,
                gpu_temp_c=gpu_temps,
                gpu_power_w=gpu_powers,
                gpu_energy_j=gpu_energies,
                tokens_generated=tokens,
                energy_per_token_j=ept,
                request_rate=self._request_rate,
                phase=self._phase,
                max_num_seqs=self._max_num_seqs,
                gpu_freq_mhz=self._gpu_freq_mhz,
                throttled=throttled,
            )

    def set_workload(self, request_rate: float, phase: str) -> None:
        """Update workload parameters (separate from serving config)."""
        with self._lock:
            self._request_rate = request_rate
            self._phase = phase
