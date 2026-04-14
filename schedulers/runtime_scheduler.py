"""
Runtime scheduler: adapts GPU serving config based on live GpuObservation data.

Unlike the static scheduler, this scheduler never consults precomputed tables.
It reacts to measured temperature, energy-per-token (from Zeus windows), and
throttle events using a three-layer feedback controller:

    Layer 1 — Thermal guard     (highest priority, overrides cooldown)
    Layer 2 — Efficiency decay  (step down when energy/token degrades)
    Layer 3 — Opportunistic     (recover frequency when thermal headroom exists)

Zeus integration
----------------
Energy measurements flow in through GpuObservation.gpu_energy_j and
GpuObservation.energy_per_token_j, which RealGpuDataSource populates from
Zeus energy windows (ZeusMonitor.end_window).  The scheduler itself never
imports Zeus — the DataSource abstraction handles hardware access.

Config space (must match run_static_profiling.py sweep)
--------------------------------------------------------
    GPU freqs  : [780, 1000, 1200, 1377] MHz   (ascending)
    max_num_seqs: driven by phase — prefill=8, decode=32
    (matching the most common within-SLO configs in the static Pareto table)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from profilers.interface import DataSource, GpuObservation

# ── Config space — must match run_static_profiling.py ────────────────────────
FREQS_MHZ: list[int] = [780, 1000, 1200, 1377]   # index 0 = most conservative

# Default max_seqs per phase, chosen to maximise goodput while staying within
# the SLO observed in static profiling for the mid-frequency range.
_SEQS_BY_PHASE: dict[str, int] = {"prefill": 8, "decode": 32}

# ── V100S thermal thresholds ──────────────────────────────────────────────────
# Throttle at 83 °C (TAPAS / profiles.py).  Guard margins chosen so the
# scheduler reacts before the hardware throttles, not after.
_THROTTLE_C = 83.0
_URGENT_C   = 81.0   # drop 2 freq steps immediately
_WARN_C     = 78.0   # drop 1 freq step
_COOL_C     = 70.0   # safe to attempt a step-up

# ── EMA smoothing ─────────────────────────────────────────────────────────────
_TEMP_ALPHA = 0.35   # faster — thermal changes matter quickly
_EPT_ALPHA  = 0.20   # slower — energy/token is noisy; filter it

# Efficiency degradation: if ept_ema exceeds baseline by this factor, step down.
# 15% chosen to exceed token-count noise (simulator ±5%, real hardware ±10%).
_EPT_DEGRADE = 1.15

# Number of observations with valid ept before locking in a baseline
_EPT_WARMUP = 3

# ── Cooldown intervals ────────────────────────────────────────────────────────
# Asymmetric: be quick to step down (thermal safety), slow to step up
# (avoid oscillation when GPU is near the warm/cool boundary).
_COOLDOWN_DOWN = 2
_COOLDOWN_UP   = 5


@dataclass
class RuntimeDecision:
    """Record of one scheduling decision, kept in RuntimeScheduler.history."""
    sim_time: float           # simulated seconds elapsed
    timestamp: float          # wall-clock time.time()
    phase: str
    freq_mhz: int
    max_seqs: int
    avg_temp_c: float         # mean across GPUs this interval
    temp_ema: float           # smoothed temperature signal
    ept_ema: Optional[float]  # smoothed energy-per-token (None until warmup)
    ept_baseline: Optional[float]
    throttled: bool
    reason: str               # what drove the decision
    config_changed: bool


class RuntimeScheduler:
    """
    Feedback-control scheduler driven by live GpuObservation data from Zeus.

    Compatible with both SimulatedGpuDataSource (for simulation experiments)
    and RealGpuDataSource (for CloudLab V100S hardware).

    Control layers
    --------------
    1. Thermal guard (overrides cooldown):
       - throttled or avg_temp_ema >= URGENT_C  →  drop freq 2 steps
       - avg_temp_ema >= WARN_C                  →  drop freq 1 step

    2. Efficiency decay (respects cooldown):
       - ept_ema > ept_baseline * EPT_DEGRADE    →  drop freq 1 step
       ept_baseline is the EMA value at warmup; ratchets down if ept improves.

    3. Opportunistic scale-up (respects longer cooldown):
       - avg_temp_ema <= COOL_C                  →  step freq up 1 level

    max_num_seqs is phase-driven (prefill=8, decode=32) rather than
    dynamically tuned, keeping the comparison with the static scheduler clean.
    """

    def __init__(
        self,
        source: DataSource,
        initial_freq_idx: int = 2,        # 1200 MHz — middle of the sweep
        cooldown_down: int = _COOLDOWN_DOWN,
        cooldown_up: int = _COOLDOWN_UP,
    ) -> None:
        self.source = source

        # Frequency state — index into FREQS_MHZ
        self._freq_idx: int = initial_freq_idx
        self._max_seqs: int = 8            # updated on first observation

        # EMA accumulators
        self._temp_ema: Optional[float] = None
        self._ept_ema:  Optional[float] = None
        self._ept_baseline: Optional[float] = None
        self._ept_obs: int = 0

        # Cooldown: positive → skip soft decisions for that many steps
        self._cooldown: int = 0
        self._cooldown_down = cooldown_down
        self._cooldown_up   = cooldown_up

        # Full decision log
        self.history: list[RuntimeDecision] = []

        # Apply initial config so the first observation reflects it
        self.source.set_config(
            max_num_seqs=self._max_seqs,
            gpu_freq_mhz=FREQS_MHZ[self._freq_idx],
        )

    # ── Public API ──────────────────────────────────────────────────────────────

    def step(self, interval_s: float) -> GpuObservation:
        """
        Advance the data source by interval_s, update EMA signals, run
        the feedback controller, and return the observation from this interval.

        set_config() is called only when the chosen config differs from the
        currently active one, minimising unnecessary nvidia-smi invocations.
        """
        obs = self.source.next_observation(interval_s)

        # Phase-driven max_seqs update
        desired_seqs = _SEQS_BY_PHASE.get(obs.phase, 8)
        seqs_changed = desired_seqs != self._max_seqs
        self._max_seqs = desired_seqs

        # Update EMA signals from this observation
        avg_temp = sum(obs.gpu_temp_c.values()) / max(len(obs.gpu_temp_c), 1)
        self._temp_ema = _ema(self._temp_ema, avg_temp, _TEMP_ALPHA)

        if obs.energy_per_token_j is not None:
            self._ept_ema = _ema(self._ept_ema, obs.energy_per_token_j, _EPT_ALPHA)
            self._ept_obs += 1
            # Lock in baseline after warmup; ratchet down if ept improves
            if self._ept_obs == _EPT_WARMUP:
                self._ept_baseline = self._ept_ema
            elif (
                self._ept_baseline is not None
                and self._ept_ema < self._ept_baseline
            ):
                self._ept_baseline = self._ept_ema

        # Run three-layer controller
        freq_changed, reason = self._control(obs.throttled)

        config_changed = freq_changed or seqs_changed
        if config_changed:
            self.source.set_config(
                max_num_seqs=self._max_seqs,
                gpu_freq_mhz=FREQS_MHZ[self._freq_idx],
            )

        self.history.append(RuntimeDecision(
            sim_time=obs.sim_time,
            timestamp=obs.timestamp,
            phase=obs.phase,
            freq_mhz=FREQS_MHZ[self._freq_idx],
            max_seqs=self._max_seqs,
            avg_temp_c=round(avg_temp, 2),
            temp_ema=round(self._temp_ema, 2) if self._temp_ema is not None else avg_temp,
            ept_ema=round(self._ept_ema, 6) if self._ept_ema is not None else None,
            ept_baseline=round(self._ept_baseline, 6) if self._ept_baseline is not None else None,
            throttled=obs.throttled,
            reason=reason,
            config_changed=config_changed,
        ))

        return obs

    def run(self, total_s: float, interval_s: float = 10.0) -> list[GpuObservation]:
        """
        Drive the scheduler for total_s seconds, stepping every interval_s.
        Returns all GpuObservations collected.
        """
        observations: list[GpuObservation] = []
        elapsed = 0.0
        while elapsed < total_s:
            obs = self.step(interval_s)
            observations.append(obs)
            elapsed += interval_s
        return observations

    # ── Internals ───────────────────────────────────────────────────────────────

    def _control(self, throttled: bool) -> tuple[bool, str]:
        """
        Three-layer feedback controller.

        Returns (freq_changed, reason_string).
        Thermal layers override the cooldown; efficiency and scale-up respect it.
        """
        temp = self._temp_ema  # None only before first observation

        # ── Layer 1: Thermal guard — bypasses cooldown ─────────────────────────
        if throttled or (temp is not None and temp >= _URGENT_C):
            changed = self._step_freq_down(steps=2)
            if changed:
                self._cooldown = self._cooldown_down
            return changed, "thermal_urgent"

        if temp is not None and temp >= _WARN_C:
            changed = self._step_freq_down(steps=1)
            if changed:
                self._cooldown = self._cooldown_down
            return changed, "thermal_warn"

        # Soft decisions respect cooldown
        if self._cooldown > 0:
            self._cooldown -= 1
            return False, "hold_cooldown"

        # ── Layer 2: Efficiency decay ──────────────────────────────────────────
        if (
            self._ept_ema is not None
            and self._ept_baseline is not None
            and self._ept_obs >= _EPT_WARMUP
            and self._ept_ema > self._ept_baseline * _EPT_DEGRADE
        ):
            changed = self._step_freq_down(steps=1)
            if changed:
                self._cooldown = self._cooldown_down
            return changed, "efficiency_decay"

        # ── Layer 3: Opportunistic scale-up ────────────────────────────────────
        if temp is not None and temp <= _COOL_C:
            changed = self._step_freq_up()
            if changed:
                self._cooldown = self._cooldown_up
            return changed, "scale_up"

        return False, "hold_steady"

    def _step_freq_down(self, steps: int = 1) -> bool:
        new_idx = max(0, self._freq_idx - steps)
        if new_idx != self._freq_idx:
            self._freq_idx = new_idx
            return True
        return False

    def _step_freq_up(self) -> bool:
        new_idx = min(len(FREQS_MHZ) - 1, self._freq_idx + 1)
        if new_idx != self._freq_idx:
            self._freq_idx = new_idx
            return True
        return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ema(current: Optional[float], new_val: float, alpha: float) -> float:
    """Exponential moving average.  Returns new_val on first call (no prior state)."""
    return new_val if current is None else alpha * new_val + (1.0 - alpha) * current
