# runtime_optimizer.py
# Perseus-inspired: use live Zeus measurements to adapt configuration.
# Queries RuntimeProfiler for current state and adjusts accordingly.

import time
import logging
from dataclasses import dataclass
from runtime_profiler import RuntimeProfiler

logger = logging.getLogger(__name__)

P99_E2EL_SLO_MS  = 500.0
THERMAL_LIMIT_C  = 80.0
THERMAL_WARN_C   = 72.0   # start backing off before hitting the hard limit
ENERGY_BUDGET_J  = 0.005  # per token


@dataclass
class ServingConfig:
    max_num_seqs: int
    gpu_power_limit_w: int
    reason: str


class RuntimeOptimizer:
    """
    Perseus-inspired: adapt serving configuration based on live GPU readings.

    Decision logic has three modes:
      NORMAL   — serve efficiently based on live energy-per-token
      WARN     — thermal pressure building, reduce batch size proactively
      CRITICAL — hard thermal/energy limit hit, shed load aggressively
    """

    def __init__(self, profiler: RuntimeProfiler):
        self.profiler = profiler
        self._mode = "NORMAL"
        self._last_decision_time = 0.0

    def recommend(self, current_p99_e2el_ms: float) -> ServingConfig:
        """
        Called periodically (e.g. every 30s) by the serving loop.
        current_p99_e2el_ms: the actual observed p99 latency from recent requests.
        """
        est = self.profiler.current_estimate()

        if est["status"] == "warming_up":
            # Not enough data yet — use a safe conservative default
            return ServingConfig(
                max_num_seqs=4,
                gpu_power_limit_w=200,
                reason="Runtime: profiler warming up, using conservative default",
            )

        temp      = est.get("peak_temp_c", 0.0)
        power_w   = est.get("recent_avg_power_w", 0.0)
        ept       = est.get("mean_energy_per_token_j")
        frontier  = est.get("frontier_points", 0)

        # ── Determine mode based on live readings ─────────────────────────────

        if temp >= THERMAL_LIMIT_C:
            self._mode = "CRITICAL"
        elif temp >= THERMAL_WARN_C:
            self._mode = "WARN"
        else:
            self._mode = "NORMAL"

        logger.info(
            "Runtime optimizer: mode=%s temp=%.1f°C power=%.1fW "
            "ept=%.4f J/tok frontier_pts=%d",
            self._mode, temp, power_w, ept or 0, frontier,
        )

        # ── Translate mode into a concrete configuration ──────────────────────

        if self._mode == "CRITICAL":
            # Hard thermal limit — minimize power draw immediately.
            # Accept latency degradation to protect hardware.
            return ServingConfig(
                max_num_seqs=1,
                gpu_power_limit_w=150,
                reason=(
                    f"Runtime CRITICAL: temp={temp:.1f}°C >= limit {THERMAL_LIMIT_C}°C. "
                    f"Dropping to single-request mode."
                ),
            )

        if self._mode == "WARN":
            # Thermal pressure building — reduce batch size proactively.
            # This is the key advantage over static: we react before hitting the limit.
            return ServingConfig(
                max_num_seqs=4,
                gpu_power_limit_w=200,
                reason=(
                    f"Runtime WARN: temp={temp:.1f}°C approaching limit. "
                    f"Reducing batch size preemptively."
                ),
            )

        # NORMAL mode — use the Pareto frontier to find the efficient point.
        # If we're meeting our latency SLO with headroom, we can afford to
        # reduce batch size (and thus power). If we're near the SLO, increase it.
        if ept and ept > ENERGY_BUDGET_J:
            # Over energy budget but SLOs are fine — reduce batch size
            return ServingConfig(
                max_num_seqs=4,
                gpu_power_limit_w=200,
                reason=(
                    f"Runtime NORMAL: ept={ept:.4f} J/tok over budget {ENERGY_BUDGET_J}. "
                    f"Reducing batch size to save energy."
                ),
            )

        latency_headroom = P99_E2EL_SLO_MS - current_p99_e2el_ms
        if latency_headroom > 150:
            # Plenty of latency headroom — serve with smaller batch to save energy
            return ServingConfig(
                max_num_seqs=4,
                gpu_power_limit_w=200,
                reason=(
                    f"Runtime NORMAL: {latency_headroom:.0f}ms latency headroom. "
                    f"Using smaller batch for energy efficiency."
                ),
            )
        elif latency_headroom < 50:
            # Close to SLO — scale up to protect latency
            return ServingConfig(
                max_num_seqs=16,
                gpu_power_limit_w=350,
                reason=(
                    f"Runtime NORMAL: only {latency_headroom:.0f}ms SLO headroom. "
                    f"Scaling up to protect latency."
                ),
            )
        else:
            return ServingConfig(
                max_num_seqs=8,
                gpu_power_limit_w=250,
                reason=f"Runtime NORMAL: balanced config. ept={ept:.4f} J/tok",
            )