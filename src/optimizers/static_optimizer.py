# static_optimizer.py
# Uses the pre-built static profile to recommend a serving configuration
# at each decision interval (e.g. every 30 seconds).

import joblib
import numpy as np
from pathlib import Path
from dataclasses import dataclass

MODEL_DIR = Path("static_profile_results/models")

# SLO targets — these come from your experimental setup
P99_E2EL_SLO_MS   = 500.0
THERMAL_LIMIT_C   = 80.0
ENERGY_BUDGET_J   = 0.005  # per token


@dataclass
class ServingConfig:
    max_num_seqs: int
    gpu_power_limit_w: int
    reason: str


@dataclass
class WorkloadObservation:
    """What you can observe about the current workload without Zeus."""
    request_rate: float          # requests/sec arriving
    avg_input_len: int           # average prompt length in tokens
    avg_output_len: int          # average output length in tokens


class StaticOptimizer:
    """
    TAPAS-inspired: consult the pre-built profile to pick a configuration.
    All decisions are made from offline measurements — no live GPU data.
    """

    # Candidate configurations to evaluate at each decision step.
    # In a real system these would come from your sweep axes.
    CANDIDATE_CONFIGS = [
        {"max_num_seqs": 1,  "gpu_power_limit_w": 200},
        {"max_num_seqs": 4,  "gpu_power_limit_w": 200},
        {"max_num_seqs": 8,  "gpu_power_limit_w": 250},
        {"max_num_seqs": 16, "gpu_power_limit_w": 300},
        {"max_num_seqs": 16, "gpu_power_limit_w": 400},  # max performance
    ]

    def __init__(self):
        self.models = {
            target: joblib.load(MODEL_DIR / f"{target}.joblib")
            for target in [
                "steady_state_energy_per_token_j",
                "mean_ttft_ms",
                "p99_e2el_ms",
                "peak_temp_c",
            ]
        }

    def _predict(self, request_rate, max_num_seqs, input_len, output_len) -> dict:
        x = np.array([[request_rate, max_num_seqs, input_len, output_len]])
        return {t: float(m.predict(x)[0]) for t, m in self.models.items()}

    def recommend(self, obs: WorkloadObservation) -> ServingConfig:
        """
        Find the most energy-efficient configuration that still meets SLOs.
        Tries configs in ascending energy order, returns first SLO-compliant one.
        """
        candidates = []
        for cfg in self.CANDIDATE_CONFIGS:
            pred = self._predict(
                obs.request_rate,
                cfg["max_num_seqs"],
                obs.avg_input_len,
                obs.avg_output_len,
            )
            candidates.append((pred["steady_state_energy_per_token_j"], cfg, pred))

        # Sort by energy efficiency (cheapest first)
        candidates.sort(key=lambda x: x[0])

        for energy_per_tok, cfg, pred in candidates:
            slo_ok   = pred["p99_e2el_ms"]  <= P99_E2EL_SLO_MS
            temp_ok  = pred["peak_temp_c"]  <= THERMAL_LIMIT_C

            if slo_ok and temp_ok:
                return ServingConfig(
                    max_num_seqs=cfg["max_num_seqs"],
                    gpu_power_limit_w=cfg["gpu_power_limit_w"],
                    reason=(
                        f"Static: best energy ({energy_per_tok:.4f} J/tok), "
                        f"predicted p99={pred['p99_e2el_ms']:.0f}ms, "
                        f"temp={pred['peak_temp_c']:.1f}°C"
                    ),
                )

        # No config meets SLOs — return max performance and accept the cost
        _, cfg, pred = candidates[-1]
        return ServingConfig(
            max_num_seqs=cfg["max_num_seqs"],
            gpu_power_limit_w=cfg["gpu_power_limit_w"],
            reason=(
                f"Static: no SLO-safe config found, using max performance. "
                f"Predicted p99={pred['p99_e2el_ms']:.0f}ms"
            ),
        )