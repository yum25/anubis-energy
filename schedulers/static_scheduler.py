"""
Static scheduler: uses the Pareto lookup table produced by
static_profiling/build_static_models.py to pick GPU serving configs
entirely offline — it never re-measures at runtime.

Artifacts consumed
------------------
  static_pareto.json       : goodput + energy for every (freq, max_seqs, phase) triple
  static_temp_model.pkl    : degree-2 poly regression → predicted steady-state temp (°C)
  static_energy_model.pkl  : degree-2 poly regression → predicted steady-state energy (J)

Interface contract
------------------
Schedulers depend only on GpuObservation and DataSource from profilers/interface.py.
The concrete DataSource (simulator or real hardware) is injected at construction time.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from profilers.interface import DataSource, GpuObservation

# ── Default artifact locations (relative to this file) ────────────────────────
_STATIC_DIR = Path(__file__).parent.parent / "static_profiling"
DEFAULT_PARETO_PATH  = _STATIC_DIR / "static_pareto.json"
DEFAULT_TEMP_MODEL   = _STATIC_DIR / "static_temp_model.pkl"
DEFAULT_ENERGY_MODEL = _STATIC_DIR / "static_energy_model.pkl"

# Normalization constants — must match build_static_models.py exactly
_MAX_FREQ_MHZ = 1377
_MAX_NUM_SEQS = 32


@dataclass
class SchedulerDecision:
    """Record of one scheduling decision, kept in StaticScheduler.history."""
    timestamp: float          # wall-clock time of the observation that triggered this
    phase: str                # "prefill" or "decode"
    chosen_freq_mhz: int
    chosen_max_seqs: int
    predicted_temp_c: float   # from static_temp_model
    predicted_energy_j: float # from static_energy_model
    predicted_goodput: float  # tokens/sec (0 if outside SLO)
    policy: str
    config_changed: bool      # True if set_config() was actually called


class StaticScheduler:
    """
    Offline-optimal scheduler backed by a precomputed Pareto table.

    At each step the scheduler:
      1. Collects a GpuObservation from the data source.
      2. Reads obs.phase to determine the current workload type.
      3. Selects the best within-SLO config from the static Pareto table
         after filtering out thermally unsafe configs.
      4. Calls source.set_config() only when the chosen config differs from
         the currently active one (avoids redundant reconfiguration).

    Policies
    --------
    "min_energy"      — lowest steady-state energy (J) among valid configs.
    "max_goodput"     — highest output throughput (tokens/sec) among valid configs.
    "best_efficiency" — highest goodput-per-joule among valid configs.

    Thermal guard
    -------------
    Any config whose predicted steady-state temperature exceeds `thermal_limit_c`
    is excluded before the policy ranking.  If every within-SLO config exceeds
    the limit the guard is relaxed so the scheduler always returns something.
    """

    def __init__(
        self,
        source: DataSource,
        pareto_path: Path = DEFAULT_PARETO_PATH,
        temp_model_path: Path = DEFAULT_TEMP_MODEL,
        energy_model_path: Path = DEFAULT_ENERGY_MODEL,
        policy: str = "min_energy",
        thermal_limit_c: float = 80.0,
    ) -> None:
        if policy not in ("min_energy", "max_goodput", "best_efficiency"):
            raise ValueError(
                f"Unknown policy {policy!r}. "
                "Choose from: 'min_energy', 'max_goodput', 'best_efficiency'"
            )

        self.source = source
        self.policy = policy
        self.thermal_limit_c = thermal_limit_c

        # Load artifacts
        with open(pareto_path) as f:
            self._pareto: list[dict] = json.load(f)

        with open(temp_model_path, "rb") as f:
            self._temp_model = pickle.load(f)

        with open(energy_model_path, "rb") as f:
            self._energy_model = pickle.load(f)

        # Pre-compute model predictions for every Pareto entry so _select_config
        # does no sklearn inference at scheduling time.
        for r in self._pareto:
            feat = _features(r["gpu_freq_mhz"], r["max_num_seqs"], r["phase"])
            r["_pred_temp_c"]   = float(self._temp_model.predict([feat])[0])
            r["_pred_energy_j"] = float(self._energy_model.predict([feat])[0])

        # Currently active config (None = not yet set)
        self._current_freq: Optional[int] = None
        self._current_seqs: Optional[int] = None

        # Full decision log for post-hoc analysis
        self.history: list[SchedulerDecision] = []

    # ── Public API ──────────────────────────────────────────────────────────────

    def step(self, interval_s: float) -> GpuObservation:
        """
        Advance the data source by `interval_s`, issue a config update for
        the *next* interval based on the static Pareto table, and return the
        GpuObservation from this interval.
        """
        obs = self.source.next_observation(interval_s)

        candidate = self._select_config(obs.phase)
        if candidate is None:
            # No within-SLO config at all — return observation unchanged.
            return obs

        freq = candidate["gpu_freq_mhz"]
        seqs = candidate["max_num_seqs"]

        changed = False
        if freq != self._current_freq or seqs != self._current_seqs:
            self.source.set_config(max_num_seqs=seqs, gpu_freq_mhz=freq)
            self._current_freq = freq
            self._current_seqs = seqs
            changed = True

        self.history.append(SchedulerDecision(
            timestamp=obs.timestamp,
            phase=obs.phase,
            chosen_freq_mhz=freq,
            chosen_max_seqs=seqs,
            predicted_temp_c=candidate["_pred_temp_c"],
            predicted_energy_j=candidate["_pred_energy_j"],
            predicted_goodput=candidate["goodput"],
            policy=self.policy,
            config_changed=changed,
        ))

        return obs

    def run(
        self,
        total_s: float,
        interval_s: float = 10.0,
    ) -> list[GpuObservation]:
        """
        Drive the scheduler for `total_s` seconds, stepping every `interval_s`.
        Returns the full list of GpuObservations collected.
        """
        observations: list[GpuObservation] = []
        elapsed = 0.0
        while elapsed < total_s:
            obs = self.step(interval_s)
            observations.append(obs)
            elapsed += interval_s
        return observations

    def best_config_for(self, phase: str) -> Optional[dict]:
        """
        Return the recommended Pareto entry for `phase` without advancing the
        data source.  Useful for inspection and unit testing.
        """
        return self._select_config(phase)

    # ── Internals ───────────────────────────────────────────────────────────────

    def _select_config(self, phase: str) -> Optional[dict]:
        """
        Filter the Pareto table to within-SLO, phase-matched, thermally safe
        configs, then rank by the active policy.

        Fallback: if all within-SLO configs exceed `thermal_limit_c`, the
        thermal guard is relaxed and policy ranking is applied to all of them.
        """
        within_slo = [
            r for r in self._pareto
            if r["phase"] == phase and r["within_slo"]
        ]
        if not within_slo:
            return None

        # Apply thermal guard
        candidates = [r for r in within_slo if r["_pred_temp_c"] <= self.thermal_limit_c]
        if not candidates:
            # Relax guard — prefer throttle over zero goodput
            candidates = within_slo

        return _rank(candidates, self.policy)


# ── Module-level helpers ───────────────────────────────────────────────────────

def _features(freq_mhz: int, max_seqs: int, phase: str) -> list[float]:
    """
    Normalized feature vector matching build_static_models.py exactly:
      [gpu_freq_mhz / 1377, max_num_seqs / 32, phase_binary]
    """
    return [
        freq_mhz / _MAX_FREQ_MHZ,
        max_seqs / _MAX_NUM_SEQS,
        1.0 if phase == "prefill" else 0.0,
    ]


def _rank(candidates: list[dict], policy: str) -> dict:
    """Return the best candidate according to `policy`."""
    if policy == "min_energy":
        return min(candidates, key=lambda r: r["steady_state_energy_j"])
    if policy == "max_goodput":
        return max(candidates, key=lambda r: r["goodput"])
    # "best_efficiency": highest goodput per joule
    return max(
        candidates,
        key=lambda r: r["goodput"] / max(r["steady_state_energy_j"], 1e-9),
    )
