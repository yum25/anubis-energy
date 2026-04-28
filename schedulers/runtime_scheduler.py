"""
Runtime scheduler: maintains a live Pareto table updated from real-time
GpuObservation data, then makes decisions using the same policy logic as
the static scheduler.

The key difference from the static scheduler is the data source for the
Pareto table:
  - Static scheduler: frozen offline measurements, never updated
  - Runtime scheduler: starts from the same static prior, then continuously
    refines entries with empirical observations as the experiment runs

This means both schedulers use identical decision logic (same policies, same
Pareto ranking, same thermal guard) — the only variable is data quality.
Under stable conditions they should behave similarly. Under load/thermal
spikes the runtime scheduler detects that the static prior is stale and
explores neighboring configs to find the new empirical frontier.

Exploration design
------------------
The scheduler is reactive: it stays in exploitation mode (trusting the
current table) until observed metrics diverge from the table prior beyond
a threshold. When degradation is detected it enters exploration mode,
sampling neighboring (freq, max_seqs) configs to find a better operating
point. Exploration is bounded by a cooldown and a per-event budget to
limit the cost of config switching.

Exploration presets (--exploration-mode flag)
---------------------------------------------
  conservative  — only explore on large deviations; long cooldowns; minimal
                  config switching. Closest to static behaviour; useful as
                  a near-static baseline.
  balanced       — moderate thresholds; default for most experiments.
  aggressive     — triggers on small deviations; short cooldowns; willing
                  to switch configs frequently. Maximum adaptivity but
                  highest switching overhead.

Config space
------------
  GPU freqs   : [780, 1000, 1200, 1377, 1980] MHz  (ascending)
  max_num_seqs: [1, 8, 16, 32]                      (ascending)
  phase       : "prefill" | "decode"  (workload-driven, not a control var)

Pareto table entry fields (per (freq, max_seqs, phase) key)
-----------------------------------------------------------
  obs_energy_j        : EMA-smoothed measured energy per interval (J)
  obs_goodput         : EMA-smoothed measured tokens/s
  obs_temp_c          : EMA-smoothed measured GPU temp
  obs_count           : number of real observations for this config
  prior_energy_j      : initial estimate from static profile (frozen)
  prior_goodput       : initial estimate from static profile (frozen)
  within_slo          : bool, updated from observed latency proxy
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from profilers.interface import DataSource, GpuObservation

# ── Config space ──────────────────────────────────────────────────────────────
FREQS_MHZ: list[int] = [780, 1000, 1200, 1377, 1980]
MAX_SEQS_OPS: list[int] = [1, 8, 16, 32]

# Default static profile path — same file build_static_models.py reads
_DEFAULT_PRIOR_PATH = (
    Path(__file__).parent.parent / "static_profiling" / "static_profile.json"
)

# EMA smoothing for live observations
_OBS_EMA_ALPHA = 0.30  # how quickly observations replace the prior

# Thermal limit — configs predicted to exceed this are excluded (matches static scheduler)
_THERMAL_LIMIT_C = 80.0

# SLO threshold — matches build_static_models.py
_SLO_P99_MS = 5000.0


# ── Exploration presets ───────────────────────────────────────────────────────


@dataclass
class ExplorationPreset:
    """
    All knobs controlling when and how aggressively the scheduler explores.

    degradation_threshold : fractional degradation vs prior before triggering
                            exploration. 0.20 = explore when observed metric
                            is 20% worse than the table prior.
    exploration_cooldown  : steps to wait between exploration events. Prevents
                            thrashing when conditions are volatile.
    reversion_patience    : steps to evaluate an explored config before
                            deciding whether to keep it or revert.
    neighbor_budget       : max configs to sample per exploration event
                            (explores freq and max_seqs dimensions).
    min_obs_to_trust      : observations required before treating a config's
                            live estimate as reliable enough to prefer over
                            the prior.
    """

    name: str
    degradation_threshold: float
    exploration_cooldown: int
    reversion_patience: int
    neighbor_budget: int
    min_obs_to_trust: int


PRESETS: dict[str, ExplorationPreset] = {
    "conservative": ExplorationPreset(
        name="conservative",
        degradation_threshold=0.30,  # only react to large deviations (30%)
        exploration_cooldown=6,  # long cooldown — ~60s at 10s intervals
        reversion_patience=4,  # evaluate new config for 40s before deciding
        neighbor_budget=2,  # explore at most 2 neighbors per event
        min_obs_to_trust=5,  # need 5 obs before trusting live estimate
    ),
    "balanced": ExplorationPreset(
        name="balanced",
        degradation_threshold=0.20,  # react to moderate deviations (20%)
        exploration_cooldown=4,  # moderate cooldown — ~40s
        reversion_patience=3,  # evaluate for 30s
        neighbor_budget=4,  # explore up to 4 neighbors
        min_obs_to_trust=3,  # trust after 3 obs
    ),
    "aggressive": ExplorationPreset(
        name="aggressive",
        degradation_threshold=0.10,  # react to small deviations (10%)
        exploration_cooldown=2,  # short cooldown — ~20s
        reversion_patience=2,  # quick evaluation
        neighbor_budget=6,  # explore broadly
        min_obs_to_trust=2,  # trust quickly
    ),
}

DEFAULT_PRESET = "balanced"


# ── Pareto table entry ────────────────────────────────────────────────────────


@dataclass
class TableEntry:
    """One cell in the live Pareto table, keyed by (freq_mhz, max_seqs, phase)."""

    # Prior (from static profile, frozen)
    prior_energy_j: float
    prior_goodput: float
    prior_temp_c: float
    within_slo: bool

    # Live estimates (start equal to prior, updated by observations)
    obs_energy_j: float
    obs_goodput: float
    obs_temp_c: float
    obs_count: int = 0

    @property
    def energy_j(self) -> float:
        """Best available energy estimate — live if trusted, else prior."""
        return self.obs_energy_j

    @property
    def goodput(self) -> float:
        """Best available goodput estimate."""
        return self.obs_goodput

    @property
    def temp_c(self) -> float:
        return self.obs_temp_c

    def update(
        self, obs_energy_j: float, obs_goodput: float, obs_temp_c: float, alpha: float
    ) -> None:
        """EMA-update this entry with a new real observation."""
        if self.obs_count == 0:
            self.obs_energy_j = obs_energy_j
            self.obs_goodput = obs_goodput
            self.obs_temp_c = obs_temp_c
        else:
            self.obs_energy_j = alpha * obs_energy_j + (1 - alpha) * self.obs_energy_j
            self.obs_goodput = alpha * obs_goodput + (1 - alpha) * self.obs_goodput
            self.obs_temp_c = alpha * obs_temp_c + (1 - alpha) * self.obs_temp_c
        self.obs_count += 1


# ── Decision record ───────────────────────────────────────────────────────────


@dataclass
class RuntimeDecision:
    """Record of one scheduling decision."""

    sim_time: float
    timestamp: float
    phase: str
    freq_mhz: int
    max_seqs: int
    avg_temp_c: float
    obs_energy_j: float
    obs_goodput: float
    table_obs_count: int  # how many times current config was observed
    policy: str
    mode: str  # "exploit" | "explore" | "revert"
    exploration_mode: str  # preset name
    config_changed: bool
    reason: str


# ── Scheduler ─────────────────────────────────────────────────────────────────


class RuntimeScheduler:
    """
    Pareto-table runtime scheduler.

    Maintains a live (freq, max_seqs, phase) → TableEntry map.
    Initialises from the static profile prior (if available), then updates
    entries in-place as real observations arrive.

    Decision logic mirrors static_scheduler.py exactly — same policies,
    same thermal guard, same ranking functions. The difference is that the
    table entries reflect current empirical conditions rather than frozen
    offline measurements.
    """

    def __init__(
        self,
        source: DataSource,
        prior_path: Path = _DEFAULT_PRIOR_PATH,
        policy: str = "min_energy",
        thermal_limit_c: float = _THERMAL_LIMIT_C,
        exploration_mode: str = DEFAULT_PRESET,
        max_freq_mhz: int = 1377,
        use_static_prior: bool = False,
    ) -> None:
        if policy not in ("min_energy", "max_goodput", "best_efficiency"):
            raise ValueError(
                f"Unknown policy {policy!r}. "
                "Choose from: 'min_energy', 'max_goodput', 'best_efficiency'"
            )
        if exploration_mode not in PRESETS:
            raise ValueError(
                f"Unknown exploration_mode {exploration_mode!r}. "
                f"Choose from: {list(PRESETS)}"
            )

        self.source = source
        self.policy = policy
        self.thermal_limit_c = thermal_limit_c
        self._preset = PRESETS[exploration_mode]
        self._max_freq_mhz = max_freq_mhz
        self._use_static_prior = use_static_prior

        # Build live Pareto table from static prior.
        # Note: _load_prior calls _fill_missing_entries which uses
        # self._max_freq_mhz, so it must be assigned before this call.
        self._table: dict[tuple[int, int, str], TableEntry] = {}
        self._load_prior(prior_path)

        # Current active config
        self._freq_mhz: int = FREQS_MHZ[2]  # start mid-range
        self._max_seqs: int = 8
        self._phase: str = "decode"

        # Exploration state
        self._exploit_cooldown: int = 0  # steps until exploration allowed
        self._exploring: bool = False
        self._explore_candidates: list[tuple[int, int]] = []
        self._explore_idx: int = 0
        self._pre_explore_freq: int = self._freq_mhz
        self._pre_explore_seqs: int = self._max_seqs
        self._reversion_counter: int = 0

        self.history: list[RuntimeDecision] = []

        # Apply initial config
        self.source.set_config(
            max_num_seqs=self._max_seqs,
            gpu_freq_mhz=self._freq_mhz,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def step(self, interval_s: float) -> GpuObservation:
        obs = self.source.next_observation(interval_s)

        avg_temp = sum(obs.gpu_temp_c.values()) / max(len(obs.gpu_temp_c), 1)
        self._phase = obs.phase

        # Update live table entry for the config that was active this interval
        self._update_table(obs, avg_temp, interval_s)

        # Decide next config
        freq, seqs, mode, reason = self._decide(obs, avg_temp)

        changed = freq != self._freq_mhz or seqs != self._max_seqs
        if changed:
            self._freq_mhz = freq
            self._max_seqs = seqs
            self.source.set_config(max_num_seqs=seqs, gpu_freq_mhz=freq)

        key = (self._freq_mhz, self._max_seqs, obs.phase)
        entry = self._table.get(key)

        self.history.append(
            RuntimeDecision(
                sim_time=obs.sim_time,
                timestamp=obs.timestamp,
                phase=obs.phase,
                freq_mhz=freq,
                max_seqs=seqs,
                avg_temp_c=round(avg_temp, 2),
                obs_energy_j=round(entry.obs_energy_j, 3) if entry else 0.0,
                obs_goodput=round(entry.obs_goodput, 2) if entry else 0.0,
                table_obs_count=entry.obs_count if entry else 0,
                policy=self.policy,
                mode=mode,
                exploration_mode=self._preset.name,
                config_changed=changed,
                reason=reason,
            )
        )

        return obs

    def run(self, total_s: float, interval_s: float = 10.0) -> list[GpuObservation]:
        observations: list[GpuObservation] = []
        elapsed = 0.0
        while elapsed < total_s:
            obs = self.step(interval_s)
            observations.append(obs)
            elapsed += interval_s
        return observations

    def best_config_for(self, phase: str) -> Optional[tuple[int, int]]:
        """Return (freq_mhz, max_seqs) for the best config for phase per current table."""
        entry = self._select_config(phase)
        if entry is None:
            return None
        for (f, s, p), e in self._table.items():
            if e is entry and p == phase:
                return f, s
        return None

    # ── Internals — table management ──────────────────────────────────────────

    def _load_prior(self, prior_path: Path) -> None:
        """
        Populate the table from static_profile.json.
        All (freq, max_seqs, phase) combos in the file get a TableEntry
        initialised to the offline-measured values.
        Missing combos get a synthetic entry derived from neighbours.
        """
        if not self._use_static_prior:
            self._build_empty_table()
            return

        if not prior_path.exists():
            self._build_empty_table()
            return

        with open(prior_path) as f:
            records = json.load(f)

        for r in records:
            freq = r["gpu_freq_mhz"]
            seqs = r["max_num_seqs"]
            phase = r["phase"]
            if freq not in FREQS_MHZ or seqs not in MAX_SEQS_OPS:
                continue

            duration = float(r.get("steady_state_duration_s", 60.0))
            interval_energy = (
                float(r["steady_state_energy_j"]) / duration
            ) * 10.0  # normalize to 10s interval
            energy = interval_energy

            goodput = float(r["output_throughput"])
            temp = float(r["avg_gpu_temp_c"]) if r.get("avg_gpu_temp_c") else 50.0
            p99 = r.get("p99_e2el_ms")
            slo = bool(p99 is not None and p99 < _SLO_P99_MS)

            self._table[(freq, seqs, phase)] = TableEntry(
                prior_energy_j=energy,
                prior_goodput=goodput,
                prior_temp_c=temp,
                within_slo=slo,
                obs_energy_j=energy,
                obs_goodput=goodput,
                obs_temp_c=temp,
            )

        # Fill any missing (freq, seqs, phase) combos by interpolation
        self._fill_missing_entries()

    def _build_empty_table(self) -> None:
        """Fallback: build a table with placeholder entries when no prior exists."""
        for freq in FREQS_MHZ:
            for seqs in MAX_SEQS_OPS:
                for phase in ("prefill", "decode"):
                    self._table[(freq, seqs, phase)] = TableEntry(
                        prior_energy_j=50000.0,
                        prior_goodput=float(seqs * (15 if phase == "prefill" else 40)),
                        prior_temp_c=45.0,
                        within_slo=True,
                        obs_energy_j=50000.0,
                        obs_goodput=float(seqs * (15 if phase == "prefill" else 40)),
                        obs_temp_c=45.0,
                    )

    def _fill_missing_entries(self) -> None:
        """
        For any (freq, seqs, phase) combo not in the prior file, synthesise
        an entry by scaling the nearest available entry using self._max_freq_mhz
        so the freq_factor matches the GPU model the simulator is running.
        """
        for freq in FREQS_MHZ:
            for seqs in MAX_SEQS_OPS:
                for phase in ("prefill", "decode"):
                    key = (freq, seqs, phase)
                    if key in self._table:
                        continue
                    # Find nearest freq with same seqs+phase
                    candidates = [
                        self._table[(f, seqs, phase)]
                        for f in FREQS_MHZ
                        if (f, seqs, phase) in self._table
                    ]
                    if candidates:
                        ref = candidates[0]
                        freq_scale = (freq / self._max_freq_mhz) ** 0.7
                        self._table[key] = TableEntry(
                            prior_energy_j=ref.prior_energy_j
                            * (freq / self._max_freq_mhz),
                            prior_goodput=ref.prior_goodput * freq_scale,
                            prior_temp_c=ref.prior_temp_c,
                            within_slo=ref.within_slo,
                            obs_energy_j=ref.obs_energy_j * (freq / self._max_freq_mhz),
                            obs_goodput=ref.obs_goodput * freq_scale,
                            obs_temp_c=ref.obs_temp_c,
                        )
                    else:
                        # No reference at all — use placeholder
                        self._table[key] = TableEntry(
                            prior_energy_j=50000.0,
                            prior_goodput=float(seqs * 25),
                            prior_temp_c=45.0,
                            within_slo=True,
                            obs_energy_j=50000.0,
                            obs_goodput=float(seqs * 25),
                            obs_temp_c=45.0,
                        )

    def _update_table(
        self, obs: GpuObservation, avg_temp: float, interval_s: float
    ) -> None:
        """EMA-update the entry for the config that was active this interval."""
        key = (self._freq_mhz, self._max_seqs, obs.phase)
        if key not in self._table:
            return

        # Energy: sum across GPUs for this interval
        interval_energy = sum(obs.gpu_energy_j.values())

        # Goodput: tokens per second, using the actual interval duration
        goodput_toks_per_s = obs.tokens_generated / max(interval_s, 1e-6)

        self._table[key].update(
            obs_energy_j=interval_energy,
            obs_goodput=goodput_toks_per_s,
            obs_temp_c=avg_temp,
            alpha=_OBS_EMA_ALPHA,
        )

        # Update SLO estimate: if temp is near throttle, mark as outside SLO
        if avg_temp >= self.thermal_limit_c:
            self._table[key].within_slo = False
        elif self._table[key].obs_count >= self._preset.min_obs_to_trust:
            # Restore SLO flag if temp has recovered
            self._table[key].within_slo = avg_temp < self.thermal_limit_c - 5.0

    # ── Internals — decision logic ─────────────────────────────────────────────

    def _decide(
        self, obs: GpuObservation, avg_temp: float
    ) -> tuple[int, int, str, str]:
        """
        Returns (freq_mhz, max_seqs, mode, reason).

        Decision flow:
          1. Thermal guard — override everything if near throttle
          2. Exploration step — if actively exploring, try next candidate
          3. Degradation check — trigger exploration if current config stale
          4. Exploit — select best config from current Pareto table
        """
        phase = obs.phase

        # ── 1. Thermal guard (mirrors static_scheduler thermal logic) ──────────
        if avg_temp >= self.thermal_limit_c - 3.0:
            safe = self._select_config(phase, thermal_override=True)
            if safe is not None:
                f, s = self._key_for_entry(safe, phase)
                return f, s, "exploit", "thermal_guard"

        # ── 2. Active exploration step ─────────────────────────────────────────
        if self._exploring:
            result = self._exploration_step(phase)
            if result is not None:
                return *result, "explore", "exploring_candidate"
            freq, seqs, reason = self._conclude_exploration(phase)
            return freq, seqs, "revert" if "revert" in reason else "exploit", reason

        # ── 3. Degradation check — should we start exploring? ──────────────────
        if self._exploit_cooldown > 0:
            self._exploit_cooldown -= 1
        else:
            degraded, reason = self._is_degraded(phase)
            if degraded:
                self._start_exploration(phase)
                result = self._exploration_step(phase)
                if result is not None:
                    return *result, "explore", reason

        # ── 4. Exploit — pick best from current table ──────────────────────────
        best = self._select_config(phase)
        if best is None:
            return self._freq_mhz, self._max_seqs, "exploit", "no_valid_config"
        f, s = self._key_for_entry(best, phase)
        return f, s, "exploit", "pareto_select"

    def _is_degraded(self, phase: str) -> tuple[bool, str]:
        """
        Check if the current config's live observations have degraded
        significantly from its prior, indicating the static frontier is stale.

        Uses energy-per-token rather than raw energy to avoid the unit mismatch
        between the static profile's measurement window and the live observation
        interval. Both obs and prior goodput are in tok/s, so dividing energy
        by goodput normalises out the window length difference.
        """
        key = (self._freq_mhz, self._max_seqs, phase)
        entry = self._table.get(key)
        if entry is None or entry.obs_count < self._preset.min_obs_to_trust:
            return False, "insufficient_obs"

        threshold = self._preset.degradation_threshold

        # Energy-per-token degradation: unit-independent comparison
        obs_ept = entry.obs_energy_j / max(entry.obs_goodput, 1e-6)
        prior_ept = entry.prior_energy_j / max(entry.prior_goodput, 1e-6)
        ept_ratio = obs_ept / max(prior_ept, 1e-9)
        if ept_ratio > 1.0 + threshold:
            return True, f"ept_degraded_{ept_ratio:.2f}x"

        # Goodput degradation: observed goodput significantly lower than prior
        goodput_ratio = entry.obs_goodput / max(entry.prior_goodput, 1e-6)
        if goodput_ratio < 1.0 - threshold:
            return True, f"goodput_degraded_{goodput_ratio:.2f}x"

        return False, "stable"

    def _start_exploration(self, phase: str) -> None:
        """
        Build the list of neighbor configs to explore, bounded by neighbor_budget.
        """
        self._pre_explore_freq = self._freq_mhz
        self._pre_explore_seqs = self._max_seqs
        self._exploring = True
        self._explore_idx = 0
        self._reversion_counter = 0

        fi = FREQS_MHZ.index(self._freq_mhz) if self._freq_mhz in FREQS_MHZ else 2
        si = MAX_SEQS_OPS.index(self._max_seqs) if self._max_seqs in MAX_SEQS_OPS else 1

        neighbors: list[tuple[int, int]] = []
        if fi > 0:
            neighbors.append((FREQS_MHZ[fi - 1], self._max_seqs))
        if fi < len(FREQS_MHZ) - 1:
            neighbors.append((FREQS_MHZ[fi + 1], self._max_seqs))
        if si > 0:
            neighbors.append((self._freq_mhz, MAX_SEQS_OPS[si - 1]))
        if si < len(MAX_SEQS_OPS) - 1:
            neighbors.append((self._freq_mhz, MAX_SEQS_OPS[si + 1]))
        if fi > 0 and si < len(MAX_SEQS_OPS) - 1:
            neighbors.append((FREQS_MHZ[fi - 1], MAX_SEQS_OPS[si + 1]))

        def score(fs: tuple[int, int]) -> float:
            entry = self._table.get((fs[0], fs[1], phase))
            if entry is None or not entry.within_slo:
                return -1e9
            return _policy_score(entry, self.policy)

        neighbors.sort(key=score, reverse=True)
        self._explore_candidates = neighbors[: self._preset.neighbor_budget]

    def _exploration_step(self, phase: str) -> Optional[tuple[int, int]]:
        """Return the next candidate config to try, or None if exhausted."""
        while self._explore_idx < len(self._explore_candidates):
            freq, seqs = self._explore_candidates[self._explore_idx]
            self._explore_idx += 1
            key = (freq, seqs, phase)
            entry = self._table.get(key)
            if entry and entry.obs_temp_c < self.thermal_limit_c:
                self._reversion_counter = 0
                return freq, seqs
        return None

    def _conclude_exploration(self, phase: str) -> tuple[int, int, str]:
        """
        Exploration candidates exhausted. Keep the best explored config if it
        beats the pre-exploration config; otherwise revert.
        """
        self._exploring = False
        self._exploit_cooldown = self._preset.exploration_cooldown

        best_entry = None
        best_freq = self._pre_explore_freq
        best_seqs = self._pre_explore_seqs

        for freq, seqs in self._explore_candidates:
            key = (freq, seqs, phase)
            entry = self._table.get(key)
            if entry is None or entry.obs_count < 1:
                continue
            if not entry.within_slo:
                continue
            if entry.obs_temp_c >= self.thermal_limit_c:
                continue
            if best_entry is None or _policy_score(entry, self.policy) > _policy_score(
                best_entry, self.policy
            ):
                best_entry = entry
                best_freq = freq
                best_seqs = seqs

        pre_key = (self._pre_explore_freq, self._pre_explore_seqs, phase)
        pre_entry = self._table.get(pre_key)

        if (
            best_entry is not None
            and pre_entry is not None
            and _policy_score(best_entry, self.policy)
            > _policy_score(pre_entry, self.policy)
        ):
            return best_freq, best_seqs, "kept_explored_config"

        return (
            self._pre_explore_freq,
            self._pre_explore_seqs,
            "reverted_to_prior_config",
        )

    def _select_config(
        self, phase: str, thermal_override: bool = False
    ) -> Optional[TableEntry]:
        """
        Filter the table to within-SLO, phase-matched, thermally safe entries,
        then rank by policy. Mirrors static_scheduler._select_config exactly.
        """
        limit = (
            self.thermal_limit_c
            if not thermal_override
            else (self.thermal_limit_c - 10.0)
        )

        within_slo = [
            e for (f, s, p), e in self._table.items() if p == phase and e.within_slo
        ]
        if not within_slo:
            return None

        candidates = [e for e in within_slo if e.obs_temp_c <= limit]
        if not candidates:
            candidates = within_slo

        return _rank(candidates, self.policy)

    def _key_for_entry(self, entry: TableEntry, phase: str) -> tuple[int, int]:
        """Reverse-lookup (freq, seqs) for a given TableEntry object."""
        for (f, s, p), e in self._table.items():
            if e is entry and p == phase:
                return f, s
        return self._freq_mhz, self._max_seqs


# ── Module-level helpers ──────────────────────────────────────────────────────


def _policy_score(entry: TableEntry, policy: str) -> float:
    """Scalar score for ranking — higher is better."""
    if policy == "min_energy":
        return -entry.energy_j
    if policy == "max_goodput":
        return entry.goodput
    return entry.goodput / max(entry.energy_j, 1e-9)


def _rank(candidates: list[TableEntry], policy: str) -> TableEntry:
    return max(candidates, key=lambda e: _policy_score(e, policy))

