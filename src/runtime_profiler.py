# runtime_profiler.py
# Lightweight wrapper around Zeus that continuously profiles GPU energy/temp
# and maintains a rolling time-energy frontier.
# Drop this into your inference serving loop alongside the vLLM server.

import time
import threading
import statistics
from collections import deque
from dataclasses import dataclass, field
from zeus.monitor import PowerMonitor, TemperatureMonitor, ZeusMonitor


@dataclass
class ProfileSnapshot:
    timestamp: float
    avg_power_w: float
    peak_temp_c: float
    energy_j: float
    tokens_generated: int
    energy_per_token_j: float | None


class RuntimeProfiler:
    """
    Continuously monitors GPU power and temperature.
    Maintains a rolling window of energy-per-token estimates
    that can be queried to inform load-balancing decisions.
    """

    WINDOW_SIZE = 60  # seconds of history to keep

    def __init__(self, power_interval: float = 0.5, temp_interval: float = 1.0):
        self._power_monitor = PowerMonitor(update_period=power_interval)
        self._temp_monitor  = TemperatureMonitor(update_period=temp_interval)
        self._zeus          = ZeusMonitor()

        self._snapshots: deque[ProfileSnapshot] = deque()
        self._token_count = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # Time-energy frontier: list of (tokens_per_sec, joules_per_token) pairs
        self._frontier: list[tuple[float, float]] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._zeus.begin_window("runtime_profiler", sync_execution=False)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        mes = self._zeus.end_window("runtime_profiler", sync_execution=False)
        return {
            "total_energy_j":     mes.total_energy,
            "duration_s":         mes.time,
            "snapshots":          list(self._snapshots),
            "frontier":           self._frontier,
        }

    def notify_tokens_generated(self, n: int):
        """Call this from your request handler each time tokens are produced."""
        with self._lock:
            self._token_count += n

    def current_estimate(self) -> dict:
        """
        Return the current runtime estimate of energy cost.
        This is what the runtime profiling strategy uses instead of the
        static lookup table.
        """
        with self._lock:
            recent = [s for s in self._snapshots
                      if time.time() - s.timestamp < 30]  # last 30s
        if not recent:
            return {"status": "warming_up"}

        epj = [s.energy_per_token_j for s in recent if s.energy_per_token_j]
        return {
            "status":                   "ready",
            "mean_energy_per_token_j":  statistics.mean(epj) if epj else None,
            "recent_avg_power_w":       statistics.mean(s.avg_power_w for s in recent),
            "peak_temp_c":              max(s.peak_temp_c for s in recent),
            "frontier_points":          len(self._frontier),
        }

    # ── Internal loop ──────────────────────────────────────────────────────────

    def _loop(self):
        prev_time    = time.time()
        prev_tokens  = 0
        window_start = time.time()

        while self._running:
            time.sleep(2.0)  # sample every 2 seconds

            now = time.time()
            elapsed = now - prev_time

            # Power and temperature from Zeus monitors
            power_data = self._power_monitor.get_all_power_timelines(
                start_time=prev_time, end_time=now
            )
            temp_data = self._temp_monitor.get_temperature_timeline(
                start_time=prev_time, end_time=now
            )

            avg_power = self._mean_from_timeline(power_data)
            peak_temp = self._peak_from_timeline(temp_data)
            energy_j  = avg_power * elapsed if avg_power else 0.0

            with self._lock:
                cur_tokens = self._token_count
            delta_tokens = cur_tokens - prev_tokens
            tps = delta_tokens / elapsed if elapsed > 0 else 0.0
            ept = energy_j / delta_tokens if delta_tokens > 0 else None

            snap = ProfileSnapshot(
                timestamp=now,
                avg_power_w=avg_power or 0.0,
                peak_temp_c=peak_temp or 0.0,
                energy_j=energy_j,
                tokens_generated=delta_tokens,
                energy_per_token_j=ept,
            )

            with self._lock:
                self._snapshots.append(snap)
                # Evict old snapshots outside the window
                cutoff = now - self.WINDOW_SIZE
                while self._snapshots and self._snapshots[0].timestamp < cutoff:
                    self._snapshots.popleft()

                # Update frontier: keep Pareto-optimal (tps, ept) pairs
                if tps > 0 and ept is not None:
                    self._update_frontier(tps, ept)

            prev_time   = now
            prev_tokens = cur_tokens

    def _update_frontier(self, tps: float, ept: float):
        """Keep only Pareto-optimal (throughput, energy-per-token) pairs."""
        self._frontier.append((tps, ept))
        # A point is dominated if another has both higher tps and lower ept
        self._frontier = [
            (t, e) for (t, e) in self._frontier
            if not any(t2 >= t and e2 <= e and (t2, e2) != (t, e)
                       for (t2, e2) in self._frontier)
        ]
        # Keep frontier bounded
        self._frontier = sorted(self._frontier)[-50:]

    @staticmethod
    def _mean_from_timeline(timeline: dict) -> float | None:
        vals = [v for gpu_vals in timeline.values() for _, v in gpu_vals]
        return statistics.mean(vals) if vals else None

    @staticmethod
    def _peak_from_timeline(timeline: dict) -> float | None:
        vals = [v for gpu_vals in timeline.values() for _, v in gpu_vals]
        return max(vals) if vals else None

# ## Suggested Experimental Workflow on CloudLab
# ```
# 1. Reserve 1-2 nodes with A100s on CloudLab
# 2. Install dependencies: vllm, zeus, mlenergy (pip install -e .)
# 3. Set env: HF_TOKEN, HF_HOME, CUDA_VISIBLE_DEVICES
# 4. Run:  python sweep.py          ← collects ~120 configs (takes several hours)
# 5. Run:  python build_static_profile.py  ← fits & saves static models
# 6. During live serving, instantiate RuntimeProfiler alongside vLLM
#    and compare its current_estimate() to static predict() at each
#    load-balancing decision point