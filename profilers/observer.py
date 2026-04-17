from __future__ import annotations

import time
import threading
from typing import Optional
from zeus.monitor import ZeusMonitor, PowerMonitor, TemperatureMonitor
import requests as _http

from interface import GpuObservation, DataSource


def _fetch_vllm_tokens(base_url: str) -> Optional[int]:
    """
    Read the cumulative generation-token counter from vLLM's Prometheus endpoint.
    Returns None on any failure so callers can fall back gracefully.
    """
    try:
        resp = _http.get(f"{base_url}/metrics", timeout=2)
        resp.raise_for_status()
        for line in resp.text.splitlines():
            if line.startswith("vllm:generation_tokens_total"):
                return int(float(line.split()[-1]))
    except Exception:
        pass
    return None


# ── Real hardware data source (swap-in for CloudLab) ──────────────────────────


class RealGpuDataSource(DataSource):
    """
    Production DataSource that reads from Zeus monitors on real hardware.
    Drop-in replacement for SimulatedGpuDataSource — identical interface.

    Token counting: if vllm_base_url is provided, tokens are measured by
    diffing vLLM's Prometheus counter across each interval.  This is the
    primary path used by experiment.py.  notify_tokens_generated() remains
    as a fallback for callers that drive the vLLM server themselves.

    Usage:
        source = RealGpuDataSource(gpu_indices=[0], vllm_base_url="http://localhost:8000")
    """

    def __init__(
        self,
        gpu_indices: list[int] = [0, 1],
        vllm_base_url: Optional[str] = None,
    ):

        self.gpu_indices = gpu_indices
        self._vllm_base_url = vllm_base_url.rstrip("/") if vllm_base_url else None
        self._zeus = ZeusMonitor(gpu_indices=gpu_indices)
        self._power = PowerMonitor(update_period=0.5)
        self._temp = TemperatureMonitor(update_period=1.0)

        self._max_num_seqs = 8
        self._gpu_freq_mhz = 1410
        self._phase = "decode"
        self._request_rate = 2.0
        self._token_count = 0
        self._lock = threading.Lock()
        self._sim_time = 0.0

        # Snapshot vLLM token counter so the first interval starts from zero delta.
        self._vllm_tokens_snapshot: Optional[int] = (
            _fetch_vllm_tokens(self._vllm_base_url) if self._vllm_base_url else None
        )

        self._zeus.begin_window("interval", sync_execution=False)
        self._interval_start = time.time()

    def set_config(self, max_num_seqs: int, gpu_freq_mhz: int) -> None:
        import subprocess

        self._max_num_seqs = max_num_seqs
        self._gpu_freq_mhz = gpu_freq_mhz
        # Best-effort frequency set — may need sudo/zeusd on real hardware
        try:
            subprocess.run(
                ["sudo", "nvidia-smi", "-ac", f"877,{gpu_freq_mhz}"],
                check=True,
                capture_output=True,
            )
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning("Could not set GPU freq: %s", e)

    def notify_tokens_generated(self, n: int) -> None:
        """Call from your request handler each time tokens are produced."""
        with self._lock:
            self._token_count += n

    def set_workload(self, request_rate: float, phase: str) -> None:
        with self._lock:
            self._request_rate = request_rate
            self._phase = phase

    def inject_event(self, event: str, **kwargs) -> None:
        """No-op on real hardware — events happen naturally."""
        pass

    def next_observation(self, interval_s: float) -> GpuObservation:
        """Block for interval_s, read Zeus, return observation."""
        time.sleep(interval_s)

        now = time.time()
        self._sim_time += interval_s

        # End Zeus energy window
        mes = self._zeus.end_window("interval", sync_execution=False)

        # Power timeline
        power_data = self._power.get_all_power_timelines(
            start_time=self._interval_start, end_time=now
        )
        temp_data = self._temp.get_temperature_timeline(
            start_time=self._interval_start, end_time=now
        )

        # Build per-GPU dicts
        gpu_temps = {}
        gpu_powers = {}
        gpu_energies = dict(mes.gpu_energy)

        for idx in self.gpu_indices:
            key = str(idx)
            # Temperature
            gpu_readings = temp_data.get(key, [])
            gpu_temps[key] = (
                sum(t for _, t in gpu_readings) / len(gpu_readings)
                if gpu_readings
                else 0.0
            )
            # Power (mean over interval)
            pw_readings = power_data.get(key, [])
            gpu_powers[key] = (
                sum(p for _, p in pw_readings) / len(pw_readings)
                if pw_readings
                else 0.0
            )

        # Prefer vLLM Prometheus counter diff; fall back to notify_tokens_generated.
        if self._vllm_base_url is not None:
            new_snapshot = _fetch_vllm_tokens(self._vllm_base_url)
            if new_snapshot is not None and self._vllm_tokens_snapshot is not None:
                tokens = max(0, new_snapshot - self._vllm_tokens_snapshot)
            else:
                with self._lock:
                    tokens = self._token_count
                    self._token_count = 0
            self._vllm_tokens_snapshot = new_snapshot
        else:
            with self._lock:
                tokens = self._token_count
                self._token_count = 0

        total_energy = sum(gpu_energies.values())
        ept = total_energy / tokens if tokens > 0 else None

        # Restart window for next interval
        self._zeus.begin_window("interval", sync_execution=False)
        self._interval_start = now

        return GpuObservation(
            timestamp=now,
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
            throttled=False,
        )
