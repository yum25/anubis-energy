from __future__ import annotations

import time
import threading

from interface import GpuObservation, DataSource

# ── Real hardware data source (swap-in for CloudLab) ──────────────────────────


class RealGpuDataSource(DataSource):
    """
    Production DataSource that reads from Zeus monitors on real hardware.
    Drop-in replacement for SimulatedGpuDataSource — identical interface.

    Usage:
        source = RealGpuDataSource(gpu_indices=[0, 1])
        # then pass to schedulers exactly as you would SimulatedGpuDataSource
    """

    def __init__(self, gpu_indices: list[int] = [0, 1]):
        # Import Zeus here so the rest of the file works without it locally
        from zeus.monitor import ZeusMonitor, PowerMonitor, TemperatureMonitor

        self.gpu_indices = gpu_indices
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
