from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
# ── Shared data contract ───────────────────────────────────────────────────────
# This is the ONLY thing schedulers are allowed to import from this file.
# When using real hardware, RealGpuDataSource produces the same type.


@dataclass
class GpuObservation:
    """
    A single snapshot of GPU state at one point in time.
    Produced by either the simulator or real Zeus monitors.
    Both schedulers receive this and nothing else.
    """

    timestamp: float  # wall-clock time (time.time())
    sim_time: float  # simulated seconds elapsed (equals wall time on real hardware)

    # Per-GPU readings (dicts keyed by GPU index string, e.g. "0", "1")
    gpu_temp_c: dict[str, float]  # current GPU temperature
    gpu_power_w: dict[str, float]  # current power draw in Watts
    gpu_energy_j: dict[str, float]  # energy consumed since last observation

    # Aggregate inference metrics for this interval
    tokens_generated: int  # output tokens produced in this interval
    energy_per_token_j: float | None  # None if no tokens yet
    request_rate: float  # requests/sec arriving in this interval
    phase: str  # "prefill" or "decode"

    # Serving config active during this interval
    max_num_seqs: int
    gpu_freq_mhz: int

    # Derived: did any limit get hit?
    throttled: bool = False  # True if GPU hit thermal throttle


class DataSource(ABC):
    """
    Protocol that both SimulatedGpuDataSource and RealGpuDataSource implement.
    Schedulers depend only on this — never on the concrete class.
    """

    @abstractmethod
    def next_observation(self, interval_s: float) -> GpuObservation:
        """
        Advance time by interval_s and return the resulting GPU state.
        On real hardware this blocks for interval_s seconds then reads Zeus.
        In simulation this advances the physics model and returns immediately.
        """
        ...

    @abstractmethod
    def set_config(self, max_num_seqs: int, gpu_freq_mhz: int) -> None:
        """
        Apply a new serving configuration.
        On real hardware this calls nvidia-smi and updates vLLM.
        In simulation this updates the workload parameters immediately.
        """
        ...

    @abstractmethod
    def inject_event(self, event: str, **kwargs) -> None:
        """
        Inject a named disturbance into the data source.
        No-op on real hardware (events happen naturally).
        In simulation this perturbs the physics model.

        Supported simulation events:
            "thermal_spike"  kwargs: delta_c (float)
            "load_spike"     kwargs: duration_s (float), multiplier (float)
            "cooling_failure" kwargs: duration_s (float)
        """
        ...
