# decision_loop.py
# Runs both optimizers side by side and logs their divergence.
# This is where your experimental comparison lives.

import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from static_optimizer  import StaticOptimizer, WorkloadObservation
from runtime_optimizer import RuntimeOptimizer
from runtime_profiler  import RuntimeProfiler

logger = logging.getLogger(__name__)
RESULTS_LOG = Path("comparison_results.jsonl")
DECISION_INTERVAL_S = 30.0   # how often to re-evaluate configuration


@dataclass
class DecisionRecord:
    timestamp: str
    workload_request_rate: float
    workload_avg_input_len: int
    workload_avg_output_len: int
    observed_p99_e2el_ms: float
    observed_energy_per_token_j: float | None
    static_max_num_seqs: int
    static_power_limit_w: int
    static_reason: str
    runtime_max_num_seqs: int
    runtime_power_limit_w: int
    runtime_reason: str
    configs_agree: bool


def run_comparison_loop(
    static_opt: StaticOptimizer,
    runtime_opt: RuntimeOptimizer,
    workload_stream,       # iterator yielding WorkloadObservation + live metrics
    duration_s: float = 600.0,
):
    """
    Main comparison loop. Runs both optimizers every DECISION_INTERVAL_S seconds
    and logs where they agree and disagree.

    workload_stream should yield dicts with keys:
        request_rate, avg_input_len, avg_output_len,
        observed_p99_e2el_ms, observed_energy_per_token_j
    """
    start = time.time()
    records = []

    while time.time() - start < duration_s:
        obs_data = next(workload_stream)
        obs = WorkloadObservation(
            request_rate=obs_data["request_rate"],
            avg_input_len=obs_data["avg_input_len"],
            avg_output_len=obs_data["avg_output_len"],
        )

        static_cfg  = static_opt.recommend(obs)
        runtime_cfg = runtime_opt.recommend(obs_data["observed_p99_e2el_ms"])

        agrees = (
            static_cfg.max_num_seqs    == runtime_cfg.max_num_seqs and
            static_cfg.gpu_power_limit_w == runtime_cfg.gpu_power_limit_w
        )

        record = DecisionRecord(
            timestamp=datetime.now().isoformat(),
            workload_request_rate=obs.request_rate,
            workload_avg_input_len=obs.avg_input_len,
            workload_avg_output_len=obs.avg_output_len,
            observed_p99_e2el_ms=obs_data["observed_p99_e2el_ms"],
            observed_energy_per_token_j=obs_data.get("observed_energy_per_token_j"),
            static_max_num_seqs=static_cfg.max_num_seqs,
            static_power_limit_w=static_cfg.gpu_power_limit_w,
            static_reason=static_cfg.reason,
            runtime_max_num_seqs=runtime_cfg.max_num_seqs,
            runtime_power_limit_w=runtime_cfg.gpu_power_limit_w,
            runtime_reason=runtime_cfg.reason,
            configs_agree=agrees,
        )
        records.append(record)

        # Log divergences — these are your most interesting data points
        if not agrees:
            logger.warning(
                "DIVERGENCE at t=%.0fs: static→(seqs=%d, W=%d) vs runtime→(seqs=%d, W=%d)",
                time.time() - start,
                static_cfg.max_num_seqs, static_cfg.gpu_power_limit_w,
                runtime_cfg.max_num_seqs, runtime_cfg.gpu_power_limit_w,
            )
            logger.warning("  Static  said: %s", static_cfg.reason)
            logger.warning("  Runtime said: %s", runtime_cfg.reason)
        else:
            logger.info("t=%.0fs: both optimizers agree → seqs=%d, W=%d",
                        time.time() - start,
                        static_cfg.max_num_seqs, static_cfg.gpu_power_limit_w)

        # Append to JSONL for later analysis
        with open(RESULTS_LOG, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

        time.sleep(DECISION_INTERVAL_S)

    return records