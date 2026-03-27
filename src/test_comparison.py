# test_comparison.py
# Tests the decision-making logic that sits on top of both profilers.
# No GPUs needed.

import numpy as np
from pathlib import Path
from test_static_profile import generate_fake_sweep_csv
from build_static_profile import load_and_clean, fit_and_save, predict
from runtime_profiler import RuntimeProfiler

SLO_LATENCY_MS    = 500.0   # p99 E2EL must stay under this
THERMAL_LIMIT_C   = 80.0    # peak temp must stay under this
ENERGY_BUDGET_J_PER_TOK = 0.005  # acceptable energy per token


def static_routing_decision(models, request_rate, max_concurrency, input_len, output_len):
    """
    TAPAS-inspired: use the pre-built static profile to decide whether
    to accept a request at these parameters or shed load.
    Returns (accept: bool, reason: str, prediction: dict)
    """
    pred = predict(models, request_rate, max_concurrency, input_len, output_len)

    if pred["p99_e2el_ms"] > SLO_LATENCY_MS:
        return False, f"Predicted p99 E2EL {pred['p99_e2el_ms']:.1f}ms exceeds SLO {SLO_LATENCY_MS}ms", pred
    if pred["peak_temp_c"] > THERMAL_LIMIT_C:
        return False, f"Predicted temp {pred['peak_temp_c']:.1f}°C exceeds limit {THERMAL_LIMIT_C}°C", pred
    if pred["steady_state_energy_per_token_j"] > ENERGY_BUDGET_J_PER_TOK:
        return False, f"Predicted {pred['steady_state_energy_per_token_j']:.4f} J/tok exceeds budget", pred

    return True, "OK", pred


def runtime_routing_decision(profiler_estimate, input_len, output_len):
    """
    Perseus-inspired: use live profiler readings to make the same decision.
    """
    if profiler_estimate["status"] == "warming_up":
        return True, "Profiler warming up, accepting request optimistically", {}

    ept = profiler_estimate.get("mean_energy_per_token_j")
    temp = profiler_estimate.get("peak_temp_c", 0)

    if temp > THERMAL_LIMIT_C:
        return False, f"Live temp {temp:.1f}°C exceeds limit", profiler_estimate
    if ept and ept > ENERGY_BUDGET_J_PER_TOK:
        return False, f"Live {ept:.4f} J/tok exceeds budget", profiler_estimate

    return True, "OK", profiler_estimate


def test_comparison():
    # Build static models from fake data
    fake_dir = Path("test_fake_results")
    csv_path = generate_fake_sweep_csv(fake_dir)
    df = load_and_clean(csv_path)
    models = fit_and_save(df)

    # Start mock runtime profiler
    profiler = RuntimeProfiler(mock=True)
    profiler.start()
    import time; time.sleep(3)  # let it collect a few samples

    print("\n── Routing decisions: static vs runtime ──")
    print(f"{'Config':<35} {'Static':>10} {'Runtime':>10}")
    print("-" * 60)

    test_cases = [
        # (request_rate, max_concurrency, input_len, output_len, label)
        (1.0,  1,  128,  64,  "light load"),
        (4.0,  8,  512, 256,  "medium load"),
        (8.0, 16, 1024, 512,  "heavy load"),
        (1.0,  1,  128,  64,  "light (repeat)"),
    ]

    for rr, mc, in_len, out_len, label in test_cases:
        s_accept, s_reason, _ = static_routing_decision(models, rr, mc, in_len, out_len)
        r_accept, r_reason, _ = runtime_routing_decision(profiler.current_estimate(), in_len, out_len)

        s_str = "ACCEPT ✓" if s_accept else "REJECT ✗"
        r_str = "ACCEPT ✓" if r_accept else "REJECT ✗"
        print(f"{label:<35} {s_str:>10} {r_str:>10}")
        if not s_accept:
            print(f"  Static reason:  {s_reason}")
        if not r_accept:
            print(f"  Runtime reason: {r_reason}")

    profiler.stop()
    print("\nComparison logic test passed ✓")


if __name__ == "__main__":
    test_comparison()