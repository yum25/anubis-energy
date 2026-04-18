#!/usr/bin/env python3
"""
experiment.py — static vs. runtime scheduler comparison.

Modes
-----
  sim   Run both schedulers against the simulator (same seed → reproducible).
        Injects a thermal spike and a load spike so the runtime scheduler has
        something to react to that the static scheduler cannot anticipate.

  real  Run one scheduler on live CloudLab hardware via RealGpuDataSource /
        Zeus.  Only one scheduler at a time (hardware constraint).

Usage examples
--------------
  # Simulation — compare both schedulers on H100, save results
  python experiment.py --mode sim --gpu-model H100 --duration 360 \
                       --output results/sim_h100.json

  # Simulation — run only the runtime scheduler with max_goodput policy
  python experiment.py --mode sim --gpu-model H100 --duration 360 \
                       --scheduler runtime --static-policy max_goodput

  # Real hardware — run static scheduler on V100S (default gpu-model)
  python experiment.py --mode real --duration 360 --scheduler static \
                       --gpu-indices 0 --vllm-url http://localhost:8000 \
                       --output results/real_static.json

Output schema
-------------
  {
    "meta": { mode, gpu, max_freq_mhz, scenario, duration_s, interval_s, timestamp },
    "static":  { summary metrics, observations[], decisions[] },   # if run
    "runtime": { summary metrics, observations[], decisions[] }    # if run
  }

Import note
-----------
  profilers/simulator.py and profilers/observer.py use bare `from interface import …`
  (designed to run from within profilers/).  This script adds both the repo root
  AND profilers/ to sys.path so both import styles resolve to the same source files.
  Python caches modules by (loader, spec) so there is no double-import of classes
  in practice for the dataclasses we use.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── sys.path setup ────────────────────────────────────────────────────────────
# Must happen before any project imports so both import styles resolve.
_REPO_ROOT = Path(__file__).parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "profilers") not in sys.path:
    sys.path.insert(1, str(_REPO_ROOT / "profilers"))

from profilers.interface import DataSource, GpuObservation  # noqa: E402
from schedulers.static_scheduler import StaticScheduler  # noqa: E402
from schedulers.runtime_scheduler import RuntimeScheduler  # noqa: E402

# ── GPU model → max frequency lookup ─────────────────────────────────────────
# Used to derive the Pareto normalisation constant for both schedulers and to
# set the simulator's physics profile.  Must stay in sync with profiles.py and
# the freq grids in the static/runtime schedulers.
GPU_MAX_FREQ: dict[str, int] = {
    "V100": 1377,
    "V100S": 1377,
    "H100": 1980,
    "B200": 2050,
}

# ── Experiment scenario ───────────────────────────────────────────────────────
# A list of (sim_time_s, action, kwargs) triples applied during the run loop.
# inject_event() is a no-op on RealGpuDataSource — safe to include in real mode.
#
# Timeline (360 s total, interval_s=10 → 36 steps):
#   t=  0  decode, steady load
#   t= 60  thermal spike +10 °C  (tests thermal guard)
#   t=120  switch to prefill phase
#   t=240  load spike ×1.5 for 60 s (tests load adaptation)
#   t=300  load spike expires; back to decode
SCENARIO: list[tuple[float, str, dict]] = [
    (0, "workload", {"request_rate": 2.0, "phase": "decode"}),
    (60, "event", {"event": "thermal_spike", "delta_c": 10.0}),
    (120, "workload", {"request_rate": 2.0, "phase": "prefill"}),
    (240, "event", {"event": "load_spike", "duration_s": 60.0, "multiplier": 1.5}),
    (300, "workload", {"request_rate": 2.0, "phase": "decode"}),
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _obs_to_dict(obs: GpuObservation) -> dict:
    return dataclasses.asdict(obs)


def _build_source(args: argparse.Namespace) -> DataSource:
    """Construct the appropriate DataSource for the requested mode."""
    if args.mode == "sim":
        from profilers.simulator import SimulatedGpuDataSource

        return SimulatedGpuDataSource(
            gpu_indices=args.gpu_indices,
            t_outside_c=22.0,
            dc_load_pct=40.0,
            gpu_model=args.gpu_model,
            seed=args.seed,
        )
    else:
        from profilers.observer import RealGpuDataSource

        return RealGpuDataSource(
            gpu_indices=args.gpu_indices,
            vllm_base_url=args.vllm_url,
        )


def _apply_scenario_events(
    source: DataSource,
    elapsed: float,
    interval_s: float,
) -> None:
    """
    Fire any scenario actions whose trigger time falls in (elapsed - interval_s, elapsed].
    Called after each step so the event is applied before the next observation.
    """
    window_start = elapsed - interval_s
    for t, action, kwargs in SCENARIO:
        if window_start < t <= elapsed:
            if action == "workload":
                if hasattr(source, "set_workload"):
                    source.set_workload(**kwargs)
            elif action == "event":
                source.inject_event(**kwargs)


def _compute_summary(
    observations: list[GpuObservation],
    interval_s: float,
) -> dict[str, Any]:
    """Aggregate metrics over an observation list."""
    if not observations:
        return {}

    total_energy_j = sum(sum(obs.gpu_energy_j.values()) for obs in observations)
    total_tokens = sum(obs.tokens_generated for obs in observations)
    epts = [obs.energy_per_token_j for obs in observations if obs.energy_per_token_j]
    all_temps = [t for obs in observations for t in obs.gpu_temp_c.values()]
    throttle_steps = sum(1 for obs in observations if obs.throttled)

    return {
        "total_energy_j": round(total_energy_j, 3),
        "total_tokens": total_tokens,
        "mean_energy_per_token_j": round(sum(epts) / len(epts), 6) if epts else None,
        "mean_tokens_per_s": round(total_tokens / (len(observations) * interval_s), 2),
        "mean_temp_c": round(sum(all_temps) / len(all_temps), 2) if all_temps else None,
        "max_temp_c": round(max(all_temps), 2) if all_temps else None,
        "throttle_steps": throttle_steps,
        "n_observations": len(observations),
    }


def _run_scheduler(
    scheduler_type: str,
    source: DataSource,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Instantiate scheduler_type on source, drive it for args.duration seconds,
    apply scenario events, and return observations + decisions + summary.
    """
    interval_s = args.interval_s

    if scheduler_type == "static":
        sched = StaticScheduler(
            source=source,
            policy=args.static_policy,
            thermal_limit_c=args.thermal_limit,
            max_freq_mhz=args.max_freq_mhz,
        )
    else:
        sched = RuntimeScheduler(
            source=source,
            policy=args.static_policy,
            exploration_mode=args.exploration_mode,
            max_freq_mhz=args.max_freq_mhz,
        )

    # Prime the workload state before the loop
    source.set_workload(request_rate=2.0, phase="decode")

    observations: list[GpuObservation] = []
    elapsed = 0.0

    print(
        f"  [{scheduler_type}] starting — duration={args.duration}s "
        f"interval={interval_s}s gpu={args.gpu_model} max_freq={args.max_freq_mhz}MHz",
        flush=True,
    )

    while elapsed < args.duration:
        elapsed += interval_s
        obs = sched.step(interval_s)
        observations.append(obs)
        _apply_scenario_events(source, elapsed, interval_s)

        if args.verbose:
            avg_temp = sum(obs.gpu_temp_c.values()) / max(len(obs.gpu_temp_c), 1)
            total_e = sum(obs.gpu_energy_j.values())
            print(
                f"    t={elapsed:>5.0f}s  phase={obs.phase:<7s}  "
                f"freq={obs.gpu_freq_mhz}MHz  seqs={obs.max_num_seqs:>2d}  "
                f"temp={avg_temp:.1f}°C  energy={total_e:.1f}J  "
                f"tokens={obs.tokens_generated}",
                flush=True,
            )

    summary = _compute_summary(observations, interval_s)
    print(
        f"  [{scheduler_type}] done — "
        f"total_energy={summary['total_energy_j']:.1f}J  "
        f"total_tokens={summary['total_tokens']}  "
        f"throttle_steps={summary['throttle_steps']}",
        flush=True,
    )

    decisions = [dataclasses.asdict(d) for d in sched.history]

    return {
        "scheduler": scheduler_type,
        "policy": args.static_policy,
        "exploration_mode": args.exploration_mode
        if scheduler_type == "runtime"
        else None,
        "summary": summary,
        "decisions": decisions,
        "observations": [_obs_to_dict(o) for o in observations],
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare static vs. runtime GPU schedulers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["sim", "real"],
        default="sim",
        help="'sim' uses SimulatedGpuDataSource; 'real' uses RealGpuDataSource+Zeus.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["static", "runtime", "both"],
        default="both",
        help="Which scheduler(s) to run.  'both' is only meaningful in sim mode.",
    )
    parser.add_argument(
        "--gpu-model",
        choices=list(GPU_MAX_FREQ.keys()),
        default="V100S",
        dest="gpu_model",
        help=(
            "GPU model. Controls simulator physics, Pareto normalisation, and "
            "output metadata. Default V100S matches CloudLab hardware. Use H100 "
            "or B200 for sim-only runs backed by ML.Energy profiles."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=360.0,
        help="Total experiment wall/sim time in seconds.",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=10.0,
        dest="interval_s",
        help="Observation interval in seconds.",
    )
    parser.add_argument(
        "--static-policy",
        choices=["min_energy", "max_goodput", "best_efficiency"],
        default="min_energy",
        dest="static_policy",
        help="Policy for the static (and runtime) scheduler.",
    )
    parser.add_argument(
        "--thermal-limit",
        type=float,
        default=80.0,
        dest="thermal_limit",
        help="Temperature ceiling (°C) for the static scheduler's thermal guard.",
    )
    parser.add_argument(
        "--exploration-mode",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        dest="exploration_mode",
        help="Runtime scheduler exploration aggressiveness.",
    )
    parser.add_argument(
        "--gpu-indices",
        type=int,
        nargs="+",
        default=[0],
        dest="gpu_indices",
        help="GPU device indices to monitor (e.g. --gpu-indices 0 1).",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000",
        dest="vllm_url",
        help="Base URL of the running vLLM server (real mode only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the simulator (ignored in real mode).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results.json"),
        help="Path to write the JSON results file.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-step observations to stdout.",
    )
    args = parser.parse_args()

    # Derive max_freq_mhz from gpu_model — single source of truth
    args.max_freq_mhz = GPU_MAX_FREQ[args.gpu_model]

    if args.scheduler == "both" and args.mode == "real":
        parser.error(
            "--scheduler both requires --mode sim "
            "(cannot run two schedulers on the same hardware simultaneously)."
        )

    results: dict[str, Any] = {
        "meta": {
            "mode": args.mode,
            "gpu": args.gpu_model,
            "max_freq_mhz": args.max_freq_mhz,
            "scenario": SCENARIO,
            "duration_s": args.duration,
            "interval_s": args.interval_s,
            "seed": args.seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    }

    schedulers_to_run = (
        ["static", "runtime"] if args.scheduler == "both" else [args.scheduler]
    )

    for sched_type in schedulers_to_run:
        print(f"\nBuilding {args.mode} data source for [{sched_type}] scheduler …")
        source = _build_source(args)
        results[sched_type] = _run_scheduler(sched_type, source, args)

    # Write results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.output}")

    # ── Quick comparison table ────────────────────────────────────────────────
    if args.scheduler == "both":
        print("\n── Summary ──────────────────────────────────────────────────")
        header = f"{'Metric':<30} {'Static':>14} {'Runtime':>14}"
        print(header)
        print("-" * len(header))
        s = results["static"]["summary"]
        r = results["runtime"]["summary"]
        rows = [
            ("Total energy (J)", s["total_energy_j"], r["total_energy_j"]),
            ("Total tokens", s["total_tokens"], r["total_tokens"]),
            (
                "Energy/token (J)",
                s["mean_energy_per_token_j"],
                r["mean_energy_per_token_j"],
            ),
            ("Mean throughput (tok/s)", s["mean_tokens_per_s"], r["mean_tokens_per_s"]),
            ("Mean temp (°C)", s["mean_temp_c"], r["mean_temp_c"]),
            ("Max temp (°C)", s["max_temp_c"], r["max_temp_c"]),
            ("Throttle steps", s["throttle_steps"], r["throttle_steps"]),
        ]
        for label, sv, rv in rows:
            sv_str = f"{sv:.4g}" if sv is not None else "N/A"
            rv_str = f"{rv:.4g}" if rv is not None else "N/A"
            print(f"  {label:<28} {sv_str:>14} {rv_str:>14}")
        print()


if __name__ == "__main__":
    main()
