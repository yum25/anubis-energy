# sweep.py
# Runs the ML.Energy LLM benchmark across a grid of configurations
# and collects results into a single CSV for static profile analysis.

import subprocess
import itertools
import json
import csv
import os
from pathlib import Path
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"   # or 13b if 2x A100 80GB
GPU_MODEL = "a100"                             # adjust to CloudLab GPU type
CONTAINER_IMAGE = "vllm/vllm-openai:v0.8.0"  # pin a version

# Sweep axes
REQUEST_RATES   = [1.0, 2.0, 4.0, 8.0, "inf"]
MAX_CONCURRENCY = [1, 4, 8, 16]
# (input_len, output_len) pairs representing short/medium/long combos
LENGTH_CONFIGS  = [
    (128,  64),
    (128, 256),
    (512, 128),
    (512, 256),
    (1024, 128),
    (1024, 512),
]
NUM_REQUESTS = 64     # enough to get steady-state, not too slow
NUM_GPUS     = 2
SEED         = 42

OUTPUT_DIR = Path("static_profile_results")
OUTPUT_DIR.mkdir(exist_ok=True)
SUMMARY_CSV = OUTPUT_DIR / "static_profile.csv"

# ── Helpers ────────────────────────────────────────────────────────────────────

def run_benchmark(
    request_rate: float | str,
    max_concurrency: int,
    input_len: int,
    output_len: int,
) -> dict | None:
    """Run one benchmark configuration and return parsed results."""

    rate_str = str(request_rate)
    tag = f"rr{rate_str}_mc{max_concurrency}_in{input_len}_out{output_len}"
    result_file = OUTPUT_DIR / f"{tag}.json"

    if result_file.exists():
        print(f"[SKIP] {tag} — result already exists")
        with open(result_file) as f:
            return json.load(f)

    print(f"[RUN ] {tag}")

    cmd = [
        "python", "-m", "mlenergy.llm.benchmark",
        # Workload config — using LengthControl to fix input/output lengths
        "--workload.model_id",       MODEL_ID,
        "--workload.gpu_model",      GPU_MODEL,
        "--workload.num_gpus",       str(NUM_GPUS),
        "--workload.num_requests",   str(NUM_REQUESTS),
        "--workload.seed",           str(SEED),
        "--workload.input_len",      str(input_len),
        "--workload.output_len",     str(output_len),
        # Benchmark args
        "--request_rate",            rate_str,
        "--max_concurrency",         str(max_concurrency),
        "--max_output_tokens",       "dataset",   # use the fixed output_len
        "--server_image",            CONTAINER_IMAGE,
        "--overwrite_results",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(NUM_GPUS))

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if proc.returncode != 0:
        print(f"[FAIL] {tag}\n{proc.stderr[-2000:]}")
        return None

    # ML.Energy writes results to a path derived from workload config;
    # copy it to our tagged output location for clarity
    # (adjust the glob pattern to match your actual output directory layout)
    candidates = list(Path("results").glob(f"**/*{MODEL_ID.split('/')[-1]}*.json"))
    if candidates:
        with open(sorted(candidates)[-1]) as f:
            data = json.load(f)
        with open(result_file, "w") as f:
            json.dump(data, f, indent=2)
        return data

    print(f"[WARN] {tag} — could not locate result file")
    return None


def extract_row(cfg: dict, data: dict) -> dict:
    """Flatten a result JSON into a single CSV row."""
    row = {**cfg}

    # Performance
    row["completed"]              = data.get("completed")
    row["request_throughput"]     = data.get("request_throughput")
    row["output_throughput"]      = data.get("output_throughput")
    row["total_token_throughput"] = data.get("total_token_throughput")

    # Latency
    row["mean_ttft_ms"]   = data.get("mean_ttft_ms")
    row["p99_ttft_ms"]    = data.get("p99_ttft_ms")
    row["mean_tpot_ms"]   = data.get("mean_tpot_ms")
    row["p99_tpot_ms"]    = data.get("p99_tpot_ms")
    row["mean_e2el_ms"]   = data.get("mean_e2el_ms")
    row["p99_e2el_ms"]    = data.get("p99_e2el_ms")

    # Energy
    ss = data.get("steady_state_energy")
    row["steady_state_energy_j"]          = ss
    row["steady_state_energy_per_token_j"] = data.get("steady_state_energy_per_token")
    entire = data.get("entire_benchmark_measurement", {})
    row["total_energy_j"] = sum(entire.get("gpu_energy", {}).values()) if entire else None

    # Temperature (peak from timeline)
    timeline = data.get("timeline", {})
    temps = timeline.get("temperature", {})
    if temps:
        all_temps = [t for gpu_temps in temps.values() for _, t in gpu_temps]
        row["peak_temp_c"] = max(all_temps) if all_temps else None
        row["mean_temp_c"] = sum(all_temps) / len(all_temps) if all_temps else None
    else:
        row["peak_temp_c"] = row["mean_temp_c"] = None

    return row


# ── Main sweep ─────────────────────────────────────────────────────────────────

def main():
    fieldnames = None
    rows = []

    configs = list(itertools.product(REQUEST_RATES, MAX_CONCURRENCY, LENGTH_CONFIGS))
    print(f"Total configurations to run: {len(configs)}")

    for request_rate, max_conc, (in_len, out_len) in configs:
        cfg = {
            "model_id":       MODEL_ID,
            "request_rate":   request_rate,
            "max_concurrency": max_conc,
            "input_len":      in_len,
            "output_len":     out_len,
            "timestamp":      datetime.now().isoformat(),
        }

        data = run_benchmark(request_rate, max_conc, in_len, out_len)
        if data is None:
            continue

        row = extract_row(cfg, data)
        rows.append(row)

        if fieldnames is None:
            fieldnames = list(row.keys())

        # Write incrementally so a crash doesn't lose everything
        write_header = not SUMMARY_CSV.exists()
        with open(SUMMARY_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    print(f"\nDone. {len(rows)} configurations collected.")
    print(f"Summary CSV: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()