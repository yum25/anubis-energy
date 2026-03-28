# run_static_profiling.py
import subprocess, json, itertools, os
from pathlib import Path
import nvidia_smi  # pip install nvidia-ml-py3

# Confirm HF token is set before doing anything
assert os.environ.get("HF_TOKEN"), (
    "HF_TOKEN not set. Run: export HF_TOKEN=your_token_here"
)

# All paths anchored to repo root (one level up from static_profiling/)
REPO_ROOT    = Path(__file__).parent.parent
RESULTS_BASE = REPO_ROOT / "results" / "static"
STATIC_PROFILE_OUT = REPO_ROOT / "static_profiling" / "static_profile.json"

GPU_MODEL    = "A100"
NUM_GPUS     = 1
NUM_REQUESTS = 100  # increase for real experiments

# Config grid — no quantization, not supported by length-control workload
GPU_FREQS = [1410, 1200, 1000, 800]
MAX_SEQS  = [1, 8, 16, 32]

# Phase control via input/output mean token lengths
# High input + low output  → stresses prefill
# Low input  + high output → stresses decode
PHASES = {
    "prefill": {"input_mean": 1000.0, "output_mean": 50.0},
    "decode":  {"input_mean": 50.0,   "output_mean": 500.0},
}

results = []

# Initialize nvidia-smi once before the loop
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

try:
    for freq, max_seqs, (phase, phase_cfg) in itertools.product(
        GPU_FREQS, MAX_SEQS, PHASES.items()
    ):
        run_id   = f"llama2-7b_{freq}_{max_seqs}_{phase}"
        base_dir = RESULTS_BASE / run_id
        base_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Profiling: {run_id} ===")

        # Set GPU frequency
        subprocess.run(
            ["sudo", "nvidia-smi", "-ac", f"1215,{freq}"],
            check=True
        )

        temp_before = nvidia_smi.nvmlDeviceGetTemperature(
            handle, nvidia_smi.NVML_TEMPERATURE_GPU
        )

        cmd = [
            "python", "-m", "mlenergy.llm.benchmark",
            "--request-rate",          "4.0",
            "--max-concurrency",       str(max_seqs),
            "--max-output-tokens",     str(int(phase_cfg["output_mean"] * 2)),
            "--percentile-metrics",    "ttft,tpot,itl,e2el",
            "--metric-percentiles",    "50,90,95,99",
            "--overwrite-results",
            "--server-image",          "vllm/vllm-openai:v0.11.1",
            "--container-runtime",     "docker",
            "workload:length-control",
            "--workload.base-dir",     str(base_dir),
            "--workload.model-id",     "meta-llama/Llama-2-7b-hf",
            "--workload.num-requests", str(NUM_REQUESTS),
            "--workload.gpu-model",    GPU_MODEL,
            "--workload.num-gpus",     str(NUM_GPUS),
            "--workload.max-num-seqs", str(max_seqs),
            "--workload.input-mean",   str(phase_cfg["input_mean"]),
            "--workload.output-mean",  str(phase_cfg["output_mean"]),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)

        temp_after = nvidia_smi.nvmlDeviceGetTemperature(
            handle, nvidia_smi.NVML_TEMPERATURE_GPU
        )

        if proc.returncode != 0:
            print(f"ERROR for {run_id}:")
            print(proc.stderr)
            continue

        # Find results file — exclude prometheus and server log files
        result_files = [
            f for f in base_dir.iterdir()
            if f.suffix == ".json"
            and "prometheus" not in f.name
            and "server" not in f.name
        ]
        if not result_files:
            print(f"No results file found for {run_id}")
            continue

        with open(result_files[0]) as f:
            bench_out = json.load(f)

        # Extract temperature from timeline
        temp_timeline = bench_out.get("timeline", {}).get("temperature", [])
        avg_temp = (
            sum(t["temperature"] for t in temp_timeline) / len(temp_timeline)
            if temp_timeline else None
        )

        results.append({
            "run_id":        run_id,
            "gpu_freq_mhz":  freq,
            "max_num_seqs":  max_seqs,
            "phase":         phase,
            "input_mean":    phase_cfg["input_mean"],
            "output_mean":   phase_cfg["output_mean"],
            # Energy — use steady state, not entire benchmark
            "steady_state_energy_j":           bench_out["steady_state_energy"],
            "steady_state_energy_per_token_j": bench_out["steady_state_energy_per_token"],
            "steady_state_duration_s":         bench_out["steady_state_duration"],
            # Latency
            "mean_e2el_ms": bench_out.get("mean_e2el_ms"),
            "p50_e2el_ms":  bench_out.get("p50_e2el_ms"),
            "p99_e2el_ms":  bench_out.get("p99_e2el_ms"),  # SLO check
            "mean_ttft_ms": bench_out.get("mean_ttft_ms"),
            "p99_ttft_ms":  bench_out.get("p99_ttft_ms"),
            # Throughput
            "request_throughput": bench_out["request_throughput"],
            "output_throughput":  bench_out["output_throughput"],
            # Temperature
            "avg_gpu_temp_c": avg_temp,
            "temp_before_c":  temp_before,
            "temp_after_c":   temp_after,
        })

        print(f"  Temp: {temp_before}°C → {temp_after}°C")
        print(f"  Energy: {bench_out['steady_state_energy']:.1f} J")
        print(f"  P99 E2EL: {bench_out.get('p99_e2el_ms'):.1f} ms")
        print(f"  Results: {result_files[0]}")

finally:
    nvidia_smi.nvmlShutdown()

# Save frozen static profile alongside the other static_profiling scripts
with open(STATIC_PROFILE_OUT, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nCollected {len(results)} configurations.")
print(f"Static profile saved to {STATIC_PROFILE_OUT}")