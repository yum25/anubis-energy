# run_static_profiling.py
import subprocess, json, itertools, os

# Your configuration grid — mirrors TAPAS Table 1 knobs
CONFIGS = {
    "model":      ["meta-llama/Llama-2-7b-hf"],
    "batch_size": [1, 8, 16, 32],
    "gpu_freq":   [1410, 1200, 1000, 800],
    "quant":      ["fp16", "int8"],
    "phase":      ["prefill", "decode"],
}

# Phase-appropriate prompt sets
PHASE_PROMPTS = {
    "prefill": "prompts/long_prompts.txt",   # long prompts, few output tokens
    "decode":  "prompts/short_prompts.txt",  # short prompts, many output tokens
}

results = []

for model, batch, freq, quant, phase in itertools.product(
    CONFIGS["model"],
    CONFIGS["batch_size"],
    CONFIGS["gpu_freq"],
    CONFIGS["quant"],
    CONFIGS["phase"],
):
    run_id = f"{model.split('/')[-1]}_{batch}_{freq}_{quant}_{phase}"
    output_file = f"profiling_results/{run_id}.json"
    os.makedirs("profiling_results", exist_ok=True)

    # Set GPU frequency before each run
    subprocess.run(
        ["sudo", "nvidia-smi", "-ac", f"1215,{freq}"],
        check=True
    )

    # Invoke ML.Energy benchmark for this configuration
    cmd = [
        "python", "-m", "benchmark.run",
        "--model",        model,
        "--batch-size",   str(batch),
        "--quantization", quant,
        "--input-file",   PHASE_PROMPTS[phase],
        "--output-file",  output_file,
        "--max-new-tokens", "5"   if phase == "prefill" else "200",
        "--backend",      "vllm",
    ]

    subprocess.run(cmd, check=True)

    # Parse benchmark output into your unified results format
    with open(output_file) as f:
        bench_out = json.load(f)

    results.append({
        "model":        model,
        "batch_size":   batch,
        "gpu_freq_mhz": freq,
        "quantization": quant,
        "phase":        phase,
        # ML.Energy benchmark output fields:
        "energy_j":     bench_out["energy_j"],
        "avg_power_w":  bench_out["avg_power_w"],
        "p50_lat_s":    bench_out["latency_p50_s"],
        "p99_lat_s":    bench_out["latency_p99_s"],
        "throughput":   bench_out["throughput_token_per_s"],
        "gpu_temp_c":   bench_out.get("avg_gpu_temp_c"),
    })

# Save frozen static profile
with open("static_profile.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Collected {len(results)} configurations.")