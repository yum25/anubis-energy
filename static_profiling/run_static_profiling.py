#!/usr/bin/env python3
"""
run_static_profiling.py  —  V100S-compatible static profiling
Uses vLLM 0.4.x (CUDA CC 7.0 compatible) + Zeus for energy measurement.
No mlenergy/Docker dependency.

CloudLab setup:
    pip install "vllm==0.4.3" requests
    export HF_TOKEN=<your_token>
    sudo nvidia-persistenced    # required for nvidia-smi -ac to stick
"""
import json, itertools, os, statistics, subprocess, sys, threading, time
import concurrent.futures
from pathlib import Path

import nvidia_smi
import requests as http
from zeus.monitor import ZeusMonitor

# ── Config ─────────────────────────────────────────────────────────────────────
assert os.environ.get("HF_TOKEN"), "HF_TOKEN not set. Run: export HF_TOKEN=<token>"

REPO_ROOT          = Path(__file__).parent.parent
STATIC_PROFILE_OUT = Path(__file__).parent / "static_profile.json"

MODEL_ID  = "meta-llama/Llama-2-7b-hf"
VLLM_PORT = 8000
BASE_URL  = f"http://localhost:{VLLM_PORT}"

NUM_REQUESTS    = 100   # benchmark requests per config (measured window)
WARMUP_REQUESTS = 10    # discarded before Zeus window opens

V100S_MEM_CLOCK_MHZ = 877  # V100S PCIe fixed HBM2 memory clock

# 4 SM freqs × 3 batch sizes × 2 phases = 24 configs
# Outer loop over MAX_SEQS → server restarts only 3 times total.
# Freq changes via nvidia-smi -ac need no server restart.
GPU_FREQS = [1377, 1200, 1000, 780]
MAX_SEQS  = [1, 8, 32]

PHASES = {
    "prefill": {"input_mean": 1000.0, "output_mean": 50.0},
    "decode":  {"input_mean": 50.0,   "output_mean": 500.0},
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_prompt(target_tokens: float) -> str:
    """Generate a prompt of approximately target_tokens tokens."""
    sentence = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump! "
    )
    # ~4 chars/token is a reasonable estimate for Llama-2 tokenizer
    target_chars = max(1, int(target_tokens * 4))
    reps = target_chars // len(sentence) + 1
    return (sentence * reps)[:target_chars]


def _wait_for_server(proc: subprocess.Popen, timeout: int = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read().decode(errors="replace")
            raise RuntimeError(f"vLLM server exited early (rc={proc.returncode}):\n{output}")
        try:
            if http.get(f"{BASE_URL}/health", timeout=2).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(3)
    output = proc.stdout.read1().decode(errors="replace") if proc.stdout else ""
    raise RuntimeError(f"vLLM server did not become ready in time.\n{output}")


def _start_server(max_num_seqs: int) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                   MODEL_ID,
        "--dtype",                   "float16",  # V100S: no bfloat16
        "--enforce-eager",                       # disable CUDA graphs (CC 7.0)
        "--max-num-seqs",            str(max_num_seqs),
        "--max-model-len",           "2048",     # caps KV cache; fits all shapes
        "--gpu-memory-utilization",  "0.85",
        "--port",                    str(VLLM_PORT),
        "--disable-log-requests",
    ]
    print(f"  [server] starting (max_num_seqs={max_num_seqs})…")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    _wait_for_server(proc)
    print("  [server] ready")
    return proc


def _stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _set_freq(freq_mhz: int) -> None:
    subprocess.run(
        ["sudo", "nvidia-smi", "-ac", f"{V100S_MEM_CLOCK_MHZ},{freq_mhz}"],
        check=True, capture_output=True,
    )


def _send_one(prompt: str, max_tokens: int) -> dict | None:
    """
    Send a single streaming completion.
    Returns e2el_ms, ttft_ms, and tokens_generated (chunk count ≈ token count).
    """
    t0 = time.monotonic()
    t_first = None
    tokens_generated = 0

    try:
        with http.post(
            f"{BASE_URL}/v1/completions",
            json={"model": MODEL_ID, "prompt": prompt,
                  "max_tokens": max_tokens, "stream": True},
            stream=True,
            timeout=max(60, max_tokens * 2),  # 2s/token worst-case budget
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw if isinstance(raw, str) else raw.decode()
                if not line.startswith("data:"):
                    continue
                s = line[5:].strip()
                if s == "[DONE]":
                    break
                try:
                    text = json.loads(s)["choices"][0].get("text", "")
                    if text:
                        # Each non-empty chunk ≈ 1 token in vLLM streaming
                        tokens_generated += 1
                        if t_first is None:
                            t_first = time.monotonic()
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception as e:
        print(f"  [warn] request failed: {e}")
        return None

    t1 = time.monotonic()
    return {
        "e2el_ms":         (t1 - t0) * 1000,
        "ttft_ms":         (t_first - t0) * 1000 if t_first else None,
        "tokens_generated": max(1, tokens_generated),
    }


def _run_requests(prompt: str, max_tokens: int, n: int, concurrency: int,
                  label: str = "") -> list[dict]:
    """Submit n requests with up to `concurrency` in flight at once."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(_send_one, prompt, max_tokens) for _ in range(n)]
        for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
            r = f.result()
            if r is not None:
                results.append(r)
                tok_s = r["tokens_generated"] / (r["e2el_ms"] / 1000)
                print(f"  {label}[{i}/{n}] {r['e2el_ms']:.0f} ms  {tok_s:.1f} tok/s",
                      flush=True)
    return results


def _percentile(vals: list[float], p: int) -> float:
    s = sorted(vals)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _poll_temp(handle, out: list, stop: threading.Event) -> None:
    while not stop.is_set():
        out.append(nvidia_smi.nvmlDeviceGetTemperature(
            handle, nvidia_smi.NVML_TEMPERATURE_GPU))
        time.sleep(5)


# ── Main loop ──────────────────────────────────────────────────────────────────

nvidia_smi.nvmlInit()
nvml_h = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
zeus   = ZeusMonitor(gpu_indices=[0])

# Load any previously completed results so we can resume
if STATIC_PROFILE_OUT.exists():
    with open(STATIC_PROFILE_OUT) as f:
        results = json.load(f)
    print(f"Resuming: {len(results)} configs already done.")
else:
    results = []

done_ids        = {r["run_id"] for r in results}
server_proc     = None
current_max_seq = None

try:
    for max_seqs, (phase, phase_cfg), freq in itertools.product(
        MAX_SEQS, PHASES.items(), GPU_FREQS
    ):
        run_id = f"llama2-7b_{freq}_{max_seqs}_{phase}"

        if run_id in done_ids:
            print(f"\n=== {run_id} (skipping — already done) ===")
            continue

        print(f"\n=== {run_id} ===")

        # Restart server only when max_seqs changes (3 restarts total)
        if max_seqs != current_max_seq:
            if server_proc:
                print("  [server] stopping…")
                _stop_server(server_proc)
            server_proc     = _start_server(max_seqs)
            current_max_seq = max_seqs

        # Frequency change needs no server restart
        _set_freq(freq)

        prompt  = _make_prompt(phase_cfg["input_mean"])
        max_out = int(phase_cfg["output_mean"])
        workers = max(1, max_seqs)

        # Warmup — not measured, lets GPU reach thermal steady state at this freq
        print(f"  warmup ({WARMUP_REQUESTS} req)…")
        _run_requests(prompt, max_out, WARMUP_REQUESTS, workers, label="warmup ")

        # Measurement window
        temp_before = nvidia_smi.nvmlDeviceGetTemperature(
            nvml_h, nvidia_smi.NVML_TEMPERATURE_GPU)

        temp_readings: list[int] = []
        stop_temp   = threading.Event()
        temp_thread = threading.Thread(
            target=_poll_temp,
            args=(nvml_h, temp_readings, stop_temp),
            daemon=True,
        )
        temp_thread.start()

        zeus.begin_window(run_id, sync_execution=False)
        t_start = time.monotonic()
        reqs    = _run_requests(prompt, max_out, NUM_REQUESTS, workers, label="req ")
        t_end   = time.monotonic()
        mes     = zeus.end_window(run_id, sync_execution=False)

        stop_temp.set()
        temp_thread.join()

        temp_after = nvidia_smi.nvmlDeviceGetTemperature(
            nvml_h, nvidia_smi.NVML_TEMPERATURE_GPU)

        if not reqs:
            print("  all requests failed — skipping")
            continue

        duration_s   = t_end - t_start
        energy_j     = sum(dict(mes.gpu_energy).values())
        total_tokens = sum(r["tokens_generated"] for r in reqs)
        e2el_vals    = [r["e2el_ms"] for r in reqs]
        ttft_vals    = [r["ttft_ms"] for r in reqs if r["ttft_ms"] is not None]
        avg_temp     = (statistics.mean(temp_readings) if temp_readings
                        else (temp_before + temp_after) / 2)

        results.append({
            "run_id":       run_id,
            "gpu_freq_mhz": freq,
            "max_num_seqs": max_seqs,
            "phase":        phase,
            "input_mean":   phase_cfg["input_mean"],
            "output_mean":  phase_cfg["output_mean"],
            # Energy — same field names as original schema; build_static_models.py unchanged
            "steady_state_energy_j":           energy_j,
            "steady_state_energy_per_token_j": energy_j / total_tokens,
            "steady_state_duration_s":         duration_s,
            # Latency
            "mean_e2el_ms": statistics.mean(e2el_vals),
            "p50_e2el_ms":  _percentile(e2el_vals, 50),
            "p99_e2el_ms":  _percentile(e2el_vals, 99),
            "mean_ttft_ms": statistics.mean(ttft_vals) if ttft_vals else None,
            "p99_ttft_ms":  _percentile(ttft_vals, 99) if ttft_vals else None,
            # Throughput
            "request_throughput": len(reqs) / duration_s,
            "output_throughput":  total_tokens / duration_s,
            # Temperature
            "avg_gpu_temp_c": avg_temp,
            "temp_before_c":  temp_before,
            "temp_after_c":   temp_after,
        })

        print(f"  temp:   {temp_before}°C → {temp_after}°C  (avg {avg_temp:.1f}°C)")
        print(f"  energy: {energy_j:.1f} J  ({energy_j/total_tokens:.4f} J/tok)")
        print(f"  p99 e2el: {_percentile(e2el_vals, 99):.0f} ms")
        print(f"  tok/s:  {total_tokens/duration_s:.1f}")

        STATIC_PROFILE_OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(STATIC_PROFILE_OUT, "w") as f:
            json.dump(results, f, indent=2)

finally:
    if server_proc:
        _stop_server(server_proc)
    nvidia_smi.nvmlShutdown()

print(f"\nCollected {len(results)}/24 configurations.")
print(f"Saved → {STATIC_PROFILE_OUT}")
