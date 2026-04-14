# LLM Inference Energy — Static vs. Runtime Scheduler Comparison

CSE 585 final project. Measures and compares the energy efficiency of two GPU
serving schedulers for LLM inference on a CloudLab V100S node:

- **Static scheduler** — picks configs from a precomputed Pareto lookup table;
  never re-measures at runtime.
- **Runtime scheduler** — adapts frequency in real time using live temperature
  and energy-per-token readings from Zeus.

---

## Prerequisites

### Hardware

A CloudLab node with an **NVIDIA Tesla V100S PCIe** (CUDA Compute Capability
7.0). The static profiling sweep and the vLLM version pinned below are both
specific to CC 7.0.

### Software

| Requirement | Version / note |
|---|---|
| Python | 3.10 or 3.11 (3.12 works; 3.13 untested) |
| CUDA | 11.8 or 12.1 |
| vLLM | **0.4.3** — newer versions drop CC 7.0 support |
| Hugging Face token | Required for `meta-llama/Llama-2-7b-hf` |
| `sudo` access | Required for `nvidia-smi -ac` frequency setting |

---

## 1 — Repository setup

```bash
git clone https://github.com/yum25/anubis-energy.git anubis-energy
cd anubis-energy

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install "vllm==0.4.3"   # separate — only needed for static profiling;
                              # newer versions drop V100S (CC 7.0) support
```

Set your Hugging Face token (required to download Llama-2-7b-hf):

```bash
export HF_TOKEN=hf_...
```

Enable GPU persistence mode so that `nvidia-smi -ac` frequency changes survive
between driver calls:

```bash
sudo nvidia-persistenced
```

Verify the GPU is visible and at its default clock:

```bash
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv
```

---

## 2 — Static profiling sweep

`run_static_profiling.py` benchmarks 24 configurations
(4 SM frequencies × 3 batch sizes × 2 phases) using vLLM and Zeus.
Each configuration runs 10 warmup requests then 100 measured requests.
**Expect roughly 60–90 minutes of wall time.**

```bash
python static_profiling/run_static_profiling.py
```

The script:
1. Starts a vLLM OpenAI-compatible server for each batch-size group
   (3 server restarts total; frequency changes via `nvidia-smi -ac` need no restart).
2. Opens a Zeus energy window around the 100 measured requests per config.
3. Records latency percentiles, throughput, temperature timeline, and energy.

Output: `static_profiling/static_profile.json` — 24 records, one per config.

> **Troubleshooting**
> - If `sudo nvidia-smi -ac` fails, confirm `nvidia-persistenced` is running.
> - If the vLLM server times out on startup, lower `--gpu-memory-utilization`
>   in the script from `0.85` to `0.80`.
> - If you see a CUDA out-of-memory error, make sure no other process is using
>   the GPU (`nvidia-smi` to check).

---

## 3 — Build the static models

`build_static_models.py` reads `static_profile.json` and fits two degree-2
polynomial regression models (temperature and energy), then writes a Pareto
lookup table used by the static scheduler at runtime.

```bash
python static_profiling/build_static_models.py
```

Outputs written to `static_profiling/`:

| File | Contents |
|---|---|
| `static_temp_model.pkl` | sklearn Pipeline → predicted steady-state GPU temp (°C) |
| `static_energy_model.pkl` | sklearn Pipeline → predicted steady-state energy (J) |
| `static_pareto.json` | All 24 configs annotated with goodput and SLO flag |

The script prints R² scores for both models and the number of configurations
within the 8 s P99 E2EL SLO.

---

## 4 — Simulated experiment

Simulation uses a physics model grounded in the TAPAS thermal equations and
ML.Energy benchmark figures for the V100S. Both schedulers run on independent
simulator instances with the same random seed, so results are directly
comparable without hardware access.

### Run both schedulers (default)

```bash
python experiment.py --mode sim --duration 360 --output results/sim.json
```

### Common variations

```bash
# Print per-step observations while running
python experiment.py --mode sim --duration 360 --verbose --output results/sim.json

# Change the static scheduler's optimisation policy
# Options: min_energy (default) | max_goodput | best_efficiency
python experiment.py --mode sim --static-policy max_goodput --output results/sim_goodput.json

# Run only one scheduler
python experiment.py --mode sim --scheduler runtime --output results/sim_runtime_only.json
```

**Simulated workload scenario** (360 s total, 36 steps at 10 s intervals):

| Sim time | Event |
|---|---|
| 0 s | Decode phase, normal load |
| 60 s | Thermal spike +10 °C injected |
| 120 s | Phase transitions to prefill |
| 240 s | Load spike ×1.5 injected for 60 s |
| 300 s | Load expires; phase returns to decode |

The thermal and load spikes are no-ops on real hardware (injected into the
simulator only), giving the runtime scheduler something to react to that the
static scheduler — with its frozen lookup table — cannot anticipate.

After the run, a summary table is printed:

```
── Summary ──────────────────────────────────────────────────
Metric                            Static        Runtime
---------------------------------------------------------
  Total energy (J)               ...            ...
  Total tokens                   ...            ...
  Energy/token (J)               ...            ...
  Mean throughput (tok/s)        ...            ...
  Mean temp (°C)                 ...            ...
  Max temp (°C)                  ...            ...
  Throttle steps                 ...            ...
```

---

## 5 — Real hardware experiment

On real hardware the two schedulers must be run **sequentially** — one per
invocation — with a cool-down period between them. Running them simultaneously
is impossible; running them back-to-back without cooling would give the second
scheduler an unfair thermal starting condition and contaminate the comparison.

### Before either run: start vLLM

The experiment script controls the scheduler (GPU frequency and batch size via
`nvidia-smi -ac`) but does not drive the vLLM server itself. Start the server
first and keep it running for both scheduler runs:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --dtype float16 \
  --enforce-eager \
  --max-num-seqs 32 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --port 8000 &
```

Send traffic to it throughout both runs (e.g. with a load generator), and call
`RealGpuDataSource.notify_tokens_generated(n)` from your request handler so
the scheduler can track throughput.

### Run 1 — static scheduler

```bash
python experiment.py \
  --mode real \
  --scheduler static \
  --static-policy min_energy \
  --duration 360 \
  --gpu-indices 0 \
  --output results/real_static.json
```

### Cool-down between runs

Wait until the GPU returns to near-idle temperature (≈ 40–50 °C) before
starting the second run. This typically takes 5–10 minutes.

```bash
# Monitor until stable
watch -n 5 "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader"
```

### Run 2 — runtime scheduler

```bash
python experiment.py \
  --mode real \
  --scheduler runtime \
  --duration 360 \
  --gpu-indices 0 \
  --output results/real_runtime.json
```

### Compare results

Load both JSON files and compare the `summary` fields in each. The full
`observations` and `decisions` arrays are included for per-step analysis.

---

## CLI reference — experiment.py

| Flag | Default | Description |
|---|---|---|
| `--mode` | `sim` | `sim` = simulator, `real` = Zeus on hardware |
| `--scheduler` | `both` | `static`, `runtime`, or `both` (`both` is sim-only) |
| `--duration` | `360` | Experiment length in seconds |
| `--interval-s` | `10` | Observation interval in seconds |
| `--static-policy` | `min_energy` | `min_energy`, `max_goodput`, `best_efficiency` |
| `--thermal-limit` | `80.0` | Static scheduler thermal ceiling (°C) |
| `--runtime-initial-freq-idx` | `2` | Runtime start freq index: 0=780 1=1000 2=1200 3=1377 MHz |
| `--gpu-indices` | `0` | GPU device indices (e.g. `--gpu-indices 0 1`) |
| `--seed` | `42` | Simulator RNG seed (ignored in real mode) |
| `--output` | `results.json` | Path for the JSON results file |
| `--verbose` / `-v` | off | Print per-step observations to stdout |

---

## Output file schema

```jsonc
{
  "meta": {
    "mode": "sim",          // "sim" or "real"
    "gpu": "V100S",
    "scenario": [...],      // list of (time, action, kwargs) triples
    "duration_s": 360,
    "interval_s": 10,
    "seed": 42,
    "timestamp": "2026-..."
  },
  "static": {               // present if static scheduler was run
    "scheduler": "static",
    "policy": "min_energy",
    "summary": {
      "total_energy_j": ...,
      "total_tokens": ...,
      "mean_energy_per_token_j": ...,
      "mean_tokens_per_s": ...,
      "mean_temp_c": ...,
      "max_temp_c": ...,
      "throttle_steps": ...
    },
    "decisions": [...],     // one SchedulerDecision per step
    "observations": [...]   // one GpuObservation per step
  },
  "runtime": { ... }        // same structure, present if runtime scheduler was run
}
```

---

## Directory layout

```
anubis-energy/
├── profilers/
│   ├── interface.py          # GpuObservation dataclass + DataSource ABC
│   ├── simulator.py          # TAPAS physics simulator (SimulatedGpuDataSource)
│   ├── observer.py           # Zeus wrapper for real hardware (RealGpuDataSource)
│   └── profiles.py           # GPU hardware specs (V100, H100, B200)
├── schedulers/
│   ├── __init__.py
│   ├── static_scheduler.py   # Pareto-table lookup scheduler
│   └── runtime_scheduler.py  # Feedback-control scheduler (Zeus-compatible)
├── static_profiling/
│   ├── run_static_profiling.py   # vLLM + Zeus benchmark sweep → static_profile.json
│   ├── build_static_models.py    # Fits sklearn models + writes Pareto table
│   ├── static_profile.json       # (generated) 24 benchmark records
│   ├── static_pareto.json        # (generated) Pareto lookup table
│   ├── static_temp_model.pkl     # (generated) temperature regression model
│   └── static_energy_model.pkl   # (generated) energy regression model
├── experiment.py             # Comparison experiment driver
├── requirements.txt
└── README.md
```
