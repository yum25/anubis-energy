# test_static_profile.py
# Generates fake sweep results and runs them through build_static_profile.py
# No GPUs, no Docker, no ML.Energy needed.

import json
import csv
import numpy as np
from pathlib import Path

# ── Generate fake sweep result JSONs ──────────────────────────────────────────

def make_fake_result(request_rate, max_concurrency, input_len, output_len, seed=42):
    """
    Simulate what a real ML.Energy result JSON looks like.
    Energy and latency scale in a physically plausible way with the inputs
    so that the regression has something real to learn.
    """
    rng = np.random.default_rng(seed)

    # Rough heuristics: more tokens = more energy, higher concurrency = more throughput
    base_energy_per_tok = 0.002 + (input_len / 10000) + (output_len / 20000)
    noise = rng.normal(0, 0.0001)
    energy_per_token = max(0.001, base_energy_per_tok + noise)

    total_tokens = max_concurrency * output_len
    total_energy = energy_per_token * total_tokens

    # Latency: TTFT grows with input_len; TPOT shrinks with concurrency
    ttft = 50 + input_len * 0.3 + rng.normal(0, 5)
    tpot = 20 + (100 / max(max_concurrency, 1)) + rng.normal(0, 2)
    e2el = ttft + tpot * output_len

    rate = 999.0 if request_rate == "inf" else float(request_rate)
    throughput = min(rate, max_concurrency) * (1000 / e2el)

    # Match the structure of a real ML.Energy result JSON
    return {
        "completed": 64,
        "request_throughput": throughput,
        "output_throughput": throughput * output_len,
        "total_token_throughput": throughput * (input_len + output_len),
        "mean_ttft_ms": ttft,
        "p99_ttft_ms":  ttft * 1.4,
        "mean_tpot_ms": tpot,
        "p99_tpot_ms":  tpot * 1.5,
        "mean_e2el_ms": e2el,
        "p99_e2el_ms":  e2el * 1.4,
        "steady_state_energy": total_energy,
        "steady_state_energy_per_token": energy_per_token,
        "entire_benchmark_measurement": {
            "gpu_energy": {"0": total_energy * 0.6, "1": total_energy * 0.4},
            "time": e2el / 1000,
        },
        "timeline": {
            "temperature": {
                "0": [(0.0, 45.0 + max_concurrency * 2), (1.0, 47.0 + max_concurrency * 2)],
                "1": [(0.0, 43.0 + max_concurrency * 2), (1.0, 44.0 + max_concurrency * 2)],
            }
        },
    }


def generate_fake_sweep_csv(output_dir: Path) -> Path:
    """Write fake per-config JSONs and a summary CSV, mirroring sweep.py output."""
    output_dir.mkdir(parents=True, exist_ok=True)

    request_rates   = [1.0, 2.0, 4.0, 8.0, "inf"]
    max_concurrency = [1, 4, 8, 16]
    length_configs  = [(128, 64), (128, 256), (512, 128), (512, 256), (1024, 512)]

    rows = []
    fieldnames = None

    for rr in request_rates:
        for mc in max_concurrency:
            for in_len, out_len in length_configs:
                data = make_fake_result(rr, mc, in_len, out_len)

                # Save individual JSON (mirrors what sweep.py writes)
                tag = f"rr{rr}_mc{mc}_in{in_len}_out{out_len}"
                with open(output_dir / f"{tag}.json", "w") as f:
                    json.dump(data, f, indent=2)

                # Flatten into a CSV row (mirrors extract_row in sweep.py)
                temps = [t for gpu in data["timeline"]["temperature"].values()
                         for _, t in gpu]
                row = {
                    "model_id":       "meta-llama/Llama-2-7b-chat-hf",
                    "request_rate":   rr,
                    "max_concurrency": mc,
                    "input_len":      in_len,
                    "output_len":     out_len,
                    "completed":      data["completed"],
                    "request_throughput":     data["request_throughput"],
                    "output_throughput":      data["output_throughput"],
                    "total_token_throughput": data["total_token_throughput"],
                    "mean_ttft_ms":   data["mean_ttft_ms"],
                    "p99_ttft_ms":    data["p99_ttft_ms"],
                    "mean_tpot_ms":   data["mean_tpot_ms"],
                    "p99_tpot_ms":    data["p99_tpot_ms"],
                    "mean_e2el_ms":   data["mean_e2el_ms"],
                    "p99_e2el_ms":    data["p99_e2el_ms"],
                    "steady_state_energy_j":           data["steady_state_energy"],
                    "steady_state_energy_per_token_j": data["steady_state_energy_per_token"],
                    "total_energy_j": sum(data["entire_benchmark_measurement"]["gpu_energy"].values()),
                    "peak_temp_c":    max(temps),
                    "mean_temp_c":    sum(temps) / len(temps),
                }
                rows.append(row)
                if fieldnames is None:
                    fieldnames = list(row.keys())

    csv_path = output_dir / "static_profile.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} fake configs → {csv_path}")
    return csv_path


# ── Run the actual build_static_profile logic against fake data ───────────────

def test_profile_pipeline():
    from build_static_profile import load_and_clean, fit_and_save, predict

    fake_dir = Path("test_fake_results")
    csv_path = generate_fake_sweep_csv(fake_dir)

    df = load_and_clean(csv_path)
    print(f"\nLoaded {len(df)} rows, columns: {list(df.columns)}")
    assert len(df) == 100, f"Expected 100 rows, got {len(df)}"

    models = fit_and_save(df)
    print(f"\nFitted models for: {list(models.keys())}")

    # Test a few predictions and sanity-check directionality
    low  = predict(models, request_rate=1.0,   max_concurrency=1,  input_len=128, output_len=64)
    high = predict(models, request_rate=8.0,   max_concurrency=16, input_len=1024, output_len=512)

    print("\nLow-load prediction:  ", {k: f"{v:.4f}" for k, v in low.items()})
    print("High-load prediction: ", {k: f"{v:.4f}" for k, v in high.items()})

    # High load should consume more energy per token and have higher latency
    assert high["steady_state_energy_per_token_j"] > low["steady_state_energy_per_token_j"], \
        "Energy per token should be higher under heavy load"
    assert high["mean_ttft_ms"] > low["mean_ttft_ms"], \
        "TTFT should be higher with longer inputs"
    assert high["peak_temp_c"] > low["peak_temp_c"], \
        "Temperature should be higher under heavy load"

    print("\nAll assertions passed ✓")


if __name__ == "__main__":
    test_profile_pipeline()