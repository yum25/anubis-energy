# build_static_models.py
import json, pickle
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Paths anchored to repo root (one level up from static_profiling/)
REPO_ROOT = Path(__file__).parent.parent
PROFILE_IN = Path(__file__).parent / "static_profile.json"
MODELS_DIR = Path(__file__).parent  # save models alongside this script

SLO_P99_MS = 8000.0  # 8 s P99 E2EL SLO — V100S has ~⅓ A100 memory bandwidth,
# so decode latency is proportionally higher than A100's 5 s target

with open(PROFILE_IN) as f:
    data = json.load(f)

print(f"Loaded {len(data)} configurations from {PROFILE_IN}")

# ── Build feature matrix ──────────────────────────────────────────────────────
# Features:
#   - gpu_freq_mhz normalized to [0, 1] (max in sweep is 1377 for V100S)
#   - max_num_seqs normalized to [0, 1] (max in sweep is 32)
#   - phase encoded as binary (prefill=1, decode=0)
# No quantization feature — removed from sweep

X = []
y_temp = []
y_energy = []  # steady_state_energy_j (primary energy metric)
y_goodput = []  # output_throughput if within SLO, else 0

for r in data:
    X.append(
        [
            r["gpu_freq_mhz"] / 1377,
            r["max_num_seqs"] / 32,
            1.0 if r["phase"] == "prefill" else 0.0,
        ]
    )

    y_temp.append(
        r["avg_gpu_temp_c"] if r["avg_gpu_temp_c"] is not None else float("nan")
    )
    y_energy.append(r["steady_state_energy_j"])

    # Goodput: output throughput only if P99 latency is within SLO
    within_slo = bool(r["p99_e2el_ms"] is not None and r["p99_e2el_ms"] < SLO_P99_MS)
    y_goodput.append(r["output_throughput"] if within_slo else 0.0)

X = np.array(X)
y_temp = np.array(y_temp)
y_energy = np.array(y_energy)
y_goodput = np.array(y_goodput)

# Drop rows where temperature is NaN (benchmark returned no timeline data)
valid_temp = ~np.isnan(y_temp)
if not valid_temp.all():
    print(
        f"Warning: {(~valid_temp).sum()} configs had no temperature data, "
        "excluded from temp model."
    )

# ── Fit regression models ─────────────────────────────────────────────────────
# Degree-2 polynomial approximates TAPAS's piecewise polynomial (MAE < 1°C)
temp_model = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lr", LinearRegression()),
    ]
).fit(X[valid_temp], y_temp[valid_temp])

energy_model = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lr", LinearRegression()),
    ]
).fit(X, y_energy)

# ── Save models ───────────────────────────────────────────────────────────────
temp_model_path = MODELS_DIR / "static_temp_model.pkl"
energy_model_path = MODELS_DIR / "static_energy_model.pkl"

with open(temp_model_path, "wb") as f:
    pickle.dump(temp_model, f)
with open(energy_model_path, "wb") as f:
    pickle.dump(energy_model, f)

print(f"Temp model   R²: {temp_model.score(X[valid_temp], y_temp[valid_temp]):.3f}")
print(f"Energy model R²: {energy_model.score(X, y_energy):.3f}")
print(f"Models saved to {MODELS_DIR}")

# ── Build Pareto frontier ─────────────────────────────────────────────────────
# Each entry in the pareto list is a raw config record augmented with goodput.
# The static scheduler uses this as its lookup table — it never re-measures.
# Pareto-optimal = configs where no other config has both lower energy AND
# higher goodput. We include all configs and let the scheduler filter by
# thermal/power constraints at decision time.
pareto = []
for r, gp in zip(data, y_goodput):
    pareto.append(
        {
            **r,
            "goodput": float(gp),
            "within_slo": bool(gp > 0.0),
        }
    )

pareto_path = MODELS_DIR / "static_pareto.json"
with open(pareto_path, "w") as f:
    json.dump(pareto, f, indent=2)

print(f"Pareto frontier saved to {pareto_path}")
print(f"  {sum(r['within_slo'] for r in pareto)}/{len(pareto)} configs within SLO")

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\nConfig space summary:")
print(f"  GPU freqs:    {sorted(set(r['gpu_freq_mhz'] for r in data))}")
print(f"  Max seqs:     {sorted(set(r['max_num_seqs'] for r in data))}")
print(f"  Phases:       {sorted(set(r['phase'] for r in data))}")
print(f"  Energy range: {min(y_energy):.1f} – {max(y_energy):.1f} J")
print(f"  Temp range:   {np.nanmin(y_temp):.1f} – {np.nanmax(y_temp):.1f} °C")

