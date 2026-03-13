# build_static_profile.py
# Fits a simple lookup/interpolation model from the collected sweep data.
# This becomes the "static profiler" — given a workload description,
# predict energy and latency without running anything.

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

PROFILE_CSV = Path("static_profile_results/static_profile.csv")
MODEL_DIR   = Path("static_profile_results/models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["request_rate_num", "max_concurrency", "input_len", "output_len"]
TARGETS  = [
    "steady_state_energy_per_token_j",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_e2el_ms",
    "peak_temp_c",
]


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Convert "inf" string to a large float for regression
    df["request_rate_num"] = df["request_rate"].apply(
        lambda x: 999.0 if str(x) == "inf" else float(x)
    )
    df = df.dropna(subset=FEATURES + TARGETS)
    return df


def fit_and_save(df: pd.DataFrame):
    X = df[FEATURES].values
    models = {}

    for target in TARGETS:
        y = df[target].values
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
            )),
        ])
        pipe.fit(X, y)
        models[target] = pipe
        out = MODEL_DIR / f"{target}.joblib"
        joblib.dump(pipe, out)
        print(f"Saved model for '{target}' → {out}")

    return models


def predict(
    models: dict,
    request_rate: float,
    max_concurrency: int,
    input_len: int,
    output_len: int,
) -> dict:
    """Query the static profile given workload parameters."""
    x = np.array([[request_rate, max_concurrency, input_len, output_len]])
    return {target: float(model.predict(x)[0]) for target, model in models.items()}


if __name__ == "__main__":
    df = load_and_clean(PROFILE_CSV)
    print(f"Loaded {len(df)} rows from sweep")
    models = fit_and_save(df)

    # Smoke test
    example = predict(models, request_rate=4.0, max_concurrency=8,
                      input_len=512, output_len=256)
    print("\nExample static prediction (rr=4, mc=8, in=512, out=256):")
    for k, v in example.items():
        print(f"  {k:45s}: {v:.4f}")