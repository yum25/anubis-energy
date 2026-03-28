# build_static_models.py
import json, pickle, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

with open("static_profile.json") as f:
    data = json.load(f)

quant_map = {"fp16": 0, "int8": 1}

X, y_temp, y_power, y_goodput = [], [], [], []

for r in data:
    X.append([
        r["gpu_freq_mhz"] / 1410,
        r["batch_size"] / 32,
        quant_map[r["quantization"]],
        1 if r["phase"] == "prefill" else 0,
    ])
    y_temp.append(r["gpu_temp_c"])
    y_power.append(r["avg_power_w"])
    # Goodput: throughput only if within SLO, else 0
    slo = r["p99_lat_s"] < 5.0  # define your SLO threshold
    y_goodput.append(r["throughput"] if slo else 0.0)

X = np.array(X)

# Piecewise polynomial is what TAPAS used for <1°C MAE
# degree-2 polynomial is a practical approximation for your scale
temp_model   = Pipeline([("poly", PolynomialFeatures(2)), ("lr", LinearRegression())]).fit(X, y_temp)
power_model  = Pipeline([("poly", PolynomialFeatures(2)), ("lr", LinearRegression())]).fit(X, y_power)

with open("static_temp_model.pkl",  "wb") as f: pickle.dump(temp_model, f)
with open("static_power_model.pkl", "wb") as f: pickle.dump(power_model, f)

# Also save the raw Pareto frontier for the scheduler to look up
pareto = []
for r, gp in zip(data, y_goodput):
    pareto.append({**r, "goodput": gp})

with open("static_pareto.json", "w") as f:
    json.dump(pareto, f, indent=2)

print(f"Temp R²:  {temp_model.score(X, y_temp):.3f}")
print(f"Power R²: {power_model.score(X, y_power):.3f}")