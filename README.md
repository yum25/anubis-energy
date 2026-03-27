# anubis-energy

> weighs the energy tradeoffs of static vs runtime profiling

## Tests

To run local tests that mock the CloudLab environment (no GPUs, no vLLM models) run:

```sh
pip install scikit-learn pandas numpy joblib

python test_static_profile.py   # Layer 1: no deps beyond sklearn/pandas
python test_runtime_profiler.py # Layer 2: tests profiler with mock monitors  
python test_comparison.py       # Layer 3: tests the actual decision logic
```