#!/usr/bin/env bash
# Setup script for LLM inference energy experiments on a CloudLab V100S node.
# Tested on Ubuntu 20.04/22.04 with CUDA 11.8.
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash setup.sh [--skip-profiling] [--skip-vllm]
#
#   --skip-profiling  Skip the 60-90 min static profiling run (use if static_profile.json
#                     already exists or you only need the simulator path).
#   --skip-vllm       Skip vLLM installation (use if static profiling is not needed).

set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
SKIP_PROFILING=false
SKIP_VLLM=false
for arg in "$@"; do
  case $arg in
    --skip-profiling) SKIP_PROFILING=true ;;
    --skip-vllm)      SKIP_VLLM=true; SKIP_PROFILING=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()   { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_ROOT/.venv"

# ── 0. Pre-flight checks ──────────────────────────────────────────────────────
info "Pre-flight checks..."

# Must be run as a user with sudo (not root directly, because nvidia-smi -ac
# works fine via sudo but some CUDA tools behave oddly as root).
if [[ $EUID -eq 0 ]]; then
  warn "Running as root. Some NVIDIA tools behave oddly as root; prefer a sudo user."
fi

command -v nvidia-smi &>/dev/null || die "nvidia-smi not found. Are NVIDIA drivers installed?"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
info "Detected GPU: $GPU_NAME"
if ! echo "$GPU_NAME" | grep -qi "V100"; then
  warn "Expected a V100S but detected '$GPU_NAME'. Continuing anyway."
fi

# CUDA version check — need 11.8 or 12.1
CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)
info "Driver-reported CUDA version: ${CUDA_VER:-unknown}"

# HF_TOKEN is required for static profiling (Llama-2 gated model)
if [[ "$SKIP_PROFILING" == false ]] && [[ -z "${HF_TOKEN:-}" ]]; then
  die "HF_TOKEN is not set. Export it before running:\n  export HF_TOKEN=hf_..."
fi

# ── 1. System packages ────────────────────────────────────────────────────────
info "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  build-essential git curl wget ca-certificates \
  python3-pip python3-venv python3-dev \
  libssl-dev libffi-dev \
  lsof \
  2>/dev/null

# Python ≥ 3.10 is required. Ubuntu 20.04 ships 3.8 by default.
PYTHON_BIN=$(command -v python3.11 || command -v python3.10 || command -v python3 || true)
PYTHON_VER=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

if python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
  PYTHON_BIN=$(command -v python3)
  info "Python ${PYTHON_VER} already available at $PYTHON_BIN"
else
  info "Python < 3.10 detected (${PYTHON_VER}). Installing Python 3.11 via deadsnakes PPA..."
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update -qq
  sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
  PYTHON_BIN=$(command -v python3.11)
  info "Installed Python 3.11 at $PYTHON_BIN"
fi

# ── 2. NVIDIA persistence daemon (required for nvidia-smi -ac freq control) ──
info "Enabling nvidia-persistenced..."
sudo systemctl enable nvidia-persistenced 2>/dev/null || true
sudo systemctl start  nvidia-persistenced 2>/dev/null || \
  sudo nvidia-persistenced --no-fork 2>/dev/null || \
  warn "Could not start nvidia-persistenced. Frequency control may not persist between calls."

# Verify -ac (application clocks) works — requires persistence mode to be on
sudo nvidia-smi -pm 1 &>/dev/null && info "Persistence mode ON" || warn "Could not enable persistence mode"

# ── 3. Virtual environment ────────────────────────────────────────────────────
info "Creating virtual environment at $VENV..."
"$PYTHON_BIN" -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel setuptools -q

# ── 4. PyTorch (CUDA 11.8 build, compatible with V100 CC 7.0) ─────────────────
# PyTorch 2.3.0+cu118 is the newest torch that ships cu118 wheels and is
# required by vLLM 0.4.3.  Must be installed before vLLM so vLLM's own
# torch requirement is already satisfied and it won't pull a CPU-only build.
info "Installing PyTorch 2.3.0 (CUDA 11.8)..."
pip install \
  torch==2.3.0+cu118 \
  torchvision==0.18.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118 \
  -q

# Quick sanity check
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available after torch install"
cc = torch.cuda.get_device_capability()
print(f"  torch {torch.__version__}  |  CUDA available  |  CC {cc[0]}.{cc[1]}")
if cc < (7, 0):
    raise RuntimeError(f"Compute capability {cc} is below the required 7.0 for V100S")
EOF
info "PyTorch CUDA check passed."

# ── 5. Project Python dependencies ───────────────────────────────────────────
info "Installing requirements.txt..."
pip install -r "$REPO_ROOT/requirements.txt" -q

# ── 6. vLLM 0.4.3 (skip if --skip-vllm) ─────────────────────────────────────
if [[ "$SKIP_VLLM" == false ]]; then
  info "Installing vLLM 0.4.3 (this may take a few minutes)..."
  # vLLM 0.4.3 is the last release with V100 (CC 7.0) support.
  # The prebuilt wheel targets sm_70+, so no source build is needed.
  pip install vllm==0.4.3 -q

  python - <<'EOF'
from vllm import LLM
print("  vLLM import OK")
EOF
  info "vLLM install verified."
else
  info "Skipping vLLM installation (--skip-vllm)."
fi

# ── 7. Hugging Face CLI login ─────────────────────────────────────────────────
if [[ "$SKIP_PROFILING" == false ]]; then
  info "Logging into Hugging Face (needed for Llama-2-7b-hf)..."
  pip install huggingface_hub -q
  python -c "
from huggingface_hub import login
import os
login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
print('  HF login OK')
"
fi

# ── 8. One-time profile calibration ──────────────────────────────────────────
# This generates profilers/profiles.py from embedded ML.Energy constants.
# Idempotent: safe to re-run.
info "Running profile calibration (generates profilers/profiles.py)..."
python "$REPO_ROOT/scripts/calibrate_profiles.py"
info "Profile calibration done."

# ── 9. Static profiling pipeline (60-90 min, real hardware) ──────────────────
STATIC_DIR="$REPO_ROOT/static_profiling"
PROFILE_JSON="$STATIC_DIR/static_profile.json"
PARETO_JSON="$STATIC_DIR/static_pareto.json"

if [[ "$SKIP_PROFILING" == true ]]; then
  info "Skipping static profiling (--skip-profiling)."
  if [[ ! -f "$PARETO_JSON" ]]; then
    warn "static_pareto.json not found. The static scheduler will not work until"
    warn "you run: python static_profiling/run_static_profiling.py"
    warn "      then: python static_profiling/build_static_models.py"
  fi
else
  if [[ -f "$PROFILE_JSON" ]]; then
    warn "static_profile.json already exists — skipping benchmark run."
    warn "Delete it and re-run this script to re-profile."
  else
    info "Starting static profiling benchmark (24 configs × 100 requests each)."
    info "Expected runtime: 60-90 minutes. Output: $PROFILE_JSON"
    python "$REPO_ROOT/static_profiling/run_static_profiling.py"
    info "Static profiling complete."
  fi

  if [[ -f "$PROFILE_JSON" ]] && [[ ! -f "$PARETO_JSON" ]]; then
    info "Building static models and Pareto table..."
    python "$REPO_ROOT/static_profiling/build_static_models.py"
    info "Static models built: static_pareto.json, *.pkl"
  fi
fi

# ── 10. Smoke test ────────────────────────────────────────────────────────────
info "Running simulator smoke test..."
python - <<'EOF'
import sys, os
sys.path.insert(0, os.environ.get("REPO_ROOT", "."))
from profilers.simulator import SimulatedGpuDataSource
from schedulers import RuntimeScheduler

src = SimulatedGpuDataSource(gpu_index=0, seed=42)
sched = RuntimeScheduler()
obs = src.observe(interval_s=10.0, freq_mhz=1377, max_num_seqs=8)
dec = sched.decide(obs)
print(f"  obs: temp={obs.gpu_temp_c[0]:.1f}°C  ept={obs.energy_per_token_j:.4f} J/tok")
print(f"  dec: freq={dec.freq_mhz} MHz  reason={dec.reason}")
print("  Simulator smoke test PASSED")
EOF
export REPO_ROOT="$REPO_ROOT"

# ── 11. Write .env for future sessions ───────────────────────────────────────
ENV_FILE="$REPO_ROOT/.env"
info "Writing $ENV_FILE for future sessions..."
cat > "$ENV_FILE" <<ENVEOF
# Source this file before running experiments:
#   source .env
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:\$PATH"
export HF_TOKEN="${HF_TOKEN:-}"
export REPO_ROOT="$REPO_ROOT"
ENVEOF
chmod 600 "$ENV_FILE"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Activate the environment:"
echo "  source $VENV/bin/activate"
echo "  # or: source .env"
echo ""
echo "Run a simulation experiment:"
echo "  python experiment.py --mode sim --scheduler both"
echo ""
if [[ "$SKIP_PROFILING" == false ]] && [[ -f "$PARETO_JSON" ]]; then
echo "Run a real-hardware experiment (requires vLLM server running):"
echo "  python experiment.py --mode real --scheduler static"
echo ""
fi
echo "Run the full static profiling pipeline manually:"
echo "  python static_profiling/run_static_profiling.py"
echo "  python static_profiling/build_static_models.py"
