#!/bin/bash
#SBATCH --job-name=ct_pipeline
#SBATCH --output=log/ct_pipeline_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --mem=128gb
#SBATCH --time=8:00:00
#
# No --partition directive is set: pass your cluster's partition on the command
# line, e.g. `sbatch --partition=<gpu-partition> ...`.
#
# Full pipeline (all five stages) for one config. Usage:
#   sbatch --export=ALL,CONFIG=main_nfcorpus scripts/slurm/pipeline.sh
# Set ENV=... to pick the conda env (default: canarytrace).

set -e
CONFIG="${CONFIG:-main_nfcorpus}"
ENV="${ENV:-canarytrace}"
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

command -v module >/dev/null 2>&1 && module load cuda || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
set -a; [ -f "$ROOT/.env" ] && source "$ROOT/.env"; set +a
export PYTHONPATH="$ROOT"

echo "===== 1. synthesize ====="; python -m canarytrace.synthesize --config "$CONFIG"
echo "===== 1b. prepare  ====="; python -m canarytrace.prepare    --config "$CONFIG"
echo "===== 2. rag       ====="; python -m canarytrace.rag        --config "$CONFIG"
echo "===== 3. evaluation ====="; python -m canarytrace.evaluation --config "$CONFIG"
echo "===== 4. detector  ====="; python -m canarytrace.detector   --config "$CONFIG"
echo "===== PIPELINE COMPLETE ====="
