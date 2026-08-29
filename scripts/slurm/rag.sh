#!/bin/bash
#SBATCH --job-name=ct_rag
#SBATCH --output=log/ct_rag_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=64gb
#SBATCH --time=2:00:00
#
# No --partition directive is set: pass your cluster's partition on the command
# line, e.g. `sbatch --partition=<gpu-partition> ...`.
#
# Stage 2 (retrieval). Usage:
#   sbatch --export=ALL,CONFIG=main_nfcorpus scripts/slurm/rag.sh

set -e
CONFIG="${CONFIG:-main_nfcorpus}"
ENV="${ENV:-canarytrace}"
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

command -v module >/dev/null 2>&1 && module load cuda || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

PYTHONPATH="$ROOT" python -m canarytrace.rag --config "$CONFIG"
