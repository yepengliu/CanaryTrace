#!/bin/bash
#SBATCH --job-name=ct_eval
#SBATCH --output=log/ct_eval_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --mem=128gb
#SBATCH --time=1-00:00:00
#
# No --partition directive is set: pass your cluster's partition on the command
# line, e.g. `sbatch --partition=<gpu-partition> ...`.
#
# Stage 3 (RA-LLM response generation + watermark scoring). Usage:
#   sbatch --export=ALL,CONFIG=main_nfcorpus scripts/slurm/evaluation.sh

set -e
CONFIG="${CONFIG:-main_nfcorpus}"
ENV="${ENV:-canarytrace}"
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

command -v module >/dev/null 2>&1 && module load cuda || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

PYTHONPATH="$ROOT" python -m canarytrace.evaluation --config "$CONFIG"
