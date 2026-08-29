#!/bin/bash
#SBATCH --job-name=ct_detect
#SBATCH --output=log/ct_detect_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32gb
#SBATCH --time=4:00:00
#
# No --partition directive is set: pass your cluster's partition on the command
# line, e.g. `sbatch --partition=<gpu-partition> ...`.
#
# Stage 4 (watermark detection across query quotas). Operates on the stage-3 CSV;
# needs only a tokenizer + the detector, so 1 GPU is plenty. Usage:
#   sbatch --export=ALL,CONFIG=main_nfcorpus scripts/slurm/detector.sh

set -e
CONFIG="${CONFIG:-main_nfcorpus}"
ENV="${ENV:-canarytrace}"
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

command -v module >/dev/null 2>&1 && module load cuda || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

PYTHONPATH="$ROOT" python -m canarytrace.detector --config "$CONFIG"
