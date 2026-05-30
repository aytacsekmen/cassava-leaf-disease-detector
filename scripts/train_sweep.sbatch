#!/bin/bash
#SBATCH --job-name=cassava_sweep
#SBATCH --partition=debug
#SBATCH --nodelist=adep1,romer3,ubem,dl4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

# ============================================================
# Cassava Leaf Disease Detector — Multi-Model Sweep
# Trains 3 architectures concurrently via SLURM array jobs.
#
# Usage:
#   mkdir -p logs && sbatch scripts/train_sweep.sbatch
#
# Models (Hydra config group selection):
#   Array Index 0 → EfficientNetV2-B0  (model=efficientnetv2_b0)
#   Array Index 1 → MobileViT-S        (model=mobilevit_s)
#   Array Index 2 → SwinV2-Tiny        (model=swinv2_tiny)
#
# Node preference: adep1 > romer3 > ubem > dl4
# ============================================================

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

# Hydra config group names (must match filenames under configs/model/)
MODELS=(
    "efficientnetv2_b0"
    "mobilevit_s"
    "swinv2_tiny"
)

MODEL_CONFIG="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "============================================"
echo "SLURM Job ID:       $SLURM_JOB_ID"
echo "SLURM Array Task:   $SLURM_ARRAY_TASK_ID"
echo "Node:               $(hostname)"
echo "Model Config:       $MODEL_CONFIG"
echo "GPU:                $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start Time:         $(date)"
echo "============================================"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

uv run python commands.py train \
    --overrides="['model=${MODEL_CONFIG}']"

echo "============================================"
echo "Training complete for: $MODEL_CONFIG"
echo "End Time:             $(date)"
echo "============================================"
