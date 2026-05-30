#!/bin/bash
#SBATCH --job-name=cassava_data
#SBATCH --partition=debug
#SBATCH --nodelist=adep1,romer3,ubem,dl4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/data_prep_%j.out
#SBATCH --error=logs/data_prep_%j.err

# ============================================================
# Cassava Leaf Disease Detector — Data Download & Preprocessing
# Downloads data from Google Drive, creates train/val/test splits,
# and tracks data artifacts with DVC.
#
# Usage:
#   mkdir -p logs && sbatch scripts/download_and_preprocess.sbatch
#
# Node preference: adep1 > romer3 > ubem > dl4
# ============================================================

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

echo "============================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Start Time:   $(date)"
echo "============================================"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "[Step 1/4] Downloading dataset from Google Drive..."
uv run python commands.py download

echo "[Step 2/4] Preprocessing (train/val/test split)..."
uv run python commands.py preprocess

echo "[Step 3/4] Tracking raw data with DVC..."
uv run dvc add data/raw
echo "  → data/raw.dvc created"

echo "[Step 4/4] Tracking processed data with DVC..."
uv run dvc add data/processed
echo "  → data/processed.dvc created"

echo ""
echo "============================================"
echo "Data preparation complete."
echo ""
echo "Next steps:"
echo "  1. git add data/raw.dvc data/processed.dvc data/.gitignore"
echo "  2. git commit -m 'Track data artifacts with DVC'"
echo "  3. uv run dvc push"
echo ""
echo "End Time: $(date)"
echo "============================================"
