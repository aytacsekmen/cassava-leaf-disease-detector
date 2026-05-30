#!/bin/bash
# ============================================================
# Generate comparative plots across all trained model architectures.
#
# Usage:
#   bash scripts/generate_comparison.sh
#
# Prerequisites:
#   - MLflow server running at http://127.0.0.1:8080
#   - At least 2 completed training runs in the experiment
# ============================================================

set -euo pipefail

echo "Generating comparison plots from MLflow..."

uv run python commands.py compare

echo "Comparison plots saved to plots/comparison/"
