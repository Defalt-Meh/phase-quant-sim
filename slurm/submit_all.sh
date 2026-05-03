#!/bin/bash
# Convenience wrapper: submits the full pipeline as a single SLURM job.
# Usage:  bash slurm/submit_all.sh
#
# For local (non-SLURM) execution, run instead:
#   bash slurm/run.sbatch
#
# To rerun only a subset of experiments, comment lines in slurm/run.sbatch.

set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p out/logs

if command -v sbatch > /dev/null 2>&1; then
    echo "Submitting via sbatch from $REPO_DIR..."
    sbatch slurm/run.sbatch
else
    echo "sbatch not found; running locally."
    bash slurm/run.sbatch
fi