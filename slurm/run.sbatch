#!/bin/bash
#SBATCH --job-name=PHASE_QUANT
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=out/logs/slurm_%j.log
#SBATCH --error=out/logs/slurm_%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$HOME/phase-quant-sims}"
VENV_PATH="${VENV_PATH:-$HOME/venv/bin/activate}"

echo "================================"
echo "PHASE_QUANT: full simulation pipeline"
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Submit dir : $(pwd)"
echo "Repo dir   : $REPO_DIR"
echo "Venv path  : $VENV_PATH"
echo "Start time : $(date)"
echo "================================"

source "$REPO_DIR/slurm/common.sh"
nvidia-smi || true

CONFIG="configs/sweeps.yaml"
SECONDS=0

# Helper to count rows in a given experiment block of the YAML.
count_rows() {
    python -c "
import yaml, sys
cfg = yaml.safe_load(open('$CONFIG'))['$1']
print(len(cfg['rows']))
"
}

# Helper to read n_seeds for an experiment.
n_seeds() {
    python -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))['$1']
print(cfg['n_seeds'])
"
}

run_experiment() {
    local EXP="$1"
    local NROWS NSEEDS
    NROWS=$(count_rows "$EXP")
    NSEEDS=$(n_seeds "$EXP")
    echo ""
    echo "---- $EXP ($NROWS rows x $NSEEDS seeds) [$(date)] ----"
    local T_START=$SECONDS
    for ((row=0; row<NROWS; row++)); do
        for ((seed=0; seed<NSEEDS; seed++)); do
            python -m "experiments.${EXP}" \
                --config "$CONFIG" --row "$row" --seed "$seed"
        done
    done
    echo "$EXP done in $(( SECONDS - T_START ))s"
}

# ---- Phase 1: Synthetic experiments (CPU, ~30 min) ----
echo ""
echo "==== Phase 1/3: Synthetic experiments (exp01-exp11) ===="

run_experiment exp01_budget_sweep
run_experiment exp02_temperature_sweep
run_experiment exp03_noise
run_experiment exp04_alphabet_heatmap
run_experiment exp05_quantized_keys
run_experiment exp06_worst_case_margin
run_experiment exp07_transition_sharpness
run_experiment exp08_qk_gap
run_experiment exp09_multihead
run_experiment exp10_nonuniform_queries
run_experiment exp11_realistic_noise

echo "Phase 1 done in ${SECONDS}s"

# ---- Phase 2: Toy Transformer (GPU, ~10 min) ----
echo ""
echo "==== Phase 2/3: Toy Transformer (exp12) ===="
T_START=$SECONDS
run_experiment exp12_toy_transformer
echo "Phase 2 done in $(( SECONDS - T_START ))s"

# ---- Phase 3: Plots and verdict (CPU, instant) ----
echo ""
echo "==== Phase 3/3: Plots and verdict ===="
T_START=$SECONDS
python scripts/make_plots.py
python scripts/verdict.py
echo "Phase 3 done in $(( SECONDS - T_START ))s"

echo ""
echo "================================"
echo "ALL PHASES COMPLETE"
echo "Total wall time: ${SECONDS}s"
echo "End time: $(date)"
echo ""
echo "Outputs:"
echo "  Verdict : out/verdict.txt"
echo "  Plots   : out/plots/"
echo "  Logs    : out/logs/"
echo ""
echo "===== VERDICT ====="
cat out/verdict.txt
echo "==================="
echo ""
ls -lh out/plots/*.png 2>/dev/null || true
echo "================================"