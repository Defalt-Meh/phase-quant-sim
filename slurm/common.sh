#!/bin/bash
# Shared environment setup, sourced by run.sbatch.
# Defines REPO_DIR, VENV_PATH, and runs pip refresh.

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$HOME/phase-quant-sims}"
VENV_PATH="${VENV_PATH:-$HOME/venv/bin/activate}"

cd "$REPO_DIR"
mkdir -p out/{exp01,exp02,exp03,exp04,exp05,exp06,exp07,exp08,exp09,exp10,exp11,exp12,plots,logs}

module load python/3.11.9 || true
source "$VENV_PATH"

PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
SITE_PKGS="$VIRTUAL_ENV/lib/python${PY_VER}/site-packages"

echo "Cleaning broken pip state from: $SITE_PKGS"
rm -rf "$SITE_PKGS"/pip \
       "$SITE_PKGS"/pip-*.dist-info \
       "$SITE_PKGS"/~ip \
       "$SITE_PKGS"/~ip-*.dist-info || true

python -m ensurepip --upgrade --default-pip
python -m pip install --upgrade --force-reinstall "pip==24.3.1" setuptools wheel

python -m pip install \
    "numpy>=1.26" \
    "scipy>=1.11" \
    "pandas>=2.0" \
    "pyarrow>=12.0" \
    "matplotlib>=3.8" \
    "pyyaml>=6.0" \
    "torch>=2.2" \
    "tqdm>=4.66"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

echo "================================"
echo "Hostname : $(hostname)"
echo "Python   : $(which python) ($(python --version))"
echo "Repo     : $REPO_DIR"
echo "================================"