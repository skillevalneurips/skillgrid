#!/bin/bash
# Run Qwen3-4B-Instruct on full AIME25 test set with existing skills
# Usage: nohup bash run_qwen3_4b_instruct_full.sh > run_qwen3_4b_instruct_full.log 2>&1 &
#
# Environment setup (run once):
#   conda env create -f environment.yaml
#   conda activate skilleval
#   pip install -e .

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate skilleval

cd "$(dirname "$0")"

# Existing skills library (TD = Trace-Derived)
# Path should be parent of skills/ directory for load_auto to work
LIBRARY_PATH="outputs/libraries/Qwen/Qwen3-4B-Instruct-2507/20260421_181047/experiments/FL_PR_FR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Starting Qwen3-4B-Instruct Full AIME Run"
echo "Timestamp: $TIMESTAMP"
echo "Library: $LIBRARY_PATH"
echo "Config: datasets/aime/config_full.yaml"
echo "=========================================="

# Backup original config and use full config
cp datasets/aime/config.yaml datasets/aime/config.yaml.bak
cp datasets/aime/config_full.yaml datasets/aime/config.yaml

# Run with library path (skip skill generation)
# This runs all 3 axes: visibility, retrieval, evolution
# With 30 episodes and 1 evolution round

python datasets/aime/run.py \
    --model qwen3_4b_instruct_vllm \
    --only-origin TD \
    --library-path "$LIBRARY_PATH"

# Restore original config
cp datasets/aime/config.yaml.bak datasets/aime/config.yaml

echo "=========================================="
echo "Run completed at $(date)"
echo "Results saved to: datasets/aime/outputs/"
echo "Debug logs in: outputs/debug/"
echo "=========================================="
