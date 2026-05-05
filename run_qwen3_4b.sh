#!/bin/bash
# Run Qwen3-4B-Instruct-2507 on AIME24, AIME25, and AMC23 sequentially
# Usage: CUDA_VISIBLE_DEVICES=0 ./run_qwen3_4b.sh

set -e

cd "$(dirname "$0")"

LIBRARY_PATH="libraries/openr1_math_skills"
MODEL="qwen3_4b_instruct_vllm"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Starting Qwen3-4B-Instruct experiments"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

echo ""
echo "[1/3] Running AIME24..."
python datasets/aime/run.py \
    --model "$MODEL" \
    --config config_aime24.yaml \
    --library-path "$LIBRARY_PATH" \
    --only-origin TD \
    2>&1 | tee "${LOG_DIR}/${MODEL}_aime24.log"

echo ""
echo "[2/3] Running AIME25..."
python datasets/aime/run.py \
    --model "$MODEL" \
    --config config_aime25.yaml \
    --library-path "$LIBRARY_PATH" \
    --only-origin TD \
    2>&1 | tee "${LOG_DIR}/${MODEL}_aime25.log"

echo ""
echo "[3/3] Running AMC23..."
python datasets/aime/run.py \
    --model "$MODEL" \
    --config config_amc23.yaml \
    --library-path "$LIBRARY_PATH" \
    --only-origin TD \
    2>&1 | tee "${LOG_DIR}/${MODEL}_amc23.log"

echo ""
echo "=========================================="
echo "Qwen3-4B-Instruct experiments complete!"
echo "=========================================="
