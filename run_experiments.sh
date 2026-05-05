#!/usr/bin/env bash
# ============================================================================
# SkillEval-Bench: Run Experiments
# ============================================================================
#
# This script demonstrates how to run the full experiment pipeline:
#   1. Install dependencies
#   2. Run baselines (no skills)
#   3. Run skill creation (top-down, bottom-up, hybrid)
#   4. Run skill composition (protocol x policy grid)
#   5. Run the full benchmark grid
#   6. Generate visualizations
#
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh [--step STEP_NAME] [--dataset DATASET] [--model MODEL]
#
# Examples:
#   ./run_experiments.sh                          # Run everything
#   ./run_experiments.sh --step baseline          # Only run baselines
#   ./run_experiments.sh --step full              # Only full benchmark
#   ./run_experiments.sh --dataset gsm8k          # Specific dataset
#   ./run_experiments.sh --step viz               # Only generate plots
# ============================================================================

set -euo pipefail

# -- Defaults ----------------------------------------------------------------
STEP="${STEP:-all}"
DATASET="${DATASET:-gsm8k}"
MODEL_PROVIDER="${MODEL_PROVIDER:-openai}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models/gpt4o.yaml}"
MAX_EPISODES="${MAX_EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# -- Interpreter discovery ----------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        echo "Error: python interpreter not found (tried python3, python)." >&2
        exit 1
    fi
fi

# -- Parse CLI args ----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)       STEP="$2"; shift 2 ;;
        --dataset)    DATASET="$2"; shift 2 ;;
        --model)      MODEL_PROVIDER="$2"; shift 2 ;;
        --config)     MODEL_CONFIG="$2"; shift 2 ;;
        --episodes)   MAX_EPISODES="$2"; shift 2 ;;
        --output)     OUTPUT_DIR="$2"; shift 2 ;;
        --log-level)  LOG_LEVEL="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--step STEP] [--dataset DS] [--model MODEL] [--config YAML]"
            echo ""
            echo "Steps: install, baseline, skills, composition, full, viz, all"
            echo "Datasets: gsm8k, math, gaia, alfworld, appworld, webwalkerqa, conversational_rec"
            echo "Models: openai, anthropic, google, local"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  SkillEval-Bench Experiment Runner"
echo "=============================================="
echo "  Step:     $STEP"
echo "  Dataset:  $DATASET"
echo "  Model:    $MODEL_PROVIDER ($MODEL_CONFIG)"
echo "  Episodes: $MAX_EPISODES"
echo "  Output:   $OUTPUT_DIR"
echo "  Python:   $PYTHON_BIN"
echo "=============================================="

# -- Step 1: Install ---------------------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "install" ]]; then
    echo ""
    echo ">>> Step 1: Installing dependencies..."
    "$PYTHON_BIN" -m pip install -e ".[all]" 2>/dev/null || "$PYTHON_BIN" -m pip install -e . 2>/dev/null || {
        echo "Falling back to requirements.txt..."
        "$PYTHON_BIN" -m pip install -r requirements.txt
    }
    echo "    Done."
fi

# -- Step 2: Baseline -------------------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "baseline" ]]; then
    echo ""
    echo ">>> Step 2: Running baseline (no skills)..."
    "$PYTHON_BIN" experiments/run_baseline.py \
        --dataset "$DATASET" \
        --model "$MODEL_PROVIDER" \
        --config "$MODEL_CONFIG" \
        --max-episodes "$MAX_EPISODES" \
        --max-steps "$MAX_STEPS" \
        --output-dir "$OUTPUT_DIR/baseline" \
        --log-level "$LOG_LEVEL"
    echo "    Baseline results in $OUTPUT_DIR/baseline/"
fi

# -- Step 3: Skill Creation -------------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "skills" ]]; then
    echo ""
    echo ">>> Step 3: Running skill creation (all strategies)..."
    "$PYTHON_BIN" experiments/run_skill_creation.py \
        --dataset "$DATASET" \
        --model "$MODEL_PROVIDER" \
        --config "$MODEL_CONFIG" \
        --strategy all \
        --output-dir "$OUTPUT_DIR/skill_creation" \
        --log-level "$LOG_LEVEL"
    echo "    Skill libraries in $OUTPUT_DIR/skill_creation/"
fi

# -- Step 4: Skill Composition -----------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "composition" ]]; then
    echo ""
    echo ">>> Step 4: Running skill composition (protocol x policy grid)..."
    "$PYTHON_BIN" experiments/run_composition.py \
        --dataset "$DATASET" \
        --model "$MODEL_PROVIDER" \
        --config "$MODEL_CONFIG" \
        --protocol all \
        --policy all \
        --max-episodes "$MAX_EPISODES" \
        --max-steps "$MAX_STEPS" \
        --output-dir "$OUTPUT_DIR/composition" \
        --log-level "$LOG_LEVEL"
    echo "    Composition results in $OUTPUT_DIR/composition/"
fi

# -- Step 5: Full Benchmark -------------------------------------------------
if [[ "$STEP" == "full" ]]; then
    echo ""
    echo ">>> Step 5: Running full benchmark grid..."
    "$PYTHON_BIN" experiments/run_full_benchmark.py \
        --config configs/experiments/default.yaml \
        --output-dir "$OUTPUT_DIR" \
        --log-level "$LOG_LEVEL"
    echo "    Full results in $OUTPUT_DIR/"
fi

# -- Step 6: Visualization --------------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "viz" ]]; then
    echo ""
    echo ">>> Step 6: Generating visualizations..."

    RESULTS_FILE=""
    for f in "$OUTPUT_DIR/results.json" "$OUTPUT_DIR/composition/"*.json "$OUTPUT_DIR/baseline/"*.json; do
        if [[ -f "$f" ]]; then
            RESULTS_FILE="$f"
            break
        fi
    done

    if [[ -n "$RESULTS_FILE" ]]; then
        "$PYTHON_BIN" visualization/plot_results.py \
            --results "$RESULTS_FILE" \
            --output-dir "$OUTPUT_DIR/figures"
        echo "    Figures in $OUTPUT_DIR/figures/"

        "$PYTHON_BIN" visualization/generate_tables.py \
            --results "$RESULTS_FILE" \
            --output-dir "$OUTPUT_DIR/tables"
        echo "    Tables in $OUTPUT_DIR/tables/"
    else
        echo "    No results file found; skipping visualization."
        echo "    Run experiments first, then: ./run_experiments.sh --step viz"
    fi
fi

echo ""
echo "=============================================="
echo "  Experiments complete!"
echo "  Results: $OUTPUT_DIR/"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  - View results: cat $OUTPUT_DIR/results.json"
echo "  - Interactive dashboard: streamlit run visualization/dashboard.py"
echo "  - Generate LaTeX tables: python visualization/generate_tables.py"
