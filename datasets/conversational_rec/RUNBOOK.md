# Conversational Recommendation Taxonomy Runbook

This runbook covers the recommendation-only taxonomy runs for Reddit-V2,
ReDial, and mixed-source experiments.

## Setup

Run from the repository root:

```bash
conda activate skilleval
```

Default dataset config:

```bash
datasets/conversational_rec/config.yaml
```

Default data path (override in config if your local layout differs):

```bash
data/reddit_v2_unzipped
```

The default config uses Reddit-V2 `small`, `clean_with_titles`, and native
`test` tasks for evaluation. With pre-generated skill libraries, the main
axis-sweep configs do not load train tasks.

## ReDial Data

ReDial uses the upstream preprocessed CSVs from
`zhouhanxie/neighborhood-based-CF-for-CRS`. Download them into the ignored
local data directory:

```bash
python datasets/conversational_rec/redial/prepare.py
```

Use `--force` to replace existing local copies:

```bash
python datasets/conversational_rec/redial/prepare.py --force
```

The ReDial-only config is:

```bash
datasets/conversational_rec/config_redial.yaml
```

ReDial test rows follow the upstream flattened format: each row has one
`test_outputs` movie, so each loaded test task has a single gold title.

Smoke configs cap train/test rows to keep `BU` taxonomy cells fast:

```bash
datasets/conversational_rec/config_smoke.yaml
datasets/conversational_rec/config_redial_smoke.yaml
```

## Reasoning Traces For Skill Authoring

Before running `LB` or `FL` cells with hand-authored skills, generate source
specific traces from train examples and use those traces to author the initial
skill library. The trace script samples Reddit-V2 and ReDial train tasks,
prompts `Qwen/Qwen3-4B-Instruct-2507` for concise step-by-step recommendation
reasoning plus final recommendation JSON, and writes one JSON artifact per
source.

Dry-run dataset loading and prompt rendering without loading the model:

```bash
python datasets/conversational_rec/generate_reasoning_traces.py \
  --samples-per-source 2 \
  --candidate-pool-size 5 \
  --dry-run
```

Generate 200 traces per source on GPU 4:

```bash
CUDA_VISIBLE_DEVICES=4 python datasets/conversational_rec/generate_reasoning_traces.py \
  --samples-per-source 200 \
  --candidate-pool-size 2000 \
  --batch-size 8 \
  --max-new-tokens 1024
```

Outputs:

```bash
datasets/conversational_rec/outputs/reasoning_traces/redditv2_qwen3_4b_instruct_2507_traces.json
datasets/conversational_rec/outputs/reasoning_traces/redial_qwen3_4b_instruct_2507_traces.json
```

The script resumes existing output files by task id. Add `--overwrite` to
regenerate from scratch.

## What The Axes Mean

Skill creation origin:

- `SD`: spec-derived skills, written from dataset/tool specification plus train examples.
- `TD`: trace-derived skills, written from probe traces on train tasks.

Visibility axis:

- `NL`: no visible library.
- `LB`: seeded random limited skill bundle, 2 skills by default.
- `FL`: full skill library.

Retrieval axis:

- `NR`: no runtime retrieval beyond visible skills.
- `RR`: retrieve-route using skill descriptions.
- `PR`: planner selects a skill by name/description, executor uses the selected template.

Evolution axis:

- `FR`: frozen library.
- `BU`: one offline batch update from eval traces, matching the AIME runner
  behavior, then evaluation on the same eval pool after updating. The updater
  rewrites existing skills from a 70/30 failure/success trace sample.

Protocols:

- `in_context`: default recommendation JSON protocol.
- `react`: recommendation-oriented Thought/Action loop using `<fetch_skill>` and `<answer>{...}</answer>`.

Metrics:

- Primary: `hit_at_k`
- Secondary: `recall_at_k`, `ndcg_at_k`
- Default `k`: `10`

## Quick Smoke Runs

Run unit and dataset-load checks before GPU jobs:

```bash
python -m pytest -q tests/test_conversational_rec.py

python - <<'PY'
from skilleval.core.config import Config
from skilleval.datasets.conversational_rec import ConversationalRecDataset

for path in [
    "datasets/conversational_rec/config.yaml",
    "datasets/conversational_rec/config_redial.yaml",
]:
    cfg = Config.from_yaml(path)
    ds = ConversationalRecDataset(cfg)
    print(path, len(ds.train_tasks()), len(ds.test_tasks()))
PY
```

Run a ReDial-only smoke after data preparation:

```bash
python datasets/conversational_rec/run.py \
  --config datasets/conversational_rec/config_redial.yaml \
  --mode single_axis \
  --only-axis visibility \
  --model gpt4o-mini \
  --only-origin SD \
  --max-episodes 3
```

### GPU 4 Full-Taxonomy Smoke

Use GPU 4 for local vLLM smoke tests. These commands run every visibility,
retrieval, and evolution taxonomy cell with one evaluation episode per cell.
They use the unbounded `qwen2.5-7b` preset because the Qwen3 preset advertises
a 262k-token context that can exceed available KV cache during vLLM startup.
They use `SD` only to keep the smoke bounded; change `--only-origin all` for
the full SD+TD origin sweep.

Reddit-V2:

```bash
CUDA_VISIBLE_DEVICES=4 python datasets/conversational_rec/run.py \
  --config datasets/conversational_rec/config_smoke.yaml \
  --mode full_grid \
  --model qwen2.5-7b \
  --only-origin SD \
  --max-episodes 1 \
  --output-dir datasets/conversational_rec/outputs/smoke_redditv2_gpu4
```

ReDial:

```bash
CUDA_VISIBLE_DEVICES=4 python datasets/conversational_rec/run.py \
  --config datasets/conversational_rec/config_redial_smoke.yaml \
  --mode full_grid \
  --model qwen2.5-7b \
  --only-origin SD \
  --max-episodes 1 \
  --output-dir datasets/conversational_rec/outputs/smoke_redial_gpu4
```

Run a tiny visibility sweep with GPT-4o-mini and SD skills:

```bash
python datasets/conversational_rec/run.py \
  --mode single_axis \
  --only-axis visibility \
  --model gpt4o-mini \
  --only-origin SD \
  --max-episodes 3
```

Run the same with ReAct:

```bash
python datasets/conversational_rec/run.py \
  --mode single_axis \
  --only-axis visibility \
  --protocol react \
  --model gpt4o-mini \
  --only-origin SD \
  --max-episodes 3
```

## Axis Sweeps

Axis sweep is the paper-style compact sweep:

- Visibility sweep: `NL/LB/FL`, holding `Retrieval=NR`, `Evolution=FR`
- Retrieval sweep: `NR/RR/PR`, holding `Visibility=FL`, `Evolution=FR`
- Evolution sweep: `FR/BU`, holding `Visibility=FL`, `Retrieval=NR`

The overlapping `FL/NR/FR` cell is deduplicated in output, so this produces 6 unique cells per model/origin.
This is the default `run.py` mode and matches the AIME runner behavior.

With a pre-generated TD library:

```bash
python datasets/conversational_rec/run.py \
  --library-path datasets/conversational_rec/skills/generated/redditv2 \
  --model qwen2.5-7b \
  --only-origin TD
```

### SD, Default Protocol

```bash
python datasets/conversational_rec/run.py \
  --mode axis_sweep \
  --model gpt4o-mini \
  --only-origin SD
```

### TD, Default Protocol

```bash
python datasets/conversational_rec/run.py \
  --mode axis_sweep \
  --model gpt4o-mini \
  --only-origin TD
```

### SD + TD

```bash
python datasets/conversational_rec/run.py \
  --mode axis_sweep \
  --model gpt4o-mini \
  --only-origin all
```

### ReAct Axis Sweep

```bash
python datasets/conversational_rec/run.py \
  --mode axis_sweep \
  --protocol react \
  --model gpt4o-mini \
  --only-origin all
```

### Mixed Reddit-V2 + ReDial

The mixed config appends train and test tasks from both sources:

```bash
python datasets/conversational_rec/run.py \
  --config datasets/conversational_rec/config_mixed.yaml \
  --mode axis_sweep \
  --model gpt4o-mini \
  --only-origin all
```

## Single-Axis Runs

Visibility only:

```bash
python datasets/conversational_rec/run.py \
  --mode single_axis \
  --only-axis visibility \
  --model gpt4o-mini \
  --only-origin SD
```

Retrieval only:

```bash
python datasets/conversational_rec/run.py \
  --mode single_axis \
  --only-axis retrieval \
  --model gpt4o-mini \
  --only-origin SD
```

Evolution only:

```bash
python datasets/conversational_rec/run.py \
  --mode single_axis \
  --only-axis evolution \
  --model gpt4o-mini \
  --only-origin SD
```

## Full 18-Cell Taxonomy Grid

This runs every combination:

```text
Visibility: NL, LB, FL
Retrieval:  NR, RR, PR
Evolution:  FR, BU
```

That is `3 x 3 x 2 = 18` cells for each model/origin.

```bash
python datasets/conversational_rec/run.py \
  --mode full_grid \
  --model gpt4o-mini \
  --only-origin SD
```

Full grid for both SD and TD:

```bash
python datasets/conversational_rec/run.py \
  --mode full_grid \
  --model gpt4o-mini \
  --only-origin all
```

Full grid with ReAct:

```bash
python datasets/conversational_rec/run.py \
  --mode full_grid \
  --protocol react \
  --model gpt4o-mini \
  --only-origin all
```

## Model Choices

Preset model names:

- `gpt4o-mini`
- `qwen3-4b`
- `qwen2.5-7b`
- `all`

Run all three model presets on the compact axis sweep:

```bash
python datasets/conversational_rec/run.py \
  --mode axis_sweep \
  --model all \
  --only-origin all
```

Run Qwen3 only:

```bash
python datasets/conversational_rec/run.py \
  --mode axis_sweep \
  --model qwen3-4b \
  --only-origin SD
```

Run Qwen2.5 only:

```bash
python datasets/conversational_rec/run.py \
  --mode axis_sweep \
  --model qwen2.5-7b \
  --only-origin SD
```

For custom model config:

```bash
python datasets/conversational_rec/run.py \
  --mode axis_sweep \
  --model custom \
  --model-provider openai \
  --model-config configs/models/gpt4o_mini.yaml \
  --only-origin SD
```

## Recommended Run Order

1. Smoke test default SD:

```bash
python datasets/conversational_rec/run.py --mode single_axis --only-axis visibility --model gpt4o-mini --only-origin SD --max-episodes 3
```

2. Smoke test ReAct SD:

```bash
python datasets/conversational_rec/run.py --mode single_axis --only-axis visibility --protocol react --model gpt4o-mini --only-origin SD --max-episodes 3
```

3. Run compact taxonomy for one model:

```bash
python datasets/conversational_rec/run.py --mode axis_sweep --model gpt4o-mini --only-origin all
```

4. Run compact taxonomy for all models:

```bash
python datasets/conversational_rec/run.py --mode axis_sweep --model all --only-origin all
```

5. Run full 18-cell grid only after the compact sweep is stable:

```bash
python datasets/conversational_rec/run.py --mode full_grid --model all --only-origin all
```

## Outputs

Default output directory:

```bash
datasets/conversational_rec/outputs
```

Each run writes JSON and CSV summaries. Skill libraries and snapshots are written under:

```bash
outputs/libraries/
```

## Notes

- `BU` uses the eval/test pool for updates, matching AIME's default behavior.
- `LB` uses seeded random selection by default, controlled by `experiment.lb_selection`, `experiment.lb_sample_size` (2 in the shipped configs), and `experiment.react.lb_seed`.
- `RR` retrieves over skill descriptions by default, controlled by `experiment.retrieval.description_only`.
- ReAct is recommendation-oriented for this dataset and expects final JSON inside `<answer>...</answer>`.
- Keep `--max-episodes` small for smoke tests; remove it for configured evaluation size.
