# GSM8K — 3-axis evaluation

First validation target for the SkillEval pipeline. Grade-school math
word problems with short integer answers; exact-match evaluation.

## Run

```bash
python datasets/gsm8k/run.py
```

Outputs land in `datasets/gsm8k/outputs/`:
- `results.json` — per-cell results via `JSONReporter`
- `results.csv` — tabular via `CSVReporter`
- Library snapshots under `outputs/libraries/gsm8k/SD/` (project root):
  `library.json` (initial), `library_round{1,2}.json` (BU), `library_final.json`

## Config knobs (edit `config.yaml`)

- `experiment.max_episodes` — tasks per axis cell (5 smoke, 200 full)
- `experiment.max_steps_per_episode` — agent steps per task (15 plenty for math)
- `experiment.num_evolution_rounds` — BU rounds (3 default)
- `model.model_id` — `gpt-4o-mini` for quick runs, `gpt-4o` for reference numbers

## Evaluation

GSM8K ships with gold answers extracted from the `#### N` marker in the
HuggingFace split. `evaluate_prediction` normalizes currency / punctuation
and falls back to "last number in the predicted string" so the agent's
free-form replies can still be matched.

Returns `{"success": 0|1, "exact_match": 0|1}`. The evaluator reads
`success` to set `trace.success`.

## Train/test split

- Cat 1 & 2: test split only (`split: test`).
- Cat 3 (BU evolution): for published numbers, switch `split: train` and run
  rounds on `train[:N]`, then re-run FR on `test[:N]` for final eval. For
  smoke/internal iteration, test split is fine.

## Tools

GSM8K declares `tools_required: [calculator]` but the protocol returns a
stub for calculator — strong models do arithmetic in CoT. If local 3B
model numbers suffer, wire a real `calculator` in a new `tools.py` here
and override `GSM8KDataset.get_tool_executors()`. No core changes needed.
