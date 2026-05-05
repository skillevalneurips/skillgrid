# SkillGrid

Evaluation framework for skill composition in LLM harness agents. SkillGrid
operationalizes three independent design choices that any deployer of a skill
library must make, and provides a per-choice attribution protocol that
isolates the contribution of each.

## What it evaluates

| Design choice | Levels | Question |
|---|---|---|
| **Visibility** (V) | NL / LB / FL | How many skills are placed in the prompt at episode start? |
| **Runtime selection** (ρ) | NR / RR / PR | Who picks the skill at a given step — the acting model, a retriever, or a planner? |
| **Evolution** (ω) | FR / BU | Is the library updated offline between episodes? |

Skill libraries are built before any experiment via one of two origins:

| Origin | Module | How it works |
|---|---|---|
| **Spec-Derived (SD)** | `skilleval/skills/creation/top_down.py` | LLM writes skills from a dataset spec + sample tasks |
| **Trace-Derived (TD)** | `skilleval/skills/creation/bottom_up.py` | LLM generalizes skills from probe interaction traces |

Per-dataset prompts and skill construction details live in
`datasets/<name>/skills/recipe.py`.

## Datasets

| Dataset | Domain | Source |
|---|---|---|
| AIME 2024 / 2025, AMC 23 | Competition mathematics | HuggingFace |
| GAIA | Open-web reasoning + tool use | HuggingFace (`gaia-benchmark/GAIA`) |
| WebWalkerQA | Web navigation | HuggingFace (`callanwu/WebWalkerQA`) |
| ReDial, Reddit-V2 | Conversational recommendation | Local download (see `datasets/conversational_rec/RUNBOOK.md`) |

## Backbones

OpenAI (`gpt-4o`, `gpt-4o-mini`, `gpt-5-mini`), local vLLM
(`Qwen3-4B-Instruct-2507`, `Qwen2.5-7B-Instruct`, etc.), HuggingFace
Transformers, and any OpenAI-compatible endpoint. Add a backbone by
dropping a model file under `datasets/<name>/models/`.

## Repository layout

```
configs/                 # Shared experiment configs (base.yaml + per-dataset overrides)
datasets/                # Per-dataset adapters, run.py entry points, model files, skill recipes
  ├── aime/              # Competition math (AIME24/25, AMC23)
  ├── conversational_rec/  # ReDial + Reddit-V2
  ├── gaia/              # GAIA tasks via HuggingFace
  └── webwalkerqa/       # WebWalkerQA
skilleval/               # Core framework
  ├── agents/            # Skill agent and protocols
  ├── datasets/          # BaseDataset and per-dataset loaders
  ├── evaluation/        # Evaluator, metrics, axis mapping
  ├── models/            # Model backends (OpenAI, Anthropic, Google, vLLM, HF)
  └── skills/            # Library, creation, retrieval, evolution
libraries/               # Pre-built skill libraries that ship with the repo
gaia/                    # GAIA-specific scripts and skill files
webwalker/               # WebWalker source code (upstream + local extensions)
```

## Installation

```bash
conda env create -f environment.yaml
conda activate skilleval
pip install -e .
```

API keys are read from environment variables: `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`. A local `.env` file at the repo root
is also auto-loaded by the dataset runners.

## Running experiments

Each dataset has its own runner under `datasets/<name>/run.py` that sweeps
the V × ρ × ω grid for all registered models.

### Math (AIME / AMC)

```bash
python datasets/aime/run.py --model qwen3_4b_instruct_vllm \
    --config config_aime24.yaml \
    --library-path libraries/openr1_math_skills \
    --only-origin TD
```

The shipped helper scripts run AIME24 / AIME25 / AMC23 sequentially:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_qwen3_4b.sh
CUDA_VISIBLE_DEVICES=1 ./run_qwen25_7b.sh
```

### GAIA

```bash
python datasets/gaia/run.py --model gpt4o_mini
```

### WebWalkerQA

```bash
python datasets/webwalkerqa/run.py --model gpt4o_mini
```

### Conversational recommendation

```bash
python datasets/conversational_rec/run.py \
    --mode axis_sweep --model gpt4o-mini --only-origin all
```

See `datasets/conversational_rec/RUNBOOK.md` for the full run book (ReDial
preparation, mixed-source configs, single-axis runs, full 18-cell grid).

### Common flags

```text
--mode {axis_sweep, single_axis, full_grid}
--only-axis {visibility, retrieval, evolution}
--only-origin {SD, TD, all}
--library-path PATH        # use a pre-built library, skip skill creation
--max-episodes N           # cap eval pool, useful for smoke tests
```

## Outputs

Every run writes timestamped artifacts under `outputs/`:

```
outputs/
├── libraries/<model>/<timestamp>/                  # Skill library snapshots per cell
├── results/<dataset>/<model>/<timestamp>/          # Per-episode JSON + summary CSV
└── reasoning_traces/<dataset>/                     # Probe traces used by TD construction
```

Logs land under `logs/` (gitignored).

## Per-choice protocol

For each design choice $a \in \{V, \rho, \omega\}$ and each alternative
instantiation $a'$, SkillGrid estimates the per-choice effect

$$\Delta_{a \to a'} = J(\theta^{\circ}\big|_{a=a'}) - J(\theta^{\circ})$$

where $\theta^{\circ} = (\mathrm{FL}, \mathrm{NR}, \mathrm{FR})$ is the
canonical reference and $J$ is mean task success on the dataset's eval pool.
This produces 8 ablation cells per backbone × benchmark pair (3 visibility
+ 3 retrieval + 2 evolution, with the shared FL/NR/FR cell de-duplicated).
