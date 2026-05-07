# SkillGrid

Evaluation framework for skill composition in LLM harness agents. SkillGrid
operationalizes three independent design choices that any deployer of a skill
library must make, and provides a per-choice attribution protocol that
isolates the contribution of each.

## What it evaluates

| Design choice | Levels | Question |
|---|---|---|
| **Visibility** (V) | NL / LB / FL | How many skills are placed in the prompt at episode start? |
| **Runtime selection** (ρ) | NR / RR / PR | Who picks the skill — the acting model, a retriever, or a planner? |
| **Evolution** (ω) | FR / BU | Is the library updated offline between episodes? |

Skill libraries can be built via one of two origins:

| Origin | Module | How it works |
|---|---|---|
| **Spec-Derived (SD)** | `skilleval/skills/creation/top_down.py` | LLM writes skills from a dataset spec + sample tasks |
| **Trace-Derived (TD)** | `skilleval/skills/creation/bottom_up.py` | LLM generalizes skills from probe interaction traces |

Per-dataset skill recipes live at `datasets/<name>/skills/recipe.py`.

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
Transformers, and any OpenAI-compatible endpoint.

**Adding a new backbone.** Drop a model file under
`datasets/<name>/models/` and reference it on the CLI via `--model
<filename-without-extension>`.
- For Python factories (AIME, GAIA, WebWalkerQA): create a `.py` file
  exposing a `create_model()` function, e.g. mirror
  `datasets/aime/models/qwen3_4b_instruct_vllm.py`.
- For YAML configs (conversational recommendation): create a `.yaml` file
  with a `model:` block, e.g. mirror
  `datasets/conversational_rec/models/gpt4o_mini.yaml`.

## Repository layout

```
configs/                 # Shared experiment configs (base.yaml + overrides)
datasets/                # Per-dataset adapters, run.py entry points, model files, skill recipes
  ├── aime/              # Competition math (AIME24/25, AMC23)
  ├── conversational_rec/  # ReDial + Reddit-V2 (pre-generated TD skills under skills/generated/)
  ├── gaia/              # GAIA tasks (recipe loads gaia/skills/* by default)
  └── webwalkerqa/       # WebWalkerQA (recipe loads webwalker/skills/* by default)
skilleval/               # Core framework
  ├── agents/            # Skill agent and protocols
  ├── datasets/          # BaseDataset and per-dataset loaders
  ├── evaluation/        # Evaluator, metrics, axis mapping
  ├── models/            # Model backends (OpenAI, Anthropic, Google, vLLM, HF)
  └── skills/            # Library, creation, retrieval, evolution
libraries/               # Pre-built shared libraries (currently: openr1_math_skills)
gaia/skills/             # Pre-built SD library for GAIA (one skill per file type)
webwalker/skills/        # Pre-built SD library for WebWalkerQA (one skill per web subtask)
```

## Installation

The repo ships three install paths. Pick one:

| File | What it gives you | When to use |
|---|---|---|
| `environment.yaml` | Pinned conda env (Python 3.12 + all CUDA/vLLM bits) | Local box with the exact same OS/CUDA stack |
| `environment_portable.yaml` | Looser conda env (no machine-specific build hashes) | Different OS or CUDA version |
| `requirements.txt` | Pure-pip dependency list | You already have a Python 3.10+ environment |

Recommended (full env):

```bash
conda env create -f environment.yaml
conda activate skilleval
pip install -e .
```

Pip-only (skip vLLM if you only plan to run API backbones):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

System prerequisites:
- Python 3.10+ (3.12 is what the conda env pins).
- CUDA 12.x and a recent NVIDIA driver if you plan to run open-weight
  backbones via vLLM. We tested on a single NVIDIA RTX A6000.
- API-only runs need no GPU.

API keys are read from environment variables: `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`. A local `.env` file at the repo root
is also auto-loaded by the dataset runners.

## Configuration

SkillGrid uses a three-layer config hierarchy. The deeper layers extend the
shallower ones via the `_base_:` key.

```
configs/base.yaml                          # global defaults (seed, output_dir, metrics)
   └── datasets/<name>/config*.yaml        # dataset-specific knobs (max_episodes, protocol, max_skills)
          └── datasets/<name>/models/*     # backbone-specific knobs (model_id, temperature, max_tokens)
```

**`configs/base.yaml`** sets project-wide defaults: `project.seed`,
`project.output_dir`, evaluation metrics, default `experiment.budget`, and
default `experiment.max_skills`.

**Dataset configs** at `datasets/<name>/config*.yaml` extend the base and
set things specific to that benchmark:

| Key | Meaning | Where to edit |
|---|---|---|
| `experiment.max_episodes` | Test pool size (per cell) | `datasets/<name>/config*.yaml` |
| `experiment.max_steps_per_episode` | Step budget per episode | same |
| `experiment.max_skills` | Cap on library size | same |
| `experiment.retrieval_bundle_size` | RR/PR top-k | same |
| `experiment.react.lb_sample_size` | Number of skills shown under LB | same |
| `experiment.skill_creation.probe_tasks` | Probe-trace pool size for TD origin | same |
| `experiment.num_evolution_rounds` | BU rounds (default 1 = single offline pass) | same |
| `dataset.test_hf` / `dataset.train_hf` | HF dataset IDs | same |

For AIME, the default `config.yaml` evaluates AIME 2025; use
`config_aime24.yaml` to evaluate AIME 2024 (and pass `--config
config_aime24.yaml` on the CLI). AMC 23 uses `config_amc23.yaml`.

**Model files** under `datasets/<name>/models/` are either Python
(`.py`, instantiated via a `create_model()` factory) or YAML (used for
recommendation runs that route through the unified API client). Edit these
to change `temperature`, `max_tokens`, `gpu_memory_utilization`,
`tensor_parallel_size`, etc.

CLI flags **override** values from these files; the runner applies overrides
in the order base → dataset → model file → CLI.

## Pre-built skill libraries

Each task family ships with at least one ready-to-use library. Pass the path
to `--library-path` to skip on-the-fly skill creation.

| Dataset | Library path | Origin | Notes |
|---|---|---|---|
| AIME 24 / 25 / AMC 23 | `libraries/openr1_math_skills/` | TD (from OpenR1 traces) | 7 skills |
| GAIA | `gaia/skills/` | SD (file-type schemas) | 9 skills (csv, docx, excel, json, pdb, pdf, search, txt, xml) |
| WebWalkerQA | `webwalker/skills/` | SD (web-task schemas) | 8 skills (search, navigation, html_parsing, web_crawling, info_extraction, screenshot, markdown, evaluation) |
| ReDial | `datasets/conversational_rec/skills/generated/redial/` | TD (from in-domain probes) | 7 skills |
| Reddit-V2 | `datasets/conversational_rec/skills/generated/redditv2/` | TD (from in-domain probes) | 7 skills |

For GAIA and WebWalkerQA, the recipe auto-loads its library from the paths
above, so `--library-path` is only needed if you want to substitute a
different one.

## Running experiments

Each dataset has its own runner under `datasets/<name>/run.py` that sweeps
the V × ρ × ω grid for all registered models.

### Math (AIME / AMC)

With the shipped TD library:

```bash
python datasets/aime/run.py --model qwen3_4b_instruct_vllm \
    --config config_aime24.yaml \
    --library-path libraries/openr1_math_skills \
    --only-origin TD
```

Generate a fresh library on the fly (no `--library-path`):

```bash
python datasets/aime/run.py --model qwen3_4b_instruct_vllm \
    --config config_aime24.yaml \
    --only-origin SD
```

The shipped helper scripts run AIME24 / AIME25 / AMC23 sequentially:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_qwen3_4b.sh
CUDA_VISIBLE_DEVICES=1 ./run_qwen25_7b.sh
```

### GAIA

The recipe auto-loads `gaia/skills/`:

```bash
python datasets/gaia/run.py --model gpt4o_mini --only-origin SD
```

To generate a TD library from probe traces instead:

```bash
python datasets/gaia/run.py --model gpt4o_mini --only-origin TD
```

### WebWalkerQA

The recipe auto-loads `webwalker/skills/`:

```bash
python datasets/webwalkerqa/run.py --model gpt4o_mini --only-origin SD
```

Override with a different library:

```bash
python datasets/webwalkerqa/run.py --model gpt4o_mini \
    --library-path webwalker/skills \
    --only-origin SD
```

### Conversational recommendation

With the pre-generated ReDial library:

```bash
python datasets/conversational_rec/run.py --mode axis_sweep \
    --model gpt4o_mini --only-origin TD \
    --library-path datasets/conversational_rec/skills/generated/redial
```

For Reddit-V2 swap the path to `.../generated/redditv2`. See
`datasets/conversational_rec/RUNBOOK.md` for the full run book (ReDial
preparation, mixed-source configs, single-axis runs, full 18-cell grid).

### Running without skills (NL cell only)

The visibility sweep already includes NL (no library). To produce only the
NL cell, restrict the axis:

```bash
python datasets/aime/run.py --model qwen3_4b_instruct_vllm \
    --config config_aime24.yaml \
    --only-axis visibility \
    --library-path libraries/openr1_math_skills
```

The runner will run NL, LB, and FL; pull the NL row from the output CSV.

### Smoke test (verify your install)

Run a single cell on a tiny pool to confirm the model and library load:

```bash
python datasets/aime/run.py --model qwen3_4b_instruct_vllm \
    --config config_aime24.yaml \
    --library-path libraries/openr1_math_skills \
    --only-origin TD --only-axis visibility
```

For a non-vLLM smoke test, use any OpenAI backbone:

```bash
python datasets/aime/run.py --model gpt4o_mini \
    --config config_aime24.yaml \
    --library-path libraries/openr1_math_skills \
    --only-origin TD --only-axis visibility
```

Successful runs land under `outputs/results/<dataset>/<model>/<timestamp>/`.

### Common flags

The flag set is similar across runners but differs in a few places. Use
`python datasets/<name>/run.py --help` for the authoritative list. The most
useful flags are:

| Flag | Where | Purpose |
|---|---|---|
| `--model NAME` | all | Run only the named model file (without `.py`) |
| `--only-origin {SD,TD}` (`{SD,TD,all}` for convrec) | all | Restrict to one library origin |
| `--only-axis {visibility,retrieval,evolution}` | all | Restrict to one design choice |
| `--library-path PATH` | all | Use a pre-built library; skip skill creation |
| `--config NAME` | aime, convrec | Choose the dataset-specific config file |
| `--max-episodes N` | gaia, webwalker, convrec | Cap eval pool, useful for smoke tests |
| `--mode {single_axis, axis_sweep, full_grid}` | convrec | Sweep mode (axis_sweep = 8-cell grid) |

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
