"""Main evaluator: orchestrates experiment runs across the taxonomy grid."""

from __future__ import annotations

import itertools
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from skilleval.agents.executor import SkillAgent
from skilleval.core.config import Config
from skilleval.core.types import (
    EvalResult,
    RuntimePolicy,
    SkillOrigin,
    SkillRetrieval,
    SkillVisibility,
    UpdateStrategy,
)
from skilleval.datasets.base import BaseDataset
from skilleval.evaluation.axis_mapping import (
    resolve_initial_library,
    resolve_retrieval_policy,
    validate_axis_combination,
)
from skilleval.evaluation.metrics import MetricsComputer
from skilleval.models.base import BaseModel
from skilleval.evaluation.splits import (
    partition_bu_batches,
    persist_bu_batches,
    persist_splits,
)
from skilleval.skills.creation.bottom_up import BottomUpCreator
from skilleval.skills.creation.loader import load_dataset_recipe
from skilleval.skills.creation.recipe import SkillRecipe
from skilleval.skills.creation.skill_updater import update_library
from skilleval.skills.creation.top_down import TopDownCreator
from skilleval.skills.library import SkillLibrary

logger = logging.getLogger(__name__)


class Evaluator:
    """Run experiments across the full taxonomy grid.

    Supports two calling conventions:

    1. **Legacy** (``run_grid`` / ``_run_single_legacy``):
       Iterates over ``SkillOrigin × RuntimePolicy``.

    2. **Taxonomy axes** (``run_single`` / axis runners):
       Uses the paper's three orthogonal axes — Visibility, Retrieval,
       Evolution — while keeping ``SkillOrigin`` as a separate library-
       construction parameter.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.metrics = MetricsComputer()
        self.results: list[EvalResult] = []
        self._recipe_cache: dict[str, SkillRecipe] = {}
        self._completed_experiments: dict[str, EvalResult] = {}
        # Timestamped run ID for output isolation
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("Evaluator initialized with run_timestamp=%s", self.run_timestamp)

    def _get_recipe(self, dataset: BaseDataset) -> SkillRecipe:
        if dataset.name not in self._recipe_cache:
            self._recipe_cache[dataset.name] = load_dataset_recipe(dataset.name)
        return self._recipe_cache[dataset.name]

    # ======================================================================
    # New axis-based API
    # ======================================================================

    def create_library_once(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        skill_origin: SkillOrigin = SkillOrigin.SPEC_DERIVED,
    ) -> SkillLibrary:
        """Generate the skill library once and save as individual files.

        Call this before running axis experiments. Pass the returned library
        to all axis runners so skills are consistent across experiments.
        Also persists train/test split manifests on first call.
        """
        # Persist splits manifest once so reruns are reproducible.
        train = dataset.train_tasks() if hasattr(dataset, "train_tasks") else []
        test = dataset.test_tasks() if hasattr(dataset, "test_tasks") else dataset.tasks()
        persist_splits(dataset.name, train, test)

        library = self._create_skill_library(dataset, model, skill_origin)

        # Save as individual SKILL.md files, tagged by origin so SD/TD
        # libraries don't clobber each other. Uses timestamped run folder.
        model_name = getattr(model, "model_id", "unknown")
        base_dir = (
            Path("outputs/libraries") / model_name / self.run_timestamp
            / f"base_{skill_origin.value}"
        )
        library.save_to_markdown_dir(base_dir)
        logger.info(
            "Base library saved: %d skills to %s", library.size, base_dir,
        )
        return library

    def run_single(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        visibility: SkillVisibility = SkillVisibility.FULL_LIBRARY,
        retrieval: SkillRetrieval = SkillRetrieval.NO_RETRIEVAL,
        evolution: UpdateStrategy = UpdateStrategy.FROZEN,
        skill_origin: SkillOrigin = SkillOrigin.SPEC_DERIVED,
        max_episodes: int = 100,
        max_steps: int = 50,
        num_evolution_rounds: int = 3,
        skill_library: SkillLibrary | None = None,
    ) -> EvalResult:
        """Run a single experiment cell defined by the three taxonomy axes.

        Args:
            dataset: Benchmark dataset.
            model: LLM backend.
            visibility: Axis 1 — what the agent sees at episode start.
            retrieval: Axis 2 — how skills are selected during the episode.
            evolution: Axis 3 — whether the library updates across episodes.
            skill_origin: How the skill library is initially constructed
                (orthogonal to the three axes).
            max_episodes: Max tasks to evaluate.
            max_steps: Max agent steps per task.
            num_evolution_rounds: Number of BU rounds (only used when
                ``evolution == BATCH_UPDATE``).
            skill_library: Pre-built library to use. If None, generates fresh.
        """
        experiment_id = (
            f"{dataset.name}_{model.model_id}"
            f"_{skill_origin.value}"
            f"_{visibility.value}_{retrieval.value}_{evolution.value}"
        )

        if experiment_id in self._completed_experiments:
            logger.info("Skipping duplicate experiment: %s", experiment_id)
            cached = self._completed_experiments[experiment_id]
            self.results.append(cached)
            return cached

        logger.info("Running experiment: %s", experiment_id)

        # Validate axis combo and log warnings.
        for warning in validate_axis_combination(visibility, retrieval, evolution):
            logger.warning(warning)

        # Use pre-built library or generate fresh.
        if visibility == SkillVisibility.NO_LIBRARY:
            full_library = SkillLibrary()
        elif skill_library is not None:
            full_library = skill_library
        else:
            full_library = self._create_skill_library(dataset, model, skill_origin)
        logger.info("Full library: %d skills", full_library.size)

        # Save experiment-specific library snapshot (timestamped run).
        model_name = getattr(model, "model_id", "unknown")
        exp_dir = (
            Path("outputs/libraries") / model_name / self.run_timestamp
            / "experiments" / f"{visibility.value}_{retrieval.value}_{evolution.value}"
        )
        full_library.save_to_markdown_dir(exp_dir)
        logger.info("Library saved to %s", exp_dir)

        # Resolve retrieval axis → runtime policy code.
        policy_code = resolve_retrieval_policy(retrieval)
        protocol = self._resolve_protocol(dataset)

        # Eval always runs on held-out test_tasks.
        test_pool = (
            dataset.test_tasks() if hasattr(dataset, "test_tasks") else dataset.tasks()
        )
        tasks = test_pool[:max_episodes]
        env_tools = dataset.get_tools()

        if evolution == UpdateStrategy.BATCH_UPDATE:
            result = self._run_batch_update(
                dataset=dataset,
                model=model,
                full_library=full_library,
                visibility=visibility,
                retrieval=retrieval,
                skill_origin=skill_origin,
                policy_code=policy_code,
                protocol=protocol,
                tasks=tasks,
                env_tools=env_tools,
                max_steps=max_steps,
                num_rounds=num_evolution_rounds,
                experiment_id=experiment_id,
            )
        else:
            result = self._run_frozen(
                dataset=dataset,
                model=model,
                full_library=full_library,
                visibility=visibility,
                retrieval=retrieval,
                evolution=evolution,
                skill_origin=skill_origin,
                policy_code=policy_code,
                protocol=protocol,
                tasks=tasks,
                env_tools=env_tools,
                max_steps=max_steps,
                experiment_id=experiment_id,
            )

        self._completed_experiments[experiment_id] = result
        self.results.append(result)
        logger.info(
            "Experiment %s: success=%.2f%%, steps=%.1f, cost=$%.4f",
            experiment_id,
            result.success_rate * 100,
            result.avg_steps,
            result.avg_cost,
        )
        return result

    # ------------------------------------------------------------------
    # Axis runner convenience methods
    # ------------------------------------------------------------------

    def run_visibility_axis(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        skill_origin: SkillOrigin = SkillOrigin.SPEC_DERIVED,
        max_episodes: int = 100,
        max_steps: int = 50,
        skill_library: SkillLibrary | None = None,
    ) -> list[EvalResult]:
        """Vary Visibility (NL / LB / FL), hold Retrieval=NR, Evolution=FR."""
        results = []
        for vis in SkillVisibility:
            r = self.run_single(
                dataset, model,
                visibility=vis,
                retrieval=SkillRetrieval.NO_RETRIEVAL,
                evolution=UpdateStrategy.FROZEN,
                skill_origin=skill_origin,
                max_episodes=max_episodes,
                max_steps=max_steps,
                skill_library=skill_library,
            )
            results.append(r)
        return results

    def run_retrieval_axis(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        skill_origin: SkillOrigin = SkillOrigin.SPEC_DERIVED,
        max_episodes: int = 100,
        max_steps: int = 50,
        skill_library: SkillLibrary | None = None,
    ) -> list[EvalResult]:
        """Vary Retrieval (NR / RR / PR), hold Visibility=FL, Evolution=FR."""
        results = []
        for ret in SkillRetrieval:
            r = self.run_single(
                dataset, model,
                visibility=SkillVisibility.FULL_LIBRARY,
                retrieval=ret,
                evolution=UpdateStrategy.FROZEN,
                skill_origin=skill_origin,
                max_episodes=max_episodes,
                max_steps=max_steps,
                skill_library=skill_library,
            )
            results.append(r)
        return results

    def run_evolution_axis(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        skill_origin: SkillOrigin = SkillOrigin.SPEC_DERIVED,
        max_episodes: int = 100,
        max_steps: int = 50,
        num_evolution_rounds: int = 3,
        skill_library: SkillLibrary | None = None,
    ) -> list[EvalResult]:
        """Vary Evolution (FR / BU), hold Visibility=FL, Retrieval=NR."""
        results = []
        for evo in [UpdateStrategy.FROZEN, UpdateStrategy.BATCH_UPDATE]:
            r = self.run_single(
                dataset, model,
                visibility=SkillVisibility.FULL_LIBRARY,
                retrieval=SkillRetrieval.NO_RETRIEVAL,
                evolution=evo,
                skill_origin=skill_origin,
                max_episodes=max_episodes,
                max_steps=max_steps,
                num_evolution_rounds=num_evolution_rounds,
                skill_library=skill_library,
            )
            results.append(r)
        return results

    def run_taxonomy_grid(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        skill_origin: SkillOrigin = SkillOrigin.SPEC_DERIVED,
        max_episodes: int = 100,
        max_steps: int = 50,
        num_evolution_rounds: int = 3,
    ) -> list[EvalResult]:
        """Run all 3 × 3 × 2 = 18 taxonomy cells."""
        evolutions = [UpdateStrategy.FROZEN, UpdateStrategy.BATCH_UPDATE]
        grid = list(itertools.product(SkillVisibility, SkillRetrieval, evolutions))
        logger.info("Running taxonomy grid: %d cells", len(grid))

        results = []
        for vis, ret, evo in grid:
            try:
                r = self.run_single(
                    dataset, model,
                    visibility=vis, retrieval=ret, evolution=evo,
                    skill_origin=skill_origin,
                    max_episodes=max_episodes, max_steps=max_steps,
                    num_evolution_rounds=num_evolution_rounds,
                )
                results.append(r)
            except Exception as e:
                logger.error(
                    "Failed: %s/%s/%s/%s: %s",
                    dataset.name, vis.value, ret.value, evo.value, e,
                )
        return results

    # ------------------------------------------------------------------
    # Internal execution methods
    # ------------------------------------------------------------------

    def _run_frozen(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        full_library: SkillLibrary,
        visibility: SkillVisibility,
        retrieval: SkillRetrieval,
        evolution: UpdateStrategy,
        skill_origin: SkillOrigin,
        policy_code: str,
        protocol: str,
        tasks: list[Any],
        env_tools: list[dict[str, Any]],
        max_steps: int,
        experiment_id: str,
    ) -> EvalResult:
        """Run all episodes against a frozen (unchanging) library."""
        all_traces = []
        tool_executors = dataset.get_tool_executors()

        answer_format = dataset.get_answer_format_prompt()

        lb_k = int(
            self.config.get(
                "experiment.lb_sample_size",
                self.config.get("experiment.react.lb_sample_size", 1),
            )
        )
        lb_seed = int(self.config.get("experiment.react.lb_seed", 0))
        lb_selection = str(self.config.get("experiment.lb_selection", "random"))

        if visibility == SkillVisibility.LIMITED_BUNDLE:
            # LB: per-task library (lb_sample_size random skills per task).
            # Save assignments for reproducibility.
            assignments = {}
            for task in tasks:
                visible = resolve_initial_library(
                    visibility, full_library, task,
                    lb_sample_size=lb_k,
                    lb_seed=lb_seed,
                    lb_selection=lb_selection,
                )
                skill_ids = [s.skill_id for s in visible.list_skills()]
                assignments[task.task_id] = skill_ids

                agent = SkillAgent(
                    model=model,
                    library=visible,
                    protocol=protocol,
                    policy=policy_code,
                    env_tools=env_tools,
                    full_library=full_library,
                    tool_executors=tool_executors,
                    config=self.config,
                    answer_format=answer_format,
                )
                trace = self._safe_solve(agent, task, max_steps)
                all_traces.append(trace)

            # Save LB assignments (timestamped run)
            model_name = getattr(model, "model_id", "unknown")
            assign_dir = (
                Path("outputs/libraries") / model_name / self.run_timestamp
                / "experiments" / experiment_id
            )
            assign_dir.mkdir(parents=True, exist_ok=True)
            import json as _json
            with open(assign_dir / "assignments.json", "w") as f:
                _json.dump({
                    "selection_method": lb_selection,
                    "lb_seed": lb_seed,
                    "lb_sample_size": lb_k,
                    "assignments": assignments,
                }, f, indent=2)
        else:
            # NL or FL: same library for all tasks — one agent suffices.
            visible = resolve_initial_library(
                visibility, full_library, tasks[0] if tasks else None,
                lb_selection=lb_selection,
            )
            agent = SkillAgent(
                model=model,
                library=visible,
                protocol=protocol,
                policy=policy_code,
                env_tools=env_tools,
                full_library=full_library,
                tool_executors=tool_executors,
                config=self.config,
                answer_format=answer_format,
            )
            all_traces = [self._safe_solve(agent, t, max_steps) for t in tasks]

        self._score_traces_against_gold(dataset, tasks, all_traces)

        return self.metrics.compute(
            traces=all_traces,
            tasks=tasks,
            experiment_id=experiment_id,
            model_id=model.model_id,
            dataset_id=dataset.name,
            skill_origin=skill_origin,
            runtime_policy=RuntimePolicy(policy_code),
            visibility=visibility,
            retrieval=retrieval,
            evolution=evolution,
        )

    def _run_batch_update(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        full_library: SkillLibrary,
        visibility: SkillVisibility,
        retrieval: SkillRetrieval,
        skill_origin: SkillOrigin,
        policy_code: str,
        protocol: str,
        tasks: list[Any],
        env_tools: list[dict[str, Any]],
        max_steps: int,
        num_rounds: int,
        experiment_id: str,
    ) -> EvalResult:
        """Run BU with one training batch per round, scored on the full test set.

        For each round k in 1..num_rounds:
          1. Run agent with library_{k-1} on ``batch_k`` (train slice) →
             collect update traces.
          2. LLM rewrites skills using those traces → library_k.
          3. Score library_k on the full ``tasks`` (test pool).

        Only round k's test eval contributes to the final metric; train
        traces drive evolution only.
        """
        current_library = full_library
        tool_executors = dataset.get_tool_executors()
        writer_model = self._build_skill_writer_model(model)
        recipe = self._get_recipe(dataset)

        # Timestamped BU output directory
        model_name = getattr(model, "model_id", "unknown")
        bu_dir = (
            Path("outputs/libraries") / model_name / self.run_timestamp
            / "experiments" / experiment_id
        )
        # Round 0 = the starting library (pre-update).
        current_library.save_to_markdown_dir(bu_dir / "round_0")

        update_source = str(
            self.config.get("experiment.evolution.update_source", "eval")
        )
        if update_source == "train":
            # Recommendation configs opt into this path so BU updates from the
            # native train split and remains comparable to FR on held-out test.
            update_pool = (
                dataset.train_tasks() if hasattr(dataset, "train_tasks") else []
            )
        else:
            # Compatibility default: historical BU adapted on the eval pool.
            update_pool = list(tasks)
        if update_source == "train" and not update_pool:
            logger.warning(
                "BU requested but dataset has no train split; using test pool "
                "for update traces as a compatibility fallback."
            )
            update_pool = list(tasks)
        split_seed = int(self.config.get("dataset.split_seed", 42))
        batches = partition_bu_batches(update_pool, num_rounds, seed=split_seed)
        persist_bu_batches(dataset.name, experiment_id, batches)
        logger.info(
            "BU: %d update tasks partitioned into %d batches of sizes %s",
            len(update_pool), len(batches), [len(b) for b in batches],
        )

        # Round 0 eval on the test pool (baseline library, no updates yet).
        final_test_traces = self._run_eval_pass(
            dataset=dataset,
            model=model,
            library=current_library,
            visibility=visibility,
            policy_code=policy_code,
            protocol=protocol,
            tasks=tasks,
            env_tools=env_tools,
            tool_executors=tool_executors,
            max_steps=max_steps,
        )
        round0_success = sum(1 for t in final_test_traces if t.success) / max(len(final_test_traces), 1)
        per_round_test_success = [round0_success]
        logger.info(
            "BU round 0 (baseline) test eval: %.1f%% success (%d/%d)",
            round0_success * 100,
            sum(1 for t in final_test_traces if t.success),
            len(final_test_traces),
        )

        for round_idx, batch in enumerate(batches, start=1):
            if not batch:
                logger.info("BU round %d: empty batch — stopping.", round_idx)
                break

            logger.info(
                "BU round %d/%d: update batch=%d tasks, library=%d skills",
                round_idx, num_rounds, len(batch), current_library.size,
            )
            logger.info(
                "BU ROUND %d START library_skills=%s batch_tasks=%s",
                round_idx,
                [s.skill_id for s in current_library.list_skills()],
                [t.task_id for t in batch],
            )

            # Update phase: run on train batch to collect evolution traces.
            update_traces = self._run_eval_pass(
                dataset=dataset,
                model=model,
                library=current_library,
                visibility=visibility,
                policy_code=policy_code,
                protocol=protocol,
                tasks=batch,
                env_tools=env_tools,
                tool_executors=tool_executors,
                max_steps=max_steps,
            )
            update_traces = self._select_update_traces(update_traces)

            # LLM rewrites skills from this batch's traces.
            current_library = update_library(
                current_library,
                update_traces,
                writer_model,
                max_new_skills=int(
                    self.config.get("experiment.evolution.max_new_skills", 0)
                ),
                allow_bootstrap=bool(
                    self.config.get("experiment.evolution.allow_bootstrap", False)
                ),
                recovery_prompt_template=recipe.td_prompt,
                trace_summarizer=recipe.trace_summarizer,
                extra_context=recipe.extra_context,
            )
            current_library.save_to_markdown_dir(bu_dir / f"round_{round_idx}")
            logger.info(
                "BU round %d: %d skills after rewrite",
                round_idx, current_library.size,
            )
            logger.info(
                "BU ROUND %d END library_skills=%s",
                round_idx,
                [s.skill_id for s in current_library.list_skills()],
            )

            # Eval phase: score the updated library on the full test pool.
            final_test_traces = self._run_eval_pass(
                dataset=dataset,
                model=model,
                library=current_library,
                visibility=visibility,
                policy_code=policy_code,
                protocol=protocol,
                tasks=tasks,
                env_tools=env_tools,
                tool_executors=tool_executors,
                max_steps=max_steps,
            )
            round_success = (
                sum(1 for t in final_test_traces if t.success)
                / max(len(final_test_traces), 1)
            )
            per_round_test_success.append(round_success)
            logger.info(
                "BU round %d test eval: %.1f%% success (%d/%d)",
                round_idx, round_success * 100,
                sum(1 for t in final_test_traces if t.success),
                len(final_test_traces),
            )

        # Final result = last round's test pass.
        result = self.metrics.compute(
            traces=final_test_traces,
            tasks=tasks,
            experiment_id=experiment_id,
            model_id=model.model_id,
            dataset_id=dataset.name,
            skill_origin=skill_origin,
            runtime_policy=RuntimePolicy(policy_code),
            visibility=visibility,
            retrieval=retrieval,
            evolution=UpdateStrategy.BATCH_UPDATE,
        )
        result.metadata["num_rounds"] = num_rounds
        result.metadata["library_size_final"] = current_library.size
        result.metadata["per_round_test_success"] = per_round_test_success
        result.metadata["batch_sizes"] = [len(b) for b in batches]
        return result

    def _run_eval_pass(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        library: SkillLibrary,
        visibility: SkillVisibility,
        policy_code: str,
        protocol: str,
        tasks: list[Any],
        env_tools: list[dict[str, Any]],
        tool_executors: dict[str, Any],
        max_steps: int,
    ) -> list[Any]:
        """Run a single agent pass over ``tasks`` with ``library``; score vs gold.

        Used by BU both for train-batch update traces and for per-round
        test scoring. Returns traces with ``success`` populated.
        """
        traces: list[Any] = []
        if not tasks:
            return traces

        answer_format = dataset.get_answer_format_prompt()

        lb_k = int(
            self.config.get(
                "experiment.lb_sample_size",
                self.config.get("experiment.react.lb_sample_size", 1),
            )
        )
        lb_seed = int(self.config.get("experiment.react.lb_seed", 0))
        lb_selection = str(self.config.get("experiment.lb_selection", "random"))

        if visibility == SkillVisibility.LIMITED_BUNDLE:
            for task in tasks:
                visible = resolve_initial_library(
                    visibility, library, task,
                    lb_sample_size=lb_k,
                    lb_seed=lb_seed,
                    lb_selection=lb_selection,
                )
                agent = SkillAgent(
                    model=model,
                    library=visible,
                    protocol=protocol,
                    policy=policy_code,
                    env_tools=env_tools,
                    full_library=library,
                    tool_executors=tool_executors,
                    config=self.config,
                    answer_format=answer_format,
                )
                traces.append(self._safe_solve(agent, task, max_steps))
        else:
            visible = resolve_initial_library(
                visibility,
                library,
                tasks[0],
                lb_selection=lb_selection,
            )
            agent = SkillAgent(
                model=model,
                library=visible,
                protocol=protocol,
                policy=policy_code,
                env_tools=env_tools,
                full_library=library,
                tool_executors=tool_executors,
                config=self.config,
                answer_format=answer_format,
            )
            traces = [self._safe_solve(agent, t, max_steps) for t in tasks]

        self._score_traces_against_gold(dataset, tasks, traces)
        return traces

    def _select_update_traces(self, traces: list[Any]) -> list[Any]:
        """Choose high-signal traces for BU skill rewriting.

        Recommendation failures are most useful when they miss all gold items,
        produce invalid JSON, or repeat titles from the context.
        """
        mode = str(
            self.config.get("experiment.evolution.update_trace_filter", "all")
        )
        if mode == "all":
            return traces

        selected = []
        for trace in traces:
            score = trace.metadata.get("gold_score", {})
            if not isinstance(score, dict):
                selected.append(trace)
                continue
            if not trace.success:
                selected.append(trace)
                continue
            if float(score.get("schema_valid", 1.0)) < 1.0:
                selected.append(trace)
                continue
            if float(score.get("context_repeat_rate", 0.0)) > 0.0:
                selected.append(trace)

        if not selected:
            logger.info("BU trace filter selected no traces; keeping all traces.")
            return traces

        logger.info(
            "BU trace filter selected %d/%d traces for skill rewriting.",
            len(selected), len(traces),
        )
        return selected

    @staticmethod
    def _safe_solve(agent: SkillAgent, task: Any, max_steps: int) -> Any:
        """Run one episode, returning a failed-but-valid trace on exception.

        Ensures that a single task's API/network/parsing error doesn't kill
        the entire sweep. The returned trace has ``success=False`` and an
        error note in metadata, so metrics treat it as a failed task.
        """
        from skilleval.core.types import EpisodeTrace
        try:
            return agent.solve(task, max_steps=max_steps)
        except Exception as exc:
            logger.warning(
                "Task %s crashed during agent.solve: %s",
                getattr(task, "task_id", "?"), exc,
            )
            return EpisodeTrace(
                task_id=getattr(task, "task_id", "unknown"),
                model_id=agent.model.model_id,
                entries=[],
                success=False,
                final_answer=None,
                metadata={"error": str(exc)},
            )

    def _score_traces_against_gold(
        self,
        dataset: BaseDataset,
        tasks: list[Any],
        traces: list[Any],
    ) -> None:
        """Set ``trace.success`` from ``dataset.evaluate_prediction()``.

        Mutates each trace in place:
          - ``trace.success`` = ``result["success"] >= 0.5``
          - ``trace.metadata["gold_score"]`` = full result dict

        This is the single source of truth for whether a task was solved,
        replacing the agent's self-reported success signal.
        """
        for task, trace in zip(tasks, traces):
            try:
                result = dataset.evaluate_prediction(task, trace.final_answer or "")
            except Exception as exc:
                logger.warning(
                    "evaluate_prediction failed for task %s: %s",
                    getattr(task, "task_id", "?"), exc,
                )
                result = {"success": 0.0, "error": str(exc)}
            trace.success = float(result.get("success", 0.0)) >= 0.5
            trace.metadata["gold_score"] = result
            logger.info(
                "GOLD_SCORE task=%s pred=%r gold=%r result=%s success=%s",
                task.task_id, trace.final_answer, task.gold_answer,
                result, trace.success,
            )

    # ======================================================================
    # Legacy API (backward-compatible)
    # ======================================================================

    def _run_single_legacy(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        skill_origin: SkillOrigin,
        runtime_policy: RuntimePolicy,
        max_episodes: int = 100,
        max_steps: int = 50,
    ) -> EvalResult:
        """Run a single (dataset, model, origin, policy) experiment cell."""
        experiment_id = (
            f"{dataset.name}_{model.model_id}_{skill_origin.value}_{runtime_policy.value}"
        )
        logger.info("Running experiment: %s", experiment_id)

        library = self._create_skill_library(dataset, model, skill_origin)

        tasks = dataset.tasks()[:max_episodes]
        env_tools = dataset.get_tools()

        agent = SkillAgent(
            model=model,
            library=library,
            protocol=self._resolve_protocol(dataset),
            policy=runtime_policy.value,
            env_tools=env_tools,
            answer_format=dataset.get_answer_format_prompt(),
        )

        traces = agent.solve_batch(tasks, max_steps=max_steps)

        result = self.metrics.compute(
            traces=traces,
            tasks=tasks,
            experiment_id=experiment_id,
            model_id=model.model_id,
            dataset_id=dataset.name,
            skill_origin=skill_origin,
            runtime_policy=runtime_policy,
        )
        self.results.append(result)
        logger.info(
            "Experiment %s: success=%.2f%%, steps=%.1f, cost=$%.4f",
            experiment_id,
            result.success_rate * 100,
            result.avg_steps,
            result.avg_cost,
        )
        return result

    def run_grid(
        self,
        datasets: list[BaseDataset],
        models: list[BaseModel],
        skill_origins: list[SkillOrigin] | None = None,
        runtime_policies: list[RuntimePolicy] | None = None,
        max_episodes: int = 100,
        max_steps: int = 50,
    ) -> list[EvalResult]:
        """Run the full SkillOrigin × RuntimePolicy grid (legacy API)."""
        origins = skill_origins or [SkillOrigin.SPEC_DERIVED, SkillOrigin.TRACE_DERIVED]
        policies = runtime_policies or [RuntimePolicy.ORACLE_BUNDLE, RuntimePolicy.RETRIEVE_ROUTE, RuntimePolicy.PLAN_VERIFY]

        grid = list(itertools.product(datasets, models, origins, policies))
        logger.info("Running grid: %d experiment cells", len(grid))

        for ds, model, origin, policy in grid:
            try:
                self._run_single_legacy(ds, model, origin, policy, max_episodes=max_episodes, max_steps=max_steps)
            except Exception as e:
                logger.error("Failed: %s/%s/%s/%s: %s", ds.name, model.model_id, origin.value, policy.value, e)

        return self.results

    # ======================================================================
    # Shared helpers
    # ======================================================================

    def _create_skill_library(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        origin: SkillOrigin,
    ) -> SkillLibrary:
        """Create a skill library using an LLM skill-writer.

        - SD (Spec-Derived): LLM writes skills from dataset + tool specs
          (plus optional recipe-supplied prompt / seed skills).
        - TD (Trace-Derived): a normal agent runs with an empty library on
          ``dataset.train_tasks()``; the LLM writes skills by generalizing
          over those probe traces.
        """
        max_skills = int(self.config.get("experiment.max_skills", 3))
        max_steps = int(self.config.get("experiment.max_steps_per_episode", 50))
        writer_model = self._build_skill_writer_model(model)
        recipe = self._get_recipe(dataset)

        if origin == SkillOrigin.SPEC_DERIVED:
            creator = TopDownCreator(
                writer_model, max_skills=max_skills, recipe=recipe,
            )
            return creator.create(dataset)

        if origin == SkillOrigin.TRACE_DERIVED:
            creator_td = BottomUpCreator(
                writer_model, max_skills=max_skills, recipe=recipe,
            )
            # Skip probe collection when max_skills==0 (seed-only library).
            effective_max = creator_td.max_skills
            if effective_max == 0:
                logger.info(
                    "TD probe skipped: max_skills=0, using seed skills only."
                )
                return creator_td.create([], SkillLibrary())

            # Probe on held-out train tasks, not on eval tasks.
            train_pool = dataset.train_tasks() if hasattr(dataset, "train_tasks") else dataset.tasks()
            probe_tasks_cfg = self.config.get(
                "experiment.skill_creation.probe_tasks"
            )
            if probe_tasks_cfg is not None:
                probe_n = max(1, int(probe_tasks_cfg))
            else:
                probe_fraction = float(
                    self.config.get("experiment.skill_creation.probe_fraction", 0.2)
                )
                pool_size = len(train_pool) or len(dataset)
                probe_n = max(1, int(pool_size * probe_fraction))

            traces = self._collect_probe_traces(
                dataset=dataset,
                model=model,
                library=SkillLibrary(),
                tasks=train_pool[:probe_n],
                max_steps=max_steps,
            )
            return creator_td.create(traces, SkillLibrary())

        return SkillLibrary()

    def _build_skill_writer_model(self, fallback_model: BaseModel) -> BaseModel:
        """Build the cheap model used to author skill templates.

        Reads ``experiment.skill_writer.provider`` / ``...model_id`` from
        config. Falls back to the agent's model if not configured.
        """
        from skilleval.core.config import Config
        from skilleval.core.registry import model_registry

        writer_cfg = self.config.get("experiment.skill_writer")
        if not writer_cfg:
            return fallback_model
        provider = writer_cfg.get("provider")
        if not provider:
            return fallback_model
        sub = Config({"model": dict(writer_cfg)})
        try:
            return model_registry.build(provider, sub)
        except Exception as exc:
            logger.warning(
                "Could not build skill_writer model %s: %s — falling back to agent model.",
                writer_cfg, exc,
            )
            return fallback_model

    def _collect_probe_traces(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        library: SkillLibrary,
        tasks: list[Any],
        max_steps: int,
    ) -> list[Any]:
        """Direct-inference probe with an optional Python subprocess escape hatch.

        The probe avoids the SkillAgent / tool-protocol stack entirely.
        Each train task runs a minimal loop against the provider:
          - System prompt: the standard GAIA system prompt.
          - At each turn, if the model emits a python fenced block,
            execute it via ``skilleval.tools.python_runner.run_python``
            and feed stdout/stderr back as the next user turn.
          - Stop when a turn contains ``FINAL ANSWER:`` or there is no
            python code to run.

        Traces are written as JSONL (one task per line) to
        ``logs/traces/<dataset>/probe_TD/probe_traces.jsonl``.
        """
        import re
        import json as _json
        from skilleval.core.types import EpisodeTrace, TraceEntry
        from skilleval.tools.python_runner import run_python, format_result

        if not tasks:
            return []

        gaia_system_prompt = (
            "You are an intelligent assistant helping with the GAIA benchmark. "
            "For each question, reasoning steps are helpful, but you MUST end your response "
            "with the final answer in the format: FINAL ANSWER: [answer]\n\n"
            "If you need to compute something, search the web, or parse data, you MAY write "
            "a single Python code block fenced with ```python ... ``` and the runner will "
            "execute it and return STDOUT/STDERR as the next user message. Only include the "
            "FINAL ANSWER line once you are confident in the answer."
        )

        max_iters = 8
        code_re = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
        final_re = re.compile(r"final\s*answer\s*[:\-]?\s*", re.IGNORECASE)

        records: list[dict[str, Any]] = []
        traces: list[EpisodeTrace] = []

        for task in tasks:
            history: list[dict[str, str]] = [
                {"role": "system", "content": gaia_system_prompt},
                {"role": "user", "content": task.instruction},
            ]
            turns: list[dict[str, Any]] = []
            final_answer: str | None = None
            entries: list[TraceEntry] = []

            for step in range(max_iters):
                try:
                    resp = model.generate(history)
                    text = (resp.text or "").strip()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Probe LLM error on task %s: %s", task.task_id, exc)
                    turns.append({"role": "error", "content": str(exc)})
                    break

                history.append({"role": "assistant", "content": text})
                turns.append({"role": "assistant", "content": text})
                entries.append(
                    TraceEntry(step=step, action="llm", observation=text[:2000])
                )

                if final_re.search(text):
                    final_answer = text
                    break

                m = code_re.search(text)
                if not m:
                    # No python, no final answer — treat full text as answer.
                    final_answer = text
                    break

                code = m.group(1).strip()
                exec_result = run_python(code)
                obs = format_result(exec_result)
                turns.append({"role": "tool", "content": obs, "code": code})
                entries.append(
                    TraceEntry(
                        step=step,
                        action="python",
                        tool_name="python_runner",
                        tool_args={"code": code[:500]},
                        observation=obs[:2000],
                        success=exec_result.get("returncode", -1) == 0,
                        error=None if exec_result.get("returncode", -1) == 0 else exec_result.get("stderr", "")[:200],
                    )
                )
                history.append({"role": "user", "content": obs})

            model_answer = final_answer or ""
            # Extract the text after FINAL ANSWER if present for scoring.
            m = final_re.search(model_answer)
            extracted = model_answer[m.end():].strip() if m else model_answer.strip()

            try:
                eval_result = dataset.evaluate_prediction(task, extracted)
            except Exception:
                eval_result = {"success": 0.0}
            success = float(eval_result.get("success", eval_result.get("exact_match", 0.0))) >= 0.5

            trace = EpisodeTrace(
                task_id=task.task_id,
                model_id=getattr(model, "model_id", "unknown"),
                entries=entries,
                success=success,
                final_answer=extracted,
                metadata={"gold_score": eval_result},
            )
            traces.append(trace)

            records.append({
                "task_id": task.task_id,
                "question": task.instruction,
                "turns": turns,
                "model_answer": extracted,
                "ground_truth": str(task.gold_answer or ""),
                "success": success,
            })

        logger.info(
            "TD probe collected %d traces (%d successful) on train pool",
            len(traces), sum(1 for t in traces if t.success),
        )

        try:
            out_dir = Path("logs/traces") / dataset.name / "probe_TD"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "probe_traces.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(_json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            logger.info("Saved probe traces: %s", out_path)
        except Exception as exc:
            logger.warning("Failed to persist probe traces: %s", exc)

        return traces

    def _resolve_protocol(self, dataset: BaseDataset) -> str:
        """Pick the right skill protocol for this dataset.

        Priority:
          1. Explicit ``experiment.protocol`` in config (user override).
          2. Auto-detect: if the dataset exposes env tools → tool_using,
             otherwise → in_context (skills as prompt text).
        """
        explicit = self.config.get("experiment.protocol")
        if explicit:
            return explicit
        if dataset.get_tools():
            return "tool_using"
        return "in_context"
