"""Persist train/test splits and BU batch assignments.

Writes ``outputs/splits/<dataset>/splits.json`` with the resolved list of
train + test task_ids for the run, so reruns on the same config produce
byte-identical assignments. BU batch partitions live alongside as
``bu_batches.json``.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def persist_splits(
    dataset_name: str,
    train_tasks: list[Any],
    test_tasks: list[Any],
    out_root: Path = Path("outputs/splits"),
) -> Path:
    """Write train/test task_ids to disk. Returns the directory path."""
    dir_path = out_root / dataset_name
    dir_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "train_task_ids": [t.task_id for t in train_tasks],
        "test_task_ids": [t.task_id for t in test_tasks],
        "num_train": len(train_tasks),
        "num_test": len(test_tasks),
    }
    path = dir_path / "splits.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        "Persisted splits for %s: %d train, %d test -> %s",
        dataset_name, len(train_tasks), len(test_tasks), path,
    )
    return dir_path


def partition_bu_batches(
    train_tasks: list[Any],
    num_rounds: int,
    seed: int,
) -> list[list[Any]]:
    """Shuffle train_tasks once and slice into ``num_rounds`` contiguous batches.

    Returns a list of ``num_rounds`` task lists. If ``len(train_tasks) <
    num_rounds``, logs a warning and falls back to ``len(train_tasks)``
    single-task batches.
    """
    pool = list(train_tasks)
    if not pool:
        return [[] for _ in range(num_rounds)]
    rng = random.Random(seed)
    rng.shuffle(pool)

    if len(pool) < num_rounds:
        logger.warning(
            "Only %d train tasks for %d BU rounds — using %d single-task batches.",
            len(pool), num_rounds, len(pool),
        )
        return [[t] for t in pool]

    n = len(pool)
    batches = []
    for k in range(num_rounds):
        lo = k * n // num_rounds
        hi = (k + 1) * n // num_rounds
        batches.append(pool[lo:hi])
    return batches


def persist_bu_batches(
    dataset_name: str,
    experiment_id: str,
    batches: list[list[Any]],
    out_root: Path = Path("outputs/splits"),
) -> Path:
    """Write BU batch assignments for one experiment."""
    dir_path = out_root / dataset_name
    dir_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "experiment_id": experiment_id,
        "num_batches": len(batches),
        "batches": {
            f"round_{k+1}": [t.task_id for t in batch]
            for k, batch in enumerate(batches)
        },
    }
    # Sanitize experiment_id to avoid path issues (e.g., "Qwen/Qwen3-4B" -> "Qwen_Qwen3-4B")
    import re
    safe_experiment_id = re.sub(r'[/\\:]', '_', experiment_id)
    path = dir_path / f"bu_batches_{safe_experiment_id}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
