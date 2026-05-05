"""Trace storage: persist and load traces from disk."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from skilleval.core.types import EpisodeTrace, TraceEntry

logger = logging.getLogger(__name__)


class TraceStorage:
    """JSON-based trace persistence."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_trace(self, trace: EpisodeTrace, filename: str | None = None) -> Path:
        """Save a single episode trace."""
        fname = filename or f"{trace.task_id}_{trace.model_id}.json"
        path = self.base_dir / fname
        with open(path, "w") as f:
            json.dump(trace.to_dict(), f, indent=2)
        return path

    def save_traces(self, traces: list[EpisodeTrace], filename: str = "all_traces.json") -> Path:
        """Save a batch of traces to a single file."""
        path = self.base_dir / filename
        data = [t.to_dict() for t in traces]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d traces to %s", len(traces), path)
        return path

    def load_traces(self, filename: str = "all_traces.json") -> list[EpisodeTrace]:
        """Load traces from a JSON file."""
        path = self.base_dir / filename
        if not path.exists():
            logger.warning("Trace file not found: %s", path)
            return []
        with open(path) as f:
            data = json.load(f)

        traces = []
        for item in data:
            entries = [
                TraceEntry(**{k: v for k, v in e.items()})
                for e in item.get("entries", [])
            ]
            trace = EpisodeTrace(
                task_id=item["task_id"],
                model_id=item["model_id"],
                entries=entries,
                total_cost=item.get("total_cost", 0.0),
                total_tokens=item.get("total_tokens", 0),
                success=item.get("success", False),
                metadata=item.get("metadata", {}),
            )
            traces.append(trace)
        return traces

    def list_trace_files(self) -> list[Path]:
        return sorted(self.base_dir.glob("*.json"))
