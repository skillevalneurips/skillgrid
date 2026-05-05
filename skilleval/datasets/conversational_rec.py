"""Conversational recommendation dataset adapter (ReDial, RedditV2, etc.)."""

from __future__ import annotations

import ast
import csv
import json
import logging
import math
import re
import string
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from skilleval.core.registry import dataset_registry
from skilleval.core.types import TaskDomain, TaskInstance
from skilleval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

REDIAL_TRAIN_FILE = "redial_train.csv"
REDIAL_TEST_FILE = "redial_test.csv"
REDIAL_RAW_BASE_URL = (
    "https://raw.githubusercontent.com/zhouhanxie/"
    "neighborhood-based-CF-for-CRS/main/datasets/redial"
)


@dataset_registry.register("conversational_rec")
class ConversationalRecDataset(BaseDataset):

    @property
    def name(self) -> str:
        return "conversational_rec"

    @property
    def domain(self) -> TaskDomain:
        return TaskDomain.CONVERSATIONAL_REC

    def load(self) -> None:
        sources = self.config.get("dataset.sources")
        if not sources:
            if self.config.get("dataset.redial.path"):
                sources = ["redial"]
            else:
                sources = ["redditv2"]
        try:
            self._load_sources(sources)
        except Exception as exc:
            if not bool(self.config.get("dataset.allow_stub", False)):
                raise
            max_samples = self.config.get("dataset.max_samples")
            logger.warning(
                "Conversational rec datasets not available; using stub data: %s",
                exc,
            )
            for idx, item in enumerate(self._stub_data()):
                if max_samples and idx >= max_samples:
                    break
                self._tasks.append(
                    TaskInstance(
                        task_id=f"convrec_{idx}",
                        domain=self.domain,
                        instruction=item["conversation_context"],
                        composition_pattern=self._assign_pattern(item),
                        tools_required=[
                            "item_search",
                            "user_preference_query",
                            "recommendation_engine",
                        ],
                        gold_answer=item.get("gold_recommendations", []),
                        metadata={
                            "source": item.get("source", "unknown"),
                            "turn_count": item.get("turn_count", 1),
                        },
                    )
                )
        logger.info(
            "Loaded %d conversational rec tasks (%d train, %d test)",
            len(self._tasks), len(self._train_tasks), len(self._test_tasks),
        )

    def _load_sources(self, sources: list[str] | str) -> None:
        if isinstance(sources, str):
            sources = [sources]
        loaded = 0
        for source in sources:
            normalized = source.lower()
            if normalized in {"redditv2", "reddit-v2", "reddit_v2"}:
                loaded += self._load_reddit_v2_native_splits()
                continue
            if normalized in {"redial", "re-dial"}:
                loaded += self._load_redial_native_splits()
                continue
            logger.warning("Unsupported conversational rec source: %s", source)

        if loaded == 0:
            raise FileNotFoundError("No conversational rec source data loaded.")

    def _load_redial_native_splits(self) -> int:
        root = Path(
            self.config.get(
                "dataset.redial.path",
                "datasets/conversational_rec/data/redial",
            )
        )
        train_file = self.config.get("dataset.redial.train_file", REDIAL_TRAIN_FILE)
        test_file = self.config.get("dataset.redial.test_file", REDIAL_TEST_FILE)
        if bool(self.config.get("dataset.redial.auto_download", False)):
            self._download_missing_redial_files(root, [train_file, test_file])

        train_split = self.config.get("dataset.train_split", "train")
        test_split = self.config.get("dataset.test_split") or self.config.get(
            "dataset.split", "test"
        )
        max_train = self.config.get("dataset.max_train_samples")
        max_test = self.config.get("dataset.max_test_samples") or self.config.get(
            "dataset.max_samples"
        )

        train = self._load_redial_split(
            root / train_file,
            train_split,
            _int_or_none(max_train),
            is_train=True,
        )
        test = self._load_redial_split(
            root / test_file,
            test_split,
            _int_or_none(max_test),
            is_train=False,
        )
        self._train_tasks.extend(train)
        self._test_tasks.extend(test)
        self._tasks.extend(test)
        return len(train) + len(test)

    @staticmethod
    def _download_missing_redial_files(root: Path, filenames: list[str]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            path = root / filename
            if path.exists():
                continue
            url = f"{REDIAL_RAW_BASE_URL}/{filename}"
            logger.info("Downloading ReDial %s to %s", filename, path)
            urllib.request.urlretrieve(url, path)

    def _load_redial_split(
        self,
        path: Path,
        split: str,
        max_samples: int | None,
        *,
        is_train: bool,
    ) -> list[TaskInstance]:
        if not path.exists():
            raise FileNotFoundError(
                f"ReDial file does not exist: {path}. Run "
                "`python datasets/conversational_rec/redial/prepare.py` "
                "or set dataset.redial.auto_download: true."
            )

        tasks = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for raw_idx, row in enumerate(reader):
                if max_samples is not None and len(tasks) >= max_samples:
                    break
                task = self._redial_row_to_task(row, raw_idx, split, is_train=is_train)
                if task is None:
                    continue
                tasks.append(task)
        return tasks

    def _redial_row_to_task(
        self,
        row: dict[str, str],
        raw_idx: int,
        split: str,
        *,
        is_train: bool,
    ) -> TaskInstance | None:
        if is_train:
            instruction = _clean_optional_text(
                row.get("full_situation") or row.get("context")
            )
            gold_titles = _parse_redial_list(row.get("movies"))
            if not gold_titles:
                gold_titles = _parse_redial_list(row.get("entity_name"))
        else:
            instruction = _clean_optional_text(row.get("test_inputs"))
            output = _clean_optional_text(row.get("test_outputs"))
            gold_titles = [output] if output else []

        if not instruction or not gold_titles:
            return None

        dialog_id = _clean_optional_text(row.get("dialog_id")) or f"{split}_{raw_idx}"
        turn_id = (
            _clean_optional_text(row.get("turn_id"))
            or _clean_optional_text(row.get(""))
            or str(raw_idx)
        )
        context_titles = _extract_titles_with_years(instruction)
        turn_count = _count_conversation_turns(instruction)

        return TaskInstance(
            task_id=(
                f"redial_{split}_{raw_idx}_"
                f"{_safe_task_id_part(dialog_id)}_{_safe_task_id_part(turn_id)}"
            ),
            domain=self.domain,
            instruction=instruction,
            composition_pattern=self._assign_pattern({"turn_count": turn_count}),
            tools_required=[
                "item_search",
                "user_preference_query",
                "recommendation_engine",
            ],
            gold_answer=gold_titles,
            metadata={
                "source": "redial",
                "split": split,
                "dialog_id": dialog_id,
                "turn_id": turn_id,
                "context_titles": context_titles,
                "context_ids": [],
                "gold_ids": [],
                "raw_row_index": raw_idx,
                "csv_index": row.get(""),
                "turn_count": turn_count,
            },
        )

    def _load_reddit_v2_native_splits(self) -> int:
        train_split = self.config.get("dataset.train_split", "train")
        test_split = self.config.get("dataset.test_split") or self.config.get(
            "dataset.split", "test"
        )
        max_train = self.config.get("dataset.max_train_samples")
        max_test = self.config.get("dataset.max_test_samples") or self.config.get(
            "dataset.max_samples"
        )

        train = self._load_reddit_v2_split(train_split, _int_or_none(max_train))
        test = self._load_reddit_v2_split(test_split, _int_or_none(max_test))
        self._train_tasks.extend(train)
        self._test_tasks.extend(test)
        self._tasks.extend(test)
        return len(train) + len(test)

    def _load_reddit_v2_split(
        self, split: str, max_samples: int | None
    ) -> list[TaskInstance]:
        data_path = self.config.get("dataset.path")
        if not data_path:
            raise FileNotFoundError("dataset.path is required for Reddit-V2.")

        size = self.config.get("dataset.redditv2.size", "small")
        variant = self.config.get("dataset.redditv2.variant", "clean_with_titles")
        member = f"reddit_v2/{size}/{split}/{split}_{variant}.jsonl"
        path = Path(data_path)

        if path.is_dir():
            records = self._iter_reddit_records_from_file(path / member)
        elif zipfile.is_zipfile(path):
            records = self._iter_reddit_records_from_zip(path, member)
        elif path.is_file():
            records = self._iter_reddit_records_from_file(path)
        else:
            raise FileNotFoundError(f"Reddit-V2 path does not exist: {path}")

        tasks = []
        for idx, item in enumerate(records):
            if max_samples is not None and len(tasks) >= max_samples:
                break
            task = self._reddit_item_to_task(item, idx, split)
            if task is None:
                continue
            tasks.append(task)
        return tasks

    def get_answer_format_prompt(self) -> str:
        return (
            "Return recommendations as JSON only, with this shape: "
            '{"recommendations":[{"title":"Movie Title","imdb_id":"tt1234567"}]}. '
            "Include up to 10 movies ranked from best to worst. If you do not know "
            "an IMDb id, use null for imdb_id. Do not include explanations outside JSON."
        )

    @classmethod
    def _iter_reddit_records_from_zip(cls, path: Path, member: str):
        with zipfile.ZipFile(path) as zf:
            if member not in zf.namelist():
                raise FileNotFoundError(f"{member} not found in {path}")
            with zf.open(member) as f:
                yield from cls._iter_json_array(f)

    @classmethod
    def _iter_reddit_records_from_file(cls, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Reddit-V2 file does not exist: {path}")
        with open(path, "rb") as f:
            yield from cls._iter_json_array(f)

    @staticmethod
    def _iter_json_array(binary_file):
        """Stream objects from Reddit-V2 files stored as one huge JSON array."""
        import io

        decoder = json.JSONDecoder()
        stream = io.TextIOWrapper(binary_file, encoding="utf-8")
        buffer = ""
        pos = 0
        in_array = False

        def ensure_buffer() -> bool:
            nonlocal buffer
            chunk = stream.read(1024 * 1024)
            if chunk:
                buffer += chunk
                return True
            return False

        while True:
            if pos >= len(buffer) and not ensure_buffer():
                return
            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1
            if pos >= len(buffer):
                continue
            if not in_array:
                if buffer[pos] != "[":
                    raise ValueError("Expected Reddit-V2 JSON array.")
                pos += 1
                in_array = True
                continue
            while pos < len(buffer) and (buffer[pos].isspace() or buffer[pos] == ","):
                pos += 1
            if pos >= len(buffer):
                continue
            if buffer[pos] == "]":
                return
            try:
                item, end = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError:
                if not ensure_buffer():
                    raise
                continue
            yield item
            pos = end
            if pos > 4 * 1024 * 1024:
                buffer = buffer[pos:]
                pos = 0

    def _reddit_item_to_task(
        self, item: dict[str, Any], idx: int, split: str
    ) -> TaskInstance | None:
        old = item.get("old", {}) if isinstance(item, dict) else {}
        context_raw = item.get("context_raw", [])
        instruction = old.get("input") or self._format_context(context_raw)
        gold_titles = item.get("clean_resp_titles") or old.get("entity_name") or []
        gold_ids = item.get("clean_resp_imdb_ids") or old.get("rec") or old.get("entity") or []
        context_titles = _flatten_once(item.get("clean_context_titles") or [])
        context_ids = _flatten_once(item.get("clean_context_imdb_ids") or [])

        if not instruction or not gold_titles:
            return None

        turn_id = item.get("turn_id") or old.get("turn_id") or f"redditv2_{idx}"
        turn_count = len(context_raw) or 1
        return TaskInstance(
            task_id=f"redditv2_{split}_{idx}_{turn_id}",
            domain=self.domain,
            instruction=instruction,
            composition_pattern=self._assign_pattern({"turn_count": turn_count}),
            tools_required=[
                "item_search",
                "user_preference_query",
                "recommendation_engine",
            ],
            gold_answer=gold_titles,
            metadata={
                "source": "redditv2",
                "split": split,
                "gold_ids": gold_ids,
                "context_titles": context_titles,
                "context_ids": context_ids,
                "dialog_id": item.get("conv_id") or old.get("dialog_id"),
                "turn_id": turn_id,
                "turn_count": turn_count,
            },
        )

    @staticmethod
    def _format_context(context_raw: list[Any]) -> str:
        lines = []
        for turn in context_raw:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                lines.append(f"{turn[0]}: {turn[1]}")
        return "\n".join(lines)

    def evaluate_prediction(
        self, task: TaskInstance, prediction: Any
    ) -> dict[str, float]:
        ks = _evaluation_ks(self.config.get("dataset.evaluation.ks"))
        primary_k = int(self.config.get("dataset.evaluation.k", max(ks)))
        if primary_k not in ks:
            ks.append(primary_k)
            ks = sorted(set(ks))
        max_k = max(ks)
        gold_titles = _as_list(task.gold_answer)
        gold_ids = _as_list(task.metadata.get("gold_ids"))
        context_titles = _as_list(task.metadata.get("context_titles"))
        context_ids = _as_list(task.metadata.get("context_ids"))
        schema_valid = is_valid_recommendation_json(prediction)
        parsed = parse_recommendation_prediction(prediction)
        ranked = parsed[:max_k]

        matched_by_k: dict[int, set[int]] = {}
        dcg_by_k: dict[int, float] = {}
        matched_gold: set[int] = set()
        running_dcg = 0.0
        for rank, pred in enumerate(ranked, start=1):
            gold_idx = _match_gold_index(
                pred,
                gold_titles=gold_titles,
                gold_ids=gold_ids,
                used=matched_gold,
            )
            if gold_idx is not None:
                matched_gold.add(gold_idx)
                running_dcg += 1.0 / math.log2(rank + 1)
            if rank in ks:
                matched_by_k[rank] = set(matched_gold)
                dcg_by_k[rank] = running_dcg

        for k in ks:
            matched_by_k.setdefault(k, set(matched_gold))
            dcg_by_k.setdefault(k, running_dcg)

        gold_count = len(gold_titles)
        pred_count_at_primary = min(len(parsed), primary_k)
        metrics: dict[str, float] = {}
        for k in ks:
            hits = len(matched_by_k[k])
            recall = hits / gold_count if gold_count else 0.0
            hit = 1.0 if hits else 0.0
            ideal_hits = min(gold_count, k)
            idcg = sum(
                1.0 / math.log2(rank + 1)
                for rank in range(1, ideal_hits + 1)
            )
            ndcg = dcg_by_k[k] / idcg if idcg else 0.0
            metrics[f"hit_at_{k}"] = hit
            metrics[f"recall_at_{k}"] = recall
            metrics[f"ndcg_at_{k}"] = ndcg

        primary_hits = len(matched_by_k[primary_k])
        repeat_count = sum(
            1
            for pred in parsed[:primary_k]
            if _matches_any_context(pred, context_titles=context_titles, context_ids=context_ids)
        )
        context_repeat_rate = (
            repeat_count / pred_count_at_primary
            if pred_count_at_primary else 0.0
        )
        return {
            "success": metrics[f"hit_at_{primary_k}"],
            # Backward-compatible aliases for the configured primary k.
            "hit_at_k": metrics[f"hit_at_{primary_k}"],
            "recall_at_k": metrics[f"recall_at_{primary_k}"],
            "ndcg_at_k": metrics[f"ndcg_at_{primary_k}"],
            **metrics,
            "schema_valid": 1.0 if schema_valid else 0.0,
            "context_repeat_rate": context_repeat_rate,
            "num_gold": float(gold_count),
            "num_predictions": float(pred_count_at_primary),
            "num_hits": float(primary_hits),
        }

    @staticmethod
    def _assign_pattern(item: dict) -> str:
        turns = item.get("turn_count", 1)
        if turns >= 5:
            return "FP"
        if turns >= 3:
            return "PO"
        return "SL"

    @staticmethod
    def _stub_data() -> list[dict]:
        return [
            {
                "conversation_context": "I'm looking for a movie like The Matrix. Something sci-fi with action.",
                "gold_recommendations": ["Inception", "Blade Runner 2049"],
                "source": "redial",
                "turn_count": 2,
            },
        ]


def parse_recommendation_prediction(prediction: Any) -> list[dict[str, str | None]]:
    """Parse ranked movie recommendations from model output.

    Preferred format is JSON from ``get_answer_format_prompt()``, but this
    also tolerates plain lists, dicts, Markdown bullets, and short prose.
    """
    if prediction is None:
        return []
    if isinstance(prediction, list):
        return _dedupe_recs([_coerce_recommendation_item(p) for p in prediction])
    if isinstance(prediction, dict):
        return _dedupe_recs(_recommendations_from_obj(prediction))

    text = str(prediction).strip()
    if not text:
        return []

    parsed = _try_parse_json_recommendations(text)
    if parsed:
        return _dedupe_recs(parsed)

    ids = re.findall(r"\btt\d{6,10}\b", text)
    candidates = _split_freeform_recommendations(text)
    recs = [_coerce_recommendation_item(c) for c in candidates]
    for rec, imdb_id in zip(recs, ids):
        rec["imdb_id"] = imdb_id
    if not recs and ids:
        recs = [{"title": None, "imdb_id": imdb_id} for imdb_id in ids]
    return _dedupe_recs(recs)


def is_valid_recommendation_json(prediction: Any) -> bool:
    if isinstance(prediction, dict):
        obj = prediction
    elif isinstance(prediction, str):
        cleaned = prediction.strip()
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            return False
    else:
        return False

    recs = obj.get("recommendations") if isinstance(obj, dict) else None
    if not isinstance(recs, list):
        return False
    return all(
        isinstance(rec, dict)
        and isinstance(rec.get("title"), str)
        and ("imdb_id" not in rec or rec.get("imdb_id") is None or isinstance(rec.get("imdb_id"), str))
        for rec in recs
    )


def _try_parse_json_recommendations(text: str) -> list[dict[str, str | None]]:
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    candidates = [cleaned]
    start_obj, end_obj = cleaned.find("{"), cleaned.rfind("}")
    if start_obj != -1 and end_obj > start_obj:
        candidates.append(cleaned[start_obj:end_obj + 1])
    start_arr, end_arr = cleaned.find("["), cleaned.rfind("]")
    if start_arr != -1 and end_arr > start_arr:
        candidates.append(cleaned[start_arr:end_arr + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        recs = _recommendations_from_obj(obj)
        if recs:
            return recs
    return []


def _recommendations_from_obj(obj: Any) -> list[dict[str, str | None]]:
    if isinstance(obj, list):
        return [_coerce_recommendation_item(item) for item in obj]
    if not isinstance(obj, dict):
        return [_coerce_recommendation_item(obj)]
    for key in ("recommendations", "movies", "titles", "items", "results"):
        if key in obj:
            return _recommendations_from_obj(obj[key])
    return [_coerce_recommendation_item(obj)]


def _coerce_recommendation_item(item: Any) -> dict[str, str | None]:
    if isinstance(item, dict):
        title = (
            item.get("title")
            or item.get("movie")
            or item.get("name")
            or item.get("movie_title")
        )
        imdb_id = item.get("imdb_id") or item.get("id") or item.get("imdb")
        return {
            "title": str(title).strip() if title else None,
            "imdb_id": str(imdb_id).strip() if imdb_id else None,
        }
    raw = str(item).strip()
    imdb_match = re.search(r"\btt\d{6,10}\b", raw)
    imdb_id = imdb_match.group(0) if imdb_match else None
    if imdb_id:
        raw = raw.replace(imdb_id, "")
    title = _clean_freeform_title(raw)
    return {"title": title or None, "imdb_id": imdb_id}


def _split_freeform_recommendations(text: str) -> list[str]:
    lines = []
    for line in re.split(r"[\n|;]+", text):
        line = line.strip()
        if not line:
            continue
        # Avoid treating a long paragraph as one title; split simple sentences.
        if len(line) > 120:
            lines.extend(part.strip() for part in re.split(r"\.(?:\s+|$)", line))
        else:
            lines.append(line)
    return [line for line in lines if line]


def _clean_freeform_title(text: str) -> str:
    text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text.strip())
    text = re.sub(r"\([^)]*recommend.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:title|movie)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip(" \t\r\n\"'`.,:;")
    # Keep release years when present; title normalization removes them for matching.
    return text


def _dedupe_recs(recs: list[dict[str, str | None]]) -> list[dict[str, str | None]]:
    seen = set()
    out = []
    for rec in recs:
        if not rec:
            continue
        title = rec.get("title")
        imdb_id = rec.get("imdb_id")
        if not title and not imdb_id:
            continue
        key = imdb_id or _normalize_title(title or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"title": title, "imdb_id": imdb_id})
    return out


def _match_gold_index(
    pred: dict[str, str | None],
    gold_titles: list[Any],
    gold_ids: list[Any],
    used: set[int],
) -> int | None:
    pred_id = (pred.get("imdb_id") or "").strip()
    if pred_id:
        for idx, gold_id in enumerate(gold_ids):
            if idx in used:
                continue
            if pred_id == str(gold_id).strip():
                return idx

    pred_title = _normalize_title(pred.get("title") or "")
    if not pred_title:
        return None
    for idx, gold_title in enumerate(gold_titles):
        if idx in used:
            continue
        gold_norm = _normalize_title(str(gold_title))
        if pred_title == gold_norm:
            return idx
        if len(gold_norm) >= 4 and _contains_title(pred_title, gold_norm):
            return idx
    return None


def _matches_any_context(
    pred: dict[str, str | None],
    context_titles: list[Any],
    context_ids: list[Any],
) -> bool:
    pred_id = (pred.get("imdb_id") or "").strip()
    if pred_id and pred_id in {str(i).strip() for i in context_ids}:
        return True
    pred_title = _normalize_title(pred.get("title") or "")
    if not pred_title:
        return False
    return any(pred_title == _normalize_title(str(title)) for title in context_titles)


def _contains_title(pred_norm: str, gold_norm: str) -> bool:
    return bool(re.search(rf"(^|\s){re.escape(gold_norm)}($|\s)", pred_norm))


def _normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"\(\s*\d{4}\s*\)", " ", title)
    title = title.replace("&", " and ")
    title = title.translate(str.maketrans("", "", string.punctuation))
    return " ".join(title.split())


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _flatten_once(value: list[Any]) -> list[Any]:
    out = []
    for item in value:
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return out


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


def _evaluation_ks(value: Any) -> list[int]:
    if value is None:
        return [10]
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = list(value)
    ks = sorted({
        int(part)
        for part in parts
        if str(part).strip() and int(part) > 0
    })
    return ks or [10]


def _clean_optional_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_redial_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [text]


def _extract_titles_with_years(text: str) -> list[str]:
    token = r"(?:[A-Z0-9][A-Za-z0-9'&:.-]*|[a-z]{1,3})"
    pattern = re.compile(rf"\b({token}(?:\s+{token})*)\s*\((?:18|19|20)\d{{2}}\)")
    titles = []
    seen = set()
    for match in pattern.finditer(text):
        title = match.group(1).strip(" \t\r\n\"'`.,:;")
        norm = _normalize_title(title)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        titles.append(title)
    return titles


def _count_conversation_turns(text: str) -> int:
    turns = len(re.findall(r"(?m)^\s*(?:User|System):", text))
    return max(1, turns)


def _safe_task_id_part(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "none"
