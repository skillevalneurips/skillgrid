import json
import logging
import os
import random
import re
import string
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Union

from tqdm import tqdm

from camel.agents import ChatAgent
from camel.benchmarks.base import BaseBenchmark
from camel.messages import BaseMessage
from camel.retrievers.auto_retriever import AutoRetriever
from camel.embeddings import SentenceTransformerEncoder

logger = logging.getLogger(__name__)


class RetrieverProtocol(Protocol):
    r"""Protocol for the retriever class. Any retriever class implementing
    this protocol can be used in the benchmark class.
    """

    def retrieve(
        self, query: str, contents: List[str], **kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        r"""Retrieve the relevant content for the query.

        Args:
            query (str): The query to retrieve the content for.
            contents (List[str]): The list of contents to search in.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Returns:
            Dict[str, Any]: The relevant content for the query.
        """
        ...

    def reset(self, **kwargs) -> bool:
        r"""Reset the retriever.
        Some benchmarks may require resetting the retriever
        after each query.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if the reset was successful, False otherwise.
        """
        ...


class DefaultGAIARetriever(AutoRetriever):
    r"""Default retriever for the GAIA benchmark.
    This retriever uses AutoRetriever in camel to retrieve the content based on
    the query.
    """

    def __init__(self, **kwargs):
        if "embedding_model" not in kwargs:
            # Use a small, fast local model by default to avoid API keys
            kwargs["embedding_model"] = SentenceTransformerEncoder(
                model_name="all-MiniLM-L6-v2"
            )
        super().__init__(**kwargs)

    def retrieve(
        self, query: str, contents: List[str], **kwargs: Any
    ) -> Dict[str, Any]:
        r"""Retrieve the content based on the query.

        Args:
            query (str): The query to search for.
            contents (List[str]): The list of contents to search from.
            **kwargs (Any): The keyword arguments to pass to the
                retriever.

        Returns:
            Dict[str, Any]: The retrieved content.
        """
        # CAMEL's vector retriever can raise unexpected runtime errors for some
        # documents (e.g. UnboundLocalError: local variable 'elements' referenced
        # before assignment). For benchmark stability we fail open and return no
        # retrieved context so the task can still be attempted.
        try:
            return self.run_vector_retriever(  # type: ignore[arg-type]
                query, contents, **kwargs
            )
        except Exception as e:
            logger.warning(
                "Retriever failed; continuing without retrieved context. "
                f"Error: {e}"
            )
            return {"Retrieved Context": [], "error": str(e)}

    def reset(self, **kwargs: Any) -> bool:
        r"""Reset the retriever.

        Args:
            **kwargs (Any): The keyword arguments to pass to the
                retriever.

        Returns:
            bool: Whether the reset was successful.
        """
        path = Path(self.vector_storage_local_path or os.getcwd())
        task_id = str(kwargs.get("task_id", uuid.uuid4()))
        retriever_dir = path / task_id
        if not retriever_dir.exists():
            try:
                retriever_dir.mkdir(parents=True)
            except Exception as e:
                logger.error(
                    "Error in creating directory: " + f"{retriever_dir}: {e!s}"
                )
                return False
        self.vector_storage_local_path = str(retriever_dir)
        return True


class GAIABenchmark(BaseBenchmark):
    r"""GAIA Benchmark adapted from `"GAIA: a benchmark for General AI
    Assistants"
    <https://huggingface.co/datasets/gaia-benchmark/GAIA>`_.

    Args:
        data_dir (str): The directory to save the data.
        save_to (str): The file to save the results.
        retriever (Optional[RetrieverProtocol]): The retriever to use.
            (default: :obj:`None`)
        processes (int, optional): The number of processes to use.
            (default: :obj:`1`)
    """

    def __init__(
        self,
        data_dir: str,
        save_to: str,
        retriever: Optional[RetrieverProtocol] = None,
        processes: int = 1,
    ):
        r"""Initialize the GAIA benchmark.

        Args:
            data_dir (str): The directory to save the data.
            save_to (str): The file to save the results.
            retriever (Optional[RetrieverProtocol], optional): The retriever to
                use. (default: :obj:`None`)
            processes (int, optional): The number of processes to use for
                parallel processing. (default: :obj:`1`)
        """
        super().__init__("gaia", data_dir, save_to, processes)
        self.retriever = retriever or DefaultGAIARetriever()

    def download(self):
        r"""Download the GAIA dataset."""
        from huggingface_hub import snapshot_download

        import os

        snapshot_download(
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
            local_dir=self.data_dir,
            local_dir_use_symlinks=True,
            token=os.environ.get("HF_TOKEN"),
        )

    def _ensure_metadata_parquet(self, path: Path, hf_path: str) -> None:
        r"""Download metadata.parquet from the repo if it is missing (e.g. gated content)."""
        metadata_file = path / "metadata.parquet"
        if metadata_file.exists():
            return
        from huggingface_hub import hf_hub_download

        import os

        local_dir = self.data_dir if isinstance(self.data_dir, str) else str(self.data_dir)
        hf_hub_download(
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
            filename=hf_path,
            local_dir=local_dir,
            local_dir_use_symlinks=True,
            token=os.environ.get("HF_TOKEN"),
        )

    def load(self, force_download=False):
        r"""Load the GAIA dataset.

        Args:
            force_download (bool, optional): Whether to
                force download the data.
        """
        import pandas as pd

        if force_download:
            logger.info("Force downloading data.")
            self.download()

        # Define validation and test directories
        valid_dir = self.data_dir / "2023/validation"
        test_dir = self.data_dir / "2023/test"

        # Check if directories exist; if not, download the data
        if not valid_dir.is_dir() or not test_dir.is_dir():
            logger.info("Data not found. Downloading data.")
            self.download()

        # Load metadata for both validation and test datasets
        for path, label in zip([valid_dir, test_dir], ["valid", "test"]):
            self._data[label] = []
            metadata_file = path / "metadata.parquet"
            if not metadata_file.exists():
                logger.info("Metadata parquet missing (gated content). Downloading %s...", metadata_file.name)
                hf_path = "2023/validation/metadata.parquet" if label == "valid" else "2023/test/metadata.parquet"
                self._ensure_metadata_parquet(path, hf_path)
            df = pd.read_parquet(metadata_file)
            for _, row in df.iterrows():
                data = row.to_dict()
                if data["task_id"] == "0-0-0-0-0":
                    continue
                # convert level to int (parquet stores as string)
                data["Level"] = int(data["Level"])
                if data["file_name"]:
                    data["file_name"] = path / data["file_name"]
                self._data[label].append(data)
        return self

    @property
    def train(self):
        r"""Get the training set."""
        raise NotImplementedError("GAIA does not have a training set.")

    def run(  # type: ignore[override]
        self,
        agent: ChatAgent,
        on: Literal["train", "valid", "test"],
        level: Union[int, List[int], Literal["all"]],
        randomize: bool = False,
        subset: Optional[int] = None,
    ) -> Dict[str, Any]:
        r"""Run the benchmark.

        Args:
            agent (ChatAgent): The agent to run the benchmark.
            on (Literal["valid", "test"]): The set to run the benchmark.
            level (Union[int, List[int], Literal["all"]]): The level to run
                the benchmark.
            randomize (bool, optional): Whether to randomize the data.
                (default: :obj:`False`)
            subset (Optional[int], optional): The subset of data to run.
                (default: :obj:`None`)

        Returns:
            Dict[str, Any]: The results of the benchmark.
        """
        # Validate inputs
        if on not in ["valid", "test"]:
            raise ValueError(
                f"Invalid value for `on`: {on}, expected 'valid' or 'test'."
            )

        levels = (
            [1, 2, 3]
            if level == "all"
            else [level]
            if isinstance(level, int)
            else level
        )
        if not all(
            isinstance(level, int) and level in [1, 2, 3] for level in levels
        ):
            raise ValueError(
                f"Invalid value for `level`: {level}, expected 1, 2, 3 "
                "or 'all'."
            )

        logger.info(f"Running benchmark on {on} set at levels {levels}.")
        datas = [data for data in self._data[on] if data["Level"] in levels]

        # Shuffle and subset data if necessary
        if randomize:
            random.shuffle(datas)
        if subset:
            datas = datas[:subset]

        logger.info(f"Number of tasks: {len(datas)}")

        # Initialize results storage
        self._results = []

        # Process tasks
        with open(self.save_to, "w") as f:
            for task in tqdm(datas, desc="Running"):
                if not self._prepare_task(task):
                    continue

                try:
                    result = agent.step(self._create_user_message(task))
                    self._process_result(agent, task, result, f)
                except Exception as e:
                    self._handle_error(task, e, f)
                finally:
                    agent.reset()

        return self._generate_summary()

    def _prepare_task(self, task: Dict[str, Any]) -> bool:
        r"""Prepare the task by validating and enriching its data."""
        if task["file_name"]:
            file_path = Path(task["file_name"])
            if not file_path.exists():
                logger.info(
                    f"Skipping task because file not found: {file_path}"
                )
                return False
            # For text-based files, try to use retriever for context
            if file_path.suffix in [".pdf", ".docx", ".doc", ".txt"]:
                if not self.retriever.reset(task_id=task["task_id"]):
                    return False
                try:
                    retrieved_info = self.retriever.retrieve(
                        query=task["Question"], contents=[task["file_name"]]
                    )
                except Exception as e:
                    # Extra safety: DefaultGAIARetriever already fails open, but
                    # if a custom retriever is used, don't let retrieval crash
                    # the whole benchmark.
                    logger.warning(
                        "Retriever raised an exception; continuing without "
                        f"retrieved context. Error: {e}"
                    )
                    retrieved_info = {"Retrieved Context": [], "error": str(e)}
                retrieved_content = [
                    item["text"]
                    for item in retrieved_info.get("Retrieved Context", [])
                ]
                if retrieved_content:
                    task["Question"] += "\n" + "\n".join(retrieved_content)
            # For all other file types, let the agent handle them with tools
            # Don't skip - the agent can use parse_file tool
            else:
                logger.info(
                    f"File format {file_path.suffix} will be handled by agent tools"
                )
        return True

    def _create_user_message(self, task: Dict[str, Any]) -> BaseMessage:
        r"""Create a user message from a task."""
        # Include filename in the prompt so agent can use parse_file tool
        question = task["Question"]
        if task.get("file_name"):
            file_path = Path(task["file_name"])
            question = f"{question}\n\n[Attached file: {file_path.name}]"
        return BaseMessage.make_user_message(
            role_name="User",
            content=question,
        )

    def _process_result(
        self,
        agent: ChatAgent,
        task: Dict[str, Any],
        result: Any,
        file_obj: Any,
    ) -> None:
        r"""Process and store the result of a task."""
        model_answer = self.get_final_answer(result.msgs[0].content)
        final_answer = task["Final answer"]
        score = self.question_scorer(model_answer, final_answer)
        tool_calls = result.info.get("tool_calls", [])

        result_data = {
            "task_id": task["task_id"],
            "question": task["Question"],
            "level": task["Level"],
            "model_answer": model_answer,
            "ground_truth": final_answer,
            "tool_calls": [tool.model_dump() for tool in tool_calls],
            "error": None,
            "score": int(score),
            "history": agent.memory.get_context(),
        }
        self._results.append(result_data)
        file_obj.write(
            json.dumps(result_data, indent=2, ensure_ascii=False) + "\n"
        )
        file_obj.flush()

    def _handle_error(
        self, task: Dict[str, Any], error: Exception, file_obj: Any
    ) -> None:
        r"""Handle errors encountered during task processing."""
        logger.warning(f"Error processing task {task['task_id']}: {error}")
        error_data = {
            "task_id": task["task_id"],
            "question": task["Question"],
            "level": task["Level"],
            "model_answer": "ERROR",
            "ground_truth": task["Final answer"],
            "tool_calls": [],
            "error": str(error),
            "score": 0,
        }
        self._results.append(error_data)
        file_obj.write(
            json.dumps(error_data, indent=2, ensure_ascii=False) + "\n"
        )
        file_obj.flush()

    def _generate_summary(self) -> Dict[str, Any]:
        r"""Generate and return a summary of the benchmark results."""
        return {
            "total": len(self._results),
            "correct": sum(result["score"] for result in self._results),
            "results": self._results,
        }

    def question_scorer(self, model_answer: str, ground_truth: str) -> bool:
        r"""Scorer for the GAIA benchmark.
        https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/
        scorer.py

        Args:
            model_answer (str): The model answer.
            ground_truth (str): The ground truth answer.

        Returns:
            bool: The score of the model
        """

        def is_float(element: Any) -> bool:
            try:
                float(element)
                return True
            except ValueError:
                return False

        if is_float(ground_truth):
            logger.info(f"Evaluating {model_answer} as a number.")
            normalized_answer = self.normalize_number_str(model_answer)
            return normalized_answer == float(ground_truth)

        elif any(char in ground_truth for char in [",", ";"]):
            logger.info(
                f"Evaluating {model_answer} as a comma separated list."
            )
            gt_elems = self.split_string(ground_truth)
            ma_elems = self.split_string(model_answer)

            if len(gt_elems) != len(ma_elems):
                # NOTE: don't pass warning categories as extra logger args;
                # that triggers %-formatting and causes noisy logging errors.
                logger.warning(
                    "Answer lists have different lengths, returning False."
                )
                return False

            comparisons = []
            for ma_elem, gt_elem in zip(ma_elems, gt_elems):
                if is_float(gt_elem):
                    normalized_ma_elem = self.normalize_number_str(ma_elem)
                    comparisons.append(normalized_ma_elem == float(gt_elem))
                else:
                    ma_elem = self.normalize_str(ma_elem, remove_punct=False)
                    gt_elem = self.normalize_str(gt_elem, remove_punct=False)
                    comparisons.append(ma_elem == gt_elem)
            return all(comparisons)
        else:
            logger.info(f"Evaluating {model_answer} as a string.")
            ma_elem = self.normalize_str(model_answer)
            gt_elem = self.normalize_str(ground_truth)
            return ma_elem == gt_elem

    def normalize_number_str(self, number_str: str) -> float:
        # Best-effort numeric parsing:
        # - strip common symbols
        # - if extra text/units exist (e.g. "0.1 m³"), extract the first number
        cleaned = number_str
        for char in ["$", "%", ","]:
            cleaned = cleaned.replace(char, "")
        cleaned = cleaned.strip()

        try:
            return float(cleaned)
        except ValueError:
            # Extract the first float-like token (supports scientific notation)
            m = re.search(
                r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
                cleaned,
            )
            if m:
                try:
                    return float(m.group(0))
                except ValueError:
                    pass
            logger.error(
                f"String {number_str} cannot be normalized to number str."
            )
            return float('inf')

    def split_string(
        self, s: str, char_list: Optional[List[str]] = None
    ) -> list[str]:
        r"""Split a string based on a list of characters.

        Args:
            s (str): The string to split.
            char_list (Optional[List[str]], optional): T
                he list of characters to split on.
                (default: :obj:`None`)
        """
        if char_list is None:
            char_list = [",", ";"]
        pattern = f"[{''.join(char_list)}]"
        return re.split(pattern, s)

    def normalize_str(self, input_str, remove_punct=True) -> str:
        r"""Normalize a string.

        Args:
            input_str: The input string to normalize.
            remove_punct: Whether to remove punctuation.

        Returns:
            str: The normalized string.
        """
        no_spaces = re.sub(r"\s", "", input_str)
        if remove_punct:
            translator = str.maketrans("", "", string.punctuation)
            return no_spaces.lower().translate(translator)
        else:
            return no_spaces.lower()

    def get_final_answer(self, content: str) -> str:
        r"""Get the final answer from the content.

        Args:
            content (str): The content to extract the final answer from.

        Returns:
            str: The final answer.
        """
        # Be robust to casing and minor formatting variations, e.g.
        # "FINAL ANSWER: 42", "Final Answer - 42", "FINAL ANSWER\n42"
        m = re.search(
            r"final\s*answer\s*[:\-]?\s*(.*)$",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return "FINAL ANSWER not found"
        return m.group(1).strip()