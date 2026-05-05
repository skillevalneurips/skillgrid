#!/usr/bin/env python3
import json
import os
import sys
from collections import defaultdict

# Get the project root (parent of scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)


def iter_json_objects(fp, chunk_size: int = 1 << 20):
    decoder = json.JSONDecoder()
    buf = ""
    while True:
        chunk = fp.read(chunk_size)
        if not chunk:
            break
        buf += chunk
        while True:
            buf = buf.lstrip()
            if not buf:
                break
            try:
                obj, idx = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                break
            yield obj
            buf = buf[idx:]

    buf = buf.lstrip()
    if buf:
        try:
            obj, _ = decoder.raw_decode(buf)
            yield obj
        except json.JSONDecodeError:
            pass


def fmt(label: str, correct: int, total: int) -> str:
    acc = (100.0 * correct / total) if total else 0.0
    return f"{label}: {correct}/{total} ({acc:.1f}%)"


def main() -> int:
    default_path = os.path.join(PROJECT_ROOT, "outputs", "qwen_local_results.jsonl")
    path = sys.argv[1] if len(sys.argv) > 1 else default_path

    per_level = defaultdict(lambda: [0, 0])  # level -> [correct, total]
    overall_correct = 0
    overall_total = 0

    with open(path, "r", encoding="utf-8") as f:
        for obj in iter_json_objects(f):
            if not isinstance(obj, dict):
                continue
            score = int(obj.get("score", 0) or 0)
            level = obj.get("level")
            overall_correct += score
            overall_total += 1
            if isinstance(level, int):
                per_level[level][0] += score
                per_level[level][1] += 1

    print(fmt("Overall accuracy", overall_correct, overall_total))
    for lvl in (1, 2, 3):
        c, t = per_level[lvl]
        print(fmt(f"Level {lvl}", c, t))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


