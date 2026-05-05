"""Subprocess Python executor used inside the TD probe loop."""

from __future__ import annotations

import subprocess


def run_python(code: str, timeout_s: int = 15) -> dict:
    try:
        proc = subprocess.run(
            ["python", "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "TIMEOUT", "returncode": -1}


def format_result(result: dict) -> str:
    return f"STDOUT:\n{result.get('stdout', '')}\nSTDERR:\n{result.get('stderr', '')}\n"
