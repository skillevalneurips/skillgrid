"""Download upstream preprocessed ReDial CSVs for conversational_rec."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
from pathlib import Path

RAW_BASE_URL = (
    "https://raw.githubusercontent.com/zhouhanxie/"
    "neighborhood-based-CF-for-CRS/main/datasets/redial"
)
FILES = ("redial_train.csv", "redial_test.csv")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "redial"


def download_file(url: str, path: Path, *, force: bool = False) -> bool:
    if path.exists() and not force:
        print(f"Exists: {path}")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(url, timeout=60) as response:
            shutil.copyfileobj(response, tmp)
    tmp_path.replace(path)
    print(f"Wrote: {path}")
    return True


def prepare(output_dir: Path = DEFAULT_OUTPUT_DIR, *, force: bool = False) -> None:
    for filename in FILES:
        download_file(f"{RAW_BASE_URL}/{filename}", output_dir / filename, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where redial_train.csv and redial_test.csv are written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing local CSVs.",
    )
    args = parser.parse_args()
    prepare(args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
