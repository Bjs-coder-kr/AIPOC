"""Register real PDF fixtures and golden stubs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES_REAL = ROOT / "tests" / "fixtures_real"
GOLDEN_REAL = ROOT / "tests" / "golden_real"


def _run(cmd: list[str]) -> list[str]:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Register real PDF fixtures.")
    parser.add_argument("path", help="PDF file path or directory.")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--redact", dest="redact", action="store_true", default=True)
    parser.add_argument("--no-redact", dest="redact", action="store_false")
    args = parser.parse_args()

    FIXTURES_REAL.mkdir(parents=True, exist_ok=True)
    GOLDEN_REAL.mkdir(parents=True, exist_ok=True)

    fixture_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "make_fixture_from_pdf.py"),
        args.path,
        "--output-dir",
        str(FIXTURES_REAL),
    ]
    if args.max_pages is not None:
        fixture_cmd.extend(["--max-pages", str(args.max_pages)])
    if args.redact:
        fixture_cmd.append("--redact")
    else:
        fixture_cmd.append("--no-redact")

    fixture_paths = _run(fixture_cmd)
    if not fixture_paths:
        print("No fixtures generated.")
        return 1

    golden_paths: list[str] = []
    for fixture_path in fixture_paths:
        golden_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "make_golden_stub.py"),
            fixture_path,
            "--output-dir",
            str(GOLDEN_REAL),
        ]
        golden_paths.extend(_run(golden_cmd))

    print("Fixtures:")
    for path in fixture_paths:
        print(f"- {path}")
    print("Goldens:")
    for path in golden_paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
