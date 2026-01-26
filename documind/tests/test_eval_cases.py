"""Evaluation fixtures tests."""

import os
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.eval import check_expectations, evaluate_pages, load_json


ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "tests" / "fixtures"
GOLDEN_DIR = ROOT / "tests" / "golden"


@pytest.mark.parametrize(
    "fixture_path",
    sorted(FIXTURES_DIR.glob("*.json")),
    ids=lambda path: Path(path).stem,
)
def test_eval_case(fixture_path: Path) -> None:
    case_name = fixture_path.stem
    golden_path = GOLDEN_DIR / f"{case_name}.json"
    fixture = load_json(fixture_path)
    expected = load_json(golden_path)
    result = evaluate_pages(fixture["pages"])
    failures = check_expectations(result, expected)
    assert not failures, f"{case_name} failed: " + " | ".join(failures)
