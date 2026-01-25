"""Evaluation fixtures tests (real PDFs)."""

import os
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.eval import check_expectations, evaluate_pages, load_json


ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "tests" / "fixtures_real"
GOLDEN_DIR = ROOT / "tests" / "golden_real"


FIXTURE_PATHS = sorted(FIXTURES_DIR.glob("*.json"))
if not FIXTURE_PATHS:
    pytest.skip("No real fixtures found.", allow_module_level=True)


@pytest.mark.parametrize(
    "fixture_path",
    FIXTURE_PATHS,
    ids=[path.stem for path in FIXTURE_PATHS],
)
def test_eval_real_case(fixture_path: Path) -> None:
    case_name = fixture_path.stem
    golden_path = GOLDEN_DIR / f"{case_name}.json"
    if not golden_path.exists():
        pytest.skip(f"Missing golden for {case_name}")
    fixture = load_json(fixture_path)
    expected = load_json(golden_path)
    result = evaluate_pages(fixture["pages"])
    failures = check_expectations(result, expected)
    assert not failures, f"{case_name} failed: " + " | ".join(failures)
