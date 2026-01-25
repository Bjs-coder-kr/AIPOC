"""Generate a golden stub JSON from a fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.eval import evaluate_pages, load_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate golden stub JSON.")
    parser.add_argument("fixture", help="Fixture JSON path.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "golden"),
        help="Output directory for golden stub JSON files.",
    )
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixture = load_json(fixture_path)
    result = evaluate_pages(fixture["pages"])

    page_types = {
        str(profile["page"]): profile["type"] for profile in result["page_profiles"]
    }
    page_confidences = {
        str(profile["page"]): round(float(profile.get("confidence", 0.0)), 2)
        for profile in result["page_profiles"]
    }

    stub = {
        "expected_dominant_type": result["dominant_type"],
        "expected_actionable_min": result["actionable_count"],
        "expected_actionable_max": result["actionable_count"],
        "expected_subtypes_present": result["subtypes"],
        "expected_subtypes_absent": [],
        "expected_page_types": page_types,
        "expected_page_type_min_confidence": page_confidences,
        "notes": "Edit these expectations to tighten recall/precision for regression.",
    }

    output_path = output_dir / f"{fixture_path.stem}.json"
    output_path.write_text(
        json.dumps(stub, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
