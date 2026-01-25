"""Evaluation harness for rule-based quality checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from documind.profile.classify import classify_pages, classify_text, dominant_type_from_pages
from documind.quality.detectors import (
    consistency,
    formatting,
    punctuation,
    readability,
    redundancy,
    spelling_ko,
)
from documind.quality.pipeline import _apply_issue_policies, _dedup_issues, _score
from documind.text.normalize import normalize_pages


FIXTURES_DIR = ROOT / "tests" / "fixtures"
GOLDEN_DIR = ROOT / "tests" / "golden"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_pages(pages: list[dict], language: str = "ko") -> dict[str, Any]:
    normalized = normalize_pages(pages)
    page_profiles = classify_pages(normalized["pages"])

    issues = readability.detect(
        normalized["pages"], language=language, page_profiles=page_profiles
    )
    issues.extend(
        redundancy.detect(
            normalized["pages"], language=language, page_profiles=page_profiles
        )
    )
    issues.extend(punctuation.detect(normalized["pages"], language=language))
    issues.extend(formatting.detect(normalized["pages"], language=language))
    issues.extend(consistency.detect(normalized["pages"], language=language))
    issues.extend(spelling_ko.detect(normalized["pages"], language=language))

    profile_text = "\n\n".join(page["text"] for page in normalized["pages"][:3])
    document_profile = classify_text(profile_text)
    dominant_type = dominant_type_from_pages(page_profiles)
    if document_profile["confidence"] < 0.6 or dominant_type == "MIXED":
        document_profile["dominant_type"] = "MIXED"
    else:
        document_profile["dominant_type"] = dominant_type

    issues = _apply_issue_policies(issues, page_profiles, language, normalized["pages"])
    issues = _dedup_issues(issues)

    actionable_count = sum(1 for issue in issues if issue.kind in {"ERROR", "WARNING"})
    note_count = sum(1 for issue in issues if issue.kind == "NOTE")
    subtypes = sorted({issue.subtype for issue in issues if issue.subtype})
    raw_score = _score(issues)

    return {
        "dominant_type": document_profile["dominant_type"],
        "actionable_count": actionable_count,
        "note_count": note_count,
        "subtypes": subtypes,
        "raw_score": raw_score,
        "issues": issues,
        "page_profiles": page_profiles,
    }


def check_expectations(result: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    expected_type = expected.get("expected_dominant_type")
    if expected_type and result["dominant_type"] != expected_type:
        failures.append(
            f"dominant_type: expected={expected_type} actual={result['dominant_type']}"
        )

    min_actionable = expected.get("expected_actionable_min")
    if min_actionable is not None and result["actionable_count"] < min_actionable:
        failures.append(
            "actionable_count: expected>=%s actual=%s"
            % (min_actionable, result["actionable_count"])
        )

    max_actionable = expected.get("expected_actionable_max")
    if max_actionable is not None and result["actionable_count"] > max_actionable:
        failures.append(
            "actionable_count: expected<=%s actual=%s"
            % (max_actionable, result["actionable_count"])
        )

    expected_present = expected.get("expected_subtypes_present", [])
    actual_subtypes = set(result["subtypes"])
    missing: list[str] = []
    for item in expected_present:
        if item.endswith("*"):
            prefix = item[:-1]
            if not any(subtype.startswith(prefix) for subtype in actual_subtypes):
                missing.append(item)
        elif item not in actual_subtypes:
            missing.append(item)
    if missing:
        failures.append("subtypes_present missing=%s" % ",".join(missing))

    expected_absent = expected.get("expected_subtypes_absent", [])
    unexpected: list[str] = []
    for item in expected_absent:
        if item.endswith("*"):
            prefix = item[:-1]
            if any(subtype.startswith(prefix) for subtype in actual_subtypes):
                unexpected.append(item)
        elif item in actual_subtypes:
            unexpected.append(item)
    if unexpected:
        failures.append("subtypes_absent unexpected=%s" % ",".join(unexpected))

    expected_page_types = expected.get("expected_page_types")
    if expected_page_types:
        expected_min_conf = expected.get("expected_page_type_min_confidence", 0.0)
        profile_map = {profile["page"]: profile for profile in result["page_profiles"]}
        for page_str, expected_type in expected_page_types.items():
            page = int(page_str)
            actual = profile_map.get(page)
            if actual is None:
                failures.append(f"page_type: page={page} missing")
                continue
            actual_type = actual.get("type")
            if actual_type != expected_type:
                failures.append(
                    f"page_type: page={page} expected={expected_type} actual={actual_type}"
                )
                continue
            confidence_raw = float(actual.get("confidence", 0.0))
            confidence = round(confidence_raw, 2)
            if isinstance(expected_min_conf, dict):
                min_conf = float(expected_min_conf.get(page_str, 0.0))
            else:
                min_conf = float(expected_min_conf)
            if confidence < min_conf:
                failures.append(
                    "page_type_confidence: page=%s expected>=%s actual_raw=%.6f actual_2dp=%.2f"
                    % (page, min_conf, confidence_raw, confidence)
                )

    return failures


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = f"{'case':<22} {'dominant':<10} {'actionable':<11} {'note':<6} {'subtypes':<32} status"
    print(header)
    print("-" * len(header))
    for row in rows:
        subtypes_text = ",".join(row["subtypes"]) if row["subtypes"] else "-"
        print(
            f"{row['case']:<22} {row['dominant']:<10} "
            f"{row['actionable']:<11} {row['note']:<6} "
            f"{subtypes_text:<32} {row['status']}"
        )
        if row["failures"]:
            for failure in row["failures"]:
                print(f"  - {failure}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate fixtures against goldens.")
    parser.add_argument(
        "--fixtures-dir",
        default=str(FIXTURES_DIR),
        help="Directory containing fixture JSON files.",
    )
    parser.add_argument(
        "--golden-dir",
        default=str(GOLDEN_DIR),
        help="Directory containing golden JSON files.",
    )
    args = parser.parse_args()

    fixtures_dir = Path(args.fixtures_dir)
    golden_dir = Path(args.golden_dir)

    rows: list[dict[str, Any]] = []
    failed = 0
    for fixture_path in sorted(fixtures_dir.glob("*.json")):
        case_name = fixture_path.stem
        golden_path = golden_dir / f"{case_name}.json"
        if not golden_path.exists():
            continue
        fixture = load_json(fixture_path)
        expected = load_json(golden_path)
        result = evaluate_pages(fixture["pages"])
        failures = check_expectations(result, expected)
        status = "OK" if not failures else "FAIL"
        if failures:
            failed += 1
        rows.append(
            {
                "case": case_name,
                "dominant": result["dominant_type"],
                "actionable": result["actionable_count"],
                "note": result["note_count"],
                "subtypes": result["subtypes"],
                "status": status,
                "failures": failures,
            }
        )

    _print_table(rows)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
