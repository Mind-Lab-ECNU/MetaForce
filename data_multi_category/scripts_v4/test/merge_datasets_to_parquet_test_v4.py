#!/usr/bin/env python3
"""Merge test_merge_v4 OOD datasets with strict validations and type normalization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from common import (
    DATA_ROOT,
    build_stats_counter,
    collect_extra_info_stats,
    decide_extra_info_policies,
    ensure_extra_info_jsonable,
    load_json,
    normalize_extra_info_by_policy,
    normalize_record_for_reuse,
    save_json,
    save_parquet,
    summarize_errors,
    validate_sample_for_merge,
)


OOD_DATASETS: List[str] = [
    "CLEVR_MATH_test_OOD",
]


def _load_group(
    data_root: Path,
    group_dir: Path,
    dataset_names: List[str],
    strict: bool,
) -> List[dict]:
    records: List[dict] = []
    errors: Dict[str, List[str]] = {}
    stats: Dict[Tuple[str, ...], Dict[str, object]] = {}

    for dataset_name in dataset_names:
        path = group_dir / dataset_name / "test.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing test source: {path}")

        data = load_json(path)
        valid_count = 0
        for idx, sample in enumerate(data):
            sample_errors = validate_sample_for_merge(sample, data_root)
            key = f"{dataset_name}[{idx}]"
            if sample_errors:
                errors[key] = sample_errors
                continue

            valid_count += 1
            item = normalize_record_for_reuse(sample)
            if isinstance(item.get("extra_info"), dict):
                collect_extra_info_stats(item["extra_info"], stats)
            records.append(item)

        print(
            f"{dataset_name}: loaded={len(data)} valid={valid_count} invalid={len(data) - valid_count}"
        )

    if errors:
        print("-" * 80)
        print(f"Validation errors: {len(errors)}")
        print(summarize_errors(errors))
        if strict:
            raise ValueError("Test merge validation failed in strict mode")

    if stats:
        policies = decide_extra_info_policies(stats)
        for rec in records:
            if isinstance(rec.get("extra_info"), dict):
                rec["extra_info"] = normalize_extra_info_by_policy(rec["extra_info"], policies)

    return [ensure_extra_info_jsonable(record) for record in records]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-base-dir", type=Path, default=None)
    parser.add_argument("--test-merge-dir-name", type=str, default="test_merge_v4")
    parser.add_argument("--out-test-ood-json", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    output_base_dir = args.output_base_dir if args.output_base_dir is not None else args.data_root
    test_root = output_base_dir / args.test_merge_dir_name
    output_root = output_base_dir / args.test_merge_dir_name
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Merge test datasets -> {args.test_merge_dir_name}/test_ood_merge_v4")
    print("=" * 80)

    ood_records = _load_group(args.data_root, test_root / "OOD", OOD_DATASETS, args.strict)

    out_test_ood_json = (
        args.out_test_ood_json
        if args.out_test_ood_json is not None
        else output_root / "test_ood_merge_v4.json"
    )

    save_json(out_test_ood_json, ood_records)
    save_parquet(out_test_ood_json.with_suffix(".parquet"), ood_records)

    print("-" * 80)
    print(f"OOD records: {len(ood_records)}")
    print(f"OOD data_source distribution: {build_stats_counter(ood_records, 'data_source')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
