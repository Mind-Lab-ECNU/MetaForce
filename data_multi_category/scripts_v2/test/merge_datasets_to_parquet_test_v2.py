#!/usr/bin/env python3
"""Merge test_merge_v2 ID/OOD datasets with strict validations and type normalization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from common import (  # noqa: E402
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

ID_DATASETS: List[str] = [
    "DocVQA_test_ID",
    "TallyQA_test_ID",
    "OCRVQA_test_ID",
    "RoBUTWikiSQL_test_ID",
    "MapQA_test_ID",
    "LocalizedNarratives_test_ID",
]

OOD_DATASETS: List[str] = [
    "ChartQA_test_OOD",
    "AI2D_test_OOD",
    "WebSight_test_OOD",
    "MathVista_test_OOD",
    "ScienceQA_test_OOD",
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
            if "extra_info" in item and isinstance(item["extra_info"], dict):
                collect_extra_info_stats(item["extra_info"], stats)
            records.append(item)

        print(
            f"{dataset_name}: loaded={len(data)} valid={valid_count} invalid={len(data)-valid_count}"
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
            if "extra_info" in rec and isinstance(rec["extra_info"], dict):
                rec["extra_info"] = normalize_extra_info_by_policy(rec["extra_info"], policies)

    return [ensure_extra_info_jsonable(r) for r in records]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-base-dir", type=Path, default=None)
    parser.add_argument("--train-merge-dir-name", type=str, default="train_merge_v2")
    parser.add_argument("--test-merge-dir-name", type=str, default="test_merge_v2")
    parser.add_argument("--out-test-all-json", type=Path, default=None)
    parser.add_argument("--out-test-id-json", type=Path, default=None)
    parser.add_argument("--out-test-ood-json", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    output_base_dir = args.output_base_dir if args.output_base_dir is not None else args.data_root
    test_root = output_base_dir / args.test_merge_dir_name
    output_root = output_base_dir / args.test_merge_dir_name
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(
        f"Merge test datasets -> {args.test_merge_dir_name}/"
        "test_id_merge_v6 / test_ood_merge_v6 / test_merge_v2"
    )
    print("=" * 80)

    print("Loading ID datasets...")
    id_records = _load_group(args.data_root, test_root / "ID", ID_DATASETS, args.strict)

    print("Loading OOD datasets...")
    ood_records = _load_group(args.data_root, test_root / "OOD", OOD_DATASETS, args.strict)

    all_records = id_records + ood_records

    out_test_id_json = (
        args.out_test_id_json
        if args.out_test_id_json is not None
        else output_root / "test_id_merge_v6.json"
    )
    out_test_ood_json = (
        args.out_test_ood_json
        if args.out_test_ood_json is not None
        else output_root / "test_ood_merge_v6.json"
    )
    out_test_all_json = (
        args.out_test_all_json
        if args.out_test_all_json is not None
        else output_root / "test_merge_v2.json"
    )

    save_json(out_test_id_json, id_records)
    save_parquet(out_test_id_json.with_suffix(".parquet"), id_records)

    save_json(out_test_ood_json, ood_records)
    save_parquet(out_test_ood_json.with_suffix(".parquet"), ood_records)

    save_json(out_test_all_json, all_records)
    save_parquet(out_test_all_json.with_suffix(".parquet"), all_records)

    print("-" * 80)
    print(f"ID records: {len(id_records)}")
    print(f"OOD records: {len(ood_records)}")
    print(f"All records: {len(all_records)}")
    print(f"ID data_source distribution: {build_stats_counter(id_records, 'data_source')}")
    print(f"OOD data_source distribution: {build_stats_counter(ood_records, 'data_source')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
