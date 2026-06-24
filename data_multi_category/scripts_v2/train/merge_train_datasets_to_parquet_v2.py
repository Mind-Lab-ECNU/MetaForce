#!/usr/bin/env python3
"""Merge train_merge_v2 datasets with strict validations and type normalization."""

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

TRAIN_DATASETS: List[str] = [
    "InfographicVQA_2000",
    # "DocVQA_2000",
    "AOKVQA_2000",
    "COCOQA_2000",
    "VQAv2_2000",
    "Visual7W_2000",
    "TQA_2000",
    # "TallyQA_2000",
    "VQARAD_2000",
    "VSR_2000",
    # "LocalizedNarratives_2000",
    "MapQA_2000",
    # "CLEVR_2000",
    # "CLEVR_MATH_2000",
    "STVQA_2000",
    "TextVQA_2000",
    "VisualMRC_2000",
    "IAM_2000",
    # "RenderedText_2000",
    "Screen2Words_2000",
    "TATQA_2000",
    "FinQA_2000",
    "HiTab_2000",
    "RoBUTWTQ_2000",
    "RoBUTSQA_2000",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-base-dir", type=Path, default=None)
    parser.add_argument("--train-merge-dir-name", type=str, default="train_merge_v2")
    parser.add_argument("--test-merge-dir-name", type=str, default="test_merge_v2")
    parser.add_argument("--out-train-json", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    output_base_dir = args.output_base_dir if args.output_base_dir is not None else args.data_root
    train_root = output_base_dir / args.train_merge_dir_name
    output_root = output_base_dir / args.train_merge_dir_name
    output_root.mkdir(parents=True, exist_ok=True)

    all_records: List[dict] = []
    errors: Dict[str, List[str]] = {}
    stats: Dict[Tuple[str, ...], Dict[str, object]] = {}

    print("=" * 80)
    print(f"Merge train datasets -> {args.train_merge_dir_name}")
    print("=" * 80)

    for dataset_name in TRAIN_DATASETS:
        path = train_root / dataset_name / "train.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing train merge source: {path}")

        records = load_json(path)
        valid_count = 0
        for idx, sample in enumerate(records):
            sample_errors = validate_sample_for_merge(sample, args.data_root)
            key = f"{dataset_name}[{idx}]"
            if sample_errors:
                errors[key] = sample_errors
                continue

            valid_count += 1
            item = normalize_record_for_reuse(sample)
            if "extra_info" in item and isinstance(item["extra_info"], dict):
                collect_extra_info_stats(item["extra_info"], stats)
            all_records.append(item)

        print(f"{dataset_name}: loaded={len(records)} valid={valid_count} invalid={len(records)-valid_count}")

    if errors:
        print("-" * 80)
        print(f"Validation errors: {len(errors)}")
        print(summarize_errors(errors))
        if args.strict:
            raise ValueError("Train merge validation failed in strict mode")

    if stats:
        policies = decide_extra_info_policies(stats)
        for rec in all_records:
            if "extra_info" in rec and isinstance(rec["extra_info"], dict):
                rec["extra_info"] = normalize_extra_info_by_policy(rec["extra_info"], policies)
            rec = ensure_extra_info_jsonable(rec)

    all_records = [ensure_extra_info_jsonable(rec) for rec in all_records]

    out_json = args.out_train_json if args.out_train_json is not None else output_root / "train_merge_v2.json"
    out_parquet = out_json.with_suffix(".parquet")
    save_json(out_json, all_records)
    save_parquet(out_parquet, all_records)

    print("-" * 80)
    print(f"Saved JSON: {out_json}")
    print(f"Saved Parquet: {out_parquet}")
    print(f"Merged records: {len(all_records)}")
    print(f"data_source distribution: {build_stats_counter(all_records, 'data_source')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
