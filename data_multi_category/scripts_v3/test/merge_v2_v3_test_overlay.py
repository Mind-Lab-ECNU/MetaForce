#!/usr/bin/env python3
"""Overlay v3 test datasets onto v2 datasets by data_source, split-aware.

Example:
    python '/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/scripts_v3/test/merge_v2_v3_test_overlay.py' \
      --data-root /inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data \
      --output-base-dir /inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data \
      --v2-test-merge-dir-name /inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/test_merge_v6_tool_id \
      --v3-test-merge-dir-name /inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/test_merge_v3_ai2d_mathvista_tool_id/ \
      --out-test-merge-dir-name test_merge_v7_tool_id \
      --strict
"""

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
    save_json,
    save_json_and_parquet,
    save_parquet,
    summarize_errors,
    validate_sample_for_merge,
)


def _collect_dataset_files(group_dir: Path) -> Dict[str, Path]:
    if not group_dir.exists():
        return {}
    return {path.parent.name: path for path in sorted(group_dir.glob("*/test.json"))}


def _overlay_dataset_paths(v2_group_dir: Path, v3_group_dir: Path) -> Tuple[Dict[str, Path], List[str]]:
    v2_paths = _collect_dataset_files(v2_group_dir)
    v3_paths = _collect_dataset_files(v3_group_dir)

    merged = dict(v2_paths)
    overridden: List[str] = []
    for dataset_name, path in v3_paths.items():
        if dataset_name in merged:
            overridden.append(dataset_name)
        merged[dataset_name] = path

    return dict(sorted(merged.items(), key=lambda x: x[0])), sorted(overridden)


def _load_group(
    data_root: Path,
    dataset_paths: Dict[str, Path],
    strict: bool,
) -> Tuple[Dict[str, List[dict]], List[dict]]:
    datasets: Dict[str, List[dict]] = {}
    records: List[dict] = []
    errors: Dict[str, List[str]] = {}
    stats: Dict[Tuple[str, ...], Dict[str, object]] = {}

    for dataset_name, path in dataset_paths.items():
        data = load_json(path)
        dataset_records: List[dict] = []
        valid_count = 0
        for idx, sample in enumerate(data):
            sample_errors = validate_sample_for_merge(sample, data_root)
            key = f"{dataset_name}[{idx}]"
            if sample_errors:
                errors[key] = sample_errors
                continue

            valid_count += 1
            if isinstance(sample.get("extra_info"), dict):
                collect_extra_info_stats(sample["extra_info"], stats)
            dataset_records.append(sample)
            records.append(sample)

        datasets[dataset_name] = dataset_records
        print(
            f"{dataset_name}: loaded={len(data)} valid={valid_count} invalid={len(data) - valid_count}"
        )

    if errors:
        print("-" * 80)
        print(f"Validation errors: {len(errors)}")
        print(summarize_errors(errors))
        if strict:
            raise ValueError("Overlay merge validation failed in strict mode")

    if stats:
        policies = decide_extra_info_policies(stats)
        normalized_records: List[dict] = []
        normalized_datasets: Dict[str, List[dict]] = {}
        for dataset_name, dataset_records in datasets.items():
            normalized_items: List[dict] = []
            for rec in dataset_records:
                item = rec
                if isinstance(rec.get("extra_info"), dict):
                    item = dict(rec)
                    item["extra_info"] = normalize_extra_info_by_policy(rec["extra_info"], policies)
                item = ensure_extra_info_jsonable(item)
                normalized_items.append(item)
                normalized_records.append(item)
            normalized_datasets[dataset_name] = normalized_items
        return normalized_datasets, normalized_records

    jsonable = {
        dataset_name: [ensure_extra_info_jsonable(record) for record in dataset_records]
        for dataset_name, dataset_records in datasets.items()
    }
    flat = [record for dataset_records in jsonable.values() for record in dataset_records]
    return jsonable, flat


def _write_group_outputs(group_root: Path, datasets: Dict[str, List[dict]]) -> None:
    for dataset_name, items in datasets.items():
        save_json_and_parquet(group_root / dataset_name, "test", items)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-base-dir", type=Path, default=None)
    parser.add_argument("--v2-test-merge-dir-name", type=str, default="test_merge_v2")
    parser.add_argument("--v3-test-merge-dir-name", type=str, default="test_merge_v3")
    parser.add_argument("--out-test-merge-dir-name", type=str, default="test_merge_v2_overlay_v3")
    parser.add_argument("--out-test-all-json", type=Path, default=None)
    parser.add_argument("--out-test-id-json", type=Path, default=None)
    parser.add_argument("--out-test-ood-json", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    output_base_dir = args.output_base_dir if args.output_base_dir is not None else args.data_root
    v2_root = output_base_dir / args.v2_test_merge_dir_name
    v3_root = output_base_dir / args.v3_test_merge_dir_name
    out_root = output_base_dir / args.out_test_merge_dir_name
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(
        f"Overlay merge test datasets -> base={args.v2_test_merge_dir_name}, "
        f"overlay={args.v3_test_merge_dir_name}, output={args.out_test_merge_dir_name}"
    )
    print("=" * 80)

    id_paths, overridden_id = _overlay_dataset_paths(v2_root / "ID", v3_root / "ID")
    ood_paths, overridden_ood = _overlay_dataset_paths(v2_root / "OOD", v3_root / "OOD")

    print(f"ID overrides: {overridden_id}")
    print(f"OOD overrides: {overridden_ood}")

    print("Loading ID datasets...")
    id_datasets, id_records = _load_group(args.data_root, id_paths, args.strict)

    print("Loading OOD datasets...")
    ood_datasets, ood_records = _load_group(args.data_root, ood_paths, args.strict)

    _write_group_outputs(out_root / "ID", id_datasets)
    _write_group_outputs(out_root / "OOD", ood_datasets)

    all_records = id_records + ood_records

    out_test_id_json = (
        args.out_test_id_json
        if args.out_test_id_json is not None
        else out_root / "test_id_merge_v2_overlay_v3.json"
    )
    out_test_ood_json = (
        args.out_test_ood_json
        if args.out_test_ood_json is not None
        else out_root / "test_ood_merge_v2_overlay_v3.json"
    )
    out_test_all_json = (
        args.out_test_all_json
        if args.out_test_all_json is not None
        else out_root / "test_merge_v2_overlay_v3.json"
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