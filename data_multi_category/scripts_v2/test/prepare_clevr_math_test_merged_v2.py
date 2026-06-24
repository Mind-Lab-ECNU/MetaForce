#!/usr/bin/env python3
"""Reuse CLEVR_MATH subsets and merge into one OOD test set (2000 total)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from common import DATA_ROOT, filter_valid_samples, load_json, save_json_and_parquet, stable_sample

SUBSETS: List[str] = ["addition", "subtraction", "subtraction_multihop", "adversarial"]
SAMPLE_PER_SUBSET = 500


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-base-dir", type=Path, default=None)
    parser.add_argument("--train-merge-dir-name", type=str, default="train_merge_v2")
    parser.add_argument("--test-merge-dir-name", type=str, default="test_merge_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-per-subset", type=int, default=SAMPLE_PER_SUBSET)
    args = parser.parse_args()

    output_base_dir = args.output_base_dir if args.output_base_dir is not None else args.data_root
    source_root = args.data_root / "math" / "CLEVR_MATH_2000"
    output_dir = output_base_dir / args.test_merge_dir_name / "OOD" / "CLEVR_MATH_test_OOD"

    merged = []

    print("=" * 80)
    print("Prepare CLEVR_MATH_test_OOD (pure reuse)")
    print("=" * 80)

    for subset in SUBSETS:
        src_path = source_root / subset / "test.json"
        if not src_path.exists():
            raise FileNotFoundError(f"Missing subset source file: {src_path}")

        raw = load_json(src_path)
        valid, errors = filter_valid_samples(raw, args.data_root)

        if len(valid) < args.sample_per_subset:
            raise ValueError(
                f"Subset {subset}: valid samples {len(valid)} < {args.sample_per_subset}"
            )

        sampled = stable_sample(valid, args.sample_per_subset, args.seed)
        for sample in sampled:
            sample["data_source"] = "CLEVR_MATH_test_OOD"
            sample["ability"] = "visual_spatial"
            if isinstance(sample.get("extra_info"), dict):
                sample["extra_info"]["split"] = "test"
                sample["extra_info"]["subset"] = subset
                sample["extra_info"]["category"] = "spatial"
        merged.extend(sampled)

        print(
            f"subset={subset}: raw={len(raw)} valid={len(valid)} selected={len(sampled)} "
            f"invalid={len(errors)}"
        )

    save_json_and_parquet(output_dir, "test", merged)

    print("-" * 80)
    print(f"CLEVR_MATH merged total: {len(merged)}")
    print(f"Saved: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
