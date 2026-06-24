#!/usr/bin/env python3
"""Build OOD test_merge_v3 datasets from existing processed files (pure reuse)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from common import DATA_ROOT, filter_valid_samples, load_json, save_json_and_parquet, stable_sample


OOD_REUSE_CONFIG: List[Dict[str, str]] = [
    {
        "output_name": "AI2D_test_OOD",
        "src_rel_dir": "diagram/AI2D_2000",
        "src_file": "test.json",
    },
    {
        "output_name": "MathVista_test_OOD",
        "src_rel_dir": "math/MathVista_2000",
        "src_file": "testmini.json",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-base-dir", type=Path, default=None)
    parser.add_argument("--test-merge-dir-name", type=str, default="test_merge_v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=int, default=5000)
    parser.add_argument("--sample-over-threshold", type=int, default=2000)
    args = parser.parse_args()

    output_base_dir = args.output_base_dir if args.output_base_dir is not None else args.data_root
    out_root = output_base_dir / args.test_merge_dir_name / "OOD"
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Prepare OOD test_merge_v3 (pure reuse)")
    print("=" * 80)

    total = 0

    for cfg in OOD_REUSE_CONFIG:
        src_path = args.data_root / cfg["src_rel_dir"] / cfg["src_file"]
        if not src_path.exists():
            raise FileNotFoundError(f"Missing source file: {src_path}")

        raw = load_json(src_path)
        valid, errors = filter_valid_samples(raw, args.data_root)
        reused = (
            stable_sample(valid, args.sample_over_threshold, args.seed)
            if len(valid) > args.threshold
            else valid
        )

        out_name = cfg["output_name"]
        for sample in reused:
            sample["data_source"] = out_name
            if isinstance(sample.get("extra_info"), dict):
                sample["extra_info"]["split"] = "test"

        dst_dir = out_root / out_name
        save_json_and_parquet(dst_dir, "test", reused)

        total += len(reused)
        print(
            f"{out_name}: raw={len(raw)} valid={len(valid)} selected={len(reused)} invalid={len(errors)}"
        )

    print("-" * 80)
    print(f"OOD total samples: {total}")
    print("=" * 80)


if __name__ == "__main__":
    main()
