#!/usr/bin/env python3
"""Update prompts for test_merge_v3 OOD datasets and rewrite parquet/json."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from common import (
    DATA_ROOT,
    build_system_prompt,
    ensure_extra_info_jsonable,
    filter_valid_samples,
    load_json,
    load_tools,
    save_json,
    save_parquet,
    standardize_prompt_fields,
)


def _find_test_files(test_merge_root: Path) -> List[Path]:
    return sorted((test_merge_root / "OOD").glob("*/test.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-base-dir", type=Path, default=None)
    parser.add_argument("--test-merge-dir-name", type=str, default="test_merge_v3")
    parser.add_argument("--tool-json", type=str, default="real_tool.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    output_base_dir = args.output_base_dir if args.output_base_dir is not None else args.data_root
    test_merge_root = output_base_dir / args.test_merge_dir_name

    tools_xml = load_tools(SCRIPT_DIR, args.tool_json)
    system_prompt = build_system_prompt(tools_xml)

    files = _find_test_files(test_merge_root)
    if not files:
        print("No test_merge_v3 OOD files found.")
        return

    print("=" * 80)
    print("Update prompts for test_merge_v3")
    print("=" * 80)

    total = 0
    dropped_total = 0

    for path in files:
        raw = load_json(path)
        standardized = [standardize_prompt_fields(sample, system_prompt) for sample in raw]
        valid, errors = filter_valid_samples(standardized, args.data_root)

        if args.strict and errors:
            raise ValueError(f"{path}: {len(errors)} invalid samples after prompt update")

        processed = [ensure_extra_info_jsonable(sample) for sample in valid]
        save_json(path, processed)
        save_parquet(path.with_suffix(".parquet"), processed)

        total += len(processed)
        dropped_total += len(errors)
        print(f"{path}: raw={len(raw)} updated={len(processed)} dropped={len(errors)}")

    print("-" * 80)
    print(f"Updated test samples: {total}")
    print(f"Dropped invalid samples: {dropped_total}")
    print("=" * 80)


if __name__ == "__main__":
    main()
