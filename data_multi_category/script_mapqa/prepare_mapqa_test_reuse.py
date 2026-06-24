#!/usr/bin/env python3
"""Build a test-only MapQA dataset from an existing processed test.json."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_V2_DIR = SCRIPT_DIR.parent / "scripts_v2"
sys.path.insert(0, str(SCRIPTS_V2_DIR))

from common import (  # noqa: E402
    DATA_ROOT,
    build_system_prompt,
    ensure_extra_info_jsonable,
    filter_valid_samples,
    load_json,
    load_tools,
    parse_extra_info,
    save_json,
    save_parquet,
    standardize_prompt_fields,
)


def _prepare_sample(sample: Dict[str, object], data_source: str) -> Dict[str, object]:
    item = copy.deepcopy(sample)
    item["data_source"] = data_source

    extra_info = parse_extra_info(item.get("extra_info"))
    extra_info["split"] = "test"
    item["extra_info"] = extra_info

    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-base-dir", type=Path, default=None)
    parser.add_argument("--src-rel-dir", type=str, default="geospatial/MapQA_2000")
    parser.add_argument("--src-file", type=str, default="test.json")
    parser.add_argument("--output-name", type=str, default="MapQA_test_ID")
    parser.add_argument("--data-source", type=str, default="MapQA_test_ID")
    parser.add_argument("--out-json-name", type=str, default="test.json")
    parser.add_argument("--out-parquet-name", type=str, default="test.parquet")
    parser.add_argument(
        "--tool-json",
        type=str,
        default=str(SCRIPTS_V2_DIR / "test" / "real_tool.json"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    output_base_dir = args.output_base_dir if args.output_base_dir is not None else args.data_root
    src_path = args.data_root / args.src_rel_dir / args.src_file
    output_dir = output_base_dir / args.output_name
    output_json_path = output_dir / args.out_json_name
    output_parquet_path = output_dir / args.out_parquet_name

    if not src_path.exists():
        raise FileNotFoundError(f"MapQA source file not found: {src_path}")

    tools_xml = load_tools(SCRIPTS_V2_DIR / "test", args.tool_json)
    system_prompt = build_system_prompt(tools_xml)

    raw = load_json(src_path)
    base_valid, base_errors = filter_valid_samples(raw, args.data_root)

    if args.strict and base_errors:
        raise ValueError(f"{src_path}: {len(base_errors)} invalid source samples before prompt update")

    prepared = [_prepare_sample(sample, args.data_source) for sample in base_valid]
    standardized = [standardize_prompt_fields(sample, system_prompt) for sample in prepared]
    final_valid, final_errors = filter_valid_samples(standardized, args.data_root)

    if args.strict and final_errors:
        raise ValueError(f"{src_path}: {len(final_errors)} invalid samples after prompt update")

    processed: List[Dict[str, object]] = [ensure_extra_info_jsonable(sample) for sample in final_valid]

    save_json(output_json_path, processed)
    save_parquet(output_parquet_path, processed)

    print("=" * 80)
    print("Prepare MapQA test-only reuse")
    print("=" * 80)
    print(f"Source: {src_path}")
    print(f"Raw samples: {len(raw)}")
    print(f"Valid source samples: {len(base_valid)}")
    print(f"Dropped before prompt update: {len(base_errors)}")
    print(f"Final samples: {len(processed)}")
    print(f"Dropped after prompt update: {len(final_errors)}")
    print(f"Output JSON: {output_json_path}")
    print(f"Output Parquet: {output_parquet_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
