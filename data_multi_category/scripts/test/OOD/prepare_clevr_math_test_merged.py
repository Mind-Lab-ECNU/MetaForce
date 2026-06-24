#!/usr/bin/env python3
"""
Merge CLEVR_MATH OOD subsets by sampling 500 examples from each existing
2000-sample subset, then combining into one dataset.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

import datasets


DATA_ROOT = Path(
    "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data"
)
OUTPUT_ROOT = DATA_ROOT / "test" / "OOD" / "CLEVR_MATH_test_OOD"

SUBSETS = [
    "addition",
    "subtraction",
    "subtraction_multihop",
    "adversarial",
]

SOURCE_ROOT = DATA_ROOT / "math" / "CLEVR_MATH_2000"
SOURCE_FILE = "test.json"

SAMPLE_PER_SUBSET = 500
SEED = 42


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_dataset(items: List[Dict[str, Any]], output_dir: Path, split_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{split_name}.parquet"
    datasets.Dataset.from_list(items).to_parquet(parquet_path.as_posix())
    print(f"Saved {len(items)} items to {parquet_path}")

    json_path = output_dir / f"{split_name}.json"
    with open(json_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(items)} items to {json_path}")


def main() -> None:
    rng = random.Random(SEED)
    all_items: List[Dict[str, Any]] = []

    for subset in SUBSETS:
        src_path = SOURCE_ROOT / subset / SOURCE_FILE
        if not src_path.exists():
            print(f"Skip missing source: {src_path}")
            continue

        items = _load_json(src_path)
        if len(items) <= SAMPLE_PER_SUBSET:
            sampled = items
        else:
            sampled = rng.sample(items, SAMPLE_PER_SUBSET)

        all_items.extend(sampled)

    dataset_name = "CLEVR_MATH_test_OOD"
    for item in all_items:
        item["data_source"] = dataset_name

    _save_dataset(all_items, OUTPUT_ROOT, "test")


if __name__ == "__main__":
    main()
