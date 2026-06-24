#!/usr/bin/env python3
"""
Reuse already processed full OOD test datasets and rewrite them under data/test/OOD
with updated dataset_name and data_source.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import datasets


DATA_ROOT = Path(
    "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data"
)
OUTPUT_ROOT = DATA_ROOT / "test" / "OOD"

DATASETS = [
    {
        "name": "MathVision_test_OOD",
        "src_dir": DATA_ROOT / "math" / "MathVision",
        "src_file": "test.json",
    },
    {
        "name": "AI2D_test_OOD",
        "src_dir": DATA_ROOT / "diagram" / "AI2D",
        "src_file": "test.json",
    },
]


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
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for cfg in DATASETS:
        src_path = cfg["src_dir"] / cfg["src_file"]
        if not src_path.exists():
            print(f"Skip missing source: {src_path}")
            continue

        items = _load_json(src_path)
        for item in items:
            item["data_source"] = cfg["name"]

        output_dir = OUTPUT_ROOT / cfg["name"]
        split_name = Path(cfg["src_file"]).stem
        _save_dataset(items, output_dir, split_name)


if __name__ == "__main__":
    main()
