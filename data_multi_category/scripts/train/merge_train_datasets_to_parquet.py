#!/usr/bin/env python3
"""
Merge training dataset JSON files into Parquet and JSON.

Usage:
    python merge_train_datasets_to_parquet.py

Behavior:
    Merge selected training datasets into:
      - train_merged.parquet (for training)
      - train_merged.json (for downstream merge/use)

Tip:
    Run update_prompts_v2.py first to update tools/skills and guidelines
    (including create_skill / skill / run_skill call conventions).
"""

import json
import os
from typing import List, Dict, Any, Tuple

import pandas as pd


# Data root directory on remote server
DATA_ROOT = "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data"

# Output directory (change as needed)
OUTPUT_DIR = "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/merged_train"

# ==============================================================================
# Training dataset configuration (20 datasets)
# ==============================================================================
TRAIN_DATASETS = [
    # ID core (8)
    ("chart/ChartQA_2000", "train.json"),
    ("chart/PlotQA_2000", "train.json"),
    ("geospatial/MapQA_2000", "train.json"),
    ("add/ocr/OCRVQA_2000", "train.json"),
    ("math/GEOQA_2000", "train.json"),
    ("math/geometry3k_2000", "train.json"),
    ("science/ScienceQA_2000", "train.json"),
    ("spatial/CLEVR_2000", "train.json"),
    # ID auxiliary (10)
    ("add/caption/LocalizedNarratives_2000", "train.json"),
    ("add/chart/DVQA_2000", "train.json"),
    ("add/code/WebSight_2000", "train.json"),
    ("add/diagram/DiagramImageToText_2000", "train.json"),
    ("general/AOKVQA_2000", "train.json"),
    ("add/general/VQAv2_2000", "train.json"),
    ("add/math/InterGPS_2000", "train.json"),
    ("add/ocr/TextVQA_2000", "train.json"),
    ("add/table/TATQA_2000", "train.json"),
    ("doc/InfographicVQA_2000", "train.json"),
]


def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_numeric_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    try:
        float(text)
        return True
    except ValueError:
        return False


def _classify_scalar(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, str):
        return "numeric_str" if _is_numeric_string(value) else "str"
    return "other"


def _collect_extra_info_stats(
    extra_info: Dict[str, Any],
    dataset_name: str,
    stats: Dict[Tuple[str, ...], Dict[str, Any]],
    path: Tuple[str, ...] = (),
) -> None:
    if not isinstance(extra_info, dict):
        return
    for key, value in extra_info.items():
        if key == "images":
            continue
        current_path = path + (key,)
        entry = stats.setdefault(
            current_path,
            {"kinds": set(), "scalar_types": set(), "list_types": set(), "datasets": set()},
        )
        entry["datasets"].add(dataset_name)

        if isinstance(value, dict):
            entry["kinds"].add("dict")
            _collect_extra_info_stats(value, dataset_name, stats, current_path)
        elif isinstance(value, list):
            entry["kinds"].add("list")
            for item in value:
                if isinstance(item, (dict, list)):
                    entry["list_types"].add("complex")
                else:
                    entry["list_types"].add(_classify_scalar(item))
        else:
            entry["kinds"].add("scalar")
            entry["scalar_types"].add(_classify_scalar(value))


def _decide_policies(
    stats: Dict[Tuple[str, ...], Dict[str, Any]]
) -> Dict[Tuple[str, ...], Dict[str, str]]:
    policies: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for path, entry in stats.items():
        kinds = entry["kinds"]
        if len(kinds) > 1:
            policies[path] = {"kind": "string"}
            continue
        if "dict" in kinds:
            policies[path] = {"kind": "dict"}
            continue
        if "list" in kinds:
            list_types = set(entry["list_types"])
            list_types.discard("none")
            if not list_types:
                policies[path] = {"kind": "list", "item": "none"}
            elif list_types == {"bool"}:
                policies[path] = {"kind": "list", "item": "bool"}
            elif list_types.issubset({"number", "numeric_str"}):
                policies[path] = {"kind": "list", "item": "number"}
            else:
                policies[path] = {"kind": "list", "item": "string"}
            continue

        scalar_types = set(entry["scalar_types"])
        scalar_types.discard("none")
        if not scalar_types:
            policies[path] = {"kind": "scalar", "type": "none"}
        elif scalar_types == {"bool"}:
            policies[path] = {"kind": "scalar", "type": "bool"}
        elif scalar_types.issubset({"number", "numeric_str"}):
            policies[path] = {"kind": "scalar", "type": "number"}
        else:
            policies[path] = {"kind": "scalar", "type": "string"}
    return policies


def _stringify_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _normalize_scalar(value: Any, target_type: str) -> Any:
    if value is None:
        return None
    if target_type == "bool":
        return bool(value) if not isinstance(value, bool) else value
    if target_type == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str) and _is_numeric_string(value):
            return float(value)
        return None
    if target_type == "string":
        return _stringify_value(value)
    return value


def _normalize_list(value: Any, item_type: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, list):
        value = [value]
    if item_type == "number":
        return [
            float(item) if isinstance(item, (int, float)) and not isinstance(item, bool)
            else (float(item) if isinstance(item, str) and _is_numeric_string(item) else None)
            for item in value
        ]
    if item_type == "bool":
        return [bool(item) if not isinstance(item, bool) else item for item in value]
    if item_type == "string":
        return [_stringify_value(item) for item in value]
    return value


def normalize_extra_info_by_policy(
    extra_info: Dict[str, Any],
    policies: Dict[Tuple[str, ...], Dict[str, str]],
    path: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    if not isinstance(extra_info, dict):
        return extra_info

    result: Dict[str, Any] = {}
    for key, value in extra_info.items():
        if key == "images":
            result[key] = value
            continue
        current_path = path + (key,)
        policy = policies.get(current_path)

        if policy is None:
            if isinstance(value, dict):
                result[key] = normalize_extra_info_by_policy(value, policies, current_path)
            else:
                result[key] = value
            continue

        kind = policy["kind"]
        if kind == "string":
            result[key] = _stringify_value(value)
        elif kind == "dict":
            if isinstance(value, dict):
                result[key] = normalize_extra_info_by_policy(value, policies, current_path)
            else:
                result[key] = _stringify_value(value)
        elif kind == "list":
            result[key] = _normalize_list(value, policy["item"])
        elif kind == "scalar":
            result[key] = _normalize_scalar(value, policy["type"])
        else:
            result[key] = value

    return result


def merge_datasets(dataset_configs: List[tuple], data_root: str) -> pd.DataFrame:
    """
    Merge multiple datasets.

    Args:
        dataset_configs: list of dataset configs, each item is (relative_path, filename)
        data_root: dataset root directory

    Returns:
        Merged DataFrame
    """
    all_records = []
    total_count = 0
    extra_info_stats: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    print(f"Start merging {len(dataset_configs)} datasets...")
    print("-" * 60)

    for rel_path, filename in dataset_configs:
        filepath = os.path.join(data_root, rel_path, filename)
        dataset_name = rel_path.split("/")[-1]

        if not os.path.exists(filepath):
            print(f"Warning: skip missing file: {filepath}")
            continue

        try:
            records = load_json_file(filepath)

            for record in records:
                if "extra_info" in record and isinstance(record["extra_info"], dict):
                    _collect_extra_info_stats(record["extra_info"], dataset_name, extra_info_stats)

            count = len(records)
            total_count += count
            all_records.extend(records)
            print(f"OK {dataset_name}: {count} records")
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")

    print("-" * 60)
    print(f"Total: {total_count} records")

    if extra_info_stats:
        policies = _decide_policies(extra_info_stats)
        for record in all_records:
            if "extra_info" in record and isinstance(record["extra_info"], dict):
                record["extra_info"] = normalize_extra_info_by_policy(record["extra_info"], policies)

    df = pd.DataFrame(all_records)
    return df


def save_to_parquet(df: pd.DataFrame, output_path: str):
    """Save DataFrame to a Parquet file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"\nSaved Parquet: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(f"   Records: {len(df)}")
    print(f"   Columns: {len(df.columns)}")


def save_to_json(df: pd.DataFrame, output_path: str):
    """Save DataFrame to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    records = df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nSaved JSON: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(f"   Records: {len(df)}")


def save_merged_data(df: pd.DataFrame, parquet_path: str, json_path: str):
    """Save merged data in both Parquet and JSON formats."""
    save_to_parquet(df, parquet_path)
    save_to_json(df, json_path)


def print_dataset_info(df: pd.DataFrame, name: str):
    """Print dataset summary information."""
    print(f"\n{'='*60}")
    print(f"{name} dataset summary")
    print(f"{'='*60}")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if "data_source" in df.columns:
        print(f"\ndata_source distribution:")
        source_counts = df["data_source"].value_counts()
        for source, count in source_counts.items():
            print(f"  - {source}: {count}")

    if "ability" in df.columns:
        print(f"\nability distribution:")
        ability_counts = df["ability"].value_counts()
        for ability, count in ability_counts.items():
            print(f"  - {ability}: {count}")


def main():
    """Main entrypoint."""
    print("=" * 60)
    print("Training dataset merge tool")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("Merge training datasets")
    print("=" * 60)

    train_df = merge_datasets(TRAIN_DATASETS, DATA_ROOT)
    train_parquet = os.path.join(OUTPUT_DIR, "train_merged.parquet")
    train_json = os.path.join(OUTPUT_DIR, "train_merged.json")
    save_merged_data(train_df, train_parquet, train_json)
    print_dataset_info(train_df, "Train")

    print("\n" + "=" * 60)
    print("Merge complete")
    print("=" * 60)
    print(f"Training output:")
    print(f"  - Parquet: {train_parquet}")
    print(f"  - JSON: {train_json}")
    print(f"  - Records: {len(train_df)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
