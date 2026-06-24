#!/usr/bin/env python3
"""
Merge test/ID and test/OOD datasets into unified files.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd


DATA_ROOT = "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data"
OUTPUT_DIR = "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/merged_test"

ID_DATASETS = [
    ("test/ID/ChartQA_test_ID", "test.json"),
    ("test/ID/PlotQA_test_ID", "test.json"),
    ("test/ID/MapQA_test_ID", "test.json"),
    ("test/ID/geometry3k_test_ID", "test.json"),
    ("test/ID/ScienceQA_test_ID", "test.json"),
    ("test/ID/OCRVQA_test_ID", "test.json"),
]

OOD_DATASETS = [
    ("test/OOD/ICON-QA_test_OOD", "test.json"),
    ("test/OOD/CLEVR_MATH_test_OOD", "test.json"),
    ("test/OOD/MathVision_test_OOD", "test.json"),
    ("test/OOD/MathVista_test_OOD", "testmini.json"),
    ("test/OOD/AI2D_test_OOD", "test.json"),
    ("test/OOD/unigeo_test_OOD", "test.json"),
]


def load_json_file(filepath: str) -> List[Dict[str, Any]]:
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


def merge_json_files(file_list: List[tuple[str, str]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    extra_info_stats: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for rel_dir, filename in file_list:
        file_path = Path(DATA_ROOT) / rel_dir / filename
        if not file_path.exists():
            print(f"Skip missing file: {file_path}")
            continue
        data = load_json_file(file_path.as_posix())
        dataset_name = Path(rel_dir).name
        for record in data:
            if "extra_info" in record and isinstance(record["extra_info"], dict):
                _collect_extra_info_stats(record["extra_info"], dataset_name, extra_info_stats)
        merged.extend(data)
        print(f"Loaded {len(data)} from {file_path}")

    if extra_info_stats:
        policies = _decide_policies(extra_info_stats)
        for record in merged:
            if "extra_info" in record and isinstance(record["extra_info"], dict):
                record["extra_info"] = normalize_extra_info_by_policy(record["extra_info"], policies)
    return merged


def save_merged(output_dir: Path, basename: str, data: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{basename}.json"
    with open(json_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON: {json_path}")

    parquet_path = output_dir / f"{basename}.parquet"
    df = pd.DataFrame(data)
    df.to_parquet(parquet_path.as_posix(), index=False)
    print(f"Saved Parquet: {parquet_path}")


def main() -> None:
    output_dir = Path(OUTPUT_DIR)

    print("Merging test ID datasets...")
    id_data = merge_json_files(ID_DATASETS)
    save_merged(output_dir, "test_id_merged", id_data)

    print("Merging test OOD datasets...")
    ood_data = merge_json_files(OOD_DATASETS)
    save_merged(output_dir, "test_ood_merged", ood_data)


if __name__ == "__main__":
    main()
