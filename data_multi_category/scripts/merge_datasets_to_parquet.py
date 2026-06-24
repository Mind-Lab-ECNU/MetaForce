#!/usr/bin/env python3
"""
合并多模态数据集的 JSON 文件到 Parquet 和 JSON 格式

用法:
    python merge_datasets_to_parquet.py

功能:
    1. 将 20 个训练数据集的 train.json 合并为一个文件
       - train_merged.parquet（用于训练）
       - train_merged.json（用于后续合并）
    2. 将 18 个测试数据集的 test.json/testmini.json 合并为一个文件
       - test_merged.parquet（用于训练）
       - test_merged.json（用于后续合并）
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd


# 远程服务器上的数据根目录
DATA_ROOT = "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data"

# 输出目录（可以修改为你想要的路径）
OUTPUT_DIR = "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/merged"

# ==============================================================================
# 训练数据集配置（20个）
# ==============================================================================
TRAIN_DATASETS = [
    # ID 核心（10个）
    ("chart/ChartQA_2000", "train.json"),
    ("chart/PlotQA_2000", "train.json"),
    ("diagram/AI2D_2000", "train.json"),
    ("geospatial/MapQA_2000", "train.json"),
    ("math/geos_processed_2000", "train.json"),
    ("math/unigeo_calculation_2000", "train.json"),
    ("math/GEOQA_2000", "train.json"),
    ("math/geometry3k_2000", "train.json"),
    ("science/ScienceQA_2000", "train.json"),
    ("spatial/CLEVR_2000", "train.json"),
    # ID 辅助（10个）
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

# ==============================================================================
# 测试数据集配置（18个）
# ==============================================================================
TEST_DATASETS = [
    # ID 测试（10个）
    ("chart/ChartQA_2000", "test.json"),
    ("chart/PlotQA_2000", "test.json"),
    ("diagram/AI2D_2000", "test.json"),
    ("geospatial/MapQA_2000", "test.json"),
    ("math/geos_processed_2000", "test.json"),
    ("math/unigeo_calculation_2000", "test.json"),
    ("math/GEOQA_2000", "test.json"),
    ("math/geometry3k_2000", "test.json"),
    ("science/ScienceQA_2000", "test.json"),
    ("spatial/CLEVR_2000", "test.json"),
    # OOD 测试（8个）
    ("diagram/ICON-QA_2000", "test.json"),
    ("math/CLEVR_MATH_2000/addition", "test.json"),
    ("math/CLEVR_MATH_2000/subtraction", "test.json"),
    ("math/CLEVR_MATH_2000/subtraction_multihop", "test.json"),
    ("math/CLEVR_MATH_2000/adversarial", "test.json"),
    ("math/MathVision_2000", "test.json"),
    ("math/MathVista_2000", "testmini.json"),  # 注意：MathVista 使用 testmini
    ("doc/DocVQA_2000", "test.json"),
]


def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    """加载 JSON 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
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
    合并多个数据集
    
    Args:
        dataset_configs: 数据集配置列表，每个元素为 (相对路径, 文件名)
        data_root: 数据根目录
        
    Returns:
        合并后的 DataFrame
    """
    all_records = []
    total_count = 0
    extra_info_stats: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    
    print(f"开始合并 {len(dataset_configs)} 个数据集...")
    print("-" * 60)
    
    for rel_path, filename in dataset_configs:
        filepath = os.path.join(data_root, rel_path, filename)
        dataset_name = rel_path.split('/')[-1]  # 获取数据集名称
        
        if not os.path.exists(filepath):
            print(f"⚠️  跳过不存在的文件: {filepath}")
            continue
        
        try:
            records = load_json_file(filepath)

            # 收集 extra_info 统计，用于后续统一字段类型（跳过 images）
            for record in records:
                if 'extra_info' in record and isinstance(record['extra_info'], dict):
                    _collect_extra_info_stats(record['extra_info'], dataset_name, extra_info_stats)
            
            count = len(records)
            total_count += count
            all_records.extend(records)
            print(f"✅ {dataset_name}: {count} 条记录")
        except Exception as e:
            print(f"❌ 加载失败 {filepath}: {e}")
    
    print("-" * 60)
    print(f"总计: {total_count} 条记录")
    
    # 统一 extra_info 字段类型（数值优先，跳过 images）
    if extra_info_stats:
        policies = _decide_policies(extra_info_stats)
        for record in all_records:
            if 'extra_info' in record and isinstance(record['extra_info'], dict):
                record['extra_info'] = normalize_extra_info_by_policy(record['extra_info'], policies)

    # 转换为 DataFrame
    # 注意：由于 extra_info 字段可能不同，我们保留所有原始字段
    df = pd.DataFrame(all_records)
    
    return df


def save_to_parquet(df: pd.DataFrame, output_path: str):
    """保存 DataFrame 到 Parquet 文件"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为 parquet
    df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"\n💾 Parquet 已保存到: {output_path}")
    print(f"   文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(f"   记录数: {len(df)}")
    print(f"   列数: {len(df.columns)}")


def save_to_json(df: pd.DataFrame, output_path: str):
    """保存 DataFrame 到 JSON 文件"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 将 DataFrame 转为字典列表并保存为 JSON
    records = df.to_dict(orient='records')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    print(f"\n📝 JSON 已保存到: {output_path}")
    print(f"   文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(f"   记录数: {len(df)}")


def save_merged_data(df: pd.DataFrame, parquet_path: str, json_path: str):
    """
    同时保存合并后的数据为 Parquet 和 JSON 格式
    
    Args:
        df: DataFrame
        parquet_path: Parquet 输出路径
        json_path: JSON 输出路径
    """
    # 保存 Parquet（用于训练）
    save_to_parquet(df, parquet_path)
    
    # 保存 JSON（用于后续合并）
    save_to_json(df, json_path)


def print_dataset_info(df: pd.DataFrame, name: str):
    """打印数据集信息"""
    print(f"\n{'='*60}")
    print(f"{name} 数据集信息")
    print(f"{'='*60}")
    print(f"总记录数: {len(df)}")
    print(f"列: {list(df.columns)}")
    
    # 统计 data_source 分布
    if 'data_source' in df.columns:
        print(f"\ndata_source 分布:")
        source_counts = df['data_source'].value_counts()
        for source, count in source_counts.items():
            print(f"  - {source}: {count}")
    
    # 统计 ability 分布
    if 'ability' in df.columns:
        print(f"\nability 分布:")
        ability_counts = df['ability'].value_counts()
        for ability, count in ability_counts.items():
            print(f"  - {ability}: {count}")


def main():
    """主函数"""
    print("=" * 60)
    print("多模态数据集合并工具")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ==============================================================================
    # 合并训练数据
    # ==============================================================================
    print("\n" + "=" * 60)
    print("【1/2】合并训练数据集")
    print("=" * 60)
    
    train_df = merge_datasets(TRAIN_DATASETS, DATA_ROOT)
    train_parquet = os.path.join(OUTPUT_DIR, "train_merged.parquet")
    train_json = os.path.join(OUTPUT_DIR, "train_merged.json")
    save_merged_data(train_df, train_parquet, train_json)
    print_dataset_info(train_df, "训练")
    
    # ==============================================================================
    # 合并测试数据
    # ==============================================================================
    print("\n" + "=" * 60)
    print("【2/2】合并测试数据集")
    print("=" * 60)
    
    test_df = merge_datasets(TEST_DATASETS, DATA_ROOT)
    test_parquet = os.path.join(OUTPUT_DIR, "test_merged.parquet")
    test_json = os.path.join(OUTPUT_DIR, "test_merged.json")
    save_merged_data(test_df, test_parquet, test_json)
    print_dataset_info(test_df, "测试")
    
    # ==============================================================================
    # 完成总结
    # ==============================================================================
    print("\n" + "=" * 60)
    print("合并完成！")
    print("=" * 60)
    print(f"训练数据:")
    print(f"  - Parquet: {train_parquet}")
    print(f"  - JSON: {train_json}")
    print(f"  - 记录数: {len(train_df)}")
    print(f"测试数据:")
    print(f"  - Parquet: {test_parquet}")
    print(f"  - JSON: {test_json}")
    print(f"  - 记录数: {len(test_df)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
