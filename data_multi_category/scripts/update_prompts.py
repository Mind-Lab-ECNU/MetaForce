#!/usr/bin/env python3
"""
批量修改数据文件中的 prompt 字段
用于更新 tool definitions、system prompt 和 guidelines，而无需重新运行整个数据处理流程
同时更新 JSON 和 Parquet 文件
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List


# ============================================================================
# 配置区域 - 修改这里的值来更新 prompt
# ============================================================================

# 新的 tool.json 路径（相对于脚本目录）
TOOL_JSON_PATH = "real_tool.json"

# 新的 System Prompt 模板
# {tools_xml} 会被替换为实际的工具定义
NEW_SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
{tools_xml}

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""

# 新的 Guidelines
NEW_GUIDELINES = (
    "Guidelines: Understand the given visual information and the user query. "
    "Determine if it is beneficial to employ the given tools. "
    "You must reason and reason with the visual information step by step within <thinking></thinking> tags. "
    "Put your final answer within <answer></answer> tags."
)

# 数据目录路径
DATA_DIR = Path("/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data")

# 是否备份原文件（默认为 True，建议开启）
BACKUP = True

# 抽样参数
SEED = 42  # 随机种子，用于可复现的抽样
SAMPLE_SIZE = 500  # 从每个文件中抽样的样本数量


# ============================================================================
# 处理逻辑
# ============================================================================

def load_tools(tool_json_path: str) -> str:
    """加载工具定义并格式化为 XML 格式"""
    script_dir = Path(__file__).parent
    full_path = script_dir / tool_json_path

    if not full_path.exists():
        raise FileNotFoundError(f"Tool JSON file not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        tools = json.load(f)

    tools_str = json.dumps(tools, ensure_ascii=False, indent=2)
    return f"<tools>\n{tools_str}\n</tools>"


def build_new_system_prompt(tools_xml: str) -> str:
    """构建新的 system prompt"""
    return NEW_SYSTEM_PROMPT_TEMPLATE.format(tools_xml=tools_xml)


def extract_question_from_user_content(user_content: str) -> str:
    """
    从 user 消息中提取问题部分
    格式通常是: <image>问题文本\n\nChoices:...\n\nGuidelines:...
    我们需要保留 Guidelines 之前的内容
    """
    # 查找 "\n\nGuidelines:" 的位置
    guidelines_pos = user_content.find("\n\nGuidelines:")
    if guidelines_pos != -1:
        # 保留 Guidelines 之前的所有内容
        return user_content[:guidelines_pos]
    else:
        # 如果没有找到 Guidelines，返回原内容
        return user_content


def build_new_user_content(original_content: str) -> str:
    """构建新的 user 消息内容"""
    question_part = extract_question_from_user_content(original_content)
    return f"{question_part}\n\n{NEW_GUIDELINES}"


def update_single_sample(sample: Dict[str, Any], new_system_prompt: str) -> Dict[str, Any]:
    """更新单个数据样本的 prompt 字段"""
    # 创建样本的副本，避免修改原始数据
    updated_sample = sample.copy()

    # 更新 system 消息
    updated_sample["prompt"] = sample["prompt"].copy()
    updated_sample["prompt"][0] = sample["prompt"][0].copy()
    updated_sample["prompt"][0]["content"] = new_system_prompt

    # 更新 user 消息中的 guidelines
    updated_sample["prompt"][1] = sample["prompt"][1].copy()
    updated_sample["prompt"][1]["content"] = build_new_user_content(sample["prompt"][1]["content"])

    return updated_sample


def find_all_json_files(data_dir: Path) -> List[Path]:
    """查找所有需要处理的 JSON 文件"""
    json_files = []

    for json_file in data_dir.rglob("*.json"):
        # 跳过备份文件
        if json_file.name.endswith(".bak"):
            continue
        # 跳过 real_tool.json 和 fake_tool.json
        if json_file.name in ["real_tool.json", "fake_tool.json"]:
            continue
        # 只处理 train.json, test.json, validation.json, testmini.json, val.json
        if json_file.stem in ["train", "test", "validation", 'testmini', 'val']:
            json_files.append(json_file)

    return sorted(json_files)


def save_json(file_path: Path, data: List[Dict]):
    """保存 JSON 文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_parquet(file_path: Path, data: List[Dict]):
    """保存 Parquet 文件"""
    try:
        import pandas as pd
        df = pd.DataFrame(data)
        df.to_parquet(file_path, index=False)
    except ImportError:
        print("  警告: pandas 不可用，跳过 parquet 保存")
        raise


def process_file(file_path: Path, new_system_prompt: str) -> int:
    """
    处理单个文件，同时更新 JSON 和 Parquet
    返回更新的样本数量
    """
    # 读取原始 JSON 数据
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 检查是否是列表格式
    if not isinstance(data, list):
        print(f"  跳过非列表格式文件: {file_path.name}")
        return 0

    # 先保存原始数据用于备份
    original_data = data

    # 抽样：如果数据量大于 SAMPLE_SIZE，则随机抽取 SAMPLE_SIZE 个样本
    if len(data) > SAMPLE_SIZE:
        rng = random.Random(SEED)
        data = rng.sample(data, SAMPLE_SIZE)
        print(f"  已抽样: {SAMPLE_SIZE} 个样本 (seed={SEED})")

    # 更新每个样本
    updated_data = [update_single_sample(sample, new_system_prompt) for sample in data]

    # 备份原文件（使用原始完整数据）
    if BACKUP:
        backup_path = file_path.with_suffix(".json.bak")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(original_data, f, ensure_ascii=False, indent=2)
        print(f"  备份已创建: {backup_path.name}")

        # 备份原 parquet 文件（如果存在）
        parquet_path = file_path.with_suffix(".parquet")
        if parquet_path.exists():
            import shutil
            backup_parquet_path = file_path.with_suffix(".parquet.bak")
            shutil.copy2(parquet_path, backup_parquet_path)
            print(f"  备份已创建: {backup_parquet_path.name}")

    # 保存 JSON
    save_json(file_path, updated_data)

    # 保存 Parquet（与 JSON 同名，只是后缀不同）
    parquet_path = file_path.with_suffix(".parquet")
    try:
        save_parquet(parquet_path, updated_data)
        print(f"  ✓ Parquet 已保存: {parquet_path.name}")
    except Exception:
        pass

    return len(updated_data)


def main():
    """主函数"""
    print("=" * 60)
    print("批量更新数据文件中的 prompt 字段")
    print("=" * 60)

    # 加载工具定义并构建新的 system prompt
    print(f"\n1. 加载工具定义: {TOOL_JSON_PATH}")
    try:
        tools_xml = load_tools(TOOL_JSON_PATH)
        print(f"   成功加载工具定义")
    except FileNotFoundError as e:
        print(f"   错误: {e}")
        print(f"   请确保 {TOOL_JSON_PATH} 文件存在于脚本目录中")
        return

    new_system_prompt = build_new_system_prompt(tools_xml)
    print(f"   新的 system prompt 已构建")

    # 显示新的 guidelines
    print(f"\n2. 新的 Guidelines:")
    print(f"   {NEW_GUIDELINES}")

    # 查找所有 JSON 文件
    print(f"\n3. 扫描数据目录: {DATA_DIR}")
    json_files = find_all_json_files(DATA_DIR)
    print(f"   找到 {len(json_files)} 个数据文件")

    if not json_files:
        print("\n没有找到需要处理的文件，程序退出")
        return

    # 确认操作
    print(f"\n4. 准备更新 {len(json_files)} 个文件")
    print(f"   备份原文件: {'是' if BACKUP else '否'}")
    print(f"   输出格式: JSON + Parquet")

    # 处理每个文件
    print(f"\n5. 开始处理...")
    total_samples = 0

    for i, file_path in enumerate(json_files, 1):
        rel_path = file_path.relative_to(DATA_DIR)
        print(f"\n[{i}/{len(json_files)}] 处理: {rel_path}")

        try:
            count = process_file(file_path, new_system_prompt)
            total_samples += count
            print(f"  ✓ 已更新 {count} 个样本")
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")

    # 总结
    print("\n" + "=" * 60)
    print(f"处理完成!")
    print(f"  处理文件数: {len(json_files)}")
    print(f"  更新样本数: {total_samples}")
    print(f"  输出格式: JSON + Parquet")
    if BACKUP:
        print(f"  备份文件位置: 原文件名 + '.json.bak'")
    print("=" * 60)


if __name__ == "__main__":
    main()
