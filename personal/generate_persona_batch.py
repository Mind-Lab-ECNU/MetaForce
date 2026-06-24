#!/usr/bin/env python3
"""Generate ChartQA personas and pref_vec values with ChatECNU."""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests


PREF_VEC_PRECISE: Dict[str, float] = {
    "extract_chart_data-1": 1,
    "extract_chart_data-2": 0,
    "calculate_statistics-1": 1,
    "calculate_statistics-2": 0,
    "compare_and_filter-1": 1,
    "compare_and_filter-2": 0,
    "analyze_curve_properties-1": 1,
    "analyze_curve_properties-2": 0,
    "detect_intersection-1": 1,
    "detect_intersection-2": 0,
    "python_code": 0,
    "accuracy": 1,
    "cost": 0,
    "latency": 0,
}

PREF_VEC_ECONOMIC: Dict[str, float] = {
    "extract_chart_data-1": 0,
    "extract_chart_data-2": 1,
    "calculate_statistics-1": 0,
    "calculate_statistics-2": 1,
    "compare_and_filter-1": 0,
    "compare_and_filter-2": 1,
    "analyze_curve_properties-1": 0,
    "analyze_curve_properties-2": 1,
    "detect_intersection-1": 0,
    "detect_intersection-2": 1,
    "python_code": 0,
    "accuracy": 1,
    "cost": 0.5,
    "latency": 0.1,
}

STYLE_CONFIGS = {
    "precise": {
        "style_description": "追求准确，愿意付出更高成本",
        "pref_vec": PREF_VEC_PRECISE,
    },
    "economic": {
        "style_description": "在保证正确的前提下控制成本",
        "pref_vec": PREF_VEC_ECONOMIC,
    },
}

PROMPT_TEMPLATE = """你需要为一个图表问答场景创造一个具体的用户人物画像，并输出中英文两个版本。

## 用户偏好风格
{style_description}

## 参考示例（来自其他数据集，仅供风格参考）

示例1（精确型）：
"一位金融分析师在核对财报图表数据时，宁可投入额外时间反复验证，也要确保每个数字绝对准确，因为任何误差都可能影响投资决策。"

示例2（精确型）：
"一位医学研究员分析临床试验图表时，愿意花费更长等待时间来换取零误差的数据解读，宁可慢一点也不能出错。"

示例3（经济型）：
"一位运营人员制作周报图表时，只要数据趋势大致正确就快速完成，没必要追求完美精度，节省时间和成本更重要。"

示例4（经济型）：
"一位市场经理评估竞品图表时，够用就行，会选择最快最便宜的方案，不会为了一点精度提升而花费额外成本。"

## 要求
- 创造一个具体的人物（职业、场景、目的）
- **直接表达用户的偏好倾向，不要隐晦！**
- 用"宁愿/宁可/愿意/倾向/更看重"等词汇明确说出权衡
- 明确说出对"成本"、"时间"、"准确性"的态度
- 1-2句，专业化表述，避免口语化
- 不要提及任何具体工具名或模型名
- 输出中文与英文两段

请直接输出 persona 描述，不要有其他内容，格式如下：

persona_zh: ...
persona: ...
"""

PROMPT_TEMPLATE_BATCH = """你需要为图表问答场景创造 {num_personas} 个不同的用户人物画像，并输出中英文两个版本。

## 用户偏好风格
{style_description}

## 参考示例（来自其他数据集，仅供风格参考）

示例1（精确型）：
"一位金融分析师在核对财报图表数据时，宁可投入额外时间反复验证，也要确保每个数字绝对准确，因为任何误差都可能影响投资决策。"

示例2（精确型）：
"一位医学研究员分析临床试验图表时，愿意花费更长等待时间来换取零误差的数据解读，宁可慢一点也不能出错。"

示例3（经济型）：
"一位运营人员制作周报图表时，只要数据趋势大致正确就快速完成，没必要追求完美精度，节省时间和成本更重要。"

示例4（经济型）：
"一位市场经理评估竞品图表时，够用就行，会选择最快最便宜的方案，不会为了一点精度提升而花费额外成本。"

## 要求
- 创造 {num_personas} 个不同的人物（职业、场景、目的各不相同）
- **直接表达用户的偏好倾向，不要隐晦！**
- 用"宁愿/宁可/愿意/倾向/更看重"等词汇明确说出权衡
- 明确说出对"成本"、"时间"、"准确性"的态度
- 每个 1-2 句，专业化表述，避免口语化
- 不要提及任何具体工具名或模型名
- 输出中文与英文两段

请直接输出 persona 描述，不要有其他内容，格式如下：

persona_zh_1: ...
persona_1: ...

persona_zh_2: ...
persona_2: ...

（依此类推，直到 {num_personas} 个）
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ChartQA personas via ChatECNU.")
    parser.add_argument(
        "--input_json",
        default="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_duo/ChartQA_2000/train.json",
        help="Path to ChartQA JSON input.",
    )
    parser.add_argument(
        "--input_parquet",
        default="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_duo/ChartQA_2000/train.parquet",
        help="Path to ChartQA Parquet input.",
    )
    parser.add_argument(
        "--output_json",
        default="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_duo/ChartQA_2000/train_with_persona.json",
        help="Output path for JSON with personas.",
    )
    parser.add_argument(
        "--output_parquet",
        default="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_duo/ChartQA_2000/train_with_persona.parquet",
        help="Output path for Parquet with personas.",
    )
    parser.add_argument(
        "--api_key",
        default="sk-cp-6gLwR8oiCK6jMUfOSj3A9UvfrsrVnK58hxhjLrr5dJs7yWUsFoBGEBTRXfp_lVezuPG4_8kctGBIHbepdOcB3GeorqJxbxMZ9aEFDVMtUOWtNF8BJ6n6NQU",
        help="API key.",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.minimaxi.com/anthropic",
        help="API base URL.",
    )
    parser.add_argument(
        "--model",
        default="MiniMax M2.1",
        help="Model name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for style assignment.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit for number of samples to process (for smoke tests).",
    )
    parser.add_argument(
        "--mock-persona",
        action="store_true",
        help="Use deterministic mock personas instead of calling ChatECNU (for dry runs).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout for ChatECNU requests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of personas to generate per batch (default: 20).",
    )
    return parser.parse_args()


def contains_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def build_prompt(style_key: str) -> str:
    style = STYLE_CONFIGS[style_key]
    return PROMPT_TEMPLATE.format(style_description=style["style_description"])


def build_prompt_batch(style_key: str, num_personas: int) -> str:
    style = STYLE_CONFIGS[style_key]
    return PROMPT_TEMPLATE_BATCH.format(
        num_personas=num_personas,
        style_description=style["style_description"]
    )


def parse_persona_output(text: str) -> Tuple[str, str]:
    persona_zh, persona_en = None, None
    for line in text.splitlines():
        clean = line.strip()
        if clean.lower().startswith("persona_zh"):
            persona_zh = clean.split(":", 1)[1].strip()
        elif clean.lower().startswith("persona"):
            persona_en = clean.split(":", 1)[1].strip()
    persona_zh = persona_zh or text.strip()
    persona_en = persona_en or text.strip()
    return persona_zh, persona_en


def parse_persona_batch_output(text: str, num_personas: int) -> List[Tuple[str, str]]:
    """Parse batch persona output with numbered keys like persona_zh_1, persona_1, etc."""
    results = []
    for i in range(1, num_personas + 1):
        persona_zh, persona_en = None, None
        for line in text.splitlines():
            clean = line.strip()
            if clean.lower().startswith(f"persona_zh_{i}"):
                persona_zh = clean.split(":", 1)[1].strip()
            elif clean.lower().startswith(f"persona_{i}") and not clean.lower().startswith(f"persona_zh_{i}"):
                persona_en = clean.split(":", 1)[1].strip()
        results.append((persona_zh or "", persona_en or ""))
    return results


def call_chat_ecnu(prompt: str, api_key: str, base_url: str, model: str, timeout: float, retries: int = 2) -> Tuple[str, str]:
    """Call API (Anthropic-compatible format) to generate a single persona."""
    url = f"{base_url}/v1/messages"
    payload = {
        "model": model,
        "max_tokens": 2000,
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
    }

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            # Extract text content from response (skip thinking blocks)
            content = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    content = block.get("text", "")
                    break
            return parse_persona_output(content)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[warn] API request failed (attempt {attempt}/{retries}): {exc}", file=sys.stderr)
    raise RuntimeError(f"API request failed after {retries} attempts: {last_error}")


def call_chat_ecnu_batch(prompt: str, num_personas: int, api_key: str, base_url: str, model: str, timeout: float, retries: int = 2) -> List[Tuple[str, str]]:
    """Call API (Anthropic-compatible format) to generate multiple personas in one request."""
    url = f"{base_url}/v1/messages"
    payload = {
        "model": model,
        "max_tokens": min(8000, max(4000, num_personas * 400)),
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
    }

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            # Extract text content from response (skip thinking blocks)
            content = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    content = block.get("text", "")
                    break
            return parse_persona_batch_output(content, num_personas)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[warn] API batch request failed (attempt {attempt}/{retries}): {exc}", file=sys.stderr)
    raise RuntimeError(f"API batch request failed after {retries} attempts: {last_error}")


def build_mock_persona(item: Dict, style_key: str) -> Tuple[str, str]:
    qid = item.get("extra_info", {}).get("qid", "unknown")
    question = item.get("extra_info", {}).get("question", "").strip()
    question_short = question if len(question) <= 60 else question[:57] + "..."
    if style_key == "precise":
        zh = f"[mock-precise] 图表问题 {qid}: 更在意准确度，会慢慢核对细节。"
        en = f"[mock-precise] For {qid}, cares about accuracy and double-checks details. Question: {question_short}"
    else:
        zh = f"[mock-economic] 图表问题 {qid}: 够用就行，偏向快速省钱的方案。"
        en = f"[mock-economic] For {qid}, prefers cheaper & fast solutions. Question: {question_short}"
    return zh, en


def _normalize(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_normalize(v) for v in value)
    if isinstance(value, dict):
        # Drop keys that are explicitly None to align parquet-expanded schemas with JSON
        return {k: _normalize(v) for k, v in value.items() if v is not None}
    if hasattr(value, "tolist"):  # numpy arrays/scalars -> python/list
        return _normalize(value.tolist())
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def ensure_alignment(json_data: List[Dict], parquet_df: pd.DataFrame) -> None:
    if len(json_data) != len(parquet_df):
        raise ValueError("JSON and Parquet sample counts do not match.")
    json_qids = [item.get("extra_info", {}).get("qid") for item in json_data]
    parquet_qids = [extra.get("qid") if isinstance(extra, dict) else None for extra in parquet_df["extra_info"]]
    if json_qids != parquet_qids:
        raise ValueError("JSON and Parquet qid ordering is inconsistent.")

    parquet_records = parquet_df.to_dict(orient="records")
    for idx, (json_item, parquet_item) in enumerate(zip(json_data, parquet_records)):
        if _normalize(json_item) != _normalize(parquet_item):
            raise ValueError(f"JSON and Parquet content mismatch at index {idx}, qid={json_qids[idx]}")


def inject_persona_into_prompt(item: Dict, persona_text: str) -> None:
    if "prompt" not in item or len(item["prompt"]) < 2:
        raise ValueError("Prompt missing user message.")
    user_msg = item["prompt"][1]
    if "information from user" in user_msg.get("content", ""):
        return
    user_msg["content"] = f"information from user: {persona_text}\n\n" + user_msg["content"]


def process_sample(
    item: Dict, style_key: str, api_key: str, base_url: str, model: str, timeout: float, use_mock: bool
) -> Tuple[str, str]:
    if style_key not in STYLE_CONFIGS:
        raise ValueError(f"Unknown style: {style_key}")
    style = STYLE_CONFIGS[style_key]
    item.setdefault("extra_info", {})
    item["extra_info"]["pref_vec"] = dict(style["pref_vec"])

    prompt = build_prompt(style_key)
    if use_mock:
        persona_zh, persona_en = build_mock_persona(item, style_key)
    else:
        persona_zh, persona_en = call_chat_ecnu(prompt, api_key=api_key, base_url=base_url, model=model, timeout=timeout)

    item["extra_info"]["persona_zh"] = persona_zh
    item["extra_info"]["persona"] = persona_en

    question_text = item.get("extra_info", {}).get("question", "")
    selected = persona_zh if contains_chinese(question_text) else persona_en
    inject_persona_into_prompt(item, selected)
    return persona_zh, persona_en


def process_batch(
    items: List[Dict], style_key: str, api_key: str, base_url: str, model: str, timeout: float, use_mock: bool
) -> List[Tuple[str, str]]:
    """Process a batch of samples, generating personas in one API call."""
    if style_key not in STYLE_CONFIGS:
        raise ValueError(f"Unknown style: {style_key}")
    style = STYLE_CONFIGS[style_key]

    num_personas = len(items)
    personas = []

    if use_mock:
        # Mock mode: generate individual mock personas
        for item in items:
            persona_zh, persona_en = build_mock_persona(item, style_key)
            personas.append((persona_zh, persona_en))
    else:
        # Batch mode: one API call for all personas
        prompt = build_prompt_batch(style_key, num_personas)
        personas = call_chat_ecnu_batch(prompt, num_personas, api_key=api_key, base_url=base_url, model=model, timeout=timeout)

    # Assign personas to items and inject into prompt
    for item, (persona_zh, persona_en) in zip(items, personas):
        item.setdefault("extra_info", {})
        item["extra_info"]["pref_vec"] = dict(style["pref_vec"])
        item["extra_info"]["persona_zh"] = persona_zh
        item["extra_info"]["persona"] = persona_en

        question_text = item.get("extra_info", {}).get("question", "")
        selected = persona_zh if contains_chinese(question_text) else persona_en
        inject_persona_into_prompt(item, selected)

    return personas


def save_outputs(output_json: Path, output_parquet: Path, data: List[Dict]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    df_out = pd.DataFrame(data)
    df_out.to_parquet(output_parquet, index=False)


def main() -> None:
    args = parse_args()
    if not args.mock_persona and not args.api_key:
        raise SystemExit("ChatECNU API key missing. Provide --api_key or set CHAT_ECNU_API_KEY.")

    random.seed(args.seed)

    input_json = Path(args.input_json)
    input_parquet = Path(args.input_parquet)
    output_json = Path(args.output_json)
    output_parquet = Path(args.output_parquet)

    with input_json.open() as f:
        data_json = json.load(f)
    parquet_df = pd.read_parquet(input_parquet)
    ensure_alignment(data_json, parquet_df)

    total = len(data_json) if args.limit is None else min(args.limit, len(data_json))
    batch_size = args.batch_size
    print(f"[info] processing {total}/{len(data_json)} samples (limit={args.limit}, batch_size={batch_size})")

    # Process in batches
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_items = data_json[start_idx:end_idx]
        batch_count = end_idx - start_idx

        # Randomly assign a style for this batch
        style_key = random.choice(list(STYLE_CONFIGS.keys()))

        process_batch(
            batch_items, style_key=style_key, api_key=args.api_key, base_url=args.base_url, model=args.model, timeout=args.timeout, use_mock=args.mock_persona
        )

        time.sleep(1)

        print(f"[info] processed {end_idx}/{total} (batch={batch_count}, style={style_key})")
    print("[info] persona generation finished, saving outputs...")

    save_outputs(output_json, output_parquet, data_json[:total] + data_json[total:])
    print(f"[info] saved JSON to {output_json}")
    print(f"[info] saved Parquet to {output_parquet}")


if __name__ == "__main__":
    main()
