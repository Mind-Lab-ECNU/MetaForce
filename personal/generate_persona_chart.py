#!/usr/bin/env python3
"""Generate ChartQA personas and pref_vec values with ChatECNU."""

import argparse
import base64
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

# 职业池 - 涵盖各个领域
PROFESSION_POOL = [
    # 商业金融
    "金融分析师", "投资顾问", "会计师", "审计师", "风险控制专员", "银行柜员",
    "保险精算师", "税务师", "财务总监", "投资银行家",
    # 医疗健康
    "医学研究员", "临床医生", "流行病学家", "药剂师", "医疗器械工程师", "医院管理员",
    "公共卫生专家", "医学影像技师", "护士长", "健康管理师",
    # 科技工程
    "软件工程师", "数据科学家", "算法工程师", "系统架构师", "测试工程师", "产品经理",
    "网络安全专家", "硬件工程师", "嵌入式开发工程师", "技术文档工程师",
    # 教育科研
    "大学教授", "中学教师", "教育研究员", "图书馆管理员", "科研助理", "博士后",
    "实验室技术员", "在线教育讲师", "教育顾问", "课程设计师",
    # 媒体传播
    "新闻记者", "编辑", "数据可视化专家", "内容运营", "社交媒体经理", "广告策划",
    "公关专员", "摄影师", "视频剪辑师", "播客主持人",
    # 政府法律
    "政策分析师", "公务员", "律师", "法官助理", "法律顾问", "统计员",
    "城市规划师", "环保专员", "市场监管员", "智库研究员",
    # 制造建筑
    "工程师", "建筑师", "施工经理", "质量控制员", "供应链经理", "生产主管",
    "工业设计师", "土木工程师", "电气工程师", "机械工程师",
    # 零售服务
    "市场经理", "销售代表", "客户服务专员", "门店经理", "采购专员", "电商运营",
    "人力资源专员", "培训师", "活动策划", "品牌经理",
    # 农业环境
    "农业技术员", "环境科学家", "气象分析师", "地质勘探员", "林业工程师", "海洋学家",
    "可持续发展顾问", "水资源管理师", "生态学家", "农业经济分析师",
    # 交通运输
    "物流经理", "交通运输规划师", "航空调度员", "港口运营员", "船舶代理", "车队经理",
    # 创意艺术
    "平面设计师", "UI/UX设计师", "插画师", "游戏设计师", "动画师", "建筑师",
    # 体育娱乐
    "体育数据分析师", "赛事策划", "娱乐经纪人", "游戏测试员", "电竞数据分析师",
]

# 场景池
SCENARIO_POOL = [
    "分析销售报表", "查看用户增长图表", "研究市场份额变化", "监控服务器性能指标",
    "评估实验数据", "调查客户满意度趋势", "追踪项目进度", "分析财务报表",
    "研究气候变化数据", "监控生产质量指标", "分析用户行为数据", "评估营销效果",
    "追踪健康状况数据", "分析库存周转率", "研究竞争对手数据", "监控流量统计",
    "分析投票民意数据", "研究教育成果统计", "评估投资回报率", "监控社交媒体数据",
]

PROMPT_TEMPLATE = """你需要为一个图表问答场景创造一个具体的用户人物画像，并输出中英文两个版本。

## 用户偏好风格
{style_description}

## 职业灵感参考（请从以下或类似职业中选择，避免重复）
{profession_hint}

## 用户问题（请根据这个问题设计合适的场景）
{question}

## 参考示例（仅供风格参考）

示例1（精确型）：
"一位金融分析师在核对财报图表数据时，宁可投入额外时间反复验证，也要确保每个数字绝对准确，因为任何误差都可能影响投资决策。"

示例2（精确型）：
"一位医学研究员分析临床试验图表时，愿意花费更长等待时间来换取零误差的数据解读，宁可慢一点也不能出错。"

示例3（经济型）：
"一位运营人员制作周报图表时，只要数据趋势大致正确就快速完成，没必要追求完美精度，节省时间和成本更重要。"

示例4（经济型）：
"一位市场经理评估竞品图表时，够用就行，会选择最快最便宜的方案，不会为了一点精度提升而花费额外成本。"

## 要求
- 根据上述问题（以及你看到的图表内容）设计一个合适的场景和人物
- **参考图表类型（如柱状图、折线图、饼图等）和问题复杂度来设计人物**
- 创造一个具体的人物（职业、场景、目的），职业选择要多样化，覆盖各行各业
- **每次请选择不同的职业，避免重复！**
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
        default="a",
        help="ChatECNU API key; defaults to CHAT_ECNU_API_KEY env var.",
    )
    parser.add_argument(
        "--base_proxy_url",
        default="https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-c6264b38-46a1-4bb6-a3a7-1db39453f6f8/vscode/cce61c7c-7a05-47cf-a7cc-8fdebea2f356/0de35202-7269-4ad0-8a57-426ff3b0b243/proxy/8086/",
        help="Base proxy URL for the open-source model.",
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
    return parser.parse_args()


def contains_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def build_prompt(style_key: str, question: str = "") -> str:
    style = STYLE_CONFIGS[style_key]
    # 随机选择10个职业作为提示
    random_professions = random.sample(PROFESSION_POOL, min(10, len(PROFESSION_POOL)))
    profession_hint = "、".join(random_professions[:10])
    return PROMPT_TEMPLATE.format(
        style_description=style["style_description"],
        profession_hint=profession_hint,
        question=question if question else "（通用图表问答场景）"
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


def call_chat_ecnu(prompt: str, base_proxy_url: str, timeout: float, image_path: str = None, seed: int = None, retries: int = 2) -> Tuple[str, str]:
    url = base_proxy_url + "v1/chat/completions"
    # 添加随机盐值增加多样性
    salt = random.randint(1000, 9999) if seed is None else seed

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer token-abc123"
    }

    # 构建消息内容
    if image_path:
        # 读取并编码图片
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()

        print(f"[debug] image_path: {image_path}, base64 length: {len(image_base64)}", file=sys.stderr)

        # 多模态消息格式：图片在前，文本在后
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    else:
        # 纯文本消息格式
        messages = [
            {"role": "user", "content": prompt}
        ]

    payload = {
        "model": "qwen",
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.95,
        "stream": False,
    }

    last_error = None
    last_resp_text = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout, verify=False)
            last_resp_text = resp.text
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return parse_persona_output(content)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[warn] Model request failed (attempt {attempt}/{retries}): {exc}", file=sys.stderr)
            print(f"[warn] Response text: {last_resp_text[:500]}", file=sys.stderr)
    raise RuntimeError(f"Model request failed after {retries} attempts: {last_error}\nResponse: {last_resp_text[:500] if last_resp_text else 'No response'}")


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
    if "infromation from user" in user_msg.get("content", ""):
        return
    user_msg["content"] = f"infromation from user: {persona_text}\n\n" + user_msg["content"]


def process_sample(
    item: Dict, style_key: str, base_proxy_url: str, timeout: float, use_mock: bool
) -> Tuple[str, str]:
    if style_key not in STYLE_CONFIGS:
        raise ValueError(f"Unknown style: {style_key}")
    style = STYLE_CONFIGS[style_key]
    item.setdefault("extra_info", {})
    item["extra_info"]["pref_vec"] = dict(style["pref_vec"])

    # 获取问题并传入prompt
    question_text = item.get("extra_info", {}).get("question", "")
    prompt = build_prompt(style_key, question=question_text)

    # 获取图片路径
    image_path = None
    if item.get("images") and len(item["images"]) > 0:
        image_path = item["images"][0].get("image", "")
    elif item.get("extra_info", {}).get("images"):
        images = item["extra_info"]["images"]
        if isinstance(images, list) and len(images) > 0:
            image_path = images[0]

    if use_mock:
        persona_zh, persona_en = build_mock_persona(item, style_key)
    else:
        persona_zh, persona_en = call_chat_ecnu(prompt, base_proxy_url=base_proxy_url, timeout=timeout, image_path=image_path)

    item["extra_info"]["persona_zh"] = persona_zh
    item["extra_info"]["persona"] = persona_en

    # 根据问题语言选择persona
    selected = persona_zh if contains_chinese(question_text) else persona_en
    inject_persona_into_prompt(item, selected)
    return persona_zh, persona_en


def save_outputs(output_json: Path, output_parquet: Path, data: List[Dict]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    df_out = pd.DataFrame(data)
    df_out.to_parquet(output_parquet, index=False)


def main() -> None:
    args = parse_args()
    if not args.mock_persona and not args.base_proxy_url:
        raise SystemExit("Base proxy URL missing. Provide --base_proxy_url.")

    random.seed(args.seed)

    # Record start time
    start_time = time.time()

    input_json = Path(args.input_json)
    input_parquet = Path(args.input_parquet)
    output_json = Path(args.output_json)
    output_parquet = Path(args.output_parquet)

    with input_json.open() as f:
        data_json = json.load(f)
    parquet_df = pd.read_parquet(input_parquet)
    ensure_alignment(data_json, parquet_df)

    total = len(data_json) if args.limit is None else min(args.limit, len(data_json))
    print(f"[info] processing {total}/{len(data_json)} samples (limit={args.limit})")

    for idx, item in enumerate(data_json[:total]):
        style_key = random.choice(list(STYLE_CONFIGS.keys()))
        persona_zh, persona_en = process_sample(
            item, style_key=style_key, base_proxy_url=args.base_proxy_url, timeout=args.timeout, use_mock=args.mock_persona
        )
        # Print generated persona and preferences for each sample
        pref_vec = item.get("extra_info", {}).get("pref_vec", {})
        print(f"\n{'='*60}")
        print(f"[{idx + 1}/{total}] Style: {style_key}")
        print(f"Preference Vector: {pref_vec}")
        print(f"Persona (ZH): {persona_zh}")
        print(f"Persona (EN): {persona_en}")
        print(f"{'='*60}\n")
        # time.sleep(0.2)
    print("[info] persona generation finished, saving outputs...")

    # Calculate and print total elapsed time
    elapsed_time = time.time() - start_time
    print(f"[info] Total time elapsed: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

    save_outputs(output_json, output_parquet, data_json[:total] + data_json[total:])
    print(f"[info] saved JSON to {output_json}")
    print(f"[info] saved Parquet to {output_parquet}")


if __name__ == "__main__":
    main()
