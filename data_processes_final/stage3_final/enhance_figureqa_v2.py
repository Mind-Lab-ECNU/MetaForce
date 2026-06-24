#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Enhance FigureQA data with persona by adding dynamic tool descriptions.
Reads data with persona and adds enhanced tools, model_mapping, tool_pricing,
and rebuilds pref_vec based on stored style.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import datasets
from tqdm import tqdm


# Guideline for multimodal content generation (same as in stage1)
guideline = (
    "Guidelines: Understand the given visual information and the user query. "
    "Determine if it is beneficial to employ the given tools. "
    "You must reason and reason with the visual information step by step within <thinking></thinking> tags. "
    "Put your final answer within <answer></answer> tags."
)


STYLE_CONFIGS = {
    "precise": {
        "model_1": 1,
        "model_2": 0,
        "accuracy": 1,
        "cost": 0,
        "latency": 0,
    },
    "economic": {
        "model_1": 0,
        "model_2": 1,
        "accuracy": 1,
        "cost": 0.5,
        "latency": 0.1,
    },
}


def _build_mm_content(question: str, image_sep: str, image_count: int) -> str:
    """Build multimodal content with image tokens."""
    if image_sep and image_sep in question:
        return question
    if image_sep and image_count > 0:
        return (image_sep * image_count) + question
    return question


def _contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _inject_persona(content: str, persona_zh: str, persona: str, question: str) -> str:
    """Inject persona at the beginning of content, selecting language based on question."""
    selected_persona = persona_zh if _contains_chinese(question) else persona
    return f"Infromation from user: {selected_persona}\n\n{content}"


def _load_tool_model_descriptions(json_path: str) -> dict:
    """Load tool model descriptions from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Convert list of tools to dict keyed by tool name
    return {tool["name"]: tool for tool in data.get("tools", [])}


def _load_tool_pricing(json_path: str, pricing_key: str = "tool_pricing", latency_key: str = "tool_latency") -> tuple:
    """Load tool pricing and latency from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(pricing_key, {}), data.get(latency_key, {})


def _merge_tool_pricing(model_pricing: dict, execute_pricing: dict) -> dict:
    """合并model和execute的tool_pricing"""
    merged = model_pricing.copy()
    merged.update(execute_pricing)
    return merged


def _is_executable_tool(tool: dict) -> bool:
    """判断工具是否有model参数"""
    props = tool.get("function", {}).get("parameters", {}).get("properties", {})
    return "model" not in props


def _build_pref_vec(tools: list, style_key: str) -> dict:
    """Build preference vector from tools and style."""
    if style_key not in STYLE_CONFIGS:
        raise ValueError(f"Unknown style: {style_key}")
    style = STYLE_CONFIGS[style_key]
    pref_vec = {}
    for tool in tools:
        tool_name = tool.get("function", {}).get("name", "")
        if not tool_name:
            continue
        if _is_executable_tool(tool):
            pref_vec[tool_name] = 0
        else:
            pref_vec[f"{tool_name}-1"] = style["model_1"]
            pref_vec[f"{tool_name}-2"] = style["model_2"]
    pref_vec["accuracy"] = style["accuracy"]
    pref_vec["cost"] = style["cost"]
    pref_vec["latency"] = style["latency"]
    return pref_vec


def _select_two_models(pricing: dict) -> tuple:
    """
    随机选择两个模型，确保model_1比model_2强（价格更高）。
    Returns (model_1, model_2).
    """
    models = list(pricing.keys())
    if len(models) >= 2:
        selected = random.sample(models, 2)
        sorted_selected = sorted(selected, key=lambda m: pricing[m]["input_tokens_per_million"] + pricing[m]["output_tokens_per_million"])
        model_1 = sorted_selected[-1]  # 价格高的更强
        model_2 = sorted_selected[0]   # 价格低的较弱
    else:
        model_1 = model_2 = models[0] if models else None
    return model_1, model_2


def _format_price(price_per_token: float) -> str:
    """Convert per-token price to million token price in dollars."""
    return f"{price_per_token * 1e6:.6f}"


def _build_tool_description(tool_def: dict, tool_name: str, model_1: str, model_2: str,
                            model_desc_data: dict, pricing: dict, latency: dict) -> dict:
    """
    Build enhanced tool description with model choices, descriptions, and pricing table.
    """
    tool_def = copy.deepcopy(tool_def)
    original_desc = tool_def.get("function", {}).get("description", "")
    original_params = tool_def.get("function", {}).get("parameters", {})
    original_properties = original_params.get("properties", {})

    # Get model descriptions for this tool
    tool_model_descs = model_desc_data.get("model_descriptions", {})
    desc_1 = tool_model_descs.get(model_1, "")
    desc_2 = tool_model_descs.get(model_2, "")

    # Get pricing and latency
    price_1 = pricing.get(model_1, {"input_tokens_per_million": 0, "output_tokens_per_million": 0})
    price_2 = pricing.get(model_2, {"input_tokens_per_million": 0, "output_tokens_per_million": 0})
    lat_1 = latency.get(model_1, {}).get("latency", 0)
    lat_2 = latency.get(model_2, {}).get("latency", 0)

    # Update model parameter description
    if "model" in original_properties:
        original_properties["model"]["description"] = f"The model used to {tool_name}. Choices: ['{tool_name}-1', '{tool_name}-2']. {tool_name}-1 {desc_1} {tool_name}-2 {desc_2} The table below shows the pricing and latency of each model: Model | price per million input tokens | price per million output tokens | average latency {tool_name}-1 | ${_format_price(price_1['input_tokens_per_million'])} | ${_format_price(price_1['output_tokens_per_million'])} | {lat_1}s {tool_name}-2 | ${_format_price(price_2['input_tokens_per_million'])} | ${_format_price(price_2['output_tokens_per_million'])} | {lat_2}s"

    new_tool = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": original_desc,
            "parameters": original_params
        }
    }
    return new_tool


def _build_tools_with_models(tools: list, model_descriptions: dict, model_pricing: dict,
                            pricing: dict, latency: dict) -> tuple:
    """
    Build tools with dynamically selected models for each sample.
    Model selection is done from model_pricing only.
    Returns (enhanced_tools, model_mapping).
    """
    enhanced_tools = []
    model_mapping = {}

    for tool in tools:
        tool_name = tool.get("function", {}).get("name", "")

        # Check if tool has model parameter - executable tools don't have model
        if _is_executable_tool(tool):
            # 可执行工具没有model参数，保留原样，不添加到model_mapping
            enhanced_tools.append(copy.deepcopy(tool))
            continue

        # 为每个工具随机选择两个模型（只从model_pricing中选择）
        model_1, model_2 = _select_two_models(model_pricing)

        # Get model description data for this tool
        tool_model_desc = model_descriptions.get(tool_name, {})

        # Build enhanced tool (使用 model_pricing 确保模型描述与选择一致)
        enhanced_tool = _build_tool_description(
            tool, tool_name, model_1, model_2, tool_model_desc, model_pricing, latency
        )
        enhanced_tools.append(copy.deepcopy(enhanced_tool))

        # Build model mapping
        model_mapping[f"{tool_name}-1"] = model_1
        model_mapping[f"{tool_name}-2"] = model_2

    return enhanced_tools, model_mapping


def _load_tools_from_list(tools: list) -> str:
    """Format tool list for system prompt."""
    tools_str = json.dumps(tools, ensure_ascii=False, indent=2)
    return f"""<tools>
{tools_str}
</tools>"""


def _build_system_prompt(tools_xml: str) -> str:
    """Build system prompt with actual tool definitions."""
    return f"""You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
{tools_xml}

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, help="Split name (e.g., train, test, validation)")
    parser.add_argument("--input_dir", default=None, help="Input directory containing data with persona")
    parser.add_argument("--output_dir", default=None, help="Output directory for enhanced data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tool_json", default=None, help="Path to base tool definitions")
    parser.add_argument("--tool_model_description", default=None, help="Path to tool model descriptions")
    parser.add_argument("--model_tool_pricing", default=None, help="Path to model tool pricing")
    parser.add_argument("--execute_tool_pricing", default=None, help="Path to execute tool pricing")
    parser.add_argument("--main_agent_tool_pricing", default=None, help="Path to main agent tool pricing")
    parser.add_argument("--image_sep", default="<image>", help="Image separator token")
    args = parser.parse_args()

    # Get script directory and construct relative paths
    script_dir = Path(__file__).parent.absolute()
    verl_duo_dir = script_dir.parent.parent

    # Set default paths using relative paths (for FigureQA)
    if args.input_dir is None:
        args.input_dir = verl_duo_dir / "data_duo_final_v2" / "figureqa_2000"
    else:
        args.input_dir = Path(args.input_dir)

    if args.output_dir is None:
        args.output_dir = verl_duo_dir / "data_duo_final_v2" / "figureqa_2000"
    else:
        args.output_dir = Path(args.output_dir)

    if args.tool_json is None:
        args.tool_json = verl_duo_dir / "real_tool.json"

    if args.tool_model_description is None:
        args.tool_model_description = verl_duo_dir / "tool_model_descriptions.json"

    if args.model_tool_pricing is None:
        args.model_tool_pricing = verl_duo_dir / "model_tool_pricing.json"

    if args.execute_tool_pricing is None:
        args.execute_tool_pricing = verl_duo_dir / "execute_tool_pricing.json"

    if args.main_agent_tool_pricing is None:
        args.main_agent_tool_pricing = verl_duo_dir / "main_agent_tool_pricing.json"

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Load model descriptions and pricing
    model_descriptions = _load_tool_model_descriptions(str(args.tool_model_description))
    model_pricing, latency = _load_tool_pricing(str(args.model_tool_pricing), "model_tool_pricing", "model_tool_latency")
    execute_pricing, _ = _load_tool_pricing(str(args.execute_tool_pricing), "execute_tool_pricing", "execute_tool_latency")
    main_agent_pricing, _ = _load_tool_pricing(
        str(args.main_agent_tool_pricing), "main_agent_tool_pricing", "main_agent_tool_latency"
    )

    # Merge model and execute pricing for tool selection and extra_info
    pricing = _merge_tool_pricing(model_pricing, execute_pricing)

    # Load base tools
    with open(args.tool_json, "r", encoding="utf-8") as f:
        base_tools = json.load(f)

    # Load existing data with persona
    input_file = args.input_dir / f"{args.split}_with_persona_v2.json"
    print(f"Loading data from {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        existing_data = json.load(f)

    print(f"Loaded {len(existing_data)} samples")

    # Process each item
    enhanced_items = []
    for item in tqdm(existing_data, desc="Enhancing tools"):
        # Build enhanced tools with model choices for this sample
        enhanced_tools, model_mapping = _build_tools_with_models(
            base_tools, model_descriptions, model_pricing, pricing, latency
        )

        # Format tools for system prompt
        tools_xml = _load_tools_from_list(enhanced_tools)
        system_prompt = _build_system_prompt(tools_xml)

        # Regenerate mm_content: persona + original question + guideline
        src_extra_info = item.get("extra_info", {})
        original_question = src_extra_info.get("question", "")
        persona_zh = src_extra_info.get("persona_zh", "")
        persona = src_extra_info.get("persona", "")

        mm_content = _build_mm_content(original_question, args.image_sep, 1)
        mm_content = f"{mm_content}\n\n{guideline}"

        # Inject persona at the beginning (select language based on question)
        mm_content = _inject_persona(mm_content, persona_zh, persona, original_question)

        # Update prompt with new system prompt and regenerated mm_content
        new_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": mm_content}
        ]

        # Update extra_info: add tools, model_mapping, tool_pricing
        # IMPORTANT: Preserve persona_zh, persona, and pref_vec from input
        extra_info = item["extra_info"].copy()
        style_key = extra_info.get("style")
        extra_info["pref_vec"] = _build_pref_vec(base_tools, style_key)
        extra_info["tools"] = enhanced_tools
        extra_info["model_mapping"] = model_mapping
        extra_info["tool_pricing"] = pricing
        extra_info["main_agent_tool_pricing"] = main_agent_pricing

        # Create enhanced data entry
        enhanced_item = {
            "data_source": item["data_source"],
            "prompt": new_prompt,
            "images": item["images"],
            "ability": item["ability"],
            "reward_model": item["reward_model"],
            "extra_info": extra_info
        }
        enhanced_items.append(enhanced_item)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save enhanced data as parquet
    parquet_path = args.output_dir / f"{args.split}_final_v2.parquet"
    datasets.Dataset.from_list(enhanced_items).to_parquet(str(parquet_path))
    print(f"Saved {len(enhanced_items)} items to {parquet_path}")

    # Save enhanced data as JSON
    json_path = args.output_dir / f"{args.split}_final_v2.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(enhanced_items, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(enhanced_items)} items to {json_path}")


if __name__ == "__main__":
    main()
