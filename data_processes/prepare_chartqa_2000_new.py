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
Preprocess the ChartQA dataset to parquet format with local images.
Uniformly sample 2000 examples from each split.
Dynamically generates tool descriptions with model choices and pricing.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import datasets
from PIL import Image


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


def _build_pref_vec(tools: list) -> dict:
    """Build preference vector with tool names-1, tool names-2 + accuracy/cost/latency = 0.
    可执行工具只保留一项，没有-1/-2后缀。
    """
    pref_vec = {}
    for tool in tools:
        tool_name = tool["function"]["name"]
        if _is_executable_tool(tool):
            # 可执行工具没有model参数，只保留一项
            pref_vec[tool_name] = 0
        else:
            pref_vec[f"{tool_name}-1"] = 0
            pref_vec[f"{tool_name}-2"] = 0
    # Add the three fixed metrics at the end
    pref_vec["accuracy"] = 0
    pref_vec["cost"] = 0
    pref_vec["latency"] = 0
    return pref_vec


def _load_tools(tool_json_path: str) -> str:
    """Load tool definitions from JSON and format for system prompt."""
    with open(tool_json_path, "r", encoding="utf-8") as f:
        tools = json.load(f)

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

For each function call, return a json object with function name and arguments within <tools></tools> XML tags:
<tools>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tools>
"""


guideline = (
    "Guidelines: Understand the given visual information and the user query. "
    "Determine if it is beneficial to employ the given tools. "
    "Reason with the visual information step by step, and put your final answer within \\boxed{}."
)


def _build_mm_content(question: str, image_sep: str, image_count: int) -> str:
    """Build multimodal content with image tokens."""
    if image_sep and image_sep in question:
        return question
    if image_sep and image_count > 0:
        return (image_sep * image_count) + question
    return question


def _save_pil_image(image: Image.Image, dest_path: Path) -> str:
    """Save PIL Image to disk, returning the absolute path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest_path.as_posix())
    return dest_path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="HuggingFaceM4/ChartQA")
    parser.add_argument("--output_dir", default="/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo")
    parser.add_argument("--image_sep", default="<image>")
    parser.add_argument("--sample_count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tool_json", default="/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes/real_tool.json")
    parser.add_argument("--tool_model_description", default="/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/tool_model_descriptions.json")
    parser.add_argument("--model_tool_pricing", default="/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/model_tool_pricing.json")
    parser.add_argument("--execute_tool_pricing", default="/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/execute_tool_pricing.json")
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Load model descriptions and pricing
    model_descriptions = _load_tool_model_descriptions(args.tool_model_description)
    model_pricing, latency = _load_tool_pricing(args.model_tool_pricing, "model_tool_pricing", "model_tool_latency")
    execute_pricing, _ = _load_tool_pricing(args.execute_tool_pricing, "execute_tool_pricing", "execute_tool_latency")

    # Merge model and execute pricing for tool selection and extra_info
    pricing = _merge_tool_pricing(model_pricing, execute_pricing)

    # Load base tools
    with open(args.tool_json, "r", encoding="utf-8") as f:
        base_tools = json.load(f)

    # Build preference vector
    pref_vec = _build_pref_vec(base_tools)

    # Load tool pricing for extra_info (single token prices)
    tool_pricing_extra = pricing

    dataset = datasets.load_dataset(args.dataset_path)
    dataset_name = "ChartQA_2000"
    output_root = Path(args.output_dir) / dataset_name
    images_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    for split_name, split_dataset in dataset.items():
        items = []
        split_image_dir = images_root / split_name
        split_image_dir.mkdir(parents=True, exist_ok=True)

        # Uniform sampling: calculate sample count and randomly select indices
        n_samples = min(args.sample_count, len(split_dataset))
        sampled_indices = random.sample(range(len(split_dataset)), n_samples)
        sampled_indices = sorted(sampled_indices)  # Sort for consistent image naming

        for local_idx, orig_idx in enumerate(sampled_indices):
            example = split_dataset[orig_idx]

            # Get question (stored as 'query')
            question = example.get("query", "")
            if not question:
                continue

            # Get answer (stored as 'label', which is a list)
            label = example.get("label", [])
            if not label:
                continue
            answer = str(label[0]) if isinstance(label, list) else str(label)

            # Get and save image (already a PIL Image)
            image = example.get("image")
            if not image or not isinstance(image, Image.Image):
                continue

            image_name = f"{split_name}_{local_idx}.png"
            image_path = _save_pil_image(image, split_image_dir / image_name)

            # Build enhanced tools with model choices for this sample
            enhanced_tools, model_mapping = _build_tools_with_models(
                base_tools, model_descriptions, model_pricing, pricing, latency
            )

            # Format tools for system prompt
            tools_xml = _load_tools_from_list(enhanced_tools)
            system_prompt = _build_system_prompt(tools_xml)

            # Build multimodal content
            mm_content = _build_mm_content(question, args.image_sep, 1)
            mm_content = f"{mm_content}\n\n{guideline}"

            # Create data entry
            data = {
                "data_source": args.dataset_path,
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": mm_content}
                ],
                "images": [{"image": image_path}],
                "ability": "visual_reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    "split": split_name,
                    "index": orig_idx,
                    "qid": f"{split_name}_{local_idx}",
                    "images": [image_path],
                    "question": question,
                    "question_type": "fill_in",
                    "human_or_machine": example.get("human_or_machine"),
                    "pref_vec": pref_vec,
                    "model_mapping": model_mapping,
                    "tool_pricing": tool_pricing_extra,
                    "tools": enhanced_tools,
                },
            }
            items.append(data)

        # Save both parquet and JSON with the same sampled items
        parquet_path = output_root / f"{split_name}.parquet"
        datasets.Dataset.from_list(items).to_parquet(parquet_path.as_posix())
        print(f"Saved {len(items)} items to {parquet_path}")

        json_path = output_root / f"{split_name}.json"
        with open(json_path.as_posix(), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(items)} items to {json_path}")


def _load_tools_from_list(tools: list) -> str:
    """Format tool list for system prompt."""
    tools_str = json.dumps(tools, ensure_ascii=False, indent=2)
    return f"""<tools>
{tools_str}
</tools>"""


if __name__ == "__main__":
    main()
