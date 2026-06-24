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
Preprocess the FigureQA dataset to parquet format with local images.
Uniformly sample 2000 examples from each split.
"""

import argparse
import json
import io
import random
from pathlib import Path

import datasets
from PIL import Image


def _load_tools(tool_json_path: str) -> str:
    """Load tool definitions from JSON and format for system prompt."""
    with open(tool_json_path, "r", encoding="utf-8") as f:
        tools = json.load(f)

    tools_str = json.dumps(tools, ensure_ascii=False, indent=2)
    return f"""<tools>
{tools_str}
</tools>"""


def _build_pref_vec(tools: list) -> dict:
    """Build preference vector with tool names + accuracy/cost/latency = 0."""
    pref_vec = {}
    for tool in tools:
        tool_name = tool["function"]["name"]
        pref_vec[tool_name] = 0
    # Add the three fixed metrics at the end
    pref_vec["accuracy"] = 0
    pref_vec["cost"] = 0
    pref_vec["latency"] = 0
    return pref_vec


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


guideline = (
    "Guidelines: Understand the given visual information and the user query. "
    "Determine if it is beneficial to employ the given tools. "
    "Reason with the visual information step by step within <thinking></thinking> tags. "
    "Put your final answer within <answer></answer> tags."
)


def _build_mm_content(question: str, image_sep: str, image_count: int) -> str:
    """Build multimodal content with image tokens."""
    if image_sep and image_sep in question:
        return question
    if image_sep and image_count > 0:
        return (image_sep * image_count) + question
    return question


def _save_binary_image(image_bytes: bytes, dest_path: Path) -> str:
    """Save binary image data to disk, returning the absolute path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(io.BytesIO(image_bytes))
    image.save(dest_path.as_posix())
    return dest_path.as_posix()


def _save_pil_image(image: Image.Image, dest_path: Path) -> str:
    """Save PIL Image to disk, returning the absolute path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest_path.as_posix())
    return dest_path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="vikhyatk/figureqa")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--image_sep", default="<image>")
    parser.add_argument("--sample_count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tool_json", default=None)
    args = parser.parse_args()

    # Get script directory and construct relative paths
    script_dir = Path(__file__).parent.absolute()
    verl_duo_dir = script_dir.parent.parent  # Go up two levels to reach verl_duo

    # Set default paths using relative paths
    if args.output_dir is None:
        args.output_dir = verl_duo_dir / "data_duo_final_v2"
    else:
        args.output_dir = Path(args.output_dir)

    if args.tool_json is None:
        args.tool_json = verl_duo_dir / "real_tool.json"

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Load tools and build preference vector
    with open(args.tool_json, "r", encoding="utf-8") as f:
        tools = json.load(f)
    tools_xml = _load_tools(str(args.tool_json))
    pref_vec = _build_pref_vec(tools)

    # Build system prompt with actual tools
    system_prompt = _build_system_prompt(tools_xml)

    dataset = datasets.load_dataset(args.dataset_path)
    dataset_name = "figureqa_2000"
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

            # Get image (dict with 'bytes' and 'path' keys)
            image_item = example.get("image")
            if not image_item:
                continue

            # Save image once (reused for all QAs)
            image_name = f"{split_name}_{local_idx}.png"
            if isinstance(image_item, dict):
                image_bytes = image_item.get("bytes")
                if not image_bytes:
                    continue
                image_path = _save_binary_image(image_bytes, split_image_dir / image_name)
            elif isinstance(image_item, Image.Image):
                image_path = _save_pil_image(image_item, split_image_dir / image_name)
            else:
                continue

            # Get QA pairs
            qa_pairs = example.get("qa", [])

            # Randomly select one QA pair and keep its index
            if qa_pairs:
                qa_idx = random.randrange(len(qa_pairs))
                qa = qa_pairs[qa_idx]

                question = qa.get("question", "")
                answer = qa.get("answer", "").lower()  # Normalize to lowercase: "yes" or "no"

                if question:
                    # Add instruction for yes/no answer
                    question = f"{question} Answer yes or no."

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
                            "qa_index": qa_idx,
                            "qid": f"{split_name}_{local_idx}_{qa_idx}",
                            "images": [image_path],
                            "question": question,
                            "question_type": "fill_in",
                            "pref_vec": pref_vec,
                        }
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


if __name__ == "__main__":
    main()
