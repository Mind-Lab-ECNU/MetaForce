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
Preprocess the MapQA test dataset from V and S subsets (raw data) to parquet format with local images.
Uniformly sample 2000 examples from the combined test split.
"""

import argparse
import json
import random
import shutil
import urllib.parse
import urllib.request
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
    "You must reason and reason with the visual information step by step within <thinking></thinking> tags. "
    "Put your final answer within <answer></answer> tags."
)


def _is_url(value: str) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"}


def _safe_ext(path: str, default_ext: str = ".png") -> str:
    ext = Path(path).suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return ext
    return default_ext


def _download_url(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest_path.as_posix())


def _save_pil_image(image: Image.Image, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest_path.as_posix())


def _copy_or_save_image(image_item, dest_dir: Path, name_prefix: str) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(image_item, str):
        if _is_url(image_item):
            ext = _safe_ext(urllib.parse.urlparse(image_item).path)
            dest_path = dest_dir / f"{name_prefix}{ext}"
            _download_url(image_item, dest_path)
            return dest_path.as_posix()

        src_path = Path(image_item)
        if src_path.exists():
            ext = _safe_ext(src_path.as_posix())
            dest_path = dest_dir / f"{name_prefix}{ext}"
            shutil.copy2(src_path, dest_path)
            return dest_path.as_posix()

        raise FileNotFoundError(f"Image path not found: {image_item}")

    if isinstance(image_item, dict):
        if "path" in image_item and image_item["path"]:
            src_path = Path(image_item["path"])
            if src_path.exists():
                ext = _safe_ext(src_path.as_posix())
                dest_path = dest_dir / f"{name_prefix}{ext}"
                shutil.copy2(src_path, dest_path)
                return dest_path.as_posix()
        if "bytes" in image_item and image_item["bytes"]:
            dest_path = dest_dir / f"{name_prefix}.png"
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path.as_posix(), "wb") as f:
                f.write(image_item["bytes"])
            return dest_path.as_posix()

        raise ValueError("Unsupported image dict format.")

    if isinstance(image_item, Image.Image):
        dest_path = dest_dir / f"{name_prefix}.png"
        _save_pil_image(image_item, dest_path)
        return dest_path.as_posix()

    raise TypeError(f"Unsupported image type: {type(image_item)}")


def _build_mm_content(question: str, image_sep: str, image_count: int) -> str:
    if image_sep and image_sep in question:
        return question
    if image_sep and image_count > 0:
        return (image_sep * image_count) + question
    return question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_root",
        default="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/scripts/raw_data/MapQA",
    )
    parser.add_argument("--output_dir", default="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/geospatial")
    parser.add_argument("--image_sep", default="<image>")
    parser.add_argument("--sample_count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tool_json", default=str(Path(__file__).parent.parent / "fake_tool.json"))
    args = parser.parse_args()

    random.seed(args.seed)

    tool_json_path = Path(args.tool_json)
    if not tool_json_path.exists():
        tool_json_path = Path(__file__).parent.parent / "fake_tool.json"

    with open(tool_json_path, "r", encoding="utf-8") as f:
        tools = json.load(f)
    tools_xml = _load_tools(str(tool_json_path))

    system_prompt = _build_system_prompt(tools_xml)

    raw_root = Path(args.raw_root)
    subset_defs = {
        "V": {
            "qa": raw_root / "MapQA_V" / "MapQA_V" / "questions" / "test-QA.json",
            "images": raw_root / "MapQA_V" / "MapQA_V" / "images",
        },
        "S": {
            "qa": raw_root / "MapQA_S" / "MapQA_S" / "questions" / "test-QA.json",
            "images": raw_root / "MapQA_S" / "MapQA_S" / "images",
        },
    }

    qa_data = []
    for subset_name, paths in subset_defs.items():
        qa_path = paths["qa"]
        image_root = paths["images"]
        if not qa_path.exists():
            print(f"QA file not found: {qa_path}")
            return
        if not image_root.exists():
            print(f"Image directory not found: {image_root}")
            return
        with open(qa_path, "r", encoding="utf-8") as f:
            subset_qa = json.load(f)
        for item in subset_qa:
            item["_subset"] = subset_name
        qa_data.extend(subset_qa)

    split_name = "test"
    dataset_name = "MapQA_2000"
    output_root = Path(args.output_dir) / dataset_name
    images_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    if not qa_data:
        print("No QA data found.")
        return

    n_samples = min(args.sample_count, len(qa_data))
    sampled_indices = random.sample(range(len(qa_data)), n_samples)
    sampled_indices = sorted(sampled_indices)

    split_image_dir = images_root / split_name
    split_image_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for local_idx, orig_idx in enumerate(sampled_indices):
        example = qa_data[orig_idx]
        question = str(example.get("question", "")).strip()
        answer = example.get("answer")
        map_id = example.get("map_id")

        if not question:
            continue
        if answer is None or str(answer).strip() == "":
            continue
        if not map_id:
            continue

        subset_name = example.get("_subset")
        subset_images = subset_defs.get(subset_name, {}).get("images")
        if subset_images is None:
            continue
        image_path = subset_images / map_id
        if not image_path.exists():
            print(f"Missing image: {image_path}")
            continue

        try:
            local_path = _copy_or_save_image(image_path.as_posix(), split_image_dir, f"{split_name}_{local_idx}")
        except Exception as e:
            print(f"Error saving image for index {orig_idx}: {e}")
            continue

        mm_content = _build_mm_content(question, args.image_sep, 1)
        mm_content = f"{mm_content}\n\n{guideline}"

        extra_info = {
            "split": split_name,
            "index": orig_idx,
            "qid": f"{split_name}_{local_idx}",
            "images": [local_path],
            "question": question,
            "subset": subset_name,
            "map_id": map_id,
            "question_type": example.get("question_type"),
            "question_template_id": example.get("question_template_id"),
            "question_id": example.get("question_id"),
            "oracle_delexicalized_question": example.get("oracle_delexicalized_question"),
            "tesseract_delexicalized_question": example.get("tesseract_delexicalized_question"),
        }

        data = {
            "data_source": dataset_name,
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": mm_content},
            ],
            "images": [{"image": local_path}],
            "ability": "visual_geospatial",
            "reward_model": {
                "style": "rule",
                "ground_truth": str(answer),
            },
            "extra_info": extra_info,
        }
        items.append(data)

    if not items:
        print("No items collected. Skipping save.")
        return

    parquet_path = output_root / f"{split_name}.parquet"
    datasets.Dataset.from_list(items).to_parquet(parquet_path.as_posix())
    print(f"Saved {len(items)} items to {parquet_path}")

    json_path = output_root / f"{split_name}.json"
    with open(json_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(items)} items to {json_path}")


if __name__ == "__main__":
    main()
