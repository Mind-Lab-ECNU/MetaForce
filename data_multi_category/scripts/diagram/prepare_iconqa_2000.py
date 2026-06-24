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
Preprocess the ICON-QA dataset to parquet format with local images.
Uniformly sample 2000 examples from each split.
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


def _format_choices(choices) -> str | None:
    if not choices:
        return None
    if isinstance(choices, dict):
        ordered = sorted(choices.items(), key=lambda x: int(x[0]))
        return "\n".join([f"{key}) {value}" for key, value in ordered])
    if isinstance(choices, list):
        return "\n".join([f"{idx + 1}) {value}" for idx, value in enumerate(choices)])
    return None


def _get_choice_answer(choices, example_answer):
    if isinstance(choices, dict):
        return str(choices.get(str(example_answer), example_answer))
    if isinstance(choices, list):
        try:
            index = int(example_answer)
            if 0 <= index < len(choices):
                return str(choices[index])
        except (TypeError, ValueError):
            pass
    return str(example_answer)


def _get_question_answer(example: dict):
    question = example.get("question") or example.get("problem") or example.get("query")
    answer = example.get("answer") or example.get("solution") or example.get("label") or example.get("answers")
    
    choices = example.get("choices")
    
    return question, answer, choices


def _get_images(example: dict):
    if "images" in example and example["images"]:
        return example["images"]
    if "image" in example and example["image"]:
        return example["image"]
    return []


def _build_mm_content(question: str, image_sep: str, image_count: int) -> str:
    if image_sep and image_sep in question:
        return question
    if image_sep and image_count > 0:
        return (image_sep * image_count) + question
    return question


def _normalize_cauldron_answer(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    lower = text.lower()
    if lower.startswith("answer:"):
        return text[len("answer:") :].strip()
    return text


def _extract_cauldron_qa(example: dict) -> tuple[str, str]:
    texts = example.get("texts")
    if not isinstance(texts, list) or not texts:
        raise ValueError("Expected 'texts' to be a non-empty list.")
    turn = random.choice(texts)
    if not isinstance(turn, dict):
        raise ValueError("Expected each item in 'texts' to be a dict.")
    if "user" not in turn or "assistant" not in turn:
        raise ValueError("Expected each 'texts' turn to contain 'user' and 'assistant'.")
    question = str(turn["user"])
    answer = _normalize_cauldron_answer(turn["assistant"])
    return question, answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="HuggingFaceM4/the_cauldron")
    parser.add_argument("--dataset_config", default="iconqa")
    parser.add_argument("--output_dir", default="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/diagram")
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

    try:
        dataset = datasets.load_dataset(args.dataset_path, args.dataset_config)
    except Exception as e:
        print(f"Error loading dataset {args.dataset_path} ({args.dataset_config}): {e}")
        return

    dataset_name = "ICON-QA_2000"
    output_root = Path(args.output_dir) / dataset_name
    images_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    for split_name, split_dataset in dataset.items():
        items = []
        if len(split_dataset) == 0:
            continue

        n_samples = min(args.sample_count, len(split_dataset))
        sampled_indices = random.sample(range(len(split_dataset)), n_samples)
        sampled_indices = sorted(sampled_indices)

        split_image_dir = images_root / split_name
        split_image_dir.mkdir(parents=True, exist_ok=True)

        for local_idx, orig_idx in enumerate(sampled_indices):
            example = split_dataset[orig_idx]
            try:
                question_text, answer = _extract_cauldron_qa(example)
            except Exception as e:
                print(f"Skipping example {orig_idx} due to schema mismatch: {e}")
                print(f"Available keys: {list(example.keys())}")
                continue

            image_items = example.get("images") or []
            if not isinstance(image_items, list):
                print(f"Skipping example {orig_idx}: expected 'images' to be a list.")
                continue
            image_items = [img for img in image_items if img is not None]

            if len(image_items) != 1:
                continue

            local_image_paths = []
            for img_idx, image_item in enumerate(image_items):
                name_prefix = f"{split_name}_{local_idx}"
                try:
                    local_path = _copy_or_save_image(image_item, split_image_dir, name_prefix)
                    local_image_paths.append(local_path)
                except Exception as e:
                    print(f"Error saving image for index {orig_idx}: {e}")
                    continue

            if not local_image_paths and image_items:
                continue

            mm_content = _build_mm_content(question_text, args.image_sep, len(local_image_paths))
            mm_content = f"{mm_content}\n\n{guideline}"

            extra_info = {
                "split": split_name,
                "index": orig_idx,
                "qid": f"{split_name}_{local_idx}",
                "images": local_image_paths,
                "question": question_text,
            }

            data = {
                "data_source": dataset_name,
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": mm_content},
                ],
                "images": [{"image": path} for path in local_image_paths],
                "ability": "visual_diagram",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(answer),
                },
                "extra_info": extra_info,
            }
                
            items.append(data)

        parquet_path = output_root / f"{split_name}.parquet"
        datasets.Dataset.from_list(items).to_parquet(parquet_path.as_posix())
        print(f"Saved {len(items)} items to {parquet_path}")

        json_path = output_root / f"{split_name}.json"
        with open(json_path.as_posix(), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(items)} items to {json_path}")


if __name__ == "__main__":
    main()
