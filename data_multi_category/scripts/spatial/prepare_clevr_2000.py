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
Preprocess the CLEVR dataset to parquet format with local images.
Uniformly sample 2000 examples from each split.
"""

import argparse
import json
import os
import random
import shutil
import urllib.parse
import urllib.request
import zipfile
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

_CLEVR_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
_CLEVR_SPLITS = {
    "train": "train",
    "validation": "val",
    "test": "test",
}

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


def _build_mm_content(question: str, image_sep: str, image_count: int) -> str:
    if image_sep and image_sep in question:
        return question
    if image_sep and image_count > 0:
        return (image_sep * image_count) + question
    return question


def _get_images(example: dict):
    if "images" in example and example["images"]:
        return example["images"]
    if "image" in example and example["image"]:
        return example["image"]
    if "image_path" in example and example["image_path"]:
        return example["image_path"]
    return []


def _extract_clevr_qa(example: dict) -> tuple[str, str, str | None]:
    """Extract question, answer, and formatted choices for CLEVR-like schemas."""
    question = example.get("question") or example.get("query") or example.get("problem")
    if not question:
        raise ValueError("Missing question in example.")

    choices = example.get("choices") or example.get("options")
    formatted_choices = _format_choices(choices)
    if formatted_choices:
        question = f"{question}\n\nChoices:\n{formatted_choices}"

    answer = example.get("answer") or example.get("label") or example.get("solution")
    answer_text = _get_choice_answer(choices, answer)
    return str(question), str(answer_text), formatted_choices


def _download_and_extract_clevr(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "CLEVR_v1.0.zip"
    extract_root = cache_dir / "CLEVR_v1.0"

    if not archive_path.exists():
        print(f"Downloading CLEVR dataset to {archive_path} ...")
        urllib.request.urlretrieve(_CLEVR_URL, archive_path.as_posix())

    if not extract_root.exists():
        print(f"Extracting CLEVR dataset to {cache_dir} ...")
        with zipfile.ZipFile(archive_path.as_posix(), "r") as zf:
            zf.extractall(cache_dir.as_posix())

    return extract_root


def _load_clevr_from_source(cache_dir: Path, dataset_config: str) -> datasets.DatasetDict:
    data_root = _download_and_extract_clevr(cache_dir)
    questions_root = data_root / "questions"
    images_root = data_root / "images"

    dataset_dict = datasets.DatasetDict()
    for split_name, split_dir in _CLEVR_SPLITS.items():
        questions_path = questions_root / f"CLEVR_{split_dir}_questions.json"
        image_folder = images_root / split_dir
        with open(questions_path.as_posix(), "r", encoding="utf-8") as f:
            questions = json.load(f)["questions"]

        items = []
        for question in questions:
            record = dict(question)
            record["split"] = split_name
            record["image"] = str(image_folder / record["image_filename"])
            if split_name == "test":
                record.setdefault("question_family_index", -1)
                if "answer" not in record or record["answer"] is None:
                    record["answer"] = -1 if dataset_config == "classification" else ""
                if not record.get("program"):
                    record["program"] = [
                        {"inputs": [], "function": "scene", "value_inputs": []}
                    ]
            items.append(record)

        dataset_dict[split_name] = datasets.Dataset.from_list(items)

    return dataset_dict


def _load_dataset_with_fallback(
    dataset_path: str, dataset_config: str, cache_dir: Path
) -> datasets.DatasetDict:
    try:
        if dataset_config:
            return datasets.load_dataset(dataset_path, dataset_config)
        return datasets.load_dataset(dataset_path)
    except Exception as e:
        err_msg = str(e)
        if "Dataset scripts are no longer supported" not in err_msg:
            raise

        print("Dataset scripts are no longer supported in datasets>=4.0.")
        print("Trying Parquet conversion branch refs/convert/parquet ...")
        try:
            if dataset_config:
                return datasets.load_dataset(
                    dataset_path, dataset_config, revision="refs/convert/parquet"
                )
            return datasets.load_dataset(dataset_path, revision="refs/convert/parquet")
        except Exception as parquet_error:
            print(f"Parquet fallback failed: {parquet_error}")
            print("Falling back to direct CLEVR download and parsing.")
            return _load_clevr_from_source(cache_dir, dataset_config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="HuggingFaceM4/clevr")
    parser.add_argument("--dataset_config", default="")
    parser.add_argument("--output_dir", default="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/spatial")
    parser.add_argument("--image_sep", default="<image>")
    parser.add_argument("--sample_count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tool_json", default=str(Path(__file__).parent.parent / "fake_tool.json"))
    parser.add_argument(
        "--raw_data_dir",
        default="",
        help="Optional cache dir for raw CLEVR data when downloading directly.",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Load tools and build preference vector
    try:
        with open(args.tool_json, "r", encoding="utf-8") as f:
            tools = json.load(f)
        tools_xml = _load_tools(args.tool_json)
    except FileNotFoundError:
        print(f"Warning: Tool JSON not found at {args.tool_json}. Using empty tools.")
        tools_xml = ""

    # Build system prompt with actual tools
    system_prompt = _build_system_prompt(tools_xml)

    cache_root = (
        Path(args.raw_data_dir).expanduser()
        if args.raw_data_dir
        else Path(
            os.environ.get(
                "HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"
            )
        )
        / "clevr_raw"
    )

    try:
        dataset = _load_dataset_with_fallback(
            args.dataset_path, args.dataset_config, cache_root
        )
    except Exception as e:
        print(f"Error loading dataset {args.dataset_path} ({args.dataset_config}): {e}")
        return

    dataset_name = "CLEVR_2000"
    output_root = Path(args.output_dir) / dataset_name
    images_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    for split_name, split_dataset in dataset.items():
        items = []

        # Uniform sampling
        n_samples = min(args.sample_count, len(split_dataset))
        sampled_indices = random.sample(range(len(split_dataset)), n_samples)
        sampled_indices = sorted(sampled_indices)

        for local_idx, orig_idx in enumerate(sampled_indices):
            example = split_dataset[orig_idx]
            try:
                question_text, answer, _ = _extract_clevr_qa(example)
            except Exception as e:
                print(f"Skipping example {orig_idx} due to schema mismatch: {e}")
                print(f"Available keys: {list(example.keys())}")
                continue

            image_items = _get_images(example)
            if isinstance(image_items, list):
                image_items = [img for img in image_items if img is not None]
            elif image_items:
                image_items = [image_items]
            else:
                image_items = []

            # Keep single-image examples to match existing output format.
            if len(image_items) != 1:
                continue

            split_image_dir = images_root / split_name
            local_image_paths = []
            for img_idx, image_item in enumerate(image_items):
                name_prefix = f"{split_name}_{local_idx}"
                try:
                    local_path = _copy_or_save_image(image_item, split_image_dir, name_prefix)
                    local_image_paths.append(local_path)
                except Exception as e:
                    print(f"Error saving image for example {orig_idx}: {e}")
            
            if not local_image_paths and image_items:
                 print(f"Skipping example {orig_idx} due to image save failure")
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
                "ability": "visual_spatial",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(answer),
                },
                "extra_info": extra_info,
            }
            items.append(data)

        # Save parquet and JSON
        parquet_path = output_root / f"{split_name}.parquet"
        datasets.Dataset.from_list(items).to_parquet(parquet_path.as_posix())
        print(f"Saved {len(items)} items to {parquet_path}")

        json_path = output_root / f"{split_name}.json"
        with open(json_path.as_posix(), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(items)} items to {json_path}")


if __name__ == "__main__":
    main()
