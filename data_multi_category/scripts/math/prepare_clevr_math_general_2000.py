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
Preprocess the CLEVR-MATH general config to parquet format with local images.
Uniformly sample 2000 examples from each split for every subset.
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

# CLEVR-Math 数据集 URL
_CLEVR_MATH_URLS = {
    "general": "data/clevr-math.zip",
    "multihop": "data/clevr-math-multihop.zip",
}
_CLEVR_IMAGES_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
_CLEVR_MATH_TEST_IMAGES_FILENAME = "data/clevr-math-test-images.zip"
_CLEVR_MATH_REPO = "dali-does/clevr-math"

_CLEVR_MATH_SPLITS = {
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


def _build_mm_content(question: str, image_sep: str, image_count: int) -> str:
    if image_sep and image_sep in question:
        return question
    if image_sep and image_count > 0:
        return (image_sep * image_count) + question
    return question


def _extract_clevr_math_fields(example: dict) -> tuple[str, str, object]:
    missing = [k for k in ("question", "image", "label") if k not in example]
    if missing:
        raise ValueError(f"Missing required fields in CLEVR-Math example: {missing}")
    question = str(example["question"])
    answer = str(example["label"])
    image_item = example["image"]
    return question, answer, image_item


def _normalize_subset_name(value) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return "default"
    safe = text.replace("/", "_").replace(" ", "_")
    safe = safe.replace("-", "_").replace("__", "_")
    return safe


def _get_subset_name(example: dict) -> str:
    for key in ("subset", "subsplit", "type", "question_type", "template"):
        if key in example and example[key] is not None:
            return _normalize_subset_name(example[key])
    return "default"


def _download_and_extract_clevr_images(cache_dir: Path) -> Path:
    """Download and extract CLEVR images if not already present."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "CLEVR_v1.0.zip"
    extract_root = cache_dir / "CLEVR_v1.0"

    if not archive_path.exists():
        print(f"Downloading CLEVR images to {archive_path} ...")
        urllib.request.urlretrieve(_CLEVR_IMAGES_URL, archive_path.as_posix())

    if not extract_root.exists():
        print(f"Extracting CLEVR images to {cache_dir} ...")
        with zipfile.ZipFile(archive_path.as_posix(), "r") as zf:
            zf.extractall(cache_dir.as_posix())

    return extract_root


def _download_clevr_math_data(cache_dir: Path, config: str) -> Path:
    """Download and extract CLEVR-Math question data."""
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = cache_dir / "extracted" / config

    if extract_dir.exists() and any(extract_dir.iterdir()):
        return extract_dir

    # Download the zip file
    data_filename = _CLEVR_MATH_URLS.get(config, _CLEVR_MATH_URLS["general"])
    zip_path = hf_hub_download(
        repo_id=_CLEVR_MATH_REPO,
        filename=data_filename,
        repo_type="dataset",
        cache_dir=str(cache_dir / "hub_cache"),
    )

    # Extract
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting CLEVR-Math data to {extract_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir.as_posix())

    return extract_dir


def _download_clevr_math_test_images(cache_dir: Path) -> Path:
    """Download and extract CLEVR-Math test images."""
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = cache_dir / "extracted"
    extract_root = extract_dir / "CLEVR_v1.0"

    if extract_root.exists() and any(extract_root.iterdir()):
        return extract_root

    zip_path = hf_hub_download(
        repo_id=_CLEVR_MATH_REPO,
        filename=_CLEVR_MATH_TEST_IMAGES_FILENAME,
        repo_type="dataset",
        cache_dir=str(cache_dir / "hub_cache"),
    )

    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting CLEVR-Math test images to {extract_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir.as_posix())

    return extract_root


def _load_clevr_math_from_source(
    cache_dir: Path,
    clevr_images_dir: Path,
    clevr_test_images_dir: Path,
    config: str,
) -> datasets.DatasetDict:
    """Load CLEVR-Math dataset from local source files."""
    data_dir = _download_clevr_math_data(cache_dir, config)
    train_val_images_root = clevr_images_dir / "images"
    test_images_root = clevr_test_images_dir / "images"

    dataset_dict = datasets.DatasetDict()

    for split_name, split_suffix in _CLEVR_MATH_SPLITS.items():
        json_filename = f"clevr-math-{split_suffix}.json"
        json_path = data_dir / json_filename

        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping {split_name} split.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        questions = json_data.get("questions", [])
        items = []

        for q in questions:
            image_filename = q.get("image_filename", "")
            q_split = q.get("split", split_suffix)

            # Construct image path
            if split_name == "test":
                image_path = test_images_root / "test" / image_filename
            else:
                image_path = train_val_images_root / q_split / image_filename

            template = q.get("template_filename", "")
            if template.endswith(".json"):
                template = template[:-5]

            record = {
                "template": template,
                "id": image_filename,
                "question": q.get("question", ""),
                "image": str(image_path) if image_path.exists() else "",
                "label": q.get("answer", 0),
            }
            items.append(record)

        dataset_dict[split_name] = datasets.Dataset.from_list(items)
        print(f"Loaded {len(items)} examples for {split_name} split.")

    return dataset_dict


def _load_dataset_with_fallback(
    dataset_path: str,
    dataset_config: str,
    cache_dir: Path,
    clevr_images_dir: Path,
    clevr_test_images_dir: Path,
) -> datasets.DatasetDict:
    """Try to load dataset from HuggingFace, fallback to local source if needed."""
    try:
        if dataset_config:
            return datasets.load_dataset(dataset_path, dataset_config)
        return datasets.load_dataset(dataset_path)
    except Exception as e:
        err_msg = str(e)
        if "Dataset scripts are no longer supported" not in err_msg:
            # Try parquet revision
            print("Direct load failed, trying parquet revision...")
            try:
                if dataset_config:
                    return datasets.load_dataset(
                        dataset_path, dataset_config, revision="refs/convert/parquet"
                    )
                return datasets.load_dataset(dataset_path, revision="refs/convert/parquet")
            except Exception:
                pass

        print("Dataset scripts are no longer supported in datasets>=4.0.")
        print("Falling back to direct CLEVR-Math download and parsing.")
        return _load_clevr_math_from_source(
            cache_dir,
            clevr_images_dir,
            clevr_test_images_dir,
            dataset_config,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="dali-does/clevr-math")
    parser.add_argument("--dataset_config", default="general")
    parser.add_argument(
        "--output_dir",
        default="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/math",
    )
    parser.add_argument("--image_sep", default="<image>")
    parser.add_argument("--sample_count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tool_json", default=str(Path(__file__).parent.parent / "fake_tool.json"))
    parser.add_argument(
        "--raw_data_dir",
        default="",
        help="Optional cache dir for raw CLEVR-Math data when downloading directly.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    tool_json_path = Path(args.tool_json)
    if not tool_json_path.exists():
        tool_json_path = Path(__file__).parent.parent / "fake_tool.json"

    with open(tool_json_path, "r", encoding="utf-8") as f:
        tools = json.load(f)
    tools_xml = _load_tools(str(tool_json_path))

    system_prompt = _build_system_prompt(tools_xml)

    # Setup cache directories
    cache_root = (
        Path(args.raw_data_dir).expanduser()
        if args.raw_data_dir
        else Path(
            os.environ.get(
                "HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"
            )
        )
        / "clevr_math_raw"
    )

    # Use existing CLEVR images from clevr_raw if available
    clevr_images_cache = Path(
        os.environ.get(
            "HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"
        )
    ) / "clevr_raw"

    # Check if CLEVR images already exist
    clevr_images_dir = clevr_images_cache / "CLEVR_v1.0"
    if not clevr_images_dir.exists():
        print("CLEVR images not found, downloading...")
        clevr_images_dir = _download_and_extract_clevr_images(clevr_images_cache)
    else:
        print(f"Using existing CLEVR images from {clevr_images_dir}")

    clevr_math_test_images_cache = (
        Path(
            os.environ.get(
                "HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"
            )
        )
        / "clevr_math_test_images_raw"
    )
    clevr_math_test_images_dir = clevr_math_test_images_cache / "extracted" / "CLEVR_v1.0"
    if not clevr_math_test_images_dir.exists():
        print("CLEVR-Math test images not found, downloading...")
        clevr_math_test_images_dir = _download_clevr_math_test_images(
            clevr_math_test_images_cache
        )
    else:
        print(f"Using existing CLEVR-Math test images from {clevr_math_test_images_dir}")

    try:
        dataset = _load_dataset_with_fallback(
            args.dataset_path,
            args.dataset_config,
            cache_root,
            clevr_images_dir,
            clevr_math_test_images_dir,
        )
    except Exception as e:
        print(f"Error loading dataset {args.dataset_path} ({args.dataset_config}): {e}")
        return

    dataset_name = "CLEVR_MATH_2000"
    output_root = Path(args.output_dir) / dataset_name
    images_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    for split_name, split_dataset in dataset.items():
        if len(split_dataset) == 0:
            continue

        subset_indices: dict[str, list[int]] = {}
        for idx in range(len(split_dataset)):
            example = split_dataset[idx]
            subset_name = _get_subset_name(example)
            subset_indices.setdefault(subset_name, []).append(idx)

        for subset_name, indices in subset_indices.items():
            items = []
            n_samples = min(args.sample_count, len(indices))
            sampled_indices = random.sample(indices, n_samples)
            sampled_indices = sorted(sampled_indices)

            split_image_dir = images_root / subset_name / split_name
            split_image_dir.mkdir(parents=True, exist_ok=True)

            for local_idx, orig_idx in enumerate(sampled_indices):
                example = split_dataset[orig_idx]
                try:
                    question, answer, image_item = _extract_clevr_math_fields(example)
                except Exception as e:
                    print(f"Skipping example {orig_idx} due to schema mismatch: {e}")
                    print(f"Available keys: {list(example.keys())}")
                    continue

                image_items = [image_item] if image_item is not None else []
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

                question_text = str(question)
                mm_content = _build_mm_content(question_text, args.image_sep, len(local_image_paths))
                mm_content = f"{mm_content}\n\n{guideline}"

                extra_info = {
                    "split": split_name,
                    "subset": subset_name,
                    "index": orig_idx,
                    "qid": f"{split_name}_{subset_name}_{local_idx}",
                    "images": local_image_paths,
                    "question": question_text,
                    "template": example.get("template"),
                    "id": example.get("id"),
                }

                data = {
                    "data_source": dataset_name,
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": mm_content},
                    ],
                    "images": [{"image": path} for path in local_image_paths],
                    "ability": "visual_math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": answer,
                    },
                    "extra_info": extra_info,
                }
                items.append(data)

            subset_root = output_root / subset_name
            subset_root.mkdir(parents=True, exist_ok=True)

            parquet_path = subset_root / f"{split_name}.parquet"
            datasets.Dataset.from_list(items).to_parquet(parquet_path.as_posix())
            print(f"Saved {len(items)} items to {parquet_path}")

            json_path = subset_root / f"{split_name}.json"
            with open(json_path.as_posix(), "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(items)} items to {json_path}")


if __name__ == "__main__":
    main()
