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
Preprocess the VQA-RAD dataset to parquet format with local images.
Uniformly sample 2000 examples from val/test splits.
"""

import argparse
import json
import random
import shutil
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import datasets
from huggingface_hub import hf_hub_download
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

def _download_hf_file(repo_id: str, filename: str, revision: str | None = None) -> Path:
    cached_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )
    return Path(cached_path)

def _save_pil_image(image: Image.Image, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest_path.as_posix())


def _resolve_relative_path(path_str: str, base_dirs):
    for base_dir in base_dirs:
        candidate = base_dir / path_str
        if candidate.exists():
            return candidate
    return None


def _resolve_under_root(path_str: str, image_root: Path | None) -> Path | None:
    if image_root is None:
        return None
    rel_path = path_str.lstrip("/")
    candidate = image_root / rel_path
    if candidate.exists():
        return candidate
    return None


def _is_image_root_ready(image_root: Path) -> bool:
    image_dir = image_root / "VQA_RAD" / "Image"
    if not image_dir.exists():
        return False
    return any(image_dir.glob("*.jpg")) or any(image_dir.glob("*.png"))


def _normalize_image_root(image_root: Path) -> None:
    if _is_image_root_ready(image_root):
        return
    target = image_root / "VQA_RAD" / "Image"
    candidates = [
        image_root / "Image",
        image_root / "images",
        image_root / "vqa_rad" / "Image",
        image_root / "VQA_RAD" / "images",
    ]
    for cand in candidates:
        if cand.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.move(cand.as_posix(), target.as_posix())
            return
    for cand in image_root.rglob("Image"):
        if cand.is_dir() and any(cand.glob("synpic*.jpg")):
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.move(cand.as_posix(), target.as_posix())
            return


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    lower = archive_path.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
        return
    if lower.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
        return
    raise ValueError(f"Unsupported archive format: {archive_path}")


def _ensure_hf_images(
    image_root: Path,
    hf_repo_id: str,
    archive_name: str,
) -> None:
    if _is_image_root_ready(image_root):
        return
    image_root.mkdir(parents=True, exist_ok=True)
    cached_path = hf_hub_download(repo_id=hf_repo_id, filename=archive_name, repo_type="dataset")
    archive_path = Path(cached_path)
    _extract_archive(archive_path, image_root)
    _normalize_image_root(image_root)


def _copy_or_save_image(
    image_item,
    dest_dir: Path,
    name_prefix: str,
    base_dirs=None,
    hf_repo_id: str | None = None,
    hf_revision: str | None = None,
    image_root: Path | None = None,
) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    base_dirs = base_dirs or []

    if isinstance(image_item, str):
        if _is_url(image_item):
            ext = _safe_ext(urllib.parse.urlparse(image_item).path)
            dest_path = dest_dir / f"{name_prefix}{ext}"
            _download_url(image_item, dest_path)
            return dest_path.as_posix()

        src_path = Path(image_item)
        if not src_path.exists() and base_dirs:
            resolved = _resolve_relative_path(image_item, base_dirs)
            if resolved is not None:
                src_path = resolved
        if not src_path.exists() and image_root is not None:
            resolved = _resolve_under_root(image_item, image_root)
            if resolved is not None:
                src_path = resolved

        if src_path.exists():
            ext = _safe_ext(src_path.as_posix())
            dest_path = dest_dir / f"{name_prefix}{ext}"
            shutil.copy2(src_path, dest_path)
            return dest_path.as_posix()

        if hf_repo_id:
            rel_path = image_item.lstrip("/")
            try:
                cached_path = _download_hf_file(hf_repo_id, rel_path, hf_revision)
                ext = _safe_ext(cached_path.as_posix())
                dest_path = dest_dir / f"{name_prefix}{ext}"
                shutil.copy2(cached_path, dest_path)
                return dest_path.as_posix()
            except Exception:
                pass

        raise FileNotFoundError(f"Image path not found: {image_item}")

    if isinstance(image_item, dict):
        if "path" in image_item and image_item["path"]:
            src_path = Path(image_item["path"])
            if not src_path.exists() and base_dirs:
                resolved = _resolve_relative_path(image_item["path"], base_dirs)
                if resolved is not None:
                    src_path = resolved
            if not src_path.exists() and image_root is not None:
                resolved = _resolve_under_root(str(image_item["path"]), image_root)
                if resolved is not None:
                    src_path = resolved
            if src_path.exists():
                ext = _safe_ext(src_path.as_posix())
                dest_path = dest_dir / f"{name_prefix}{ext}"
                shutil.copy2(src_path, dest_path)
                return dest_path.as_posix()
            if hf_repo_id:
                rel_path = str(image_item["path"]).lstrip("/")
                try:
                    cached_path = _download_hf_file(hf_repo_id, rel_path, hf_revision)
                    ext = _safe_ext(cached_path.as_posix())
                    dest_path = dest_dir / f"{name_prefix}{ext}"
                    shutil.copy2(cached_path, dest_path)
                    return dest_path.as_posix()
                except Exception:
                    pass
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


def _normalize_answer(answer) -> str:
    if answer is None:
        return ""
    if isinstance(answer, (list, tuple)):
        for item in answer:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                return text
        return ""
    if isinstance(answer, dict):
        for key in ("answer", "answer_text", "text", "value", "final"):
            if key in answer:
                return _normalize_answer(answer[key])
        return json.dumps(answer, ensure_ascii=False)
    return str(answer)


def _pick_text(value, default: str = "") -> str:
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    if value is None:
        return default
    return str(value)


def _format_text_list(texts) -> str:
    if not texts:
        return ""
    if isinstance(texts, str):
        return texts
    if isinstance(texts, list):
        parts = []
        for item in texts:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(texts)


def _format_table(table) -> str:
    if table is None:
        return ""
    if isinstance(table, dict):
        for key in ("table", "table_content", "table_data", "texts", "rows", "cells"):
            if key in table:
                return _format_table(table[key])
        return json.dumps(table, ensure_ascii=False)
    if isinstance(table, list):
        if table and all(isinstance(row, list) for row in table):
            return "\n".join(" | ".join(str(cell) for cell in row) for row in table)
        return "\n".join(str(row) for row in table)
    return str(table)

def _extract_sample(example: dict) -> tuple[str, str, list, str, dict]:
    question = _pick_text(example.get("query") or example.get("question"), default="")
    answer = example.get("response") or example.get("answer")
    images = example.get("images") or example.get("image")
    if images is None:
        image_items = []
    elif isinstance(images, list):
        image_items = images
    else:
        image_items = [images]
    return question, _normalize_answer(answer), image_items, "", {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="dz-osamu/VQA-RAD")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--output_dir", default="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/add/general")
    parser.add_argument("--image_sep", default="<image>")
    parser.add_argument("--sample_count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tool_json", default=str(Path(__file__).parent.parent / "fake_tool.json"))
    parser.add_argument("--image_root", default=None, help="Root directory that contains VQA_RAD/Image/...")
    parser.add_argument(
        "--auto_download_images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto download VQA-RAD images when missing",
    )
    parser.add_argument("--image_archive", default="image.zip", help="Archive filename in HF dataset repo")
    args = parser.parse_args()

    random.seed(args.seed)

    try:
        with open(args.tool_json, "r", encoding="utf-8") as f:
            tools = json.load(f)
        tools_xml = _load_tools(args.tool_json)
    except FileNotFoundError:
        print(f"Warning: Tool JSON not found at {args.tool_json}. Using empty tools.")
        tools_xml = ""

    system_prompt = _build_system_prompt(tools_xml)

    hf_repo_id = None
    try:
        if not Path(args.dataset_path).exists() and "/" in args.dataset_path:
            hf_repo_id = args.dataset_path
    except Exception:
        hf_repo_id = None
    image_root = Path(args.image_root).expanduser() if args.image_root else None
    if image_root is None:
        image_root = (Path(args.output_dir) / "vqarad_raw").expanduser()
    if args.auto_download_images:
        if hf_repo_id is None:
            print("Warning: auto download images skipped (dataset_path is local, no HF repo_id).")
        else:
            try:
                _ensure_hf_images(
                    image_root,
                    hf_repo_id=hf_repo_id,
                    archive_name=args.image_archive,
                )
            except Exception as e:
                print(f"Warning: auto download images failed: {e}")

    try:
        if args.dataset_config is None:
            dataset = datasets.load_dataset(args.dataset_path)
        else:
            dataset = datasets.load_dataset(args.dataset_path, args.dataset_config)
    except Exception as e:
        print(f"Error loading dataset {args.dataset_path} ({args.dataset_config}): {e}")
        return

    dataset_name = "VQARAD_2000"
    output_root = Path(args.output_dir) / dataset_name
    images_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    split_aliases = {
        "validation": "val",
        "val": "val",
        "dev": "val",
        "dev_seen": "val",
        "dev_unseen": "val",
        "test": "test",
        "test_seen": "test",
        "test_unseen": "test",
    }
    items_by_split = {"val": [], "test": []}

    for split_name, split_dataset in dataset.items():
        if split_name not in split_aliases:
            continue
        output_split = split_aliases[split_name]
        items = items_by_split[output_split]

        base_dirs = []
        try:
            for cache in split_dataset.cache_files:
                base_dirs.append(Path(cache["filename"]).parent)
        except Exception:
            base_dirs = []

        n_samples = min(args.sample_count, len(split_dataset))
        sampled_indices = random.sample(range(len(split_dataset)), n_samples)
        sampled_indices = sorted(sampled_indices)

        for local_idx, orig_idx in enumerate(sampled_indices):
            example = split_dataset[orig_idx]
            try:
                question_text, answer, image_items, context_text, extra_fields = _extract_sample(example)
            except Exception as e:
                print(f"Skipping example {orig_idx} due to schema mismatch: {e}")
                print(f"Available keys: {list(example.keys())}")
                continue

            if not isinstance(image_items, list):
                image_items = [image_items] if image_items is not None else []
            image_items = [img for img in image_items if img is not None]

            split_image_dir = images_root / output_split
            local_image_paths = []
            for img_idx, image_item in enumerate(image_items):
                name_prefix = f"{split_name}_{local_idx}"
                try:
                    local_path = _copy_or_save_image(
                        image_item,
                        split_image_dir,
                        name_prefix,
                        base_dirs=base_dirs,
                        hf_repo_id=hf_repo_id,
                        image_root=image_root,
                    )
                    local_image_paths.append(local_path)
                except Exception as e:
                    print(f"Error saving image for example {orig_idx}: {e}")

            if image_items and not local_image_paths:
                print(f"Skipping example {orig_idx} due to image save failure")
                continue

            question_with_context = question_text
            if context_text:
                question_with_context = f"{question_text}\n\n{context_text}"

            mm_content = _build_mm_content(question_with_context, args.image_sep, len(local_image_paths))
            mm_content = f"{mm_content}\n\n{guideline}"

            extra_info = {
                "split": output_split,
                "index": orig_idx,
                "qid": f"{split_name}_{local_idx}",
                "images": local_image_paths,
                "question": question_text,
            }
            if extra_fields:
                extra_info.update(extra_fields)

            data = {
                "data_source": dataset_name,
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": mm_content},
                ],
                "images": [{"image": path} for path in local_image_paths],
                "ability": "visual_general",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(answer),
                },
                "extra_info": extra_info,
            }
            items.append(data)

    for output_split, items in items_by_split.items():
        if not items:
            continue
        parquet_path = output_root / f"{output_split}.parquet"
        datasets.Dataset.from_list(items).to_parquet(parquet_path.as_posix())
        print(f"Saved {len(items)} items to {parquet_path}")

        json_path = output_root / f"{output_split}.json"
        with open(json_path.as_posix(), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(items)} items to {json_path}")


if __name__ == "__main__":
    main()
