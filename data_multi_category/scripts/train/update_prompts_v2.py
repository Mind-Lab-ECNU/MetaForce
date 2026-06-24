#!/usr/bin/env python3
"""
Batch-update prompt fields in data files and normalize data format for merged multi-dataset training.
This script updates tool definitions, system prompts, and guidelines, and standardizes all data fields.

Workflow:
1. Run prepare_xxx.py to generate raw data.
2. Run this script to update prompts and normalize data format.
3. Use the output directly for training (no need to run standardize_data.py).

Normalization details:
- extra_info: keep dictionary format (do not serialize to JSON string)
- extra_info["images"]: list of strings ["path/to/img.png"]
- extra_info["question"]: question text extracted from prompt
- images: top-level field, keep original format [{"image": "path/to/img.png"}]
- prompt: standard chat format
- reward_model: {"style": "rule", "ground_truth": "..."}
- data_source: string
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================================
# Configuration section: adjust values here to update prompts
# ============================================================================

# New tool.json path (relative to script directory)
TOOL_JSON_PATH = "real_tool.json"

# New system prompt template
# {tools_xml} will be replaced with the actual tool definitions
NEW_SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
{tools_xml}

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""

# New guidelines
NEW_GUIDELINES = (
    "Guidelines: Use tools/skills only when helpful. "
    "Call a skill to read its full content of SKILL.md: <tool_call>{\"name\": \"skill_name\", \"arguments\": null}</tool_call>. Then you will get the full content of the skill definition and the following executable scripts and their corresponding parameters. "
    "If you must execute a skill script, call "
    "<tool_call>{\"name\": \"run_skill\", \"arguments\": {\"skill_name\": \"skill_name\", "
    "\"entrypoint\": \"scripts/run.py\", \"args\": {\"...\": \"...\"}}}</tool_call> "
    "For run_skill image input: pass `image_index` inside `args` only when the target entrypoint requires image input; "
    "image_index starts from 1, maximum value is the number of images in the current environment. Choose 1 to operate on the first image. "
    "if the skill does not require image input, do not pass `image_index`. "
    "and follow SKILL.md for parameters. "
    "Create a new skill only if no existing tool/skill fits and it is reusable: "
    "<tool_call>{\"name\": \"create_skill\", \"arguments\": {\"description\": \"...\"}}</tool_call>. "
    "Do not create skills casually; scripts (.py/.sh) only if needed. "
    "Format: tool calls must be wrapped in <tool_call>...</tool_call> JSON. "
    "Reasoning must be inside <thinking>...</thinking>; final answer inside <answer>...</answer>."
)

# Data directory path (remote server)
DATA_DIR = Path("/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data")

# Whether to back up original files (default True, recommended)
BACKUP = True

# Sampling parameters
SEED = 42  # Random seed for reproducible sampling
SAMPLE_SIZE = 500  # Number of samples to draw from each file


# ============================================================================
# Processing logic
# ============================================================================

def load_tools(tool_json_path: str) -> str:
    """Load tool definitions and format them as XML."""
    script_dir = Path(__file__).parent
    full_path = script_dir / tool_json_path

    if not full_path.exists():
        raise FileNotFoundError(f"Tool JSON file not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        tools = json.load(f)

    tools_str = json.dumps(tools, ensure_ascii=False, indent=2)
    return f"<tools>\n{tools_str}\n</tools>"


def build_new_system_prompt(tools_xml: str) -> str:
    """Build a new system prompt."""
    return NEW_SYSTEM_PROMPT_TEMPLATE.format(tools_xml=tools_xml)


def extract_question_from_user_content(user_content: str) -> str:
    """
    Extract the question portion from a user message.
    The typical format is: <image>question text\n\nChoices:...\n\nGuidelines:...
    We keep content before the Guidelines section.
    """
    # Find the position of "\n\nGuidelines:"
    guidelines_pos = user_content.find("\n\nGuidelines:")
    if guidelines_pos != -1:
        # Keep everything before Guidelines
        return user_content[:guidelines_pos]
    else:
        # If Guidelines is not found, return original content
        return user_content


def build_new_user_content(original_content: str) -> str:
    """Build new user message content."""
    question_part = extract_question_from_user_content(original_content)
    return f"{question_part}\n\n{NEW_GUIDELINES}"


def normalize_images_field(images: Any) -> List[str]:
    """
    Normalize images field to a list of string paths.
    Supports multiple input formats:
    - ["path/to/img.png"] (already a string list)
    - [{"image": "path/to/img.png"}] (list of objects)
    - "path/to/img.png" (single string)
    - None
    """
    if images is None:
        return []
    
    # Wrap string as a list
    if isinstance(images, str):
        return [images]
    
    # Handle list-like values
    if isinstance(images, (list, tuple)):
        result = []
        for item in images:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                # Try extracting path from dict
                if "image" in item:
                    result.append(item["image"])
                elif "path" in item:
                    result.append(item["path"])
                elif "url" in item:
                    result.append(item["url"])
            else:
                # Convert other types to string
                result.append(str(item))
        return result
    
    # Convert any other type to a one-element string list
    return [str(images)]


def parse_extra_info(extra_info: Any) -> Dict[str, Any]:
    """
    Parse extra_info field from either dict or JSON string format.
    Return a normalized dictionary.
    """
    if extra_info is None:
        return {}
    
    if isinstance(extra_info, dict):
        return extra_info
    
    if isinstance(extra_info, str):
        try:
            parsed = json.loads(extra_info)
            if isinstance(parsed, dict):
                return parsed
            return {}
        except (json.JSONDecodeError, TypeError):
            return {}
    
    return {}


def extract_question_from_prompt(prompt: List[Dict[str, str]]) -> str:
    """
    Extract the user question from prompt messages.
    Find role=user content and remove the Guidelines part.
    """
    for message in prompt:
        if message.get("role") == "user":
            content = message.get("content", "")
            # Remove the Guidelines part
            return extract_question_from_user_content(content)
    return ""


def standardize_sample(sample: Dict[str, Any], new_system_prompt: str) -> Dict[str, Any]:
    """
    Standardize a single data sample.
    
    Normalization steps:
    1. Update system prompt and user content.
    2. Ensure extra_info is a dictionary.
    3. Normalize extra_info["images"] to string list (for reward manager).
    4. Ensure extra_info["question"] exists.
    5. Sync top-level images field (keep format [{"image": path}]).
    6. Ensure data_source is a string.
    7. Ensure reward_model format is correct.
    """
    # Create a copy of the sample
    std_sample = sample.copy()
    
    # 1. Update prompt field
    if "prompt" in sample and isinstance(sample["prompt"], list) and len(sample["prompt"]) >= 2:
        std_sample["prompt"] = sample["prompt"].copy()
        # Update system message
        std_sample["prompt"][0] = sample["prompt"][0].copy()
        std_sample["prompt"][0]["content"] = new_system_prompt
        # Update user message
        std_sample["prompt"][1] = sample["prompt"][1].copy()
        std_sample["prompt"][1]["content"] = build_new_user_content(sample["prompt"][1]["content"])
    
    # 2. Parse and normalize extra_info
    extra_info = parse_extra_info(sample.get("extra_info"))
    
    # 3. Process images field (prefer extra_info, then top-level images)
    # Collect image path list
    if "images" in extra_info:
        image_paths = normalize_images_field(extra_info["images"])
    elif "images" in sample:
        image_paths = normalize_images_field(sample["images"])
    else:
        image_paths = []
    
    # 4. Ensure extra_info["images"] exists and is correctly formatted
    extra_info["images"] = image_paths
    
    # 5. Ensure extra_info["question"] exists
    if "question" not in extra_info or not extra_info["question"]:
        # Extract question from prompt
        if "prompt" in sample:
            extra_info["question"] = extract_question_from_prompt(sample["prompt"])
        else:
            extra_info["question"] = ""
    
    # 6. Ensure required fields exist in extra_info
    if "split" not in extra_info:
        extra_info["split"] = "train"
    if "index" not in extra_info:
        extra_info["index"] = 0
    if "qid" not in extra_info:
        extra_info["qid"] = f"{extra_info['split']}_{extra_info['index']}"
    
    # 7. Update sample extra_info (keep dictionary format, no serialization)
    std_sample["extra_info"] = extra_info
    
    # 8. Sync top-level images field (keep original format: [{"image": path}])
    std_sample["images"] = [{"image": path} for path in image_paths]
    
    # 9. Ensure data_source is a string
    if "data_source" in sample:
        std_sample["data_source"] = str(sample["data_source"])
    else:
        std_sample["data_source"] = "unknown"
    
    # 10. Ensure reward_model format is correct
    if "reward_model" in sample:
        reward_model = sample["reward_model"]
        if isinstance(reward_model, dict):
            std_sample["reward_model"] = {
                "style": reward_model.get("style", "rule"),
                "ground_truth": str(reward_model.get("ground_truth", "")),
            }
        else:
            std_sample["reward_model"] = {"style": "rule", "ground_truth": str(reward_model)}
    else:
        std_sample["reward_model"] = {"style": "rule", "ground_truth": ""}
    
    # 11. Process ability field (if present)
    if "ability" in sample:
        std_sample["ability"] = str(sample["ability"])
    
    return std_sample


# Training dataset config (kept consistent with merge_train_datasets_to_parquet.py)
TRAIN_DATASETS = [
    # ID core (8)
    ("chart/ChartQA_2000", "train.json"),
    ("chart/PlotQA_2000", "train.json"),
    ("geospatial/MapQA_2000", "train.json"),
    ("add/ocr/OCRVQA_2000", "train.json"),
    ("math/GEOQA_2000", "train.json"),
    ("math/geometry3k_2000", "train.json"),
    ("science/ScienceQA_2000", "train.json"),
    ("spatial/CLEVR_2000", "train.json"),
    # ID auxiliary (10)
    ("add/caption/LocalizedNarratives_2000", "train.json"),
    ("add/chart/DVQA_2000", "train.json"),
    ("add/code/WebSight_2000", "train.json"),
    ("add/diagram/DiagramImageToText_2000", "train.json"),
    ("general/AOKVQA_2000", "train.json"),
    ("add/general/VQAv2_2000", "train.json"),
    ("add/math/InterGPS_2000", "train.json"),
    ("add/ocr/TextVQA_2000", "train.json"),
    ("add/table/TATQA_2000", "train.json"),
    ("doc/InfographicVQA_2000", "train.json"),
]


def find_all_json_files(data_dir: Path) -> List[Path]:
    """Find JSON files to process based on TRAIN_DATASETS config."""
    json_files = []

    for rel_path, filename in TRAIN_DATASETS:
        file_path = data_dir / rel_path / filename
        if file_path.exists():
            json_files.append(file_path)
        else:
            print(f"  Warning: file does not exist, skipping: {file_path.relative_to(data_dir)}")

    return sorted(json_files)


def normalize_extra_info_for_json(extra_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize types inside extra_info for JSON saving.
    Convert numeric values to float for cross-dataset type consistency.
    """
    if not isinstance(extra_info, dict):
        return extra_info
    
    result = {}
    for key, value in extra_info.items():
        # Keep images field unchanged
        if key == "images":
            result[key] = value
        # Recursively process nested dictionaries
        elif isinstance(value, dict):
            result[key] = normalize_extra_info_for_json(value)
        # Keep bool unchanged
        elif isinstance(value, bool):
            result[key] = value
        # Convert numeric types to float
        elif isinstance(value, (int, float)):
            result[key] = float(value)
        # Process list values
        elif isinstance(value, list):
            result[key] = [
                float(item) if isinstance(item, (int, float)) and not isinstance(item, bool) else item
                for item in value
            ]
        # Keep other types unchanged
        else:
            result[key] = value
    
    return result


def save_json(file_path: Path, data: List[Dict]):
    """Save JSON file while normalizing extra_info types."""
    # Normalize data types (deep copy to avoid mutating input)
    import copy
    processed_data = copy.deepcopy(data)
    for record in processed_data:
        if 'extra_info' in record and isinstance(record['extra_info'], dict):
            record['extra_info'] = normalize_extra_info_for_json(record['extra_info'])
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)


def save_parquet(file_path: Path, data: List[Dict]) -> tuple[int, int]:
    """Save Parquet file and normalize numeric types inside extra_info.
    
    Returns:
        (missing_image_count, total_image_count)
    """
    import pandas as pd
    
    print(f"    [DEBUG] Start creating DataFrame, record count: {len(data)}")
    df = pd.DataFrame(data)
    print(f"    [DEBUG] DataFrame created, shape: {df.shape}")
    print(f"    [DEBUG] DataFrame columns: {list(df.columns)}")
    
    # Normalize numeric types only inside extra_info (skip images field)
    print(f"    [DEBUG] Normalizing numeric types in extra_info...")
    
    def normalize_extra_info(extra_info):
        """Recursively normalize numeric types in extra_info (keep bool unchanged)."""
        if not isinstance(extra_info, dict):
            return extra_info
        
        result = {}
        for key, value in extra_info.items():
            # Skip images field (path list does not need conversion)
            if key == "images":
                result[key] = value
                continue
            
            # Recursively process nested dictionaries
            if isinstance(value, dict):
                result[key] = normalize_extra_info(value)
            # Keep bool unchanged
            elif isinstance(value, bool):
                result[key] = value
            # Process numeric types (int/float, excluding bool)
            elif isinstance(value, (int, float)):
                result[key] = float(value)
            # Process list elements
            elif isinstance(value, list):
                result[key] = [
                    item if isinstance(item, bool) else
                    (float(item) if isinstance(item, (int, float)) else item)
                    for item in value
                ]
            else:
                result[key] = value
        
        return result
    
    # Apply type conversion to extra_info column
    if 'extra_info' in df.columns:
        df['extra_info'] = df['extra_info'].apply(normalize_extra_info)
        print(f"    [DEBUG] extra_info type normalization complete")
    
    # Validate key fields
    if 'extra_info' in df.columns:
        sample_extra = df['extra_info'].iloc[0] if len(df) > 0 else None
        print(f"    [DEBUG] extra_info type: {type(sample_extra)}")
        if isinstance(sample_extra, dict):
            print(f"    [DEBUG] extra_info example: {sample_extra}")
    if 'images' in df.columns:
        sample_images = df['images'].iloc[0] if len(df) > 0 else None
        print(f"    [DEBUG] images type: {type(sample_images)}")
    
    # Validate that image files actually exist
    print(f"    [DEBUG] Validating image file existence...")
    missing_images = []
    total_images = 0
    for sample in data:
        # Check images from extra_info
        extra_images = sample.get('extra_info', {}).get('images', [])
        for img_path in extra_images:
            total_images += 1
            # Validate using path resolved against DATA_DIR
            if img_path:
                # Resolve relative paths under DATA_DIR
                full_path = Path(img_path)
                if not full_path.is_absolute():
                    full_path = DATA_DIR / full_path
                if not full_path.exists():
                    missing_images.append(img_path)
    
    if missing_images:
        print(f"    Warning: found {len(missing_images)}/{total_images} missing image files")
        print(f"    Example missing paths: {missing_images[:3]}")
    else:
        print(f"    All {total_images} image files exist")
    
    print(f"    [DEBUG] Saving parquet to: {file_path}")
    df.to_parquet(file_path, index=False)
    
    # Verify that the file was created successfully
    if file_path.exists():
        file_size = file_path.stat().st_size
        print(f"    [DEBUG] Parquet file created, size: {file_size / 1024:.2f} KB")
    else:
        raise RuntimeError(f"Parquet file was not created successfully: {file_path}")
    
    return len(missing_images), total_images


def process_file(file_path: Path, new_system_prompt: str) -> tuple[int, int, int]:
    """
    Process one file and update both JSON and Parquet.
    
    Returns:
        (sample_count, missing_image_count, total_image_count)
    """
    # Read raw JSON data
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure list format
    if not isinstance(data, list):
        print(f"  Skipping non-list format file: {file_path.name}")
        return 0, 0, 0

    # Keep original data for backup
    original_data = data

    # Sampling: if data size exceeds SAMPLE_SIZE, randomly sample SAMPLE_SIZE entries
    if len(data) > SAMPLE_SIZE:
        rng = random.Random(SEED)
        data = rng.sample(data, SAMPLE_SIZE)
        print(f"  Sampled {SAMPLE_SIZE} entries (seed={SEED})")

    # Normalize and update each sample
    print(f"  [DEBUG] Start normalizing {len(data)} samples...")
    updated_data = [standardize_sample(sample, new_system_prompt) for sample in data]
    print(f"  [DEBUG] Normalization complete")
    
    # Validate the first sample
    if updated_data:
        sample = updated_data[0]
        print(f"  [DEBUG] Sample validation:")
        print(f"    - data_source: {sample.get('data_source')}")
        print(f"    - images type: {type(sample.get('images'))}")
        print(f"    - extra_info type: {type(sample.get('extra_info'))}")
        if isinstance(sample.get('extra_info'), dict):
            print(f"    - extra_info['images'] type: {type(sample['extra_info'].get('images'))}")
            print(f"    - extra_info['images'] content: {sample['extra_info'].get('images')}")
            print(f"    - extra_info['question'] exists: {'question' in sample['extra_info']}")
        # Verify that the two images fields are aligned
        top_images = sample.get('images', [])
        extra_images = sample.get('extra_info', {}).get('images', [])
        print(f"    - top-level images count: {len(top_images)}")
        print(f"    - extra_info['images'] count: {len(extra_images)}")
        if top_images and extra_images:
            # Extract paths from top-level images
            top_paths = [img.get('image') if isinstance(img, dict) else img for img in top_images]
            print(f"    - top-level images paths: {top_paths}")
            print(f"    - extra_info images paths: {extra_images}")
            if top_paths == extra_images:
                print(f"    Images fields are consistent")
            else:
                print(f"    Warning: images fields are inconsistent")

    # Back up original JSON file (full original data)
    if BACKUP:
        backup_path = file_path.with_suffix(".json.bak")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(original_data, f, ensure_ascii=False, indent=2)
        print(f"  JSON backup created: {backup_path.name}")

    # Save JSON
    print(f"  [DEBUG] Saving JSON to: {file_path}")
    save_json(file_path, updated_data)
    if file_path.exists():
        json_size = file_path.stat().st_size
        print(f"  [DEBUG] JSON file saved, size: {json_size / 1024:.2f} KB")
    
    # Save Parquet (same base name as JSON, different suffix)
    parquet_path = file_path.with_suffix(".parquet")
    print(f"  [DEBUG] Parquet path: {parquet_path}")
    print(f"  [DEBUG] Parquet filename: {parquet_path.name}")
    missing_count = 0
    total_count = 0
    try:
        missing_count, total_count = save_parquet(parquet_path, updated_data)
        print(f"  Parquet saved: {parquet_path.name}")
    except Exception as e:
        print(f"  Warning: failed to save Parquet: {e}")

    return len(updated_data), missing_count, total_count


def main():
    """Main entrypoint."""
    print("=" * 70)
    print("Batch update data files and normalize data format")
    print("Function: update prompts + standardize data fields")
    print("=" * 70)

    # Load tool definitions and build new system prompt
    print(f"\n1. Loading tool definitions: {TOOL_JSON_PATH}")
    try:
        tools_xml = load_tools(TOOL_JSON_PATH)
        print(f"   Tool definitions loaded successfully")
    except FileNotFoundError as e:
        print(f"   Error: {e}")
        print(f"   Ensure {TOOL_JSON_PATH} exists in the script directory")
        return

    new_system_prompt = build_new_system_prompt(tools_xml)
    print(f"   New system prompt built")

    # Show new guidelines
    print(f"\n2. New Guidelines:")
    print(f"   {NEW_GUIDELINES}")

    # Find all JSON files
    print(f"\n3. Scanning data directory: {DATA_DIR}")
    json_files = find_all_json_files(DATA_DIR)
    print(f"   Found {len(json_files)} data files")

    if not json_files:
        print("\nNo files to process. Exiting.")
        return

    # Processing summary
    print(f"\n4. Preparing to process {len(json_files)} files")
    print(f"   Backup original files: {'yes' if BACKUP else 'no'}")
    print(f"   Output format: JSON + Parquet")
    print(f"   Data normalization:")
    print(f"     - extra_info: dictionary format")
    print(f"     - extra_info['images']: list of strings (used by reward manager)")
    print(f"     - extra_info['question']: extracted automatically")
    print(f"     - images: keep original format [{'{'}'image': path{'}'}]")

    # Process each file
    print(f"\n5. Processing...")
    total_samples = 0
    total_missing_images = 0
    total_image_count = 0

    for i, file_path in enumerate(json_files, 1):
        rel_path = file_path.relative_to(DATA_DIR)
        print(f"\n[{i}/{len(json_files)}] Processing: {rel_path}")

        try:
            count, missing_count, img_count = process_file(file_path, new_system_prompt)
            total_samples += count
            total_missing_images += missing_count
            total_image_count += img_count
            print(f"  Updated {count} samples")
            if missing_count > 0:
                print(f"  Warning: this file has {missing_count}/{img_count} missing images")
        except Exception as e:
            print(f"  Failed to process: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print(f"Processing complete")
    print(f"  Files processed: {len(json_files)}")
    print(f"  Samples updated: {total_samples}")
    print(f"  Image validation: {total_missing_images}/{total_image_count} missing")
    if total_missing_images == 0:
        print(f"  All image files exist")
    else:
        print(f"  Warning: found {total_missing_images} missing image files")
    print(f"  Output format: JSON + Parquet")
    print(f"  Data is normalized and ready for training")
    if BACKUP:
        print(f"  Backup location: original filename + '.json.bak'")
    print("=" * 70)


if __name__ == "__main__":
    main()
