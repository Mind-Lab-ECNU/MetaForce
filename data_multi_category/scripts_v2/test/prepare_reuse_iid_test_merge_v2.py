#!/usr/bin/env python3
"""Build IID test_merge_v2 datasets from existing processed files (pure reuse)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from common import (
    DATA_ROOT,
    filter_valid_samples,
    load_json,
    sample_uid,
    save_json_and_parquet,
    stable_sample,
)


IID_REUSE_CONFIG = [
    {
        "output_name": "OCRVQA_test_ID",
        "src_rel_dir": "add/ocr/OCRVQA_2000",
        "preferred_files": ["test.json", "validation.json", "val.json", "train.json"],
        "target_count": 2000,
    },
    {
        "output_name": "RoBUTWikiSQL_test_ID",
        "src_rel_dir": "add/table/RoBUTWikiSQL_2000",
        "preferred_files": ["test.json", "validation.json", "val.json", "train.json"],
        "target_count": 2000,
    },
]

SPLIT_FROM_TRAIN_CONFIG = [
    # {
    #     "dataset_name": "DocVQA_2000",
    #     "src_rel_dir": "doc/DocVQA_2000",
    #     "source_candidates": ["validation.json", "val.json"],
    #     "sample_arg": "train_sample",
    #     "output_name": "DocVQA_test_ID",
    #     "target_count": 2000,
    # },
    # {
    #     "dataset_name": "TallyQA_2000",
    #     "src_rel_dir": "add/general/TallyQA_2000",
    #     "source_candidates": ["train.json"],
    #     "sample_arg": "train_sample",
    #     "output_name": "TallyQA_test_ID",
    #     "target_count": 2000,
    # },
    {
        "dataset_name": "MapQA_2000",
        "src_rel_dir": "geospatial/MapQA_2000",
        "source_candidates": ["train.json"],
        "sample_arg": "mapqa_train_sample",
        "output_name": "MapQA_test_ID",
        "target_count": 2000,
    },
    # {
    #     "dataset_name": "LocalizedNarratives_2000",
    #     "src_rel_dir": "add/caption/LocalizedNarratives_2000",
    #     "source_candidates": ["train.json"],
    #     "sample_arg": "train_sample",
    #     "output_name": "LocalizedNarratives_test_ID",
    #     "target_count": 1400,
    # },
]


def _pick_source_file(data_root: Path, rel_dir: str, candidates: List[str]) -> Tuple[Path, List[dict]]:
    found: List[Tuple[int, int, int, str, Path, List[dict]]] = []
    for rank, filename in enumerate(candidates):
        p = data_root / rel_dir / filename
        if not p.exists():
            continue
        data = load_json(p)
        # Prefer split close to 2000, then candidate order.
        score = abs(len(data) - 2000)
        found.append((score, rank, -len(data), filename, p, data))

    if not found:
        raise FileNotFoundError(f"No source split found under {data_root / rel_dir}")

    found.sort(key=lambda x: (x[0], x[1], x[2]))
    _, _, _, _, path, data = found[0]
    return path, data


def _load_manifest_selected_ids(train_merge_root: Path, dataset_name: str) -> Optional[Set[str]]:
    manifest_path = train_merge_root / "_selection_manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    entry = manifest.get(dataset_name)
    if not isinstance(entry, dict):
        return None
    ids = entry.get("selected_ids")
    if not isinstance(ids, list):
        return None
    return {str(v) for v in ids}


def _compute_train_selected_ids(
    data_root: Path,
    src_rel_dir: str,
    source_candidates: List[str],
    seed: int,
    train_sample: int,
) -> Set[str]:
    src_path, raw = _pick_source_file(data_root, src_rel_dir, source_candidates)
    valid, _ = filter_valid_samples(raw, data_root)
    if len(valid) < train_sample:
        raise ValueError(f"{src_path}: valid train samples {len(valid)} < {train_sample}")

    selected = stable_sample(valid, train_sample, seed)
    return {sample_uid(s) for s in selected}


def _get_train_sample(args: argparse.Namespace, cfg: Dict[str, object]) -> int:
    return int(getattr(args, str(cfg["sample_arg"])))


def _reuse_existing_2000(
    data_root: Path,
    cfg: Dict[str, object],
    seed: int,
) -> List[dict]:
    path, raw = _pick_source_file(
        data_root,
        str(cfg["src_rel_dir"]),
        list(cfg["preferred_files"]),
    )
    valid, errors = filter_valid_samples(raw, data_root)

    target_count = int(cfg["target_count"])
    if len(valid) < target_count:
        print(
            f"{cfg['output_name']}: valid samples {len(valid)} < target {target_count} from {path}"
        )
        valid = stable_sample(valid, len(valid), seed)

    if len(valid) > target_count:
        valid = stable_sample(valid, target_count, seed)

    out_name = str(cfg["output_name"])
    for sample in valid:
        sample["data_source"] = out_name
        if isinstance(sample.get("extra_info"), dict):
            sample["extra_info"]["split"] = "test"

    print(
        f"{out_name}: src={path.name} raw={len(raw)} valid={len(valid)} invalid={len(errors)}"
    )
    return valid


def _build_split_from_train(
    data_root: Path,
    train_merge_root: Path,
    cfg: Dict[str, object],
    seed: int,
    args: argparse.Namespace,
) -> List[dict]:
    dataset_name = str(cfg["dataset_name"])
    src_path, raw = _pick_source_file(
        data_root,
        str(cfg["src_rel_dir"]),
        list(cfg["source_candidates"]),
    )
    valid, errors = filter_valid_samples(raw, data_root)

    selected_ids = _load_manifest_selected_ids(train_merge_root, dataset_name)
    if selected_ids is None:
        selected_ids = _compute_train_selected_ids(
            data_root,
            str(cfg["src_rel_dir"]),
            list(cfg["source_candidates"]),
            seed,
            _get_train_sample(args, cfg),
        )

    remaining = [item for item in valid if sample_uid(item) not in selected_ids]

    target_count = int(cfg["target_count"])
    if len(remaining) < target_count:
        print(
            f"{cfg['output_name']}: remaining samples {len(remaining)} < target {target_count}"
        )
        remaining = stable_sample(remaining, len(remaining), seed + 100)

    # Keep exactly target_count and deterministic order.
    if len(remaining) > target_count:
        remaining = stable_sample(remaining, target_count, seed + 100)

    overlap = selected_ids.intersection({sample_uid(s) for s in remaining})
    if overlap:
        raise ValueError(f"{cfg['output_name']}: found train/test overlap size={len(overlap)}")

    out_name = str(cfg["output_name"])
    for sample in remaining:
        sample["data_source"] = out_name
        if isinstance(sample.get("extra_info"), dict):
            sample["extra_info"]["split"] = "test"

    print(
        f"{out_name}: src={src_path.name} raw={len(raw)} valid={len(valid)} remaining={len(remaining)} "
        f"invalid={len(errors)}"
    )

    return remaining


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-base-dir", type=Path, default=None)
    parser.add_argument("--train-merge-dir-name", type=str, default="train_merge_v2")
    parser.add_argument("--test-merge-dir-name", type=str, default="test_merge_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sample", type=int, default=600)
    parser.add_argument("--mapqa-train-sample", type=int, default=600)
    args = parser.parse_args()

    output_base_dir = args.output_base_dir if args.output_base_dir is not None else args.data_root
    out_root = output_base_dir / args.test_merge_dir_name / "ID"
    out_root.mkdir(parents=True, exist_ok=True)

    train_merge_root = output_base_dir / args.train_merge_dir_name

    print("=" * 80)
    print("Prepare IID test_merge_v2 (pure reuse)")
    print("=" * 80)

    total = 0

    for cfg in IID_REUSE_CONFIG:
        items = _reuse_existing_2000(args.data_root, cfg, args.seed)
        dst_dir = out_root / str(cfg["output_name"])
        save_json_and_parquet(dst_dir, "test", items)
        total += len(items)

    for cfg in SPLIT_FROM_TRAIN_CONFIG:
        items = _build_split_from_train(
            args.data_root,
            train_merge_root,
            cfg,
            args.seed,
            args,
        )
        dst_dir = out_root / str(cfg["output_name"])
        save_json_and_parquet(dst_dir, "test", items)
        total += len(items)

    print("-" * 80)
    print(f"IID total samples: {total}")
    print("=" * 80)


if __name__ == "__main__":
    main()
