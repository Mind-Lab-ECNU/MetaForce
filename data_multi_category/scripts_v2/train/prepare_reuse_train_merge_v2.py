#!/usr/bin/env python3
"""Build train_merge_v2 from existing processed train JSON files (pure reuse)."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple

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

TRAIN_DATASETS: List[Dict[str, object]] = [
    {"dataset_name": "InfographicVQA_2000", "src_rel_dir": "doc/InfographicVQA_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    # {"dataset_name": "DocVQA_2000", "src_rel_dir": "doc/DocVQA_2000", "source_candidates": ["validation.json", "val.json"], "sample_arg": "train_sample"},
    {"dataset_name": "AOKVQA_2000", "src_rel_dir": "general/AOKVQA_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    {"dataset_name": "COCOQA_2000", "src_rel_dir": "add/general/COCOQA_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    {"dataset_name": "VQAv2_2000", "src_rel_dir": "add/general/VQAv2_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    {"dataset_name": "Visual7W_2000", "src_rel_dir": "add/general/Visual7W_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    {"dataset_name": "TQA_2000", "src_rel_dir": "add/general/TQA_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    # {"dataset_name": "TallyQA_2000", "src_rel_dir": "add/general/TallyQA_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    {"dataset_name": "VQARAD_2000", "src_rel_dir": "add/general/VQARAD_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    {"dataset_name": "VSR_2000", "src_rel_dir": "add/general/VSR_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    # {"dataset_name": "LocalizedNarratives_2000", "src_rel_dir": "add/caption/LocalizedNarratives_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    {"dataset_name": "MapQA_2000", "src_rel_dir": "geospatial/MapQA_2000", "source_candidates": ["train.json"], "sample_arg": "mapqa_train_sample"},
    # {"dataset_name": "CLEVR_2000", "src_rel_dir": "spatial/CLEVR_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    # {"dataset_name": "CLEVR_MATH_2000", "src_rel_dir": "math/CLEVR_MATH_2000", "subset_names": ["addition", "subtraction", "subtraction_multihop", "adversarial"], "source_candidates": ["train.json"], "sample_arg": "clevr_math_train_sample_per_subset", "merge_subsets": True},
    {"dataset_name": "STVQA_2000", "src_rel_dir": "add/ocr/STVQA_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    {"dataset_name": "TextVQA_2000", "src_rel_dir": "add/ocr/TextVQA_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    {"dataset_name": "VisualMRC_2000", "src_rel_dir": "add/ocr/VisualMRC_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    {"dataset_name": "IAM_2000", "src_rel_dir": "add/ocr/IAM_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    # {"dataset_name": "RenderedText_2000", "src_rel_dir": "add/ocr/RenderedText_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    {"dataset_name": "Screen2Words_2000", "src_rel_dir": "add/ocr/Screen2Words_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    {"dataset_name": "TATQA_2000", "src_rel_dir": "add/table/TATQA_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    {"dataset_name": "FinQA_2000", "src_rel_dir": "add/table/FinQA_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    {"dataset_name": "HiTab_2000", "src_rel_dir": "add/table/HiTab_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    {"dataset_name": "RoBUTWTQ_2000", "src_rel_dir": "add/table/RoBUTWTQ_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
    {"dataset_name": "RoBUTSQA_2000", "src_rel_dir": "add/table/RoBUTSQA_2000", "source_candidates": ["train.json"], "sample_arg": "train_sample"},
]


def _pick_source_file(data_root: Path, rel_dir: str, candidates: List[str]) -> Tuple[Path, List[dict]]:
    for filename in candidates:
        path = data_root / rel_dir / filename
        if path.exists():
            return path, load_json(path)
    raise FileNotFoundError(f"No source split found under {data_root / rel_dir}")


def _get_requested_count(args: argparse.Namespace, cfg: Dict[str, object]) -> int:
    sample_arg = str(cfg["sample_arg"])
    return int(getattr(args, sample_arg))


def _upsample_valid_samples(valid: List[dict], requested_count: int, seed: int) -> List[dict]:
    selected = list(valid)
    remaining = requested_count - len(selected)
    if remaining <= 0 or not valid:
        return selected

    rng = random.Random(seed)
    selected.extend(copy.deepcopy(valid[rng.randrange(len(valid))]) for _ in range(remaining))
    return selected


def _sample_regular_dataset(
    data_root: Path,
    seed: int,
    cfg: Dict[str, object],
    requested_count: int,
) -> Tuple[List[dict], Dict[str, object]]:
    src_path, raw = _pick_source_file(
        data_root,
        str(cfg["src_rel_dir"]),
        list(cfg["source_candidates"]),
    )
    valid, errors = filter_valid_samples(raw, data_root)
    if len(valid) < requested_count:
        print(f"{cfg['dataset_name']}: valid samples {len(valid)} < requested {requested_count}")
        selected = _upsample_valid_samples(valid, requested_count, seed)
    else:
        selected = stable_sample(valid, requested_count, seed)

    return selected, {
        "source": str(src_path),
        "source_file": src_path.name,
        "raw_count": len(raw),
        "valid_count": len(valid),
        "invalid_dropped": len(errors),
    }


def _sample_merged_subset_dataset(
    data_root: Path,
    seed: int,
    cfg: Dict[str, object],
    requested_per_subset: int,
) -> Tuple[List[dict], Dict[str, object]]:
    merged: List[dict] = []
    subset_stats: List[Dict[str, object]] = []
    for subset in list(cfg["subset_names"]):
        src_path, raw = _pick_source_file(
            data_root,
            str(Path(str(cfg["src_rel_dir"])) / str(subset)),
            list(cfg["source_candidates"]),
        )
        valid, errors = filter_valid_samples(raw, data_root)
        if len(valid) < requested_per_subset:
            raise ValueError(
                f"{cfg['dataset_name']} subset {subset}: valid samples {len(valid)} < "
                f"requested {requested_per_subset}"
            )

        sampled = stable_sample(valid, requested_per_subset, seed)
        for sample in sampled:
            if isinstance(sample.get("extra_info"), dict):
                sample["extra_info"]["subset"] = subset
                sample["extra_info"]["category"] = "spatial"
        merged.extend(sampled)
        subset_stats.append(
            {
                "subset": subset,
                "source": str(src_path),
                "source_file": src_path.name,
                "raw_count": len(raw),
                "valid_count": len(valid),
                "invalid_dropped": len(errors),
                "selected_count": len(sampled),
            }
        )
        print(
            f"{cfg['dataset_name']}[{subset}]: raw={len(raw)} valid={len(valid)} "
            f"selected={len(sampled)} invalid={len(errors)}"
        )

    return merged, {
        "source": [item["source"] for item in subset_stats],
        "source_file": "train.json",
        "raw_count": sum(int(item["raw_count"]) for item in subset_stats),
        "valid_count": sum(int(item["valid_count"]) for item in subset_stats),
        "invalid_dropped": sum(int(item["invalid_dropped"]) for item in subset_stats),
        "subset_stats": subset_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-base-dir", type=Path, default=None)
    parser.add_argument("--train-merge-dir-name", type=str, default="train_merge_v2")
    parser.add_argument("--test-merge-dir-name", type=str, default="test_merge_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sample", type=int, default=600)
    parser.add_argument("--mapqa-train-sample", type=int, default=600)
    parser.add_argument("--clevr-math-train-sample-per-subset", type=int, default=150)
    args = parser.parse_args()

    output_base_dir = args.output_base_dir if args.output_base_dir is not None else args.data_root
    output_root = output_base_dir / args.train_merge_dir_name
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Dict[str, object]] = {}
    total = 0

    print("=" * 80)
    print("Prepare train_merge_v2 (pure reuse)")
    print("=" * 80)

    for cfg in TRAIN_DATASETS:
        dataset_name = str(cfg["dataset_name"])
        requested_count = _get_requested_count(args, cfg)
        if bool(cfg.get("merge_subsets")):
            selected, stats = _sample_merged_subset_dataset(
                args.data_root,
                args.seed,
                cfg,
                requested_count,
            )
            manifest_requested = requested_count * len(list(cfg["subset_names"]))
        else:
            selected, stats = _sample_regular_dataset(
                args.data_root,
                args.seed,
                cfg,
                requested_count,
            )
            manifest_requested = requested_count

        for sample in selected:
            sample["data_source"] = dataset_name
            if isinstance(sample.get("extra_info"), dict):
                sample["extra_info"]["split"] = "train"

        dst_dir = output_root / dataset_name
        save_json_and_parquet(dst_dir, "train", selected)

        selected_ids = [sample_uid(s) for s in selected]
        manifest[dataset_name] = {
            "requested": manifest_requested,
            "requested_per_subset": requested_count if bool(cfg.get("merge_subsets")) else None,
            "sample_arg": str(cfg["sample_arg"]),
            "selected_count": len(selected),
            "selected_ids": selected_ids,
            **stats,
        }
        total += len(selected)

        print(
            f"{dataset_name}: raw={stats['raw_count']} valid={stats['valid_count']} "
            f"selected={len(selected)} invalid={stats['invalid_dropped']}"
        )

    manifest_path = output_root / "_selection_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("-" * 80)
    print(f"Saved manifest: {manifest_path}")
    print(f"Total train samples: {total}")
    print("=" * 80)


if __name__ == "__main__":
    main()
