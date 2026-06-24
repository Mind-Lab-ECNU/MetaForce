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
Split final data into train (90%) and val (10%) subsets.
Reads {split}_final.json and outputs {split}_final_09 and {split}_final_01.
"""

import argparse
import json
import random
from pathlib import Path

import datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Split final data into train/val subsets")
    parser.add_argument("--split", required=True, help="Split name (e.g., train, val, test)")
    parser.add_argument("--input_dir", default=None, help="Input directory containing final data")
    parser.add_argument("--output_dir", default=None, help="Output directory for split data")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio for training set (default: 0.9)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Get script directory and construct relative paths
    script_dir = Path(__file__).parent.absolute()
    verl_duo_dir = script_dir.parent.parent

    # Set default paths using relative paths
    if args.input_dir is None:
        args.input_dir = verl_duo_dir / "data_duo_final" / "ChartQA_2000"
    else:
        args.input_dir = Path(args.input_dir)

    if args.output_dir is None:
        args.output_dir = verl_duo_dir / "data_duo_final" / "ChartQA_2000"
    else:
        args.output_dir = Path(args.output_dir)

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Load final data
    input_file = args.input_dir / f"{args.split}_final.json"
    print(f"Loading data from {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    # Shuffle data
    shuffled = data.copy()
    random.shuffle(shuffled)

    # Split data
    split_idx = int(len(shuffled) * args.train_ratio)
    train_data = shuffled[:split_idx]
    val_data = shuffled[split_idx:]

    print(f"Split into {len(train_data)} train samples ({len(train_data)/len(data)*100:.1f}%) "
          f"and {len(val_data)} val samples ({len(val_data)/len(data)*100:.1f}%)")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Output suffixes
    train_suffix = f"{args.split}_final_09"
    val_suffix = f"{args.split}_final_01"

    # Save train data (90%)
    train_json_path = args.output_dir / f"{train_suffix}.json"
    train_parquet_path = args.output_dir / f"{train_suffix}.parquet"

    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(train_data)} items to {train_json_path}")

    datasets.Dataset.from_list(train_data).to_parquet(str(train_parquet_path))
    print(f"Saved {len(train_data)} items to {train_parquet_path}")

    # Save val data (10%)
    val_json_path = args.output_dir / f"{val_suffix}.json"
    val_parquet_path = args.output_dir / f"{val_suffix}.parquet"

    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(val_data)} items to {val_json_path}")

    datasets.Dataset.from_list(val_data).to_parquet(str(val_parquet_path))
    print(f"Saved {len(val_data)} items to {val_parquet_path}")

    print(f"Total: {len(train_data) + len(val_data)} == {len(data)} samples")


if __name__ == "__main__":
    main()
