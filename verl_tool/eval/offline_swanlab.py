import argparse
import json
import os
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from verl_tool.trainer.ppo.metric_util import process_validation_metrics


DEFAULT_ORDERED_DATASET_SPECS: list[tuple[str, int]] = [
    ("DocVQA_test_ID", 2000),
    ("TallyQA_test_ID", 2000),
    ("OCRVQA_test_ID", 2000),
    ("RoBUTWikiSQL_test_ID", 2000),
    ("MapQA_test_ID", 2000),
    ("LocalizedNarratives_test_ID", 1400),
    ("ChartQA_test_OOD", 2500),
    ("AI2D_test_OOD", 3088),
    ("WebSight_test_OOD", 2000),
    ("MathVista_test_OOD", 1000),
    ("ScienceQA_test_OOD", 2017),
    ("CLEVR_MATH_test_OOD", 2000),

]

TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill SwanLab metrics from standalone validation rollout jsonl files."
    )
    parser.add_argument(
        "--validation-data-dir",
        required=True,
        help="Directory containing validation jsonl files dumped during standalone eval.",
    )
    parser.add_argument(
        "-p",
        "--project-name",
        required=True,
        help="Target SwanLab project name.",
    )
    parser.add_argument(
        "-n",
        "--experiment-name",
        default=None,
        help="Target SwanLab experiment name. Defaults to validation dir name plus timestamp.",
    )
    parser.add_argument(
        "--dataset-spec",
        action="append",
        default=[],
        help="Override ordered dataset specs with entries like Name=Count. Repeatable.",
    )
    parser.add_argument(
        "--swanlab-mode",
        default="cloud",
        help="SwanLab mode. Defaults to cloud.",
    )
    parser.add_argument(
        "--swanlab-logdir",
        default=None,
        help="Optional SwanLab local log directory.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="Logging step for the backfilled metrics. Defaults to 0.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print metrics without uploading to SwanLab.",
    )
    return parser


def parse_dataset_specs(raw_specs: list[str]) -> list[tuple[str, int]]:
    if not raw_specs:
        return list(DEFAULT_ORDERED_DATASET_SPECS)

    parsed: list[tuple[str, int]] = []
    for item in raw_specs:
        if "=" not in item:
            raise ValueError(f"Invalid --dataset-spec '{item}'. Expected format Name=Count.")
        name, count_text = item.split("=", 1)
        name = name.strip()
        count = int(count_text.strip())
        if not name:
            raise ValueError(f"Invalid --dataset-spec '{item}'. Name must be non-empty.")
        if count <= 0:
            raise ValueError(f"Invalid --dataset-spec '{item}'. Count must be positive.")
        parsed.append((name, count))
    return parsed


def _jsonl_sort_key(path: Path) -> tuple[int, Any]:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def load_validation_entries(validation_data_dir: str) -> list[dict[str, Any]]:
    directory = Path(validation_data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Validation data directory not found: {directory}")

    entries: list[dict[str, Any]] = []
    jsonl_files = sorted(
        [
            path
            for path in directory.glob("*.jsonl")
            if path.name != "failures.jsonl"
        ],
        key=_jsonl_sort_key,
    )
    if not jsonl_files:
        raise FileNotFoundError(f"No validation jsonl files found under {directory}")

    for path in jsonl_files:
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Failed to parse {path}:{line_no}: {exc}") from exc
    if not entries:
        raise ValueError(f"No validation entries loaded from {directory}")
    return entries


def build_expected_prompt_sources(dataset_specs: list[tuple[str, int]]) -> list[str]:
    expanded: list[str] = []
    for name, count in dataset_specs:
        expanded.extend([name] * count)
    return expanded


def ordered_unique_uids(entries: list[dict[str, Any]]) -> list[str]:
    seen: OrderedDict[str, None] = OrderedDict()
    for entry in entries:
        uid = entry.get("uid")
        if uid is None:
            raise KeyError("Validation entry is missing 'uid'; cannot restore data_source by prompt order.")
        uid = str(uid)
        if uid not in seen:
            seen[uid] = None
    return list(seen.keys())


def assign_data_source_by_uid(entries: list[dict[str, Any]], dataset_specs: list[tuple[str, int]]) -> dict[str, str]:
    expected_sources = build_expected_prompt_sources(dataset_specs)
    unique_uids = ordered_unique_uids(entries)

    if len(unique_uids) != len(expected_sources):
        raise ValueError(
            "Unique uid count does not match expected prompt count from dataset specs. "
            f"unique_uids={len(unique_uids)} expected_prompts={len(expected_sources)}"
        )

    return {uid: expected_sources[idx] for idx, uid in enumerate(unique_uids)}


def is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)) and not isinstance(value, str)


def extract_numeric_infos(entries: list[dict[str, Any]]) -> dict[str, list[float]]:
    keys: OrderedDict[str, None] = OrderedDict()
    for entry in entries:
        for key, value in entry.items():
            if key in {"step"}:
                continue
            if is_numeric_scalar(value):
                keys.setdefault(str(key), None)

    infos: dict[str, list[float]] = {key: [] for key in keys}
    for entry in entries:
        for key in infos:
            value = entry.get(key)
            infos[key].append(float(value) if is_numeric_scalar(value) else None)

    return {key: values for key, values in infos.items() if any(value is not None for value in values)}


def infer_num_turns(entry: dict[str, Any]) -> tuple[float | None, str | None]:
    for key in ("num_turns", "num_turns_completed", "__num_turns__"):
        value = entry.get(key)
        if is_numeric_scalar(value):
            return float(value), f"field:{key}"

    verl_tool_metrics = entry.get("verl_tool_metrics")
    if isinstance(verl_tool_metrics, dict):
        value = verl_tool_metrics.get("num_turns")
        if is_numeric_scalar(value):
            return float(value), "field:verl_tool_metrics.num_turns"

    output = entry.get("output")
    if isinstance(output, str):
        tool_turns = output.count("</tool_call>")
        residual = TOOL_CALL_BLOCK_RE.sub("", output).strip()
        if tool_turns > 0 or residual:
            return float(tool_turns + (1 if residual else 0)), "heuristic:output"

    return None, None


def collect_num_turns(entries: list[dict[str, Any]]) -> tuple[list[float], str]:
    turns: list[float] = []
    sources: OrderedDict[str, None] = OrderedDict()
    for entry in entries:
        value, source = infer_num_turns(entry)
        if value is None:
            continue
        turns.append(float(value))
        if source:
            sources.setdefault(source, None)

    if not turns:
        return [], "unavailable"
    return turns, ",".join(sources.keys())


def select_core_var(var_names: list[str]) -> str | None:
    for candidate in ("acc", "is_correct", "accuracy", "reward", "score", "total_reward"):
        if candidate in var_names:
            return candidate
    return var_names[0] if var_names else None


def to_builtin_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_builtin_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin_jsonable(item) for item in value]
    return value


def build_metric_dict(
    *,
    entries: list[dict[str, Any]],
    dataset_specs: list[tuple[str, int]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    uid_to_data_source = assign_data_source_by_uid(entries, dataset_specs)
    sample_uids = [str(entry["uid"]) for entry in entries]
    data_sources = [uid_to_data_source[uid] for uid in sample_uids]
    infos_dict = extract_numeric_infos(entries)

    if not infos_dict:
        raise ValueError("No numeric scalar fields were found in validation entries.")

    data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, infos_dict)
    metric_dict: dict[str, Any] = {}
    for data_source, var2metric2val in data_src2var2metric2val.items():
        core_var = select_core_var(list(var2metric2val.keys()))
        for var_name, metric2val in var2metric2val.items():
            n_max = max(int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys())
            for metric_name, metric_val in metric2val.items():
                if (
                    core_var == var_name
                    and any(metric_name.startswith(prefix) for prefix in ("mean", "maj", "best"))
                    and f"@{n_max}" in metric_name
                ):
                    metric_sec = "val-core"
                else:
                    metric_sec = "val-aux"
                metric_dict[f"{metric_sec}/{data_source}/{var_name}/{metric_name}"] = to_builtin_jsonable(metric_val)

    num_turns, num_turns_source = collect_num_turns(entries)
    if num_turns:
        turns_arr = np.asarray(num_turns, dtype=np.float64)
        metric_dict["val-aux/num_turns/min"] = float(turns_arr.min())
        metric_dict["val-aux/num_turns/max"] = float(turns_arr.max())
        metric_dict["val-aux/num_turns/mean"] = float(turns_arr.mean())

    metric_dict["offline-backfill/trajectory_count"] = int(len(entries))
    metric_dict["offline-backfill/prompt_count"] = int(len(uid_to_data_source))
    metric_dict["offline-backfill/data_source_count"] = int(len(dataset_specs))

    metadata = {
        "unique_uid_count": len(uid_to_data_source),
        "numeric_keys": sorted(infos_dict.keys()),
        "num_turns_source": num_turns_source,
        "dataset_specs": [{"name": name, "count": count} for name, count in dataset_specs],
    }
    return metric_dict, metadata


def upload_to_swanlab(
    *,
    metric_dict: dict[str, Any],
    metadata: dict[str, Any],
    project_name: str,
    experiment_name: str,
    swanlab_mode: str,
    swanlab_logdir: str | None,
    step: int,
) -> None:
    api_key = os.environ.get("SWANLAB_API_KEY")
    if swanlab_mode != "local" and not api_key:
        raise EnvironmentError("SWANLAB_API_KEY is not set.")

    import swanlab

    if api_key:
        swanlab.login(api_key)

    init_kwargs = {
        "project": project_name,
        "experiment_name": experiment_name,
        "config": {
            "backfill_source": "validation_rollout",
            "validation_fields": metadata["numeric_keys"],
            "num_turns_source": metadata["num_turns_source"],
            "dataset_specs": metadata["dataset_specs"],
        },
        "mode": swanlab_mode,
    }
    if swanlab_logdir:
        init_kwargs["logdir"] = swanlab_logdir

    swanlab.init(**init_kwargs)
    swanlab.log(metric_dict, step=step)
    swanlab.finish()


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_specs = parse_dataset_specs(args.dataset_spec)
    entries = load_validation_entries(args.validation_data_dir)
    metric_dict, metadata = build_metric_dict(entries=entries, dataset_specs=dataset_specs)

    experiment_name = args.experiment_name
    if not experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{Path(args.validation_data_dir).name}_backfill_{timestamp}"

    print("Detected validation fields:")
    print(", ".join(sorted(metadata["numeric_keys"])))
    print(f"num_turns_source: {metadata['num_turns_source']}")
    print(json.dumps(metric_dict, ensure_ascii=False, indent=2, sort_keys=True))

    if args.dry_run:
        return

    upload_to_swanlab(
        metric_dict=metric_dict,
        metadata=metadata,
        project_name=args.project_name,
        experiment_name=experiment_name,
        swanlab_mode=args.swanlab_mode,
        swanlab_logdir=args.swanlab_logdir,
        step=args.step,
    )


if __name__ == "__main__":
    main()