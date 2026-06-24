import math
import numpy as np
from typing import Any, Optional
import torch
from verl.protocol import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics as verl_compute_data_metrics
from verl.trainer.ppo.metric_utils import bootstrap_metric, calc_maj_val
from functools import partial
from collections import defaultdict


def _safe_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _to_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _add_numeric_stats(result: dict[str, Any], prefix: str, values: list[float]) -> None:
    if not values:
        return
    arr = np.asarray(values, dtype=np.float64)
    result[f"{prefix}/count"] = int(arr.size)
    result[f"{prefix}/mean"] = float(arr.mean())
    result[f"{prefix}/std"] = float(arr.std())
    result[f"{prefix}/min"] = float(arr.min())
    result[f"{prefix}/max"] = float(arr.max())


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    result = verl_compute_data_metrics(batch, use_critic)

    verl_tool_metrics = batch.non_tensor_batch.get("verl_tool_metrics", [])
    if isinstance(verl_tool_metrics, np.ndarray):
        verl_tool_metrics = verl_tool_metrics.tolist()
    if not isinstance(verl_tool_metrics, list):
        verl_tool_metrics = []

    all_keys = set()
    for item in verl_tool_metrics:
        if isinstance(item, dict):
            all_keys.update(str(k) for k in item.keys())
    for key in all_keys:
        values = [
            fv
            for fv in (
                _safe_to_float(item.get(key)) for item in verl_tool_metrics if isinstance(item, dict) and key in item
            )
            if fv is not None
        ]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        result[f"verl_tool/{key}/mean"] = float(arr.mean())
        result[f"verl_tool/{key}/max"] = float(arr.max())
        result[f"verl_tool/{key}/min"] = float(arr.min())

    train_acc_values: list[float] = []
    for acc_key in ("acc", "accuracy", "is_correct"):
        raw_values = batch.non_tensor_batch.get(acc_key, None)
        if raw_values is None:
            continue
        if isinstance(raw_values, np.ndarray):
            raw_values = raw_values.tolist()
        if not isinstance(raw_values, (list, tuple)):
            raw_values = [raw_values]

        train_acc_values = [val for val in (_safe_to_float(v) for v in raw_values) if val is not None]
        if train_acc_values:
            break

    if train_acc_values:
        values = np.array(train_acc_values, dtype=np.float64)
        result["train/acc/mean"] = float(values.mean())
        result["train/acc/std"] = float(values.std())
        result["train/acc/min"] = float(values.min())
        result["train/acc/max"] = float(values.max())

    # Capture reward_extra_info-like payloads as train metrics without exploding dynamic keys.
    # Keys in this skip set are base batch metadata and not reward_extra content.
    skip_reward_extra_keys = {
        "reward_model",
        "data_source",
        "prompts",
        "responses",
        "input_ids",
        "attention_mask",
        "position_ids",
        "raw_prompt",
        "raw_prompt_ids",
        "images",
        "question",
        "format_messages",
        "tool_interact_info",
        "traj_stop_reason",
        "__num_turns__",
        "verl_tool_metrics",
    }
    for key, raw_value in batch.non_tensor_batch.items():
        key = str(key)
        if key in skip_reward_extra_keys:
            continue

        if isinstance(raw_value, dict):
            continue

        prefix = f"train/reward_extra/{key}"
        values = _to_list(raw_value)
        if any(isinstance(v, dict) for v in values):
            continue
        numeric_values = [fv for fv in (_safe_to_float(v) for v in values) if fv is not None]
        _add_numeric_stats(result, prefix, numeric_values)

    return result

def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2uid2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        for uid, var2vals in uid2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue
                    
                var_vals = [x for x in var_vals if x is not None]
                if not var_vals:
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        )
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"], strict=True)
                            ]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2uid2var2metric[data_source][uid][var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)

    return data_src2var2metric2val
