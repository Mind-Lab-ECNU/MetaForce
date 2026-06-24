#!/usr/bin/env python3
"""
【第二步：工具动态与轨迹质量指标】
从 unified.parquet 计算训练/评测过程中的工具使用、失败模式和轨迹质量指标。

输出：
  results/tool_dynamics.csv
  results/tool_dynamics.json
  results/tool_dynamics_by_step.csv
  results/tool_failure_breakdown.csv
  results/trajectory_quality_by_step.csv
  results/trajectory_quality_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from common import ensure_dir, truncate_text, write_json


def _step_local_tool_rows(df: pd.DataFrame) -> pd.DataFrame:
    tool_call = df[df["row_type"] == "tool_call"].copy()
    tool_example = df[df["row_type"] == "tool_rollout_example"].copy()
    tool_summary = df[df["row_type"] == "tool_summary"].copy()

    picked: List[pd.DataFrame] = []
    call_steps = set(pd.to_numeric(tool_call["step"], errors="coerce").dropna().tolist())
    if len(tool_call):
        picked.append(tool_call)

    tool_example = tool_example[~tool_example["step"].isin(call_steps)].copy()
    example_steps = call_steps | set(pd.to_numeric(tool_example["step"], errors="coerce").dropna().tolist())
    if len(tool_example):
        picked.append(tool_example)

    tool_summary = tool_summary[~tool_summary["step"].isin(example_steps)].copy()
    if len(tool_summary):
        picked.append(tool_summary)

    if not picked:
        return df.iloc[0:0].copy()
    return pd.concat(picked, ignore_index=True)


def _ensure_trajectory_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "trajectory_key" in out.columns:
        out["trajectory_key"] = out["trajectory_key"].fillna("").astype(str).str.strip()
        return out
    request_id = (
        out["request_id"].fillna("").astype(str).str.strip()
        if "request_id" in out.columns
        else pd.Series("", index=out.index, dtype=object)
    )
    uid = (
        out["uid"].fillna("").astype(str).str.strip()
        if "uid" in out.columns
        else pd.Series("", index=out.index, dtype=object)
    )
    out["trajectory_key"] = request_id.where(request_id != "", uid)
    return out


def _merge_tool_call_counts(samples: pd.DataFrame, tool_calls: pd.DataFrame) -> pd.DataFrame:
    merged_samples = _ensure_trajectory_key(samples)
    call_rows = _ensure_trajectory_key(tool_calls)
    if len(merged_samples) == 0:
        return merged_samples

    if len(call_rows) == 0:
        merged_samples["num_tool_calls"] = float("nan")
        return merged_samples

    call_cnt = (
        call_rows.groupby(["step", "trajectory_key"], as_index=False)
        .size()
        .rename(columns={"size": "num_tool_calls"})
    )
    merged = merged_samples.merge(call_cnt, on=["step", "trajectory_key"], how="left")
    steps_with_calls = set(pd.to_numeric(call_rows["step"], errors="coerce").dropna().tolist())
    has_call_coverage = pd.to_numeric(merged["step"], errors="coerce").isin(list(steps_with_calls))
    merged.loc[has_call_coverage, "num_tool_calls"] = merged.loc[has_call_coverage, "num_tool_calls"].fillna(0.0)
    return merged


def _pick_tool_event_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 unified 表中按 step 选出工具事件行，优先级：tool_call > tool_rollout_example > tool_summary。
    """
    return _step_local_tool_rows(df)


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def _quantile(values: pd.Series, q: float) -> float:
    s = pd.to_numeric(values, errors="coerce").dropna()
    if len(s) == 0:
        return float("nan")
    return float(s.quantile(q))


def _ensure_tool_success_columns(tool_rows: pd.DataFrame) -> pd.DataFrame:
    t = tool_rows.copy()
    t["invalid_reason"] = t["invalid_reason"].fillna("").astype(str)
    if "tool_raw_success" not in t.columns:
        t["tool_raw_success"] = float("nan")
    t["tool_raw_success"] = pd.to_numeric(t["tool_raw_success"], errors="coerce")
    t["tool_raw_success"] = t["tool_raw_success"].where(
        t["tool_raw_success"].notna(),
        (t["invalid_reason"].str.strip() == "").astype(float),
    )

    if "tool_effective_success" not in t.columns:
        t["tool_effective_success"] = float("nan")
    t["tool_effective_success"] = pd.to_numeric(t["tool_effective_success"], errors="coerce")
    t["tool_effective_success"] = t["tool_effective_success"].where(
        t["tool_effective_success"].notna(),
        t["tool_raw_success"],
    )

    if "tool_obs_has_error" not in t.columns:
        t["tool_obs_has_error"] = float("nan")
    t["tool_obs_has_error"] = pd.to_numeric(t["tool_obs_has_error"], errors="coerce").fillna(0.0)
    t["invalid_rate"] = 1.0 - t["tool_raw_success"]
    return t


def _call_weight(tool_rows: pd.DataFrame) -> pd.Series:
    weights = pd.Series(1.0, index=tool_rows.index, dtype=float)
    if "row_type" not in tool_rows.columns:
        return weights
    summary_mask = tool_rows["row_type"] == "tool_summary"
    if summary_mask.any():
        weights.loc[summary_mask] = pd.to_numeric(
            tool_rows.loc[summary_mask, "tool_used_count"],
            errors="coerce",
        ).fillna(0.0)
    return weights


def _tool_share_by_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个 step 各工具类型占总调用量的比例（份额）。
    """
    tool_rows = _step_local_tool_rows(df)
    if len(tool_rows) == 0:
        return pd.DataFrame(columns=["step", "tool_kind", "calls", "total_calls", "share", "section"])
    tool_rows = tool_rows.copy()
    tool_rows["calls"] = _call_weight(tool_rows)
    grouped = tool_rows.groupby(["step", "tool_kind"], as_index=False)["calls"].sum()
    total = grouped.groupby("step", as_index=False)["calls"].sum().rename(columns={"calls": "total_calls"})
    out = grouped.merge(total, on="step", how="left")
    out["share"] = out["calls"] / out["total_calls"].replace(0, pd.NA)
    out["section"] = "tool_share_by_step"
    return out


def _datasource_preference(tool_rows: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个数据集来源对各工具类型的偏好（调用份额）。
    """
    t = tool_rows.copy()
    t["data_source"] = t["data_source"].fillna("").astype(str)
    t = t[t["data_source"].str.strip() != ""]
    if len(t) == 0:
        return pd.DataFrame(columns=["section", "data_source", "tool_kind", "calls", "share"])
    grouped = t.groupby(["data_source", "tool_kind"], as_index=False).size().rename(columns={"size": "calls"})
    total = grouped.groupby("data_source", as_index=False)["calls"].sum().rename(columns={"calls": "total_calls"})
    out = grouped.merge(total, on="data_source", how="left")
    out["share"] = out["calls"] / out["total_calls"].replace(0, pd.NA)
    out["section"] = "datasource_tool_preference"
    return out


def _success_failure_metrics(tool_rows: pd.DataFrame) -> Dict[str, Any]:
    if len(tool_rows) == 0:
        return {
            "overall_raw_success_rate": float("nan"),
            "overall_effective_success_rate": float("nan"),
            "overall_obs_error_rate": float("nan"),
            "failure_reason_distribution": {},
        }
    t = _ensure_tool_success_columns(tool_rows)
    fail = t[t["tool_raw_success"] == 0.0].copy()
    failure_dist = fail["invalid_reason"].value_counts(normalize=True).to_dict() if len(fail) else {}
    return {
        "overall_raw_success_rate": float(t["tool_raw_success"].mean()),
        "overall_failure_rate": float((1.0 - t["tool_raw_success"]).mean()),
        "overall_effective_success_rate": float(t["tool_effective_success"].mean()),
        "overall_obs_error_rate": float(t["tool_obs_has_error"].mean()),
        "failure_reason_distribution": failure_dist,
    }


def _judge_rollout_rows(df: pd.DataFrame) -> pd.DataFrame:
    judge_rows = df[df["row_type"] == "tool_rollout_example"].copy()
    if len(judge_rows) == 0:
        return judge_rows

    numeric_cols = ["judge_correct_raw", "judge_allow_numeric_tolerance", "is_correct"]
    for col in numeric_cols:
        if col not in judge_rows.columns:
            judge_rows[col] = float("nan")
        judge_rows[col] = pd.to_numeric(judge_rows[col], errors="coerce")

    text_cols = [
        "judge_final_source",
        "judge_reason",
        "judge_llm_reason",
        "judge_parse_status",
        "judge_api_status",
        "judge_numeric_extraction_answer",
        "judge_numeric_extraction_reason",
        "judge_numeric_extraction_parse_status",
        "judge_numeric_extraction_api_status",
    ]
    for col in text_cols:
        if col not in judge_rows.columns:
            judge_rows[col] = ""
        judge_rows[col] = judge_rows[col].fillna("").astype(str).str.strip()
    return judge_rows


def _status_distribution(series: pd.Series) -> Dict[str, float]:
    s = series.fillna("").astype(str).str.strip()
    s = s[s != ""]
    if len(s) == 0:
        return {}
    return s.value_counts(normalize=True).to_dict()


def _top_text_counts(series: pd.Series, top_k: int) -> Dict[str, int]:
    s = series.fillna("").astype(str).str.strip()
    s = s[s != ""]
    if len(s) == 0:
        return {}
    return s.value_counts().head(top_k).to_dict()


def _judge_summary_from_rows(judge_rows: pd.DataFrame, *, top_k_reasons: int) -> Dict[str, Any]:
    if len(judge_rows) == 0:
        return {
            "num_judge_rows": 0,
            "llm_raw_correct_rate": float("nan"),
            "final_correct_rate": float("nan"),
            "judge_available_rate": float("nan"),
            "judge_parse_failed_rate": float("nan"),
            "judge_api_failed_rate": float("nan"),
            "numeric_extraction_attempt_rate": float("nan"),
            "numeric_extraction_success_rate": float("nan"),
            "numeric_extraction_parse_failed_rate": float("nan"),
            "numeric_extraction_api_failed_rate": float("nan"),
            "visual_math_numeric_extracted_fallback_rate": float("nan"),
            "judge_unavailable_rate": float("nan"),
            "judge_final_source_distribution": {},
            "judge_parse_status_distribution": {},
            "judge_api_status_distribution": {},
            "judge_numeric_extraction_parse_status_distribution": {},
            "judge_numeric_extraction_api_status_distribution": {},
            "top_judge_reasons": {},
            "top_judge_llm_reasons": {},
            "top_numeric_extraction_reasons": {},
        }

    t = judge_rows.copy()
    judge_available = (t["judge_parse_status"] == "parsed") & (t["judge_api_status"] == "success")
    judge_api_failed = ~t["judge_api_status"].isin(["", "success", "not_requested"])
    extraction_attempted = ~(
        t["judge_numeric_extraction_parse_status"].isin(["", "not_requested"])
        & t["judge_numeric_extraction_api_status"].isin(["", "not_requested"])
    )
    extraction_success = (
        (t["judge_numeric_extraction_parse_status"] == "parsed")
        & (t["judge_numeric_extraction_api_status"] == "success")
    )
    extraction_api_failed = ~t["judge_numeric_extraction_api_status"].isin(["", "success", "not_requested"])
    final_source = t["judge_final_source"]
    extracted_fallback = final_source.isin(
        ["visual_math_numeric_extracted_fallback", "judge_unavailable_visual_math_numeric_extracted_fallback"]
    )
    judge_unavailable = final_source.str.startswith("judge_unavailable")

    return {
        "num_judge_rows": int(len(t)),
        "llm_raw_correct_rate": float(t["judge_correct_raw"].mean()),
        "final_correct_rate": float(t["is_correct"].mean()),
        "judge_available_rate": float(judge_available.mean()),
        "judge_parse_failed_rate": float((t["judge_parse_status"] == "parse_failed").mean()),
        "judge_api_failed_rate": float(judge_api_failed.mean()),
        "numeric_extraction_attempt_rate": float(extraction_attempted.mean()),
        "numeric_extraction_success_rate": float(extraction_success.mean()),
        "numeric_extraction_parse_failed_rate": float((t["judge_numeric_extraction_parse_status"] == "parse_failed").mean()),
        "numeric_extraction_api_failed_rate": float(extraction_api_failed.mean()),
        "visual_math_numeric_extracted_fallback_rate": float(extracted_fallback.mean()),
        "judge_unavailable_rate": float(judge_unavailable.mean()),
        "judge_final_source_distribution": _status_distribution(t["judge_final_source"]),
        "judge_parse_status_distribution": _status_distribution(t["judge_parse_status"]),
        "judge_api_status_distribution": _status_distribution(t["judge_api_status"]),
        "judge_numeric_extraction_parse_status_distribution": _status_distribution(
            t["judge_numeric_extraction_parse_status"]
        ),
        "judge_numeric_extraction_api_status_distribution": _status_distribution(
            t["judge_numeric_extraction_api_status"]
        ),
        "top_judge_reasons": _top_text_counts(t["judge_reason"], top_k_reasons),
        "top_judge_llm_reasons": _top_text_counts(t["judge_llm_reason"], top_k_reasons),
        "top_numeric_extraction_reasons": _top_text_counts(t["judge_numeric_extraction_reason"], top_k_reasons),
    }


def _judge_diagnostics_by_step(judge_rows: pd.DataFrame) -> pd.DataFrame:
    if len(judge_rows) == 0:
        return pd.DataFrame(
            columns=[
                "step",
                "num_judge_rows",
                "llm_raw_correct_rate",
                "final_correct_rate",
                "judge_available_rate",
                "judge_parse_failed_rate",
                "judge_api_failed_rate",
                "numeric_extraction_attempt_rate",
                "numeric_extraction_success_rate",
                "numeric_extraction_parse_failed_rate",
                "numeric_extraction_api_failed_rate",
                "visual_math_numeric_extracted_fallback_rate",
                "judge_unavailable_rate",
                "judge_final_source_distribution_json",
            ]
        )

    rows: List[Dict[str, Any]] = []
    for step, grp in judge_rows.groupby("step", dropna=False):
        summary = _judge_summary_from_rows(grp, top_k_reasons=10)
        rows.append(
            {
                "step": step,
                "num_judge_rows": summary["num_judge_rows"],
                "llm_raw_correct_rate": summary["llm_raw_correct_rate"],
                "final_correct_rate": summary["final_correct_rate"],
                "judge_available_rate": summary["judge_available_rate"],
                "judge_parse_failed_rate": summary["judge_parse_failed_rate"],
                "judge_api_failed_rate": summary["judge_api_failed_rate"],
                "numeric_extraction_attempt_rate": summary["numeric_extraction_attempt_rate"],
                "numeric_extraction_success_rate": summary["numeric_extraction_success_rate"],
                "numeric_extraction_parse_failed_rate": summary["numeric_extraction_parse_failed_rate"],
                "numeric_extraction_api_failed_rate": summary["numeric_extraction_api_failed_rate"],
                "visual_math_numeric_extracted_fallback_rate": summary["visual_math_numeric_extracted_fallback_rate"],
                "judge_unavailable_rate": summary["judge_unavailable_rate"],
                "judge_final_source_distribution_json": json.dumps(
                    summary["judge_final_source_distribution"],
                    ensure_ascii=False,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("step", kind="stable")


def _judge_diagnostics_by_data_source(judge_rows: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if len(judge_rows) == 0:
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    data = judge_rows.copy()
    data["data_source"] = data["data_source"].fillna("").astype(str).str.strip()
    data = data[data["data_source"] != ""]
    for data_source, grp in data.groupby("data_source", dropna=False):
        result[str(data_source)] = _judge_summary_from_rows(grp, top_k_reasons=5)
    return result


def _tool_metrics_by_step(tool_rows: pd.DataFrame) -> pd.DataFrame:
    t = _ensure_tool_success_columns(tool_rows)
    if len(t) == 0:
        return pd.DataFrame(
            columns=[
                "step",
                "tool_name",
                "tool_kind",
                "calls",
                "invalid_rate",
                "raw_success_rate",
                "effective_success_rate",
                "obs_error_rate",
                "mean_processing_time_ms",
                "p50_processing_time_ms",
                "p90_processing_time_ms",
                "mean_turn_index",
            ]
        )
    rows: List[Dict[str, Any]] = []
    for (step, tool_name, tool_kind), grp in t.groupby(["step", "tool_name", "tool_kind"], dropna=False):
        call_weight = _call_weight(grp)
        rows.append(
            {
                "step": step,
                "tool_name": tool_name,
                "tool_kind": tool_kind,
                "calls": float(call_weight.sum()),
                "invalid_rate": float(grp["invalid_rate"].mean()),
                "raw_success_rate": float(grp["tool_raw_success"].mean()),
                "effective_success_rate": float(grp["tool_effective_success"].mean()),
                "obs_error_rate": float(grp["tool_obs_has_error"].mean()),
                "mean_processing_time_ms": float(pd.to_numeric(grp["processing_time_ms"], errors="coerce").mean()),
                "p50_processing_time_ms": _quantile(grp["processing_time_ms"], 0.5),
                "p90_processing_time_ms": _quantile(grp["processing_time_ms"], 0.9),
                "mean_turn_index": float(pd.to_numeric(grp["tool_turn_index"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["step", "calls", "tool_name"], ascending=[True, False, True], kind="stable")


def _tool_failure_breakdown(tool_rows: pd.DataFrame) -> pd.DataFrame:
    t = _ensure_tool_success_columns(tool_rows)
    t = t[t["invalid_reason"].astype(str).str.strip() != ""].copy()
    if len(t) == 0:
        return pd.DataFrame(
            columns=[
                "step",
                "tool_name",
                "tool_kind",
                "data_source",
                "invalid_reason",
                "count",
                "share_within_step_tool",
                "share_within_step_tool_datasource",
            ]
        )
    grouped = (
        t.groupby(["step", "tool_name", "tool_kind", "data_source", "invalid_reason"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    tool_total = grouped.groupby(["step", "tool_name"], as_index=False)["count"].sum().rename(columns={"count": "tool_total"})
    tool_ds_total = (
        grouped.groupby(["step", "tool_name", "data_source"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "tool_ds_total"})
    )
    out = grouped.merge(tool_total, on=["step", "tool_name"], how="left").merge(
        tool_ds_total, on=["step", "tool_name", "data_source"], how="left"
    )
    out["share_within_step_tool"] = out["count"] / out["tool_total"].replace(0, pd.NA)
    out["share_within_step_tool_datasource"] = out["count"] / out["tool_ds_total"].replace(0, pd.NA)
    return out.sort_values(["step", "count", "tool_name", "invalid_reason"], ascending=[True, False, True, True], kind="stable")


def _correlation_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    samples = df[df["row_type"] == "sample"].copy()
    tool_calls = df[df["row_type"] == "tool_call"].copy()
    if len(samples) == 0:
        return {}
    merged = _merge_tool_call_counts(samples, tool_calls)
    merged["is_correct"] = pd.to_numeric(merged["is_correct"], errors="coerce")
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce")
    merged["total_reward"] = pd.to_numeric(merged["total_reward"], errors="coerce")
    return {
        "corr_num_tool_calls_vs_score": _safe_corr(merged["num_tool_calls"], merged["score"]),
        "corr_num_tool_calls_vs_is_correct": _safe_corr(merged["num_tool_calls"], merged["is_correct"]),
        "corr_num_tool_calls_vs_total_reward": _safe_corr(merged["num_tool_calls"], merged["total_reward"]),
        "samples_for_corr": int(len(merged)),
    }


def _trajectory_quality_by_step(df: pd.DataFrame) -> pd.DataFrame:
    samples = df[df["row_type"] == "sample"].copy()
    tool_calls = df[df["row_type"] == "tool_call"].copy()
    if len(samples) == 0:
        return pd.DataFrame(
            columns=[
                "step",
                "num_samples",
                "avg_num_turns",
                "valid_traj_rate",
                "no_loss_on_traj_rate",
                "avg_tool_calls",
                "avg_tool_penalty",
                "avg_latency_penalty",
                "avg_skill_reward",
                "avg_format_reward",
                "avg_total_reward",
                "avg_tool_processing_time_sec",
                "avg_tool_queue_time_sec",
                "avg_response_size_mb",
                "traj_stop_reason_distribution_json",
            ]
        )

    merged = _merge_tool_call_counts(samples, tool_calls)

    metric_cols = [
        "num_turns",
        "valid_traj",
        "no_loss_on_traj",
        "tool_penalty",
        "latency_penalty",
        "skill_reward",
        "format_reward",
        "total_reward",
        "tool_processing_time_sec",
        "tool_queue_time_sec",
        "response_size_mb",
    ]
    for col in metric_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
        else:
            merged[col] = float("nan")

    rows: List[Dict[str, Any]] = []
    for step, grp in merged.groupby("step", dropna=False):
        stop_reasons = (
            grp["traj_stop_reason"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        stop_reasons = stop_reasons[stop_reasons != ""]
        rows.append(
            {
                "step": step,
                "num_samples": int(len(grp)),
                "avg_num_turns": float(grp["num_turns"].mean()),
                "valid_traj_rate": float(grp["valid_traj"].mean()),
                "no_loss_on_traj_rate": float(grp["no_loss_on_traj"].mean()),
                "avg_tool_calls": float(grp["num_tool_calls"].mean()),
                "avg_tool_penalty": float(grp["tool_penalty"].mean()),
                "avg_latency_penalty": float(grp["latency_penalty"].mean()),
                "avg_skill_reward": float(grp["skill_reward"].mean()),
                "avg_format_reward": float(grp["format_reward"].mean()),
                "avg_total_reward": float(grp["total_reward"].mean()),
                "avg_tool_processing_time_sec": float(grp["tool_processing_time_sec"].mean()),
                "avg_tool_queue_time_sec": float(grp["tool_queue_time_sec"].mean()),
                "avg_response_size_mb": float(grp["response_size_mb"].mean()),
                "traj_stop_reason_distribution_json": json.dumps(stop_reasons.value_counts(normalize=True).to_dict(), ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows).sort_values("step", kind="stable")


def _trajectory_quality_summary(samples: pd.DataFrame, by_step: pd.DataFrame) -> Dict[str, Any]:
    traj_stop_reason_dist = (
        samples["traj_stop_reason"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    traj_stop_reason_dist = traj_stop_reason_dist[traj_stop_reason_dist != ""]
    return {
        "num_samples": int(len(samples)),
        "mean_num_turns": float(pd.to_numeric(samples["num_turns"], errors="coerce").mean()) if len(samples) else float("nan"),
        "valid_traj_rate": float(pd.to_numeric(samples["valid_traj"], errors="coerce").mean()) if len(samples) else float("nan"),
        "no_loss_on_traj_rate": float(pd.to_numeric(samples["no_loss_on_traj"], errors="coerce").mean()) if len(samples) else float("nan"),
        "traj_stop_reason_distribution": traj_stop_reason_dist.value_counts(normalize=True).to_dict(),
        "num_steps": int(len(by_step)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute tool dynamics and trajectory quality metrics.")
    parser.add_argument(
        "--unified-parquet",
        default="scripts/paper_tool_analysis/analysis_cache/unified.parquet",
        help="Path to unified.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="scripts/paper_tool_analysis/results",
        help="Directory for metrics outputs",
    )
    parser.add_argument("--top-k-tools", type=int, default=15, help="Top-K tools to keep in summaries")
    parser.add_argument("--rolling-window", type=int, default=5, help="Reserved for downstream smoothing/plots")
    parser.add_argument("--top-k-invalid-reasons", type=int, default=12, help="Top-K invalid reasons to keep in summary")
    args = parser.parse_args()

    df = pd.read_parquet(args.unified_parquet)
    out_dir = ensure_dir(args.output_dir)

    tool_event_rows = _pick_tool_event_rows(df)
    tool_event_rows = _ensure_tool_success_columns(tool_event_rows)
    judge_rows = _judge_rollout_rows(df)
    share_df = _tool_share_by_step(df)
    pref_df = _datasource_preference(tool_event_rows)
    success_metrics = _success_failure_metrics(tool_event_rows)
    corr_metrics = _correlation_metrics(df)
    by_step_df = _tool_metrics_by_step(tool_event_rows)
    failure_breakdown_df = _tool_failure_breakdown(tool_event_rows)
    judge_summary = _judge_summary_from_rows(judge_rows, top_k_reasons=args.top_k_invalid_reasons)
    judge_by_step_df = _judge_diagnostics_by_step(judge_rows)
    judge_by_data_source = _judge_diagnostics_by_data_source(judge_rows)

    samples = df[df["row_type"] == "sample"].copy()
    trajectory_by_step_df = _trajectory_quality_by_step(df)
    trajectory_summary = _trajectory_quality_summary(samples, trajectory_by_step_df)

    csv_rows: List[pd.DataFrame] = [share_df]
    if len(pref_df) > 0:
        csv_rows.append(pref_df)
    csv_out = pd.concat(csv_rows, ignore_index=True) if csv_rows else pd.DataFrame()
    csv_path = out_dir / "tool_dynamics.csv"
    csv_out.to_csv(csv_path, index=False)

    by_step_path = out_dir / "tool_dynamics_by_step.csv"
    by_step_df.to_csv(by_step_path, index=False)

    failure_breakdown_path = out_dir / "tool_failure_breakdown.csv"
    failure_breakdown_df.to_csv(failure_breakdown_path, index=False)

    judge_by_step_path = out_dir / "judge_diagnostics_by_step.csv"
    judge_by_step_df.to_csv(judge_by_step_path, index=False)

    trajectory_by_step_path = out_dir / "trajectory_quality_by_step.csv"
    trajectory_by_step_df.to_csv(trajectory_by_step_path, index=False)

    trajectory_summary_path = out_dir / "trajectory_quality_summary.json"
    write_json(trajectory_summary_path, trajectory_summary)

    top_tools_by_calls = (
        tool_event_rows.assign(call_weight=_call_weight(tool_event_rows))
        .groupby("tool_name")["call_weight"]
        .sum()
        .sort_values(ascending=False)
        .head(args.top_k_tools)
        .to_dict()
        if len(tool_event_rows)
        else {}
    )
    top_tool_kinds_by_calls = (
        tool_event_rows.assign(call_weight=_call_weight(tool_event_rows))
        .groupby("tool_kind")["call_weight"]
        .sum()
        .sort_values(ascending=False)
        .head(args.top_k_tools)
        .to_dict()
        if len(tool_event_rows)
        else {}
    )
    top_invalid_reasons = (
        tool_event_rows[tool_event_rows["invalid_reason"].astype(str).str.strip() != ""]["invalid_reason"]
        .value_counts()
        .head(args.top_k_invalid_reasons)
        .to_dict()
        if len(tool_event_rows)
        else {}
    )

    result = {
        "summary": {
            "num_rows_unified": int(len(df)),
            "num_tool_event_rows": int(len(tool_event_rows)),
            "steps": sorted(pd.to_numeric(df["step"], errors="coerce").dropna().astype(int).unique().tolist()),
            "rolling_window": int(args.rolling_window),
        },
        "success_failure": success_metrics,
        "correlations": corr_metrics,
        "judge_diagnostics": judge_summary,
        "judge_diagnostics_by_data_source": judge_by_data_source,
        "trajectory_quality": trajectory_summary,
        "top_tools_by_calls": top_tools_by_calls,
        "top_tool_kinds_by_calls": top_tool_kinds_by_calls,
        "top_invalid_reasons": top_invalid_reasons,
        "output_files": {
            "tool_dynamics_csv": str(csv_path),
            "tool_dynamics_by_step_csv": str(by_step_path),
            "tool_failure_breakdown_csv": str(failure_breakdown_path),
            "judge_diagnostics_by_step_csv": str(judge_by_step_path),
            "trajectory_quality_by_step_csv": str(trajectory_by_step_path),
            "trajectory_quality_summary_json": str(trajectory_summary_path),
        },
    }
    json_path = out_dir / "tool_dynamics.json"
    write_json(json_path, result)

    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "by_step_csv": str(by_step_path),
                "failure_breakdown_csv": str(failure_breakdown_path),
                "judge_by_step_csv": str(judge_by_step_path),
                **result["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
