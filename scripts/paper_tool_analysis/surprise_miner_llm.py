#!/usr/bin/env python3
"""
【第五步："惊喜发现"挖掘器】
从 unified 数据和前序指标文件中自动发现值得写进论文的规律。

输出：
  results/surprises.json
  results/surprises.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from common import call_llm_json, ensure_dir, entropy, truncate_text, write_json


def _series_mean_in_steps(df: pd.DataFrame, step_col: str, value_col: str, steps: List[int]) -> float:
    s = df[df[step_col].isin(steps)][value_col]
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else float("nan")


def _split_early_late(steps: List[int], frac: float = 0.25) -> tuple[List[int], List[int]]:
    if not steps:
        return [], []
    k = max(1, int(len(steps) * frac))
    return steps[:k], steps[-k:]


def _step_local_tool_rows(unified: pd.DataFrame) -> pd.DataFrame:
    calls = unified[unified["row_type"] == "tool_call"].copy()
    examples = unified[unified["row_type"] == "tool_rollout_example"].copy()
    summaries = unified[unified["row_type"] == "tool_summary"].copy()

    picked: List[pd.DataFrame] = []
    call_steps = set(pd.to_numeric(calls["step"], errors="coerce").dropna().tolist())
    if len(calls):
        picked.append(calls)

    examples = examples[~examples["step"].isin(call_steps)].copy()
    example_steps = call_steps | set(pd.to_numeric(examples["step"], errors="coerce").dropna().tolist())
    if len(examples):
        picked.append(examples)

    summaries = summaries[~summaries["step"].isin(example_steps)].copy()
    if len(summaries):
        picked.append(summaries)

    if not picked:
        return unified.iloc[0:0].copy()
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


def _pick_tool_rows(unified: pd.DataFrame) -> pd.DataFrame:
    return _step_local_tool_rows(unified)


def _best_window_drop(series_df: pd.DataFrame, value_col: str, min_effect_size: float) -> Optional[Dict[str, Any]]:
    if len(series_df) < 4:
        return None
    steps = series_df["step"].astype(int).tolist()
    window = max(1, min(5, len(steps) // 4))
    best: Optional[Dict[str, Any]] = None
    for i in range(0, max(0, len(steps) - 2 * window + 1)):
        prev_steps = steps[i : i + window]
        next_steps = steps[i + window : i + 2 * window]
        prev_mean = _series_mean_in_steps(series_df, "step", value_col, prev_steps)
        next_mean = _series_mean_in_steps(series_df, "step", value_col, next_steps)
        if pd.isna(prev_mean) or pd.isna(next_mean):
            continue
        drop = prev_mean - next_mean
        if drop < min_effect_size:
            continue
        if best is None or drop > best["drop"]:
            best = {
                "window_size": window,
                "prev_steps": prev_steps,
                "next_steps": next_steps,
                "prev_mean": prev_mean,
                "next_mean": next_mean,
                "drop": drop,
            }
    return best


def _candidate_tool_calls_vs_perf(unified: pd.DataFrame) -> Optional[Dict[str, Any]]:
    samples = unified[unified["row_type"] == "sample"].copy()
    calls = unified[unified["row_type"] == "tool_call"].copy()
    if len(samples) == 0 or len(calls) == 0:
        return None
    steps = sorted(samples["step"].dropna().astype(int).unique().tolist())
    early, late = _split_early_late(steps, frac=0.25)
    if not early or not late:
        return None
    call_per_step = calls.groupby("step", as_index=False).size().rename(columns={"size": "tool_calls"})
    early_calls = _series_mean_in_steps(call_per_step, "step", "tool_calls", early)
    late_calls = _series_mean_in_steps(call_per_step, "step", "tool_calls", late)
    early_score = _series_mean_in_steps(samples, "step", "score", early)
    late_score = _series_mean_in_steps(samples, "step", "score", late)
    early_acc = _series_mean_in_steps(samples, "step", "is_correct", early)
    late_acc = _series_mean_in_steps(samples, "step", "is_correct", late)
    if not (pd.notna(early_calls) and pd.notna(late_calls)) or late_calls <= early_calls:
        return None
    if pd.notna(early_score) and pd.notna(late_score) and late_score < early_score:
        return None
    if pd.notna(early_acc) and pd.notna(late_acc) and late_acc < early_acc:
        return None
    refs = calls[calls["step"].isin(late)]["log_ref"].dropna().astype(str).head(5).tolist()
    return {
        "type": "tool_adoption_with_non_degrading_performance",
        "magnitude": float(late_calls - early_calls),
        "stats": {
            "early_avg_calls": early_calls,
            "late_avg_calls": late_calls,
            "early_avg_score": early_score,
            "late_avg_score": late_score,
            "early_avg_acc": early_acc,
            "late_avg_acc": late_acc,
            "early_steps": early,
            "late_steps": late,
        },
        "step_window": {"early_steps": early, "late_steps": late},
        "tool_name": "",
        "skill_name": "",
        "data_source": "",
        "evidence_refs": refs,
    }


def _candidate_failure_mode_improvement(unified: pd.DataFrame) -> Optional[Dict[str, Any]]:
    calls = _pick_tool_rows(unified)
    if len(calls) == 0:
        return None
    calls["invalid_reason"] = calls["invalid_reason"].fillna("").astype(str)
    calls["is_fail"] = (calls["invalid_reason"].str.strip() != "").astype(float)
    steps = sorted(calls["step"].dropna().astype(int).unique().tolist())
    early, late = _split_early_late(steps, frac=0.25)
    if not early or not late:
        return None
    early_fail = _series_mean_in_steps(calls, "step", "is_fail", early)
    late_fail = _series_mean_in_steps(calls, "step", "is_fail", late)
    if not (pd.notna(early_fail) and pd.notna(late_fail) and late_fail < early_fail):
        return None
    refs = calls[calls["step"].isin(early + late)]["log_ref"].dropna().astype(str).head(6).tolist()
    return {
        "type": "failure_mode_improves_over_time",
        "magnitude": float(early_fail - late_fail),
        "stats": {
            "early_fail_rate": early_fail,
            "late_fail_rate": late_fail,
            "early_steps": early,
            "late_steps": late,
        },
        "step_window": {"early_steps": early, "late_steps": late},
        "tool_name": "",
        "skill_name": "",
        "data_source": "",
        "evidence_refs": refs,
    }


def _candidate_skill_specialization(unified: pd.DataFrame, min_calls: int) -> List[Dict[str, Any]]:
    calls = unified[unified["row_type"] == "tool_call"].copy()
    if len(calls) == 0:
        return []
    calls["skill_name_final"] = calls["skill_name"].where(
        calls["skill_name"].astype(str).str.strip() != "",
        calls["tool_name"],
    )
    calls = calls[
        (calls["skill_name_final"].astype(str).str.strip() != "")
        & (calls["data_source"].astype(str).str.strip() != "")
    ]
    out: List[Dict[str, Any]] = []
    for skill_name, grp in calls.groupby("skill_name_final"):
        if len(grp) < min_calls:
            continue
        cnt = grp["data_source"].value_counts()
        dom_source = str(cnt.index[0])
        dom_ratio = float(cnt.iloc[0] / cnt.sum())
        ent = entropy([float(x) for x in cnt.values.tolist()])
        if dom_ratio < 0.6:
            continue
        refs = grp["log_ref"].dropna().astype(str).head(4).tolist()
        out.append(
            {
                "type": "skill_datasource_specialization",
                "magnitude": dom_ratio,
                "skill_name": str(skill_name),
                "tool_name": "",
                "data_source": dom_source,
                "stats": {
                    "dominant_source": dom_source,
                    "dominant_ratio": dom_ratio,
                    "source_entropy": ent,
                    "num_calls": int(len(grp)),
                },
                "step_window": {},
                "evidence_refs": refs,
            }
        )
    return sorted(out, key=lambda x: x["magnitude"], reverse=True)[:3]


def _candidate_skill_switch(unified: pd.DataFrame) -> Optional[Dict[str, Any]]:
    calls = unified[unified["row_type"] == "tool_call"].copy()
    if len(calls) == 0:
        return None
    calls["skill_name_final"] = calls["skill_name"].where(
        calls["skill_name"].astype(str).str.strip() != "",
        calls["tool_name"],
    )
    calls = calls[calls["skill_name_final"].astype(str).str.strip() != ""]
    if len(calls) == 0:
        return None
    step_skill = (
        calls.groupby(["step", "skill_name_final"], as_index=False)
        .size()
        .rename(columns={"size": "calls"})
        .sort_values(["step", "calls"], ascending=[True, False], kind="stable")
    )
    top = step_skill.drop_duplicates(subset=["step"], keep="first").sort_values("step", kind="stable")
    if len(top) < 4:
        return None
    top["prev_skill"] = top["skill_name_final"].shift(1)
    switches = top[top["skill_name_final"] != top["prev_skill"]]
    if len(switches) < 2:
        return None
    sw = switches.iloc[-1]
    new_skill = str(sw["skill_name_final"])
    switch_step = int(sw["step"])
    prev_skill = str(sw["prev_skill"]) if isinstance(sw["prev_skill"], str) else ""
    post = top[top["step"] >= switch_step].head(3)
    sustained = int((post["skill_name_final"] == new_skill).sum())
    if sustained < 2:
        return None
    refs = calls[(calls["step"] >= switch_step) & (calls["skill_name_final"] == new_skill)]["log_ref"].dropna().astype(str).head(5).tolist()
    return {
        "type": "abrupt_and_sustained_top_skill_switch",
        "magnitude": float(sustained),
        "stats": {
            "switch_step": switch_step,
            "previous_top_skill": prev_skill,
            "new_top_skill": new_skill,
            "sustained_steps_in_next3": sustained,
        },
        "step_window": {"switch_step": switch_step},
        "tool_name": "",
        "skill_name": new_skill,
        "data_source": "",
        "evidence_refs": refs,
    }


def _candidate_obs_error_drop(tool_rows: pd.DataFrame, min_effect_size: float) -> Optional[Dict[str, Any]]:
    rows = tool_rows.copy()
    if "tool_obs_has_error" not in rows.columns or len(rows) == 0:
        return None
    rows["tool_obs_has_error"] = pd.to_numeric(rows["tool_obs_has_error"], errors="coerce")
    series = rows.groupby("step", as_index=False)["tool_obs_has_error"].mean().sort_values("step", kind="stable")
    best = _best_window_drop(series, "tool_obs_has_error", min_effect_size)
    if not best:
        return None
    refs = rows[rows["step"].isin(best["prev_steps"] + best["next_steps"])]["log_ref"].dropna().astype(str).head(6).tolist()
    return {
        "type": "obs_error_largest_drop_window",
        "magnitude": float(best["drop"]),
        "stats": {
            "prev_obs_error_rate": best["prev_mean"],
            "next_obs_error_rate": best["next_mean"],
            "drop": best["drop"],
            "window_size": best["window_size"],
        },
        "step_window": {"prev_steps": best["prev_steps"], "next_steps": best["next_steps"]},
        "tool_name": "",
        "skill_name": "",
        "data_source": "",
        "evidence_refs": refs,
    }


def _candidate_invalid_reason_drop(tool_rows: pd.DataFrame, min_effect_size: float, min_calls: int) -> Optional[Dict[str, Any]]:
    rows = tool_rows.copy()
    rows["invalid_reason"] = rows["invalid_reason"].fillna("").astype(str)
    rows = rows[rows["invalid_reason"].str.strip() != ""]
    if len(rows) == 0:
        return None
    best_candidate = None
    for reason, grp in rows.groupby("invalid_reason"):
        if len(grp) < min_calls:
            continue
        steps = sorted(tool_rows["step"].dropna().astype(int).unique().tolist())
        series_rows: List[Dict[str, Any]] = []
        for step in steps:
            total_calls = int((tool_rows["step"] == step).sum())
            if total_calls == 0:
                continue
            count = int((grp["step"] == step).sum())
            series_rows.append({"step": step, "rate": count / total_calls})
        series = pd.DataFrame(series_rows)
        best = _best_window_drop(series, "rate", min_effect_size)
        if not best:
            continue
        if best_candidate is None or best["drop"] > best_candidate["magnitude"]:
            refs = grp[grp["step"].isin(best["prev_steps"] + best["next_steps"])]["log_ref"].dropna().astype(str).head(6).tolist()
            best_candidate = {
                "type": "invalid_reason_largest_drop_window",
                "magnitude": float(best["drop"]),
                "stats": {
                    "invalid_reason": reason,
                    "prev_rate": best["prev_mean"],
                    "next_rate": best["next_mean"],
                    "drop": best["drop"],
                    "window_size": best["window_size"],
                },
                "step_window": {"prev_steps": best["prev_steps"], "next_steps": best["next_steps"]},
                "tool_name": "",
                "skill_name": "",
                "data_source": "",
                "evidence_refs": refs,
            }
    return best_candidate


def _candidate_first_successful_new_skill_reuse(unified: pd.DataFrame, lifecycle_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if len(lifecycle_df) == 0:
        return None
    cand = lifecycle_df[
        lifecycle_df["first_pending_create_step"].notna()
        & lifecycle_df["first_effective_reuse_step"].notna()
    ].copy()
    if len(cand) == 0:
        return None
    cand["lag"] = cand["first_effective_reuse_step"] - cand["first_pending_create_step"]
    cand = cand[cand["lag"] >= 0].sort_values(["lag", "first_effective_reuse_step"], kind="stable")
    if len(cand) == 0:
        return None
    row = cand.iloc[0]
    skill_name = str(row["skill_name"])
    calls = unified[
        (unified["row_type"] == "tool_call")
        & (
            (unified["skill_name"].astype(str) == skill_name)
            | (unified["tool_name"].astype(str) == skill_name)
        )
        & (pd.to_numeric(unified["tool_effective_success"], errors="coerce") == 1.0)
    ]
    refs = calls[calls["step"] == row["first_effective_reuse_step"]]["log_ref"].dropna().astype(str).head(4).tolist()
    return {
        "type": "first_successful_new_skill_reuse",
        "magnitude": float(-row["lag"]),
        "stats": {
            "skill_name": skill_name,
            "create_step": float(row["first_pending_create_step"]),
            "first_effective_reuse_step": float(row["first_effective_reuse_step"]),
            "lag_steps": float(row["lag"]),
        },
        "step_window": {
            "create_step": float(row["first_pending_create_step"]),
            "reuse_step": float(row["first_effective_reuse_step"]),
        },
        "tool_name": "",
        "skill_name": skill_name,
        "data_source": "",
        "evidence_refs": refs,
    }


def _candidate_hero_tool(unified: pd.DataFrame, min_calls: int) -> Optional[Dict[str, Any]]:
    samples = _ensure_trajectory_key(unified[unified["row_type"] == "sample"].copy())
    calls = _ensure_trajectory_key(unified[unified["row_type"] == "tool_call"].copy())
    if len(samples) == 0 or len(calls) == 0:
        return None
    overall_acc = float(pd.to_numeric(samples["is_correct"], errors="coerce").mean())
    best = None
    for tool_name, grp in calls.groupby("tool_name"):
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue
        call_support = int(len(grp))
        if call_support < min_calls:
            continue
        sample_keys = grp[["step", "trajectory_key"]].drop_duplicates()
        used = samples.merge(sample_keys, on=["step", "trajectory_key"], how="inner")
        used_acc = float(pd.to_numeric(used["is_correct"], errors="coerce").mean()) if len(used) else float("nan")
        if pd.isna(used_acc):
            continue
        delta = used_acc - overall_acc
        if best is None or delta > best["magnitude"]:
            refs = grp.sort_values(["step"], kind="stable")["log_ref"].dropna().astype(str).head(5).tolist()
            best = {
                "type": "hero_tool",
                "magnitude": float(delta),
                "stats": {
                    "tool_name": tool_name,
                    "overall_correct_rate": overall_acc,
                    "correct_rate_when_used": used_acc,
                    "delta": delta,
                    "call_support": call_support,
                    "sample_support": int(len(used)),
                },
                "step_window": {},
                "tool_name": tool_name,
                "skill_name": "",
                "data_source": "",
                "evidence_refs": refs,
            }
    return best if best and best["magnitude"] > 0 else None


def _candidate_hero_skill(unified: pd.DataFrame, min_calls: int) -> Optional[Dict[str, Any]]:
    calls = unified[unified["row_type"] == "tool_call"].copy()
    if len(calls) == 0:
        return None
    calls["skill_name_final"] = calls["skill_name"].where(
        calls["skill_name"].astype(str).str.strip() != "",
        calls["tool_name"],
    )
    calls = calls[calls["skill_name_final"].astype(str).str.strip() != ""]
    best = None
    for skill_name, grp in calls.groupby("skill_name_final"):
        if len(grp) < min_calls:
            continue
        acc = float(pd.to_numeric(grp["is_correct"], errors="coerce").mean())
        eff = float(pd.to_numeric(grp["tool_effective_success"], errors="coerce").mean())
        score_uplift = float(pd.to_numeric(grp["score"], errors="coerce").mean())
        magnitude = acc * 0.7 + eff * 0.3
        if best is None or magnitude > best["magnitude"]:
            refs = grp.sort_values(["step"], kind="stable")["log_ref"].dropna().astype(str).head(5).tolist()
            best = {
                "type": "hero_skill",
                "magnitude": float(magnitude),
                "stats": {
                    "skill_name": skill_name,
                    "mean_sample_correct_rate": acc,
                    "effective_success_rate": eff,
                    "mean_score_on_calls": score_uplift,
                    "call_support": int(len(grp)),
                },
                "step_window": {},
                "tool_name": "",
                "skill_name": skill_name,
                "data_source": "",
                "evidence_refs": refs,
            }
    return best


def _candidate_failure_spike_then_recovery(tool_rows: pd.DataFrame, min_effect_size: float) -> Optional[Dict[str, Any]]:
    rows = tool_rows.copy()
    rows["invalid_reason"] = rows["invalid_reason"].fillna("").astype(str)
    rows["fail"] = (rows["invalid_reason"].str.strip() != "").astype(float)
    series = rows.groupby("step", as_index=False)["fail"].mean().sort_values("step", kind="stable")
    if len(series) < 4:
        return None
    best = None
    for idx in range(1, len(series) - 1):
        peak_rate = float(series.iloc[idx]["fail"])
        prev_rate = float(series.iloc[idx - 1]["fail"])
        if peak_rate <= prev_rate:
            continue
        tail = series.iloc[idx + 1 : idx + 4]
        if len(tail) == 0:
            continue
        recovery_row = tail.loc[tail["fail"].idxmin()]
        recovery_rate = float(recovery_row["fail"])
        drop = peak_rate - recovery_rate
        if drop < min_effect_size:
            continue
        if best is None or drop > best["magnitude"]:
            peak_step = int(series.iloc[idx]["step"])
            recovery_step = int(recovery_row["step"])
            refs = rows[rows["step"].isin([peak_step, recovery_step])]["log_ref"].dropna().astype(str).head(6).tolist()
            best = {
                "type": "failure_spike_then_recovery",
                "magnitude": float(drop),
                "stats": {
                    "peak_step": peak_step,
                    "peak_fail_rate": peak_rate,
                    "recovery_step": recovery_step,
                    "recovery_fail_rate": recovery_rate,
                    "drop": drop,
                },
                "step_window": {"peak_step": peak_step, "recovery_step": recovery_step},
                "tool_name": "",
                "skill_name": "",
                "data_source": "",
                "evidence_refs": refs,
            }
    return best


def _candidate_skill_churn_spike(unified: pd.DataFrame, ecology_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if len(ecology_df) == 0:
        return None
    eco = ecology_df.copy()
    eco["active_skill_count"] = pd.to_numeric(eco["active_skill_count"], errors="coerce").fillna(0)
    eco["new_skill_count"] = pd.to_numeric(eco["new_skill_count"], errors="coerce").fillna(0)
    eco["deleted_skill_count"] = pd.to_numeric(eco["deleted_skill_count"], errors="coerce").fillna(0)
    eco["churn_rate"] = (eco["new_skill_count"] + eco["deleted_skill_count"]) / eco["active_skill_count"].replace(0, 1)
    if len(eco) == 0:
        return None
    row = eco.sort_values(["churn_rate", "step"], ascending=[False, True], kind="stable").iloc[0]
    if float(row["churn_rate"]) <= 0:
        return None
    step = int(row["step"])
    refs = unified[unified["step"] == step]["log_ref"].dropna().astype(str).head(6).tolist()
    return {
        "type": "skill_churn_spike",
        "magnitude": float(row["churn_rate"]),
        "stats": {
            "step": step,
            "churn_rate": float(row["churn_rate"]),
            "new_skill_count": float(row["new_skill_count"]),
            "deleted_skill_count": float(row["deleted_skill_count"]),
            "active_skill_count": float(row["active_skill_count"]),
        },
        "step_window": {"step": step},
        "tool_name": "",
        "skill_name": "",
        "data_source": "",
        "evidence_refs": refs,
    }


def _candidate_created_never_reused(unified: pd.DataFrame, lifecycle_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if len(lifecycle_df) == 0:
        return None
    cand = lifecycle_df[
        lifecycle_df["first_pending_create_step"].notna()
        & pd.to_numeric(lifecycle_df["total_calls"], errors="coerce").fillna(0).eq(0)
    ].copy()
    if len(cand) == 0:
        return None
    cand = cand.sort_values(["first_pending_create_step", "skill_name"], kind="stable")
    row = cand.iloc[0]
    skill_name = str(row["skill_name"])
    refs = unified[
        (unified["skill_name"].astype(str) == skill_name)
        & (unified["row_type"].isin(["skill_new_event", "skill_archive_event"]))
    ]["log_ref"].dropna().astype(str).head(4).tolist()
    return {
        "type": "created_but_never_reused_skill",
        "magnitude": float(-row["first_pending_create_step"]),
        "stats": {
            "skill_name": skill_name,
            "create_step": float(row["first_pending_create_step"]),
            "death_reason": str(row.get("death_reason", "")),
        },
        "step_window": {"create_step": float(row["first_pending_create_step"])},
        "tool_name": "",
        "skill_name": skill_name,
        "data_source": "",
        "evidence_refs": refs,
    }


def _candidate_longest_positive_uplift_skill(unified: pd.DataFrame, lifecycle_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if len(lifecycle_df) == 0:
        return None
    cand = lifecycle_df[
        (pd.to_numeric(lifecycle_df["lifespan_steps"], errors="coerce") > 0)
        & (
            (pd.to_numeric(lifecycle_df["score_uplift"], errors="coerce") > 0)
            | (pd.to_numeric(lifecycle_df["is_correct_uplift"], errors="coerce") > 0)
        )
    ].copy()
    if len(cand) == 0:
        return None
    cand = cand.sort_values(["lifespan_steps", "score_uplift"], ascending=[False, False], kind="stable")
    row = cand.iloc[0]
    skill_name = str(row["skill_name"])
    refs = unified[
        (unified["row_type"] == "tool_call")
        & (
            (unified["skill_name"].astype(str) == skill_name)
            | (unified["tool_name"].astype(str) == skill_name)
        )
    ]["log_ref"].dropna().astype(str).head(5).tolist()
    return {
        "type": "longest_lifespan_positive_uplift_skill",
        "magnitude": float(row["lifespan_steps"]),
        "stats": {
            "skill_name": skill_name,
            "lifespan_steps": float(row["lifespan_steps"]),
            "score_uplift": float(row.get("score_uplift", float("nan"))),
            "is_correct_uplift": float(row.get("is_correct_uplift", float("nan"))),
        },
        "step_window": {
            "first_seen_step": float(row.get("first_seen_step", float("nan"))),
            "last_seen_step": float(row.get("last_seen_step", float("nan"))),
        },
        "tool_name": "",
        "skill_name": skill_name,
        "data_source": "",
        "evidence_refs": refs,
    }


def _llm_rewrite(
    *,
    base_url: str,
    model: str,
    api_key: str,
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    system_prompt = (
        "You are writing cautious, evidence-grounded research observations. "
        "Do not invent facts. Return strict JSON only."
    )
    user_prompt = (
        "Given one candidate finding, produce JSON with keys:\n"
        "- claim\n"
        "- evidence_refs (copy from input)\n"
        "- confidence (0~1)\n"
        "- counter_hypothesis\n"
        "- paper_ready_sentence\n\n"
        f"candidate={json.dumps(candidate, ensure_ascii=False)}"
    )
    data = call_llm_json(
        base_url=base_url,
        model=model,
        api_key=api_key,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        timeout=90,
        max_tokens=500,
    )
    if not isinstance(data, dict):
        return {}
    return data


def _heuristic_rewrite(candidate: Dict[str, Any]) -> Dict[str, Any]:
    ctype = candidate.get("type", "unknown")
    stats = candidate.get("stats", {})
    claim = f"Observed pattern: {ctype}."
    return {
        "claim": claim,
        "evidence_refs": candidate.get("evidence_refs", []),
        "confidence": 0.6,
        "counter_hypothesis": "Pattern may be driven by data mix or sampling noise rather than capability change.",
        "paper_ready_sentence": f"{ctype}: {json.dumps(stats, ensure_ascii=False)}",
    }


def _build_md(findings: List[Dict[str, Any]]) -> str:
    lines = ["# Surprising Findings", ""]
    if not findings:
        lines.append("No strong surprise candidates found under current thresholds.")
        return "\n".join(lines) + "\n"
    for i, finding in enumerate(findings, start=1):
        lines.append(f"## {i}. {finding.get('type', 'unknown')}")
        lines.append(f"- Claim: {finding.get('claim', '')}")
        lines.append(f"- Confidence: {finding.get('confidence', '')}")
        lines.append(f"- Counter-hypothesis: {finding.get('counter_hypothesis', '')}")
        lines.append(f"- Paper sentence: {finding.get('paper_ready_sentence', '')}")
        if finding.get("tool_name"):
            lines.append(f"- Tool: {finding.get('tool_name')}")
        if finding.get("skill_name"):
            lines.append(f"- Skill: {finding.get('skill_name')}")
        if finding.get("data_source"):
            lines.append(f"- Data source: {finding.get('data_source')}")
        step_window = finding.get("step_window", {})
        if isinstance(step_window, dict) and step_window:
            lines.append(f"- Step/window: `{json.dumps(step_window, ensure_ascii=False)}`")
        refs = finding.get("evidence_refs", [])
        if isinstance(refs, list) and refs:
            lines.append("- Evidence refs:")
            for ref in refs[:8]:
                lines.append(f"  - {truncate_text(ref, 180)}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine surprising findings from unified dataset.")
    parser.add_argument(
        "--unified-parquet",
        default="scripts/paper_tool_analysis/analysis_cache/unified.parquet",
        help="Path to unified.parquet",
    )
    parser.add_argument("--output-dir", default="scripts/paper_tool_analysis/results", help="Output directory")
    parser.add_argument("--mode", choices=["auto", "heuristic", "llm"], default="auto")
    parser.add_argument("--llm-base-url", default="", help="OpenAI-compatible base URL")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="Model name")
    parser.add_argument("--llm-api-key", default="", help="API key (or OPENAI_API_KEY)")
    parser.add_argument("--min-effect-size", type=float, default=0.05, help="Minimum effect size for drop/recovery findings")
    parser.add_argument("--min-calls", type=int, default=8, help="Minimum support for specialization/hero findings")
    parser.add_argument("--max-findings", type=int, default=20, help="Maximum findings to keep")
    args = parser.parse_args()

    unified = pd.read_parquet(args.unified_parquet)
    out_dir = ensure_dir(args.output_dir)
    tool_rows = _pick_tool_rows(unified)

    lifecycle_path = Path(out_dir) / "skill_lifecycle.csv"
    ecology_path = Path(out_dir) / "skill_ecology_by_step.csv"
    lifecycle_df = pd.read_csv(lifecycle_path) if lifecycle_path.exists() else pd.DataFrame()
    ecology_df = pd.read_csv(ecology_path) if ecology_path.exists() else pd.DataFrame()

    candidates: List[Dict[str, Any]] = []
    for item in (
        _candidate_tool_calls_vs_perf(unified),
        _candidate_failure_mode_improvement(unified),
        _candidate_skill_switch(unified),
        _candidate_obs_error_drop(tool_rows, args.min_effect_size),
        _candidate_invalid_reason_drop(tool_rows, args.min_effect_size, args.min_calls),
        _candidate_first_successful_new_skill_reuse(unified, lifecycle_df),
        _candidate_hero_tool(unified, args.min_calls),
        _candidate_hero_skill(unified, args.min_calls),
        _candidate_failure_spike_then_recovery(tool_rows, args.min_effect_size),
        _candidate_skill_churn_spike(unified, ecology_df),
        _candidate_created_never_reused(unified, lifecycle_df),
        _candidate_longest_positive_uplift_skill(unified, lifecycle_df),
    ):
        if item:
            candidates.append(item)
    candidates.extend(_candidate_skill_specialization(unified, args.min_calls))

    candidates = sorted(candidates, key=lambda x: (x.get("magnitude", 0.0), x.get("type", "")), reverse=True)
    trimmed_candidates = candidates[: args.max_findings]

    use_llm = args.mode == "llm" or (args.mode == "auto" and bool(args.llm_base_url))
    findings: List[Dict[str, Any]] = []
    for candidate in trimmed_candidates:
        if use_llm:
            rewrite = _llm_rewrite(
                base_url=args.llm_base_url,
                model=args.llm_model,
                api_key=args.llm_api_key,
                candidate=candidate,
            )
        else:
            rewrite = {}
        if not rewrite:
            rewrite = _heuristic_rewrite(candidate)
        finding = {**candidate, **rewrite}
        finding.pop("magnitude", None)
        findings.append(finding)

    payload = {"mode": "llm" if use_llm else "heuristic", "num_findings": len(findings), "findings": findings}
    json_path = Path(out_dir) / "surprises.json"
    md_path = Path(out_dir) / "surprises.md"
    write_json(json_path, payload)
    md_path.write_text(_build_md(findings), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": str(json_path),
                "output_md": str(md_path),
                "mode": payload["mode"],
                "num_findings": len(findings),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
