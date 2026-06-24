#!/usr/bin/env python3
"""
【第三步：Skill 生命周期 / 生态指标】
分析每个 skill 从“创建/选中/复用/消亡”的完整过程，并输出逐步生态统计。

输出：
  results/skill_lifecycle.csv
  results/skill_creation_rate.csv
  results/skill_lifecycle_summary.json
  results/skill_ecology_by_step.csv
  results/skill_archive_events.csv
  results/skill_monitoring_by_step.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from common import (
    ensure_dir,
    normalize_archive_reason_group,
    parse_maybe_json_object,
    write_json,
)


def _skill_call_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    取出所有和 skill 复用相关的调用。
    direct skill call: tool_name 既不是 create_skill / run_skill / forever_tool，又被视作 skill 名
    run_skill: 使用 skill_name 作为目标 skill
    """
    calls = df[df["row_type"] == "tool_call"].copy()
    if len(calls) == 0:
        return calls

    calls["tool_name"] = calls["tool_name"].fillna("").astype(str)
    calls["tool_kind"] = calls["tool_kind"].fillna("").astype(str)
    calls["skill_name"] = calls["skill_name"].fillna("").astype(str)
    calls["tool_effective_success"] = pd.to_numeric(calls["tool_effective_success"], errors="coerce")
    calls["tool_obs_has_error"] = pd.to_numeric(calls["tool_obs_has_error"], errors="coerce")

    is_run = calls["tool_name"] == "run_skill"
    is_direct = (
        (calls["tool_name"].str.strip() != "")
        & (calls["tool_name"] != "create_skill")
        & (calls["tool_name"] != "run_skill")
        & (calls["tool_kind"] != "forever_tool")
    )
    calls = calls[is_run | is_direct].copy()
    calls["call_type"] = "direct"
    calls.loc[is_run.loc[calls.index], "call_type"] = "run_skill"
    calls["skill_name_final"] = calls["skill_name"].where(
        calls["skill_name"].str.strip() != "",
        calls["tool_name"],
    )
    calls = calls[calls["skill_name_final"].str.strip() != ""]
    return calls


def _build_skill_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    汇总 skill 的基础信息。
    优先使用 skill_snapshot 的 first/last/description/used_times；
    再用 archive / create / tool_call 补齐缺失 skill。
    """
    rows: List[Dict[str, Any]] = []

    snap = df[df["row_type"] == "skill_snapshot"].copy()
    if len(snap) > 0:
        snap["skill_name"] = snap["skill_name"].fillna("").astype(str)
        snap = snap[snap["skill_name"].str.strip() != ""]
        rows.extend(
            snap.groupby("skill_name", as_index=False)
            .agg(
                first_selected_step=("step", "min"),
                last_selected_step=("step", "max"),
                selected_steps=("step", "nunique"),
                latest_used_times=("skill_used_times", "max"),
                description=("skill_description", "last"),
            )
            .to_dict(orient="records")
        )

    base = pd.DataFrame(rows)
    if len(base) == 0:
        base = pd.DataFrame(
            columns=[
                "skill_name",
                "first_selected_step",
                "last_selected_step",
                "selected_steps",
                "latest_used_times",
                "description",
            ]
        )

    supplemental_names = set(base["skill_name"].tolist())
    for row_type in ("skill_new_event", "skill_archive_event"):
        sub = df[df["row_type"] == row_type]
        for name in sub["skill_name"].fillna("").astype(str).tolist():
            if name.strip():
                supplemental_names.add(name.strip())

    calls = _skill_call_rows(df)
    for name in calls["skill_name_final"].fillna("").astype(str).tolist():
        if name.strip():
            supplemental_names.add(name.strip())

    missing = sorted(supplemental_names - set(base["skill_name"].tolist()))
    if missing:
        extra = pd.DataFrame(
            {
                "skill_name": missing,
                "first_selected_step": float("nan"),
                "last_selected_step": float("nan"),
                "selected_steps": 0,
                "latest_used_times": float("nan"),
                "description": "",
            }
        )
        base = pd.concat([base, extra], ignore_index=True)
    return base.sort_values("skill_name", kind="stable").reset_index(drop=True)


def _skill_monitoring_by_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 skills_*.json 回填到 unified 的 skill_monitoring_metrics_json 中提取逐步监控指标。
    """
    sub = df[df["skill_monitoring_metrics_json"].fillna("").astype(str).str.strip() != ""].copy()
    if len(sub) == 0:
        return pd.DataFrame(columns=["step"])

    rows: List[Dict[str, Any]] = []
    for step, grp in sub.groupby("step", dropna=False):
        payload = {}
        for value in grp["skill_monitoring_metrics_json"].tolist():
            payload = parse_maybe_json_object(value)
            if payload:
                break
        row = {"step": step}
        row["selected_skill_count"] = float(pd.to_numeric(grp["selected_skill_count"], errors="coerce").max())
        for key, value in payload.items():
            row[str(key)] = value
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("step", kind="stable")
    for col in out.columns:
        if col != "step":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _archive_events(df: pd.DataFrame) -> pd.DataFrame:
    events = df[df["row_type"] == "skill_archive_event"].copy()
    if len(events) == 0:
        return pd.DataFrame(
            columns=[
                "skill_name",
                "skill_archive_event",
                "skill_archive_reason",
                "skill_archive_timestamp",
                "skill_archive_global_step",
                "skill_archive_snapshot_copied",
                "skill_archive_source_path",
                "skill_archive_runtime_path",
                "skill_archive_path",
                "archive_metadata_json",
                "death_group",
            ]
        )
    events["skill_name"] = events["skill_name"].fillna("").astype(str)
    events["skill_archive_event"] = events["skill_archive_event"].fillna("").astype(str)
    events["skill_archive_reason"] = events["skill_archive_reason"].fillna("").astype(str)
    events["death_group"] = events["skill_archive_reason"].map(normalize_archive_reason_group)
    events["skill_archive_global_step"] = pd.to_numeric(events["skill_archive_global_step"], errors="coerce")
    return events.sort_values(["skill_archive_global_step", "skill_archive_timestamp"], kind="stable")


def _window_uplift(samples: pd.DataFrame, first_seen: int, window: int) -> Dict[str, float]:
    pre = samples[(samples["step"] >= first_seen - window) & (samples["step"] <= first_seen - 1)]
    post = samples[(samples["step"] >= first_seen + 1) & (samples["step"] <= first_seen + window)]
    out: Dict[str, float] = {}
    for col in ("total_reward", "is_correct", "score"):
        pre_mean = float(pre[col].mean()) if len(pre) else float("nan")
        post_mean = float(post[col].mean()) if len(post) else float("nan")
        out[f"{col}_pre"] = pre_mean
        out[f"{col}_post"] = post_mean
        out[f"{col}_uplift"] = post_mean - pre_mean if pd.notna(pre_mean) and pd.notna(post_mean) else float("nan")
    return out


def _lag(anchor: float, target: float) -> float:
    if pd.isna(anchor) or pd.isna(target):
        return float("nan")
    return float(target - anchor)


def _last_non_nan(*values: Any) -> float:
    last = float("nan")
    for value in values:
        if pd.notna(value):
            last = float(value)
    return last


def _first_non_nan(*values: Any) -> float:
    for value in values:
        if pd.notna(value):
            return float(value)
    return float("nan")


def _skill_ecology_by_step(
    *,
    steps: List[int],
    lifecycle_df: pd.DataFrame,
    monitoring_df: pd.DataFrame,
    archive_df: pd.DataFrame,
) -> pd.DataFrame:
    if not steps:
        return pd.DataFrame(columns=["step"])

    selected_map = monitoring_df.set_index("step")["selected_skill_count"].to_dict() if "selected_skill_count" in monitoring_df else {}
    monitoring_cols = [c for c in monitoring_df.columns if c != "step"]
    monitoring_rows = monitoring_df.set_index("step").to_dict(orient="index") if len(monitoring_df) else {}

    new_counts = lifecycle_df["first_pending_create_step"].dropna().astype(int).value_counts().to_dict()
    promoted_counts = lifecycle_df["first_promoted_step"].dropna().astype(int).value_counts().to_dict()

    deleted_counts_by_step: Dict[int, int] = {}
    deleted_by_reason: Dict[str, Dict[int, int]] = {}
    deleted = archive_df[archive_df["skill_archive_event"] == "deleted"].copy()
    for _, row in deleted.iterrows():
        step = row.get("skill_archive_global_step")
        if pd.isna(step):
            continue
        step_i = int(step)
        deleted_counts_by_step[step_i] = deleted_counts_by_step.get(step_i, 0) + 1
        reason_group = str(row.get("death_group", "unknown"))
        deleted_by_reason.setdefault(reason_group, {})
        deleted_by_reason[reason_group][step_i] = deleted_by_reason[reason_group].get(step_i, 0) + 1

    lifecycle_alive = lifecycle_df.copy()
    if len(lifecycle_alive):
        lifecycle_alive["active_start"] = lifecycle_alive["first_promoted_step"].where(
            lifecycle_alive["first_promoted_step"].notna(),
            lifecycle_alive["first_selected_step"],
        )
        lifecycle_alive["active_start"] = lifecycle_alive["active_start"].where(
            lifecycle_alive["active_start"].notna(),
            lifecycle_alive["first_pending_create_step"],
        )
        lifecycle_alive["active_end"] = lifecycle_alive["death_step"].where(
            lifecycle_alive["death_step"].notna(),
            lifecycle_alive["last_seen_step"],
        )
    rows: List[Dict[str, Any]] = []
    reason_groups = [
        "reset",
        "validation_cleanup",
        "incorrect_trajectory",
        "dedup",
        "duplicate_existing",
        "fit_exit",
        "snapshot_disappearance",
        "unknown",
    ]
    for step in steps:
        if len(lifecycle_alive):
            active_mask = (
                pd.to_numeric(lifecycle_alive["active_start"], errors="coerce").fillna(float("inf")) <= step
            ) & (
                pd.to_numeric(lifecycle_alive["active_end"], errors="coerce").fillna(step) >= step
            )
            active_count = int(active_mask.sum())
        else:
            active_count = 0
        row = {
            "step": step,
            "selected_skill_count": float(selected_map.get(step, float("nan"))),
            "active_skill_count": active_count if active_count > 0 else float(selected_map.get(step, float("nan"))),
            "new_skill_count": int(new_counts.get(step, 0)),
            "promoted_skill_count": int(promoted_counts.get(step, 0)),
            "deleted_skill_count": int(deleted_counts_by_step.get(step, 0)),
        }
        for group in reason_groups:
            row[f"deleted_by_reason_{group}"] = int(deleted_by_reason.get(group, {}).get(step, 0))
        for col in monitoring_cols:
            row[col] = monitoring_rows.get(step, {}).get(col, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("step", kind="stable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute skill lifecycle/ecology metrics.")
    parser.add_argument(
        "--unified-parquet",
        default="scripts/paper_tool_analysis/analysis_cache/unified.parquet",
        help="Path to unified.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="scripts/paper_tool_analysis/results",
        help="Output directory",
    )
    parser.add_argument(
        "--uplift-window",
        type=int,
        default=5,
        help="Window size (steps) for pre/post uplift around skill first seen step",
    )
    parser.add_argument(
        "--reuse-window",
        type=int,
        default=10,
        help="Window size used for reporting creation-to-reuse lag contexts",
    )
    parser.add_argument(
        "--min-calls-for-skill-stats",
        type=int,
        default=3,
        help="Minimum calls to keep a skill in headline summaries",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.unified_parquet)
    out_dir = ensure_dir(args.output_dir)
    steps = sorted(pd.to_numeric(df["step"], errors="coerce").dropna().astype(int).unique().tolist())
    max_step = max(steps) if steps else -1

    samples = df[df["row_type"] == "sample"].copy()
    for col in ("total_reward", "is_correct", "score"):
        samples[col] = pd.to_numeric(samples[col], errors="coerce")

    base = _build_skill_base(df)
    calls = _skill_call_rows(df)
    pending_create = df[df["row_type"] == "skill_new_event"].copy()
    pending_create["skill_name"] = pending_create["skill_name"].fillna("").astype(str)
    pending_create = pending_create[pending_create["skill_name"].str.strip() != ""]
    create_tool_calls = df[
        (df["row_type"] == "tool_call")
        & (df["tool_name"].astype(str) == "create_skill")
        & (df["invalid_reason"].fillna("").astype(str).str.strip() == "")
        & (df["skill_name"].fillna("").astype(str).str.strip() != "")
    ].copy()
    if len(create_tool_calls) > 0:
        pending_create = pd.concat([pending_create, create_tool_calls], ignore_index=True)
        pending_create = pending_create.sort_values(["step", "skill_name"], kind="stable").drop_duplicates(
            subset=["step", "skill_name"],
            keep="first",
        )

    archive_df = _archive_events(df)
    promoted = archive_df[
        (archive_df["skill_archive_event"] == "created")
        & (archive_df["skill_archive_reason"] == "promoted")
        & (archive_df["skill_name"].str.strip() != "")
    ].copy()
    deleted = archive_df[
        (archive_df["skill_archive_event"] == "deleted")
        & (archive_df["skill_name"].str.strip() != "")
    ].copy()

    rows: List[Dict[str, Any]] = []
    for _, skill in base.iterrows():
        name = str(skill["skill_name"]).strip()
        if not name:
            continue

        s_calls = calls[calls["skill_name_final"] == name].copy()
        s_pending = pending_create[pending_create["skill_name"].astype(str) == name].copy()
        s_promoted = promoted[promoted["skill_name"].astype(str) == name].copy()
        s_deleted = deleted[deleted["skill_name"].astype(str) == name].copy()

        first_pending_create_step = float(pd.to_numeric(s_pending["step"], errors="coerce").min()) if len(s_pending) else float("nan")
        first_promoted_step = float(pd.to_numeric(s_promoted["skill_archive_global_step"], errors="coerce").min()) if len(s_promoted) else float("nan")
        first_selected_step = float(skill.get("first_selected_step", float("nan")))

        first_used_step = float(pd.to_numeric(s_calls["step"], errors="coerce").min()) if len(s_calls) else float("nan")
        successful_calls = s_calls[pd.to_numeric(s_calls["tool_effective_success"], errors="coerce") == 1.0].copy()
        first_effective_reuse_step = float(pd.to_numeric(successful_calls["step"], errors="coerce").min()) if len(successful_calls) else float("nan")

        direct_calls = s_calls[s_calls["call_type"] == "direct"].copy()
        run_calls = s_calls[s_calls["call_type"] == "run_skill"].copy()
        total_calls = int(len(s_calls))
        total_direct_calls = int(len(direct_calls))
        total_run_skill_calls = int(len(run_calls))

        peak_calls = 0
        peak_step = -1
        mean_calls = 0.0
        burst_score = 0.0
        is_burst = 0
        last_seen_step = _last_non_nan(skill.get("last_selected_step", float("nan")), first_used_step)
        if len(s_calls):
            cps = (
                s_calls.groupby("step", as_index=False)
                .size()
                .rename(columns={"size": "calls"})
                .sort_values("step", kind="stable")
            )
            total_calls = int(cps["calls"].sum())
            peak_calls = int(cps["calls"].max())
            peak_step = int(cps.loc[cps["calls"].idxmax(), "step"])
            mean_calls = float(cps["calls"].mean())
            burst_score = float(peak_calls / max(mean_calls, 1e-6)) if peak_calls > 0 else 0.0
            after_peak = cps[cps["step"] > peak_step]["calls"].mean()
            after_peak = float(after_peak) if pd.notna(after_peak) else 0.0
            is_burst = int(peak_calls >= 3 and burst_score >= 2.0 and after_peak <= peak_calls * 0.5)
            last_seen_step = _last_non_nan(last_seen_step, float(pd.to_numeric(cps["step"], errors="coerce").max()))

        latest_delete = None
        if len(s_deleted):
            s_deleted = s_deleted.sort_values(
                ["skill_archive_global_step", "skill_archive_timestamp"], ascending=[True, True], kind="stable"
            )
            latest_delete = s_deleted.iloc[-1]
        if latest_delete is not None:
            raw_delete_step = pd.to_numeric(latest_delete.get("skill_archive_global_step"), errors="coerce")
            death_step = float(raw_delete_step) if pd.notna(raw_delete_step) else float(last_seen_step)
            death_reason = str(latest_delete.get("skill_archive_reason", ""))
            death_group = str(latest_delete.get("death_group", "unknown"))
        elif pd.notna(last_seen_step) and last_seen_step < max_step:
            death_step = float(last_seen_step)
            death_reason = "snapshot_disappearance"
            death_group = "snapshot_disappearance"
        else:
            death_step = float("nan")
            death_reason = ""
            death_group = ""
        is_dead = int(pd.notna(death_step))

        first_anchor = _first_non_nan(first_promoted_step, first_selected_step, first_pending_create_step)
        first_seen_for_uplift = _first_non_nan(first_selected_step, first_promoted_step, first_pending_create_step, first_used_step)
        uplift = _window_uplift(samples, int(first_seen_for_uplift), args.uplift_window) if pd.notna(first_seen_for_uplift) else {
            "total_reward_pre": float("nan"),
            "total_reward_post": float("nan"),
            "total_reward_uplift": float("nan"),
            "is_correct_pre": float("nan"),
            "is_correct_post": float("nan"),
            "is_correct_uplift": float("nan"),
            "score_pre": float("nan"),
            "score_post": float("nan"),
            "score_uplift": float("nan"),
        }

        rows.append(
            {
                "skill_name": name,
                "first_pending_create_step": first_pending_create_step,
                "first_promoted_step": first_promoted_step,
                "first_selected_step": first_selected_step,
                "first_used_step": first_used_step,
                "first_effective_reuse_step": first_effective_reuse_step,
                "promotion_lag_steps": _lag(first_pending_create_step, first_promoted_step),
                "first_use_lag_steps": _lag(first_anchor, first_used_step),
                "first_success_lag_steps": _lag(first_anchor, first_effective_reuse_step),
                "reuse_window": int(args.reuse_window),
                "first_seen_step": first_seen_for_uplift,
                "last_seen_step": last_seen_step,
                "lifespan_steps": _lag(first_seen_for_uplift, death_step if pd.notna(death_step) else last_seen_step) + 1 if pd.notna(first_seen_for_uplift) else float("nan"),
                "is_dead": is_dead,
                "death_step": death_step,
                "death_reason": death_reason,
                "death_group": death_group,
                "selected_steps": int(skill.get("selected_steps", 0)),
                "latest_used_times": float(skill.get("latest_used_times", float("nan"))),
                "description": str(skill.get("description", "")),
                "total_calls": total_calls,
                "total_direct_calls": total_direct_calls,
                "total_run_skill_calls": total_run_skill_calls,
                "raw_success_rate": float(pd.to_numeric(s_calls["tool_raw_success"], errors="coerce").mean()) if len(s_calls) else float("nan"),
                "effective_success_rate": float(pd.to_numeric(s_calls["tool_effective_success"], errors="coerce").mean()) if len(s_calls) else float("nan"),
                "obs_error_rate": float(pd.to_numeric(s_calls["tool_obs_has_error"], errors="coerce").mean()) if len(s_calls) else float("nan"),
                "peak_calls": peak_calls,
                "peak_step": peak_step,
                "mean_calls": mean_calls,
                "burst_score": burst_score,
                "is_burst": is_burst,
                **uplift,
            }
        )

    lifecycle_df = pd.DataFrame(rows).sort_values(
        ["first_seen_step", "skill_name"], ascending=[True, True], kind="stable"
    )
    lifecycle_path = Path(out_dir) / "skill_lifecycle.csv"
    lifecycle_df.to_csv(lifecycle_path, index=False)

    creation_anchor = lifecycle_df["first_pending_create_step"].where(
        lifecycle_df["first_pending_create_step"].notna(),
        lifecycle_df["first_selected_step"],
    )
    creation_rate = (
        creation_anchor.dropna().astype(int).value_counts().rename_axis("first_seen_step").reset_index(name="new_skills")
        .sort_values("first_seen_step", kind="stable")
    )
    creation_rate_path = Path(out_dir) / "skill_creation_rate.csv"
    creation_rate.to_csv(creation_rate_path, index=False)

    monitoring_df = _skill_monitoring_by_step(df)
    monitoring_path = Path(out_dir) / "skill_monitoring_by_step.csv"
    monitoring_df.to_csv(monitoring_path, index=False)

    archive_out = archive_df[
        [
            "skill_name",
            "skill_archive_event",
            "skill_archive_reason",
            "death_group",
            "skill_archive_timestamp",
            "skill_archive_global_step",
            "skill_archive_snapshot_copied",
            "skill_archive_source_path",
            "skill_archive_runtime_path",
            "skill_archive_path",
            "archive_metadata_json",
        ]
    ].copy()
    archive_path = Path(out_dir) / "skill_archive_events.csv"
    archive_out.to_csv(archive_path, index=False)

    ecology_df = _skill_ecology_by_step(
        steps=steps,
        lifecycle_df=lifecycle_df,
        monitoring_df=monitoring_df,
        archive_df=archive_df,
    )
    ecology_path = Path(out_dir) / "skill_ecology_by_step.csv"
    ecology_df.to_csv(ecology_path, index=False)

    headline = lifecycle_df[lifecycle_df["total_calls"] >= args.min_calls_for_skill_stats].copy()
    summary = {
        "num_skills": int(len(lifecycle_df)),
        "num_headline_skills": int(len(headline)),
        "num_burst_skills": int(lifecycle_df["is_burst"].sum()) if len(lifecycle_df) else 0,
        "mean_lifespan_steps": float(lifecycle_df["lifespan_steps"].mean()) if len(lifecycle_df) else float("nan"),
        "dead_skill_ratio": float(lifecycle_df["is_dead"].mean()) if len(lifecycle_df) else float("nan"),
        "mean_effective_success_rate": float(headline["effective_success_rate"].mean()) if len(headline) else float("nan"),
        "mean_obs_error_rate": float(headline["obs_error_rate"].mean()) if len(headline) else float("nan"),
        "top_death_groups": headline["death_group"].fillna("").astype(str).value_counts().to_dict() if len(headline) else {},
        "output_skill_lifecycle_csv": str(lifecycle_path),
        "output_skill_creation_rate_csv": str(creation_rate_path),
        "output_skill_ecology_by_step_csv": str(ecology_path),
        "output_skill_archive_events_csv": str(archive_path),
        "output_skill_monitoring_by_step_csv": str(monitoring_path),
    }
    write_json(Path(out_dir) / "skill_lifecycle_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
