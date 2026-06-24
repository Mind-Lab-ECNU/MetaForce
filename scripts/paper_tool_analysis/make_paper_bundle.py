#!/usr/bin/env python3
"""
【第六步：论文资产打包器】
读取分析结果，生成图片、LaTeX 表格、案例文档和总览报告。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import ensure_dir, truncate_text, write_json


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


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


def _tool_calls_by_step(unified: pd.DataFrame) -> pd.DataFrame:
    calls = _step_local_tool_rows(unified)
    if len(calls) == 0:
        return pd.DataFrame(columns=["step", "tool_kind", "calls"])
    calls = calls.copy()
    calls["calls"] = 1.0
    summary_mask = calls["row_type"] == "tool_summary"
    calls.loc[summary_mask, "calls"] = pd.to_numeric(calls.loc[summary_mask, "tool_used_count"], errors="coerce")
    return calls.groupby(["step", "tool_kind"], as_index=False)["calls"].sum()


def _save_tool_adoption_curve(unified: pd.DataFrame, out_path: Path) -> None:
    g = _tool_calls_by_step(unified)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if len(g) == 0:
        ax.text(0.5, 0.5, "No tool call data", ha="center", va="center")
        ax.set_axis_off()
    else:
        pivot = g.pivot_table(index="step", columns="tool_kind", values="calls", aggfunc="sum", fill_value=0).sort_index()
        share = pivot.div(pivot.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0.0)
        for col in share.columns:
            ax.plot(share.index, share[col], label=str(col))
        ax.set_title("Tool Adoption Curve (Share by Tool Kind)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Share")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_skill_lifespan_hist(skill_lifecycle: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(skill_lifecycle) == 0 or "lifespan_steps" not in skill_lifecycle.columns:
        ax.text(0.5, 0.5, "No skill lifecycle data", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = pd.to_numeric(skill_lifecycle["lifespan_steps"], errors="coerce").dropna()
        ax.hist(x, bins=min(20, max(5, int(len(x) ** 0.5))), color="#4C78A8", alpha=0.9)
        ax.set_title("Skill Lifespan Distribution")
        ax.set_xlabel("Lifespan (steps)")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.2)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_tool_success_reward(unified: pd.DataFrame, out_path: Path) -> None:
    calls = _step_local_tool_rows(unified)
    samples = unified[unified["row_type"] == "sample"].copy()
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    if len(calls) == 0 or len(samples) == 0:
        ax1.text(0.5, 0.5, "Insufficient data for dual-axis plot", ha="center", va="center")
        ax1.set_axis_off()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    calls["tool_effective_success"] = pd.to_numeric(calls["tool_effective_success"], errors="coerce")
    samples["score"] = pd.to_numeric(samples["score"], errors="coerce")
    success_step = calls.groupby("step", as_index=False)["tool_effective_success"].mean()
    score_step = samples.groupby("step", as_index=False)["score"].mean()
    merged = success_step.merge(score_step, on="step", how="inner").sort_values("step")
    if len(merged) == 0:
        ax1.text(0.5, 0.5, "No overlapping step data", ha="center", va="center")
        ax1.set_axis_off()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    ax1.plot(merged["step"], merged["tool_effective_success"], color="#1f77b4", label="Effective Tool Success")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Effective Tool Success", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0, 1.0)
    ax1.grid(alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(merged["step"], merged["score"], color="#ff7f0e", label="Avg Score")
    ax2.set_ylabel("Average Score", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax1.set_title("Effective Tool Success vs Average Score")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_datasource_tool_heatmap(unified: pd.DataFrame, out_path: Path) -> None:
    calls = _step_local_tool_rows(unified)
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(calls) == 0:
        ax.text(0.5, 0.5, "No tool call data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    calls["data_source"] = calls["data_source"].fillna("").astype(str)
    calls = calls[calls["data_source"].str.strip() != ""]
    if len(calls) == 0:
        ax.text(0.5, 0.5, "No data_source labels", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    pivot = (
        calls.groupby(["data_source", "tool_kind"], as_index=False)
        .size()
        .pivot(index="data_source", columns="tool_kind", values="size")
        .fillna(0.0)
    )
    if len(pivot) > 12:
        pivot = pivot.iloc[:12]
    if len(pivot.columns) > 12:
        pivot = pivot.iloc[:, :12]
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_title("Data Source vs Tool Kind (Call Count)")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([truncate_text(i, 28) for i in pivot.index], fontsize=8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([truncate_text(c, 18) for c in pivot.columns], rotation=35, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_obs_error_curve(tool_dynamics_by_step: pd.DataFrame, out_path: Path, top_k: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if len(tool_dynamics_by_step) == 0:
        ax.text(0.5, 0.5, "No stepwise tool metrics", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    metric = tool_dynamics_by_step.copy()
    metric["calls"] = pd.to_numeric(metric["calls"], errors="coerce").fillna(0)
    keep = metric.groupby("tool_name")["calls"].sum().sort_values(ascending=False).head(top_k).index.tolist()
    metric = metric[metric["tool_name"].isin(keep)]
    for tool_name, grp in metric.groupby("tool_name"):
        ax.plot(grp["step"], grp["obs_error_rate"], label=truncate_text(tool_name, 28))
    ax.set_title("Observation Error Rate by Tool")
    ax.set_xlabel("Step")
    ax.set_ylabel("Obs Error Rate")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=7)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_invalid_reason_stacked_area(failure_breakdown: pd.DataFrame, out_path: Path, top_k: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if len(failure_breakdown) == 0:
        ax.text(0.5, 0.5, "No failure breakdown data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    grouped = failure_breakdown.groupby(["step", "invalid_reason"], as_index=False)["count"].sum()
    keep = grouped.groupby("invalid_reason")["count"].sum().sort_values(ascending=False).head(top_k).index.tolist()
    grouped["invalid_reason"] = grouped["invalid_reason"].where(grouped["invalid_reason"].isin(keep), "other")
    grouped = grouped.groupby(["step", "invalid_reason"], as_index=False)["count"].sum()
    pivot = grouped.pivot(index="step", columns="invalid_reason", values="count").fillna(0.0).sort_index()
    if len(pivot) == 0:
        ax.text(0.5, 0.5, "No failure breakdown data", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.stackplot(pivot.index, pivot.T.values, labels=[truncate_text(c, 24) for c in pivot.columns], alpha=0.85)
        ax.set_title("Invalid Reasons Over Time")
        ax.set_xlabel("Step")
        ax.set_ylabel("Failure Count")
        ax.legend(loc="upper left", fontsize=7)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_skill_birth_death_timeline(skill_ecology: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if len(skill_ecology) == 0:
        ax.text(0.5, 0.5, "No skill ecology data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    eco = skill_ecology.sort_values("step", kind="stable")
    ax.plot(eco["step"], eco["new_skill_count"], label="New skills", color="#2ca02c")
    ax.plot(eco["step"], eco["promoted_skill_count"], label="Promoted skills", color="#1f77b4")
    ax.plot(eco["step"], eco["deleted_skill_count"], label="Deleted skills", color="#d62728")
    ax.set_title("Skill Birth / Promotion / Death Timeline")
    ax.set_xlabel("Step")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_trajectory_quality_curve(trajectory_quality: pd.DataFrame, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    if len(trajectory_quality) == 0:
        ax1.text(0.5, 0.5, "No trajectory quality data", ha="center", va="center")
        ax1.set_axis_off()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    tq = trajectory_quality.sort_values("step", kind="stable")
    ax1.plot(tq["step"], tq["valid_traj_rate"], label="Valid traj rate", color="#1f77b4")
    ax1.plot(tq["step"], tq["no_loss_on_traj_rate"], label="No-loss traj rate", color="#2ca02c")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0, 1.0)
    ax1.grid(alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(tq["step"], tq["avg_num_turns"], label="Avg num turns", color="#ff7f0e")
    ax2.set_ylabel("Avg Num Turns")
    ax1.set_title("Trajectory Quality Over Time")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_judge_diagnostics_curve(judge_by_step: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if len(judge_by_step) == 0:
        ax.text(0.5, 0.5, "No judge diagnostics data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    jd = judge_by_step.sort_values("step", kind="stable")
    for col, label, color in [
        ("final_correct_rate", "Final correct rate", "#1f77b4"),
        ("judge_parse_failed_rate", "Judge parse failed rate", "#d62728"),
        ("numeric_extraction_success_rate", "Numeric extraction success rate", "#2ca02c"),
    ]:
        if col in jd.columns:
            ax.plot(jd["step"], pd.to_numeric(jd[col], errors="coerce"), label=label, color=color)
    ax.set_title("LLM Judge Diagnostics Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _stage_summary_table(unified: pd.DataFrame) -> pd.DataFrame:
    samples = unified[unified["row_type"] == "sample"].copy()
    calls = unified[unified["row_type"] == "tool_call"].copy()
    if len(samples) == 0:
        return pd.DataFrame(columns=["stage", "avg_score", "avg_acc", "avg_tool_calls", "effective_success_rate"])
    steps = sorted(samples["step"].dropna().astype(int).unique().tolist())
    if not steps:
        return pd.DataFrame(columns=["stage", "avg_score", "avg_acc", "avg_tool_calls", "effective_success_rate"])
    k = max(1, len(steps) // 3)
    stage_map = {
        "early": steps[:k],
        "middle": steps[k : 2 * k] if len(steps) >= 2 * k else steps[k:],
        "late": steps[-k:],
    }
    samples["score"] = pd.to_numeric(samples["score"], errors="coerce")
    samples["is_correct"] = pd.to_numeric(samples["is_correct"], errors="coerce")
    merged = _merge_tool_call_counts(samples, calls)
    calls["tool_effective_success"] = pd.to_numeric(calls["tool_effective_success"], errors="coerce")
    rows: List[Dict[str, Any]] = []
    for stage, step_set in stage_map.items():
        m = merged[merged["step"].isin(step_set)]
        c = calls[calls["step"].isin(step_set)]
        rows.append(
            {
                "stage": stage,
                "avg_score": float(m["score"].mean()) if len(m) else float("nan"),
                "avg_acc": float(m["is_correct"].mean()) if len(m) else float("nan"),
                "avg_tool_calls": float(m["num_tool_calls"].mean()) if len(m) else float("nan"),
                "effective_success_rate": float(c["tool_effective_success"].mean()) if len(c) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _ablation_delta_table(stage_df: pd.DataFrame) -> pd.DataFrame:
    if len(stage_df) == 0:
        return pd.DataFrame(columns=["stage", "delta_score_vs_early", "delta_acc_vs_early", "delta_tool_calls_vs_early"])
    early = stage_df[stage_df["stage"] == "early"]
    if len(early) == 0:
        return pd.DataFrame(columns=["stage", "delta_score_vs_early", "delta_acc_vs_early", "delta_tool_calls_vs_early"])
    early_row = early.iloc[0]
    rows: List[Dict[str, Any]] = []
    for _, row in stage_df.iterrows():
        rows.append(
            {
                "stage": row["stage"],
                "delta_score_vs_early": row["avg_score"] - early_row["avg_score"],
                "delta_acc_vs_early": row["avg_acc"] - early_row["avg_acc"],
                "delta_tool_calls_vs_early": row["avg_tool_calls"] - early_row["avg_tool_calls"],
            }
        )
    return pd.DataFrame(rows)


def _write_latex_table(path: Path, df: pd.DataFrame, caption: str, label: str) -> None:
    if len(df) == 0:
        path.write_text("% empty table\n", encoding="utf-8")
        return
    cols = list(df.columns)
    header = " & ".join(cols) + r" \\"
    body = []
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.4f}" if pd.notna(value) else "nan")
            else:
                vals.append(str(value))
        body.append(" & ".join(vals) + r" \\")
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + "l" * len(cols) + "}",
        r"\hline",
        header,
        r"\hline",
        *body,
        r"\hline",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(tex), encoding="utf-8")


def _top_failure_reasons_table(failure_breakdown: pd.DataFrame) -> pd.DataFrame:
    if len(failure_breakdown) == 0:
        return pd.DataFrame(columns=["invalid_reason", "count"])
    return (
        failure_breakdown.groupby("invalid_reason", as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False, kind="stable")
        .head(10)
        .reset_index(drop=True)
    )


def _top_judge_reasons_table(tool_dynamics: Dict[str, Any]) -> pd.DataFrame:
    judge = tool_dynamics.get("judge_diagnostics", {}) or {}
    items = list((judge.get("top_judge_reasons", {}) or {}).items())[:10]
    if not items:
        return pd.DataFrame(columns=["judge_reason", "count"])
    return pd.DataFrame(items, columns=["judge_reason", "count"])


def _top_hero_skills_table(skill_lifecycle: pd.DataFrame) -> pd.DataFrame:
    if len(skill_lifecycle) == 0:
        return pd.DataFrame(columns=["skill_name", "total_calls", "effective_success_rate", "lifespan_steps", "score_uplift"])
    df = skill_lifecycle.copy()
    for col in ("total_calls", "effective_success_rate", "lifespan_steps", "score_uplift"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return (
        df.sort_values(
            ["effective_success_rate", "total_calls", "lifespan_steps"],
            ascending=[False, False, False],
            kind="stable",
        )
        .head(10)[["skill_name", "total_calls", "effective_success_rate", "lifespan_steps", "score_uplift"]]
        .reset_index(drop=True)
    )


def _build_qualitative_cases(unified: pd.DataFrame, surprises: Dict[str, Any], out_path: Path) -> None:
    priority_types = [
        "hero_tool",
        "hero_skill",
        "failure_spike_then_recovery",
        "first_successful_new_skill_reuse",
        "created_but_never_reused_skill",
        "skill_churn_spike",
    ]
    findings = surprises.get("findings", [])
    refs: List[str] = []
    for finding_type in priority_types:
        for item in findings:
            if isinstance(item, dict) and item.get("type") == finding_type:
                refs.extend([str(r) for r in item.get("evidence_refs", []) if isinstance(r, str)])

    rows = unified[
        unified["row_type"].isin(["tool_call", "tool_rollout_example", "skill_archive_event"])
    ].copy()
    selected = rows[rows["log_ref"].astype(str).isin(refs)].copy() if refs else pd.DataFrame()
    if len(selected) < 6:
        fallback = rows.head(6 - len(selected)) if len(rows) else pd.DataFrame()
        selected = pd.concat([selected, fallback], ignore_index=True).drop_duplicates(subset=["log_ref"], keep="first")

    lines = ["# Qualitative Cases", ""]
    if len(selected) == 0:
        lines.append("No cases available.")
    else:
        for idx, (_, row) in enumerate(selected.head(8).iterrows(), start=1):
            lines.append(f"## Case {idx}")
            lines.append(f"- Row type: {row.get('row_type', '')}")
            lines.append(f"- Step: {row.get('step', '')}")
            if str(row.get("tool_name", "")).strip():
                lines.append(f"- Tool: {row.get('tool_name', '')}")
            if str(row.get("skill_name", "")).strip():
                lines.append(f"- Skill: {row.get('skill_name', '')}")
            if str(row.get("data_source", "")).strip():
                lines.append(f"- Data source: {row.get('data_source', '')}")
            if str(row.get("invalid_reason", "")).strip():
                lines.append(f"- Invalid reason: {row.get('invalid_reason', '')}")
            if str(row.get("skill_archive_reason", "")).strip():
                lines.append(f"- Archive reason: {row.get('skill_archive_reason', '')}")
            if str(row.get("input_text", "")).strip():
                lines.append(f"- Input: {truncate_text(row.get('input_text', ''), 400)}")
            if str(row.get("output_text", "")).strip():
                lines.append(f"- Output: {truncate_text(row.get('output_text', ''), 400)}")
            if str(row.get("tool_obs", "")).strip():
                lines.append(f"- Tool obs: {truncate_text(row.get('tool_obs', ''), 400)}")
            if str(row.get("archive_metadata_json", "")).strip():
                lines.append(f"- Archive metadata: `{truncate_text(row.get('archive_metadata_json', ''), 300)}`")
            lines.append(f"- log_ref: `{row.get('log_ref', '')}`")
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _build_analysis_overview(
    *,
    tool_dynamics: Dict[str, Any],
    trajectory_summary: Dict[str, Any],
    skill_lifecycle_summary: Dict[str, Any],
    surprises: Dict[str, Any],
) -> str:
    lines = ["# Analysis Overview", ""]
    lines.append("## Core Metrics")
    success = tool_dynamics.get("success_failure", {})
    lines.append(f"- Raw tool success rate: {success.get('overall_raw_success_rate', 'nan')}")
    lines.append(f"- Effective tool success rate: {success.get('overall_effective_success_rate', 'nan')}")
    lines.append(f"- Obs error rate: {success.get('overall_obs_error_rate', 'nan')}")
    lines.append(f"- Valid traj rate: {trajectory_summary.get('valid_traj_rate', 'nan')}")
    lines.append(f"- No-loss traj rate: {trajectory_summary.get('no_loss_on_traj_rate', 'nan')}")
    lines.append(f"- Num skills: {skill_lifecycle_summary.get('num_skills', 'nan')}")
    lines.append(f"- Dead skill ratio: {skill_lifecycle_summary.get('dead_skill_ratio', 'nan')}")
    lines.append("")

    judge = tool_dynamics.get("judge_diagnostics", {}) or {}
    lines.append("## Judge Diagnostics")
    lines.append(f"- LLM raw correct rate: {judge.get('llm_raw_correct_rate', 'nan')}")
    lines.append(f"- Final correct rate: {judge.get('final_correct_rate', 'nan')}")
    lines.append(f"- Judge parse failed rate: {judge.get('judge_parse_failed_rate', 'nan')}")
    lines.append(f"- Judge API failed rate: {judge.get('judge_api_failed_rate', 'nan')}")
    lines.append(f"- Numeric extraction success rate: {judge.get('numeric_extraction_success_rate', 'nan')}")
    lines.append(f"- Numeric extracted fallback rate: {judge.get('visual_math_numeric_extracted_fallback_rate', 'nan')}")
    lines.append("")

    lines.append("## Top Failures")
    for reason, count in list(tool_dynamics.get("top_invalid_reasons", {}).items())[:5]:
        lines.append(f"- {reason}: {count}")
    if not tool_dynamics.get("top_invalid_reasons"):
        lines.append("- No invalid reason data")
    lines.append("")

    lines.append("## Top Findings")
    findings = surprises.get("findings", [])
    if findings:
        for item in findings[:5]:
            lines.append(f"- {item.get('type', 'unknown')}: {item.get('claim', '')}")
    else:
        lines.append("- No surprise findings")
    lines.append("")

    lines.append("## Top Hero Signals")
    hero_found = False
    for item in findings:
        if item.get("type") in {"hero_tool", "hero_skill"}:
            hero_found = True
            label = item.get("tool_name") or item.get("skill_name") or item.get("type")
            lines.append(f"- {item.get('type')}: {label}")
    if not hero_found:
        lines.append("- No hero tool/skill finding")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper bundle assets from analysis outputs.")
    parser.add_argument(
        "--unified-parquet",
        default="scripts/paper_tool_analysis/analysis_cache/unified.parquet",
        help="Path to unified.parquet",
    )
    parser.add_argument(
        "--tool-dynamics-json",
        default="scripts/paper_tool_analysis/results/tool_dynamics.json",
        help="Path to tool_dynamics.json",
    )
    parser.add_argument(
        "--skill-lifecycle-csv",
        default="scripts/paper_tool_analysis/results/skill_lifecycle.csv",
        help="Path to skill_lifecycle.csv",
    )
    parser.add_argument(
        "--surprises-json",
        default="scripts/paper_tool_analysis/results/surprises.json",
        help="Path to surprises.json",
    )
    parser.add_argument(
        "--output-dir",
        default="scripts/paper_tool_analysis/paper_bundle",
        help="Output directory for paper bundle",
    )
    parser.add_argument("--top-k-plot-items", type=int, default=12, help="Top-K items to keep in plots")
    args = parser.parse_args()

    unified = pd.read_parquet(args.unified_parquet)
    out_dir = ensure_dir(args.output_dir)
    results_dir = Path(args.tool_dynamics_json).expanduser().resolve().parent

    tool_dynamics = _read_json_if_exists(Path(args.tool_dynamics_json))
    skill_lifecycle = _read_csv_if_exists(Path(args.skill_lifecycle_csv))
    surprises = _read_json_if_exists(Path(args.surprises_json))
    tool_dynamics_by_step = _read_csv_if_exists(results_dir / "tool_dynamics_by_step.csv")
    judge_diagnostics_by_step = _read_csv_if_exists(results_dir / "judge_diagnostics_by_step.csv")
    failure_breakdown = _read_csv_if_exists(results_dir / "tool_failure_breakdown.csv")
    trajectory_quality = _read_csv_if_exists(results_dir / "trajectory_quality_by_step.csv")
    trajectory_summary = _read_json_if_exists(results_dir / "trajectory_quality_summary.json")
    skill_ecology = _read_csv_if_exists(results_dir / "skill_ecology_by_step.csv")
    skill_lifecycle_summary = _read_json_if_exists(results_dir / "skill_lifecycle_summary.json")

    outputs = {
        "fig_tool_adoption_curve": out_dir / "fig_tool_adoption_curve.png",
        "fig_skill_lifecycle_hist": out_dir / "fig_skill_lifecycle_hist.png",
        "fig_tool_success_reward_dual_axis": out_dir / "fig_tool_success_reward_dual_axis.png",
        "fig_datasource_tool_heatmap": out_dir / "fig_datasource_tool_heatmap.png",
        "fig_obs_error_curve": out_dir / "fig_obs_error_curve.png",
        "fig_invalid_reason_stacked_area": out_dir / "fig_invalid_reason_stacked_area.png",
        "fig_skill_birth_death_timeline": out_dir / "fig_skill_birth_death_timeline.png",
        "fig_trajectory_quality_curve": out_dir / "fig_trajectory_quality_curve.png",
        "fig_judge_diagnostics_curve": out_dir / "fig_judge_diagnostics_curve.png",
        "table_main": out_dir / "table_main.tex",
        "table_ablation": out_dir / "table_ablation.tex",
        "table_top_failure_reasons": out_dir / "table_top_failure_reasons.tex",
        "table_top_judge_reasons": out_dir / "table_top_judge_reasons.tex",
        "table_top_hero_skills": out_dir / "table_top_hero_skills.tex",
        "qualitative_cases": out_dir / "qualitative_cases.md",
        "analysis_overview": out_dir / "analysis_overview.md",
    }

    _save_tool_adoption_curve(unified, outputs["fig_tool_adoption_curve"])
    _save_skill_lifespan_hist(skill_lifecycle, outputs["fig_skill_lifecycle_hist"])
    _save_tool_success_reward(unified, outputs["fig_tool_success_reward_dual_axis"])
    _save_datasource_tool_heatmap(unified, outputs["fig_datasource_tool_heatmap"])
    _save_obs_error_curve(tool_dynamics_by_step, outputs["fig_obs_error_curve"], args.top_k_plot_items)
    _save_invalid_reason_stacked_area(failure_breakdown, outputs["fig_invalid_reason_stacked_area"], args.top_k_plot_items)
    _save_skill_birth_death_timeline(skill_ecology, outputs["fig_skill_birth_death_timeline"])
    _save_trajectory_quality_curve(trajectory_quality, outputs["fig_trajectory_quality_curve"])
    _save_judge_diagnostics_curve(judge_diagnostics_by_step, outputs["fig_judge_diagnostics_curve"])

    stage_df = _stage_summary_table(unified)
    _write_latex_table(outputs["table_main"], stage_df, "Stage-wise summary metrics.", "tab:tool-stage-summary")
    _write_latex_table(
        outputs["table_ablation"],
        _ablation_delta_table(stage_df),
        "Relative changes against the early stage.",
        "tab:tool-stage-delta",
    )
    _write_latex_table(
        outputs["table_top_failure_reasons"],
        _top_failure_reasons_table(failure_breakdown),
        "Top invalid reasons across the run.",
        "tab:top-failure-reasons",
    )
    _write_latex_table(
        outputs["table_top_judge_reasons"],
        _top_judge_reasons_table(tool_dynamics),
        "Top LLM judge reasons across rollout examples.",
        "tab:top-judge-reasons",
    )
    _write_latex_table(
        outputs["table_top_hero_skills"],
        _top_hero_skills_table(skill_lifecycle),
        "Top skills by success and longevity.",
        "tab:top-hero-skills",
    )
    _build_qualitative_cases(unified, surprises, outputs["qualitative_cases"])

    overview = _build_analysis_overview(
        tool_dynamics=tool_dynamics,
        trajectory_summary=trajectory_summary,
        skill_lifecycle_summary=skill_lifecycle_summary,
        surprises=surprises,
    )
    outputs["analysis_overview"].write_text(overview, encoding="utf-8")
    # 同步写一份到 results 目录，方便用户直接在结果目录查看。
    (results_dir / "analysis_overview.md").write_text(overview, encoding="utf-8")

    manifest = {name: str(path) for name, path in outputs.items()}
    manifest["bundle_manifest"] = str(out_dir / "bundle_manifest.json")
    write_json(out_dir / "bundle_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
