#!/usr/bin/env python3
"""
【第四步：Skill 语义分类】
给训练过程中产生的所有 skill 打上语义类别标签，构建 skill 能力分类体系。

支持两种模式（通过 --mode 参数选择）：
  heuristic — 纯规则匹配，完全离线，不需要 API，适合快速运行
  llm       — 调用 LLM，理解能力更强，分类更准确，需要 OpenAI 兼容接口
  auto      — 有 base_url 时走 LLM，否则降级到 heuristic

输出：
  results/skill_taxonomy.json — 每个 skill 的类别、置信度、证据引用、分类理由
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from common import call_llm_json, ensure_dir, heuristic_skill_category, truncate_text, write_json


def _non_empty(s: Any) -> bool:
    """判断字符串是否非空（去掉空白后还有内容）。"""
    return isinstance(s, str) and s.strip() != ""


def _build_skill_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建 skill 清单，包含每个 skill 的描述、首末出现 step 和总调用次数。

    数据来源：
      - skill_snapshot 行 → 描述文字、首末 step
      - tool_call 行 → 实际调用次数（两个来源做 outer join 合并）

    按调用次数从高到低排序，后续会截取 top-N 送去分类。
    """
    snap = df[df["row_type"] == "skill_snapshot"].copy()
    snap = snap[snap["skill_name"].astype(str).str.strip() != ""]
    if len(snap):
        inv = (
            snap.sort_values(["step"], kind="stable")
            .groupby("skill_name", as_index=False)
            .agg(
                description=("skill_description", "last"),  # 取最新版的描述
                first_seen_step=("step", "min"),
                last_seen_step=("step", "max"),
            )
        )
    else:
        inv = pd.DataFrame(columns=["skill_name", "description", "first_seen_step", "last_seen_step"])

    # 从 tool_call 行统计实际调用次数（skill_name 为空时用 tool_name 代替）
    calls = df[df["row_type"].isin(["tool_call", "tool_rollout_example"])].copy()
    if len(calls):
        calls["skill_name_final"] = calls["skill_name"].where(
            calls["skill_name"].astype(str).str.strip() != "",
            calls["tool_name"],
        )
        counts = (
            calls[calls["skill_name_final"].astype(str).str.strip() != ""]
            .groupby("skill_name_final", as_index=False)
            .size()
            .rename(columns={"skill_name_final": "skill_name", "size": "total_calls"})
        )
    else:
        counts = pd.DataFrame(columns=["skill_name", "total_calls"])

    # outer join：有快照但没有调用记录的 skill 调用次数填 0，反之亦然
    out = inv.merge(counts, on="skill_name", how="outer")
    out["description"] = out["description"].fillna("").astype(str)
    out["total_calls"] = out["total_calls"].fillna(0).astype(int)
    out["first_seen_step"] = pd.to_numeric(out["first_seen_step"], errors="coerce")
    out["last_seen_step"] = pd.to_numeric(out["last_seen_step"], errors="coerce")
    out = out.sort_values(["total_calls", "skill_name"], ascending=[False, True], kind="stable")
    return out


def _examples_for_skill(df: pd.DataFrame, skill_name: str, k: int = 3) -> List[Dict[str, str]]:
    """
    为某个 skill 收集最多 k 条实际使用样例，用于辅助 LLM 理解该 skill 的实际用途。
    同时匹配 skill_name 列和 tool_name 列，兼容两种记录方式。
    """
    calls = df[df["row_type"].isin(["tool_call", "tool_rollout_example"])].copy()
    if len(calls) == 0:
        return []
    mask = (calls["skill_name"].astype(str) == skill_name) | (calls["tool_name"].astype(str) == skill_name)
    sub = calls[mask].head(k)
    examples: List[Dict[str, str]] = []
    for _, r in sub.iterrows():
        examples.append(
            {
                "input": truncate_text(r.get("input_text", ""), 220),
                "output": truncate_text(r.get("output_text", ""), 220),
                "tool_prompt": truncate_text(r.get("tool_prompt", ""), 180),
                "tool_obs": truncate_text(r.get("tool_obs", ""), 180),
                "log_ref": str(r.get("log_ref", "")),
            }
        )
    return examples


def _llm_label(
    *,
    base_url: str,
    model: str,
    api_key: str,
    skill_name: str,
    description: str,
    examples: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    调用 LLM 对单个 skill 进行语义分类。
    要求模型返回严格 JSON，包含：
      category          — 短横线命名的能力类别
      subcategory       — 可选的更细粒度标签
      confidence        — 分类置信度 [0, 1]
      evidence_sentences — 最多 2 条支持分类的证据句子
      rationale         — 一句话分类理由
    """
    system_prompt = (
        "You are analyzing tool-learning skills for a research paper. "
        "Classify the skill into a concise capability taxonomy. "
        "Return strict JSON only."
    )
    user_prompt = (
        "Given a skill, output JSON with keys:\n"
        "- category: short snake_case category\n"
        "- subcategory: optional finer label\n"
        "- confidence: float in [0,1]\n"
        "- evidence_sentences: list of <=2 short evidence sentences\n"
        "- rationale: one short sentence\n\n"
        f"skill_name: {skill_name}\n"
        f"description: {description}\n"
        f"examples: {json.dumps(examples, ensure_ascii=False)}"
    )
    data = call_llm_json(
        base_url=base_url,
        model=model,
        api_key=api_key,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout=90,
        temperature=0.0,  # 关闭随机性，保证分类结果可复现
        max_tokens=600,
    )
    if not isinstance(data, dict):
        return {}
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build skill semantic taxonomy via optional LLM.")
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
    parser.add_argument("--mode", choices=["auto", "heuristic", "llm"], default="auto")
    parser.add_argument("--llm-base-url", default="", help="OpenAI-compatible base URL")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="Model name for semantic labeling")
    parser.add_argument("--llm-api-key", default="", help="API key (or use OPENAI_API_KEY)")
    parser.add_argument("--max-skills", type=int, default=100, help="Max skills to label")
    parser.add_argument("--min-calls", type=int, default=1, help="Minimum total calls to keep skill")
    args = parser.parse_args()

    df = pd.read_parquet(args.unified_parquet)
    out_dir = ensure_dir(args.output_dir)

    inv = _build_skill_inventory(df)
    # 过滤掉调用次数不足的 skill（证据太少，分类不可靠）
    inv = inv[(inv["skill_name"].astype(str).str.strip() != "") & (inv["total_calls"] >= args.min_calls)]
    # 只取调用量最多的 top-N 个 skill，避免 LLM 调用太多
    inv = inv.head(args.max_skills).reset_index(drop=True)

    # 决定是否使用 LLM：llm 模式 或 auto 模式且提供了 base_url
    use_llm = args.mode == "llm" or (args.mode == "auto" and bool(args.llm_base_url))
    results: List[Dict[str, Any]] = []
    for _, row in inv.iterrows():
        skill_name = str(row["skill_name"])
        description = str(row.get("description", ""))
        total_calls = int(row.get("total_calls", 0))
        examples = _examples_for_skill(df, skill_name, k=3)

        if use_llm:
            llm = _llm_label(
                base_url=args.llm_base_url,
                model=args.llm_model,
                api_key=args.llm_api_key,
                skill_name=skill_name,
                description=description,
                examples=examples,
            )
        else:
            llm = {}

        if llm:
            # LLM 分类成功：使用 LLM 结果，category 为空时回退到 heuristic
            category = str(llm.get("category", "")).strip() or heuristic_skill_category(skill_name, description)
            subcategory = str(llm.get("subcategory", "")).strip()
            confidence = float(llm.get("confidence", 0.0))
            evidence_quotes = llm.get("evidence_sentences", [])
            if not isinstance(evidence_quotes, list):
                evidence_quotes = []
            rationale = str(llm.get("rationale", "")).strip()
        else:
            # heuristic 模式：用关键词规则分类，置信度固定为 0.5
            category = heuristic_skill_category(
                skill_name,
                description,
                samples=" ".join([e.get("tool_obs", "") for e in examples]),
            )
            subcategory = ""
            confidence = 0.5
            evidence_quotes = [truncate_text(e.get("tool_obs", ""), 120) for e in examples if _non_empty(e.get("tool_obs"))][:2]
            rationale = "Heuristic label from skill name/description/examples."

        results.append(
            {
                "skill_name": skill_name,
                "category": category,
                "subcategory": subcategory,
                "confidence": confidence,
                "total_calls": total_calls,
                "description": description,
                "evidence_refs": [e.get("log_ref", "") for e in examples if e.get("log_ref")],
                "evidence_quotes": evidence_quotes,
                "rationale": rationale,
                "first_seen_step": int(row["first_seen_step"]) if pd.notna(row["first_seen_step"]) else None,
                "last_seen_step": int(row["last_seen_step"]) if pd.notna(row["last_seen_step"]) else None,
            }
        )

    payload = {
        "mode": "llm" if use_llm else "heuristic",
        "num_skills": len(results),
        "skills": results,
    }
    out_path = Path(out_dir) / "skill_taxonomy.json"
    write_json(out_path, payload)
    print(json.dumps({"output": str(out_path), "mode": payload["mode"], "num_skills": payload["num_skills"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
