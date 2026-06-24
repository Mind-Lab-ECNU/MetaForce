#!/usr/bin/env python3
"""
【第七步：各数据集 create_skill 调用量统计】
统计在训练过程中，哪些数据集最容易触发模型创建新 skill（调用 create_skill 工具）。

三种统计口径（重要区别）：
  total  — rollout 样例中所有触发 create_skill 的调用（无论答对答错）
  correct — 答对了的样本（is_correct=True）中触发的 create_skill 调用
  valid  — 答对了 且 无 invalid_reason 的调用（即真正被收录进 skill 库的调用）

数据来源：直接读取 tool_rollouts_*.json，不依赖 unified.parquet。

重要限制：
  - tool_info.used_count 记录该步骤 create_skill 的全量调用次数（不含 data_source 维度）
  - rollouts[] 是样例记录，最多 5 条/步，因此 data_source 分布统计基于样例，不代表全量

输出：
  results/create_skill_total_by_datasource.csv    — 按数据集汇总的三口径统计
  results/create_skill_by_step_datasource.csv     — 按步骤 × 数据集的宽表（方便透视分析）
  results/create_skill_used_count_by_step.json    — 每步 create_skill 的全量调用总数

Usage:
    python scripts/paper_tool_analysis/create_skill_by_datasource.py \\
        --tool_log_dir /path/to/tool_log_dir \\
        --output results/

    # 只打印不保存
    python scripts/paper_tool_analysis/create_skill_by_datasource.py \\
        --tool_log_dir /path/to/tool_log_dir
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_step(filename: str) -> int:
    """从文件名提取 step 编号，如 tool_rollouts_4.json → 4"""
    match = re.search(r"tool_rollouts_(\d+)", Path(filename).name)
    return int(match.group(1)) if match else -1


def _load_json(path: str) -> dict:
    """读取 JSON 文件，返回字典。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# core collection
# ---------------------------------------------------------------------------

def collect_create_skill_stats(tool_log_dir: str):
    """
    遍历所有 tool_rollouts_*.json，收集 create_skill 相关统计数据。

    每个文件的结构：
      data["tools"][tool_key]["tool_kind"] == "create_skill"  → 找到 create_skill 工具
      data["tools"][tool_key]["used_count"]                   → 该步全量调用次数
      data["tools"][tool_key]["rollouts"][]                   → 具体样例（含 data_source / is_correct / invalid_reason）

    返回:
        records     : list[dict]       每条 rollout 样例（step / data_source / is_correct / invalid_reason / uid）
        step_totals : dict[int, int]   每步 create_skill 的 used_count（全量，不含 data_source 维度）
    """
    pattern = os.path.join(tool_log_dir, "tool_rollouts_*.json")
    files = sorted(glob.glob(pattern), key=_parse_step)

    if not files:
        print(f"[WARN] 未找到任何 tool_rollouts_*.json，目录: {tool_log_dir}")
        return [], {}

    records = []
    step_totals: dict = {}

    for fpath in files:
        step = _parse_step(fpath)
        try:
            data = _load_json(fpath)
        except Exception as e:
            print(f"[WARN] 读取失败 {fpath}: {e}")
            continue

        tools = data.get("tools", {})
        for _tool_key, tool_info in tools.items():
            # 只处理 create_skill 类型的工具，跳过其他工具
            if tool_info.get("tool_kind") != "create_skill":
                continue

            # 全量调用次数：来自 tool_info.used_count，覆盖该步骤所有 rollout（不含 data_source）
            used_count = int(tool_info.get("used_count", 0))
            step_totals[step] = step_totals.get(step, 0) + used_count

            # 逐条样例收集（最多 5 条/步）：含 data_source 维度，用于分布统计
            for rollout in tool_info.get("rollouts", []):
                records.append({
                    "step": step,
                    "data_source": str(rollout.get("data_source", "unknown")),
                    "is_correct": bool(rollout.get("is_correct", False)),
                    "invalid_reason": rollout.get("invalid_reason"),  # None 表示没有无效原因（即成功入库）
                    "uid": str(rollout.get("uid", "")),
                })

    return records, step_totals


# ---------------------------------------------------------------------------
# display helpers
# ---------------------------------------------------------------------------

def _print_ranking(counter: dict, title: str, top_n: int = 10) -> None:
    """
    在终端打印带可视化柱状条的排名榜。
    柱状条长度按最大值归一化到 30 格，直观展示数量差异。
    """
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    max_count = sorted_items[0][1] if sorted_items else 1
    for rank, (key, count) in enumerate(sorted_items[:top_n], 1):
        bar_len = max(1, round(count / max_count * 30))
        bar = "█" * bar_len
        print(f"  {rank:>3}. {str(key):<45} {count:>6}  {bar}")
    if len(sorted_items) > top_n:
        print(f"       ... (共 {len(sorted_items)} 个数据集，只显示前 {top_n})")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="统计各数据集触发 create_skill 的次数（从 tool_rollouts_*.json）"
    )
    parser.add_argument(
        "--tool_log_dir", type=str, required=True,
        help="包含 tool_rollouts_*.json 文件的目录"
    )
    _default_output = str(Path(__file__).parent / "results")
    parser.add_argument(
        "--output", type=str, default=_default_output,
        help=f"输出目录（默认: {_default_output}）；传入空字符串则只打印，不保存文件"
    )
    parser.add_argument(
        "--top_n", type=int, default=10,
        help="打印 Top-N 排名（默认 10）"
    )
    args = parser.parse_args()

    print(f"[INFO] 扫描目录: {args.tool_log_dir}")
    records, step_totals = collect_create_skill_stats(args.tool_log_dir)

    if not step_totals and not records:
        print("[ERROR] 没有找到任何 create_skill 数据，请检查路径。")
        return

    # ── Section 1: 全量 used_count（按步骤）──────────────────────────────────
    # 这里的数字来自 tool_info.used_count，是最准确的全量统计，但不含 data_source 维度
    print("\n" + "="*65)
    print("  create_skill 全量调用次数（来自 tool_info.used_count，不含 data_source 维度）")
    print("="*65)
    print(f"  {'step':<10} {'used_count':>15}")
    print(f"  {'-'*28}")
    for step in sorted(step_totals.keys()):
        print(f"  step {step:<6} {step_totals[step]:>15}")
    total_used = sum(step_totals.values())
    print(f"  {'合 计':<10} {total_used:>15}")

    if not records:
        print("\n[WARN] rollout 样例为空，无法统计 data_source 分布。")
        return

    # ── Section 2: 基于样例的 data_source 分布统计 ────────────────────────────
    # 注意：样例最多 5 条/步，以下统计不代表全量分布，仅供参考
    print(f"\n[INFO] 共收集到 {len(records)} 条 rollout 样例（每步最多 5 条）")
    print("[NOTE] 以下统计基于样例，不代表全量分布\n")

    total_by_ds:   dict = defaultdict(int)  # 口径1：所有调用
    correct_by_ds: dict = defaultdict(int)  # 口径2：is_correct=True 的调用
    valid_by_ds:   dict = defaultdict(int)  # 口径3：is_correct=True AND invalid_reason=None（真正入库）

    step_ds_matrix: dict = defaultdict(lambda: defaultdict(int))  # 按步骤 × 数据集统计

    for r in records:
        ds   = r["data_source"]
        step = r["step"]
        total_by_ds[ds] += 1
        step_ds_matrix[step][ds] += 1
        if r["is_correct"]:
            correct_by_ds[ds] += 1
        # valid = 答对 且 没有无效原因（即 invalid_reason 为 None，说明 skill 被成功收录）
        if r["is_correct"] and r["invalid_reason"] is None:
            valid_by_ds[ds] += 1

    _print_ranking(
        dict(total_by_ds),
        f"【全部】create_skill 调用次数排名（Top {args.top_n}）",
        args.top_n,
    )
    _print_ranking(
        dict(correct_by_ds),
        f"【答对样本】is_correct=True 的 create_skill 排名（Top {args.top_n}）",
        args.top_n,
    )
    _print_ranking(
        dict(valid_by_ds),
        f"【真正入库】is_correct=True & invalid_reason=null 排名（Top {args.top_n}）",
        args.top_n,
    )

    # ── Section 3: 按步骤 × 数据集明细 ──────────────────────────────────────
    # 显示每个 step 中各数据集的 create_skill 样例计数，便于定位具体训练阶段
    print(f"\n{'='*65}")
    print("  按步骤统计（样例，total / correct / valid）")
    print(f"{'='*65}")
    all_steps = sorted(step_ds_matrix.keys())
    all_ds = sorted({r["data_source"] for r in records})
    for step in all_steps:
        step_records = [r for r in records if r["step"] == step]
        step_total   = sum(1 for r in step_records)
        step_correct = sum(1 for r in step_records if r["is_correct"])
        step_valid   = sum(1 for r in step_records if r["is_correct"] and r["invalid_reason"] is None)
        print(f"\n  step {step}  (used_count={step_totals.get(step, '?')}, "
              f"样例: total={step_total} correct={step_correct} valid={step_valid})")
        for ds in all_ds:
            cnt = step_ds_matrix[step].get(ds, 0)
            if cnt > 0:
                print(f"          {ds:<45} {cnt}")

    # ── Section 4: 保存文件 ───────────────────────────────────────────────────
    if args.output and args.output.strip():
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        if HAS_PANDAS:
            # 全局排名表：三个口径合并成一张宽表，便于 Excel 筛选查看
            total_df = (
                pd.DataFrame(
                    list(total_by_ds.items()),
                    columns=["data_source", "total_count"]
                )
                .sort_values("total_count", ascending=False)
                .reset_index(drop=True)
            )
            total_df["correct_count"] = total_df["data_source"].map(correct_by_ds).fillna(0).astype(int)
            total_df["valid_count"]   = total_df["data_source"].map(valid_by_ds).fillna(0).astype(int)
            total_path = out_dir / "create_skill_total_by_datasource.csv"
            total_df.to_csv(total_path, index=False)
            print(f"\n[INFO] 已保存全局排名表: {total_path}")

            # 宽表：行=步骤，列=数据集，值=样例计数，便于透视分析
            wide_rows = []
            for step in all_steps:
                row: dict = {"step": step, "used_count_full": step_totals.get(step, 0)}
                for ds in all_ds:
                    row[ds] = step_ds_matrix[step].get(ds, 0)
                wide_rows.append(row)
            wide_df = pd.DataFrame(wide_rows)
            wide_path = out_dir / "create_skill_by_step_datasource.csv"
            wide_df.to_csv(wide_path, index=False)
            print(f"[INFO] 已保存按步骤宽表: {wide_path}")
        else:
            # 无 pandas 时手写 CSV（纯 Python 实现，无依赖）
            total_path = out_dir / "create_skill_total_by_datasource.csv"
            with open(total_path, "w", encoding="utf-8") as f:
                f.write("data_source,total_count,correct_count,valid_count\n")
                for ds, cnt in sorted(total_by_ds.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{ds},{cnt},{correct_by_ds.get(ds,0)},{valid_by_ds.get(ds,0)}\n")
            print(f"\n[INFO] 已保存全局排名表（无 pandas）: {total_path}")

        # 全量 used_count JSON：这是最精确的数字，记录每步实际被调用的总次数
        used_count_path = out_dir / "create_skill_used_count_by_step.json"
        with open(used_count_path, "w", encoding="utf-8") as f:
            json.dump(
                {"step_totals": {str(k): v for k, v in step_totals.items()}, "total_all_steps": total_used},
                f, indent=2, ensure_ascii=False,
            )
        print(f"[INFO] 已保存全量 used_count JSON: {used_count_path}")

    print("\n[完成]")


if __name__ == "__main__":
    main()
