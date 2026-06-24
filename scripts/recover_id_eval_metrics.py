#!/usr/bin/env python3
"""
从未完成的 ID eval 运行中恢复每个数据集的准确率指标。

数据来源：per-batch jsonl 文件 + failures.jsonl
"""

import json
import glob
import os
from collections import defaultdict

# ============================================================
# 配置
# ============================================================
VAL_DATA_DIR = (
    "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/"
    "verl_step_records/validation_data/val/"
    "batch_eval_external_model_id_gemini-2.5-flash-v8-data_20260411_111415"
)

# 数据集顺序和预期总量
DATASET_ORDER = [
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
EXPECTED_SIZES = {name: size for name, size in DATASET_ORDER}


def load_success_samples(val_data_dir: str) -> list[dict]:
    """读取所有 batch jsonl 文件（成功样本）。"""
    pattern = os.path.join(val_data_dir, "[0-9]*.jsonl")
    files = sorted(glob.glob(pattern), key=lambda f: int(os.path.basename(f).split(".")[0]))
    print(f"找到 {len(files)} 个 batch jsonl 文件")

    samples = []
    for f in files:
        with open(f, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
    print(f"成功样本总数: {len(samples)}")
    return samples


def load_failures(val_data_dir: str) -> list[dict]:
    """读取 failures.jsonl。"""
    path = os.path.join(val_data_dir, "failures.jsonl")
    failures = []
    if os.path.exists(path):
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                failures.append(json.loads(line))
    print(f"失败样本总数: {len(failures)}")
    return failures


def compute_metrics(samples: list[dict], failures: list[dict]):
    """按 data_source 分组计算指标。"""
    # 成功样本按 data_source 分组
    by_source = defaultdict(list)
    for s in samples:
        ds = s.get("data_source", "unknown")
        by_source[ds].append(s)

    # 失败样本计数
    fail_counts = defaultdict(int)
    for f in failures:
        ds = f.get("data_source", "unknown")
        fail_counts[ds] += 1

    results = {}
    for ds_name, expected_size in DATASET_ORDER:
        group = by_source.get(ds_name, [])
        n_success = len(group)
        n_fail = fail_counts.get(ds_name, 0)
        n_total_evaluated = n_success + n_fail

        # 准确率（仅基于成功评估的样本）
        if n_success > 0:
            accuracy = sum(s.get("accuracy", 0.0) for s in group) / n_success
            is_correct_count = sum(1 for s in group if s.get("accuracy", 0.0) > 0.5)
            total_reward_mean = sum(s.get("total_reward", 0.0) for s in group) / n_success
            format_reward_mean = sum(s.get("format_reward", 0.0) for s in group) / n_success
            tool_penalty_mean = sum(s.get("tool_penalty", 0.0) for s in group) / n_success
            skill_reward_mean = sum(s.get("skill_reward", 0.0) for s in group) / n_success
        else:
            accuracy = 0.0
            is_correct_count = 0
            total_reward_mean = 0.0
            format_reward_mean = 0.0
            tool_penalty_mean = 0.0
            skill_reward_mean = 0.0

        coverage = n_total_evaluated / expected_size if expected_size > 0 else 0.0

        results[ds_name] = {
            "accuracy": accuracy,
            "is_correct_count": is_correct_count,
            "n_success": n_success,
            "n_fail": n_fail,
            "n_total_evaluated": n_total_evaluated,
            "expected_size": expected_size,
            "coverage": coverage,
            "total_reward_mean": total_reward_mean,
            "format_reward_mean": format_reward_mean,
            "tool_penalty_mean": tool_penalty_mean,
            "skill_reward_mean": skill_reward_mean,
            "reliable": coverage >= 0.9,
        }

    return results


def print_table(results: dict):
    """打印格式化的表格。"""
    header = (
        f"{'数据集':<32s} {'准确率':>8s} {'正确/成功':>12s} "
        f"{'失败':>6s} {'已评估/总量':>14s} {'覆盖率':>8s} {'可靠':>4s}"
    )
    print("\n" + "=" * 100)
    print("ID 运行 per-dataset 指标恢复结果 (gemini-2.5-flash, v8 data)")
    print("=" * 100)
    print(header)
    print("-" * 100)

    total_correct = 0
    total_success = 0
    total_fail = 0
    total_expected = 0

    for ds_name, _ in DATASET_ORDER:
        r = results[ds_name]
        reliable_mark = "Y" if r["reliable"] else "N"
        print(
            f"{ds_name:<32s} {r['accuracy']:>7.2%} "
            f"{r['is_correct_count']:>5d}/{r['n_success']:<5d} "
            f"{r['n_fail']:>6d} "
            f"{r['n_total_evaluated']:>6d}/{r['expected_size']:<6d} "
            f"{r['coverage']:>7.1%} "
            f"{'  ' + reliable_mark:>4s}"
        )
        total_correct += r["is_correct_count"]
        total_success += r["n_success"]
        total_fail += r["n_fail"]
        total_expected += r["expected_size"]

    print("-" * 100)
    overall_acc = total_correct / total_success if total_success > 0 else 0
    overall_cov = (total_success + total_fail) / total_expected if total_expected > 0 else 0
    print(
        f"{'总计':<32s} {overall_acc:>7.2%} "
        f"{total_correct:>5d}/{total_success:<5d} "
        f"{total_fail:>6d} "
        f"{total_success + total_fail:>6d}/{total_expected:<6d} "
        f"{overall_cov:>7.1%}"
    )
    print("=" * 100)
    print("注: 覆盖率 < 90% 的数据集标记为不可靠 (N)")


def print_json(results: dict):
    """输出与 OOD 运行日志格式一致的 JSON。"""
    output = {}
    for ds_name, _ in DATASET_ORDER:
        r = results[ds_name]
        prefix = f"val-aux/{ds_name}"
        output[f"{prefix}/accuracy/mean@1"] = r["accuracy"]
        output[f"{prefix}/is_correct/mean@1"] = r["accuracy"]
        output[f"{prefix}/format_reward/mean@1"] = r["format_reward_mean"]
        output[f"{prefix}/tool_penalty/mean@1"] = r["tool_penalty_mean"]
        output[f"{prefix}/skill_reward/mean@1"] = r["skill_reward_mean"]
        output[f"{prefix}/total_reward/mean@1"] = r["total_reward_mean"]
        output[f"{prefix}/coverage"] = r["coverage"]
        output[f"{prefix}/reliable"] = r["reliable"]

    # 失败统计
    total_fail = sum(r["n_fail"] for r in results.values())
    total_eval = sum(r["n_total_evaluated"] for r in results.values())
    output["val-failure/total"] = total_fail
    output["val-failure/ratio"] = total_fail / total_eval if total_eval > 0 else 0

    for ds_name, _ in DATASET_ORDER:
        r = results[ds_name]
        if r["n_fail"] > 0:
            output[f"val-failure/by_data_source/{ds_name}"] = r["n_fail"]

    print("\n--- JSON 格式输出（与 OOD 运行格式一致）---")
    print(json.dumps(output, indent=2, ensure_ascii=False))


def main():
    print(f"数据目录: {VAL_DATA_DIR}")
    print()

    samples = load_success_samples(VAL_DATA_DIR)
    failures = load_failures(VAL_DATA_DIR)
    results = compute_metrics(samples, failures)

    print_table(results)
    print_json(results)


if __name__ == "__main__":
    main()
