#!/usr/bin/env python3
"""
【总调度器：流水线入口】
读取 pipeline_config.yaml，按顺序依次调用所有分析脚本，把整个分析流程串联起来。

执行顺序：
  1. build_unified_dataset.py     → 构建统一数据集（unified.parquet）
  2. tool_dynamics_metrics.py     → 计算工具动态指标与轨迹质量指标
  3. skill_lifecycle_metrics.py   → 计算 skill 生命周期/生态指标
  4. skill_semantic_clustering_llm.py → 给 skill 打语义类别标签
  5. surprise_miner_llm.py        → 挖掘"惊喜发现"
  6. make_paper_bundle.py         → 打包生成论文图表、表格和总览报告
  7. create_skill_by_datasource.py → 统计各数据集触发 create_skill 的次数

使用方式：
  修改 pipeline_config.yaml，然后运行：
  python run_pipeline.py [--config /path/to/config.yaml] [--resume]
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    """加载 YAML 配置文件，要求顶层必须是字典格式（键值对）。"""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return data


def _require_str(cfg: Dict[str, Any], key: str) -> str:
    """从配置字典中读取必填字符串，为空时报错（防止用户忘记填写路径）。"""
    value = cfg.get(key, "")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Required config key is missing or empty: {key}")
    return value.strip()


def _resolve_output_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    根据配置文件中的 outputs 节解析所有输出目录路径，并预先创建它们。
    返回一个包含所有关键路径的字典，供后续步骤直接使用。
    """
    outputs = config.get("outputs", {}) or {}
    root_dir = Path(outputs.get("root_dir", "scripts/paper_tool_analysis")).expanduser().resolve()
    analysis_cache_dir = root_dir / str(outputs.get("analysis_cache_dir", "analysis_cache"))
    results_dir = root_dir / str(outputs.get("results_dir", "results"))
    paper_bundle_dir = root_dir / str(outputs.get("paper_bundle_dir", "paper_bundle"))

    # 提前创建所有输出目录，避免子脚本因目录不存在而失败
    analysis_cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    paper_bundle_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root_dir": root_dir,
        "analysis_cache_dir": analysis_cache_dir,
        "results_dir": results_dir,
        "paper_bundle_dir": paper_bundle_dir,
        # 各步骤的关键输出文件路径，用于 resume 时检查是否已存在
        "unified_parquet": analysis_cache_dir / "unified.parquet",
        "tool_dynamics_json": results_dir / "tool_dynamics.json",
        "skill_lifecycle_csv": results_dir / "skill_lifecycle.csv",
        "surprises_json": results_dir / "surprises.json",
    }


def _build_llm_mode(config: Dict[str, Any]) -> str:
    """
    根据 llm.enabled 配置决定传给子脚本的 --mode 参数。
    enabled=true  → mode_when_enabled（默认 "llm"）
    enabled=false → mode_when_disabled（默认 "heuristic"）
    """
    llm_cfg = config.get("llm", {}) or {}
    enabled = bool(llm_cfg.get("enabled", False))
    if enabled:
        return str(llm_cfg.get("mode_when_enabled", "llm"))
    return str(llm_cfg.get("mode_when_disabled", "heuristic"))


def _append_llm_args(cmd: List[str], config: Dict[str, Any]) -> None:
    """
    如果 llm.enabled=true，把 LLM 相关参数追加到命令行中。
    base_url 和 model 是必填项，api_key 可选（可从环境变量读取）。
    """
    llm_cfg = config.get("llm", {}) or {}
    enabled = bool(llm_cfg.get("enabled", False))
    if not enabled:
        return
    base_url = str(llm_cfg.get("base_url", "")).strip()
    model = str(llm_cfg.get("model", "")).strip()
    api_key = str(llm_cfg.get("api_key", "")).strip()

    if not base_url:
        raise ValueError("llm.enabled=true but llm.base_url is empty")
    if not model:
        raise ValueError("llm.enabled=true but llm.model is empty")

    cmd.extend(["--llm-base-url", base_url, "--llm-model", model])
    if api_key:
        cmd.extend(["--llm-api-key", api_key])


def _run_step(cmd: List[str], fail_fast: bool, step_name: str = "", output_path: Path | None = None, resume: bool = False) -> None:
    """
    执行单个流水线步骤（以 subprocess 方式运行子脚本）。

    resume 模式下的跳过逻辑：
      - 指定了 output_path 且该路径已存在有效内容（文件非空，或目录非空）→ 跳过
      - 这样在断点后继续运行时不会重新计算已有结果

    fail_fast=True 时，任何步骤失败会立刻抛出异常中断整个流水线。
    fail_fast=False 时，失败只打印警告，继续执行后续步骤。
    """
    if step_name:
        print(f"[pipeline] === Step: {step_name} ===")

    if resume and output_path and output_path.exists():
        # 对于目录，检查是否非空（空目录不算有效输出）
        is_valid = output_path.is_file() or (output_path.is_dir() and any(output_path.iterdir()))
        if is_valid:
            print(f"[pipeline] skipping {step_name} (output exists: {output_path})")
            return

    print(f"[pipeline] running: {shlex.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode == 0:
        return
    msg = f"Step failed with exit code {proc.returncode}: {shlex.join(cmd)}"
    if fail_fast:
        raise RuntimeError(msg)
    print(f"[pipeline] warning: {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper tool analysis with one config file.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("pipeline_config.yaml")),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing outputs (skip steps with existing output files)",
    )
    args = parser.parse_args()

    # __file__ 所在目录即各子脚本所在目录
    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config).expanduser().resolve()
    config = _load_yaml(config_path)

    # 从配置文件中读取各节
    inputs = config.get("inputs", {}) or {}
    pipeline = config.get("pipeline", {}) or {}
    params = config.get("params", {}) or {}
    runtime = config.get("runtime", {}) or {}
    fail_fast = bool(runtime.get("fail_fast", True))
    # python_bin 为空时使用当前环境的 Python，保证子进程使用相同的 Python 环境
    python_bin = str(runtime.get("python_bin", "")).strip() or sys.executable

    # resume 优先级：命令行 --resume 参数 > 配置文件中的 runtime.resume
    resume_from_config = bool(runtime.get("resume", False))
    is_resume = args.resume if "resume" in args and args.resume is not None else resume_from_config

    # 验证输入目录是否存在
    tool_log_dir = Path(_require_str(inputs, "tool_log_dir")).expanduser().resolve()
    rollout_data_dir = Path(_require_str(inputs, "rollout_data_dir")).expanduser().resolve()

    if not tool_log_dir.exists():
        raise FileNotFoundError(f"inputs.tool_log_dir does not exist: {tool_log_dir}")
    if not rollout_data_dir.exists():
        raise FileNotFoundError(f"inputs.rollout_data_dir does not exist: {rollout_data_dir}")

    out = _resolve_output_paths(config)
    llm_mode = _build_llm_mode(config)

    # 读取各步骤的参数配置
    build_cfg = params.get("build_unified", {}) or {}
    tool_cfg = params.get("tool_dynamics", {}) or {}
    lifecycle_cfg = params.get("skill_lifecycle", {}) or {}
    semantic_cfg = params.get("skill_semantic_clustering", {}) or {}
    surprise_cfg = params.get("surprise_miner", {}) or {}
    paper_cfg = params.get("paper_bundle", {}) or {}
    create_skill_cfg = params.get("create_skill_by_datasource", {}) or {}

    max_rollout_files = int(build_cfg.get("max_rollout_files", 0))
    include_skill_archive = str(build_cfg.get("include_skill_archive", "auto")).strip() or "auto"
    obs_error_markers = build_cfg.get("obs_error_markers", ["[stderr]"])
    if not isinstance(obs_error_markers, list):
        obs_error_markers = ["[stderr]"]

    top_k_tools = int(tool_cfg.get("top_k_tools", 15))
    rolling_window = int(tool_cfg.get("rolling_window", 5))
    top_k_invalid_reasons = int(tool_cfg.get("top_k_invalid_reasons", 12))

    uplift_window = int(lifecycle_cfg.get("uplift_window", 5))
    reuse_window = int(lifecycle_cfg.get("reuse_window", 10))
    min_calls_for_skill_stats = int(lifecycle_cfg.get("min_calls_for_skill_stats", 3))

    max_skills = int(semantic_cfg.get("max_skills", 100))
    min_calls = int(semantic_cfg.get("min_calls", 1))

    surprise_min_effect_size = float(surprise_cfg.get("min_effect_size", 0.05))
    surprise_min_calls = int(surprise_cfg.get("min_calls", 8))
    surprise_max_findings = int(surprise_cfg.get("max_findings", 20))

    paper_top_k_plot_items = int(paper_cfg.get("top_k_plot_items", 12))
    create_skill_top_n = int(create_skill_cfg.get("top_n", 10))

    # experiment 节：phase 和 tool_variant（均默认 "auto"）
    experiment_cfg = config.get("experiment", {}) or {}
    exp_phase = str(experiment_cfg.get("phase", "auto")).strip() or "auto"
    exp_tool_variant = str(experiment_cfg.get("tool_variant", "auto")).strip() or "auto"

    # ── Step 1: 构建统一数据集 ──────────────────────────────────────────────
    if pipeline.get("build_unified", True):
        cmd = [
            python_bin,
            str(script_dir / "build_unified_dataset.py"),
            "--tool-log-dir", str(tool_log_dir),
            "--rollout-data-dir", str(rollout_data_dir),
            "--output-dir", str(out["root_dir"]),
            "--max-rollout-files", str(max_rollout_files),
            "--phase", exp_phase,
            "--tool-variant", exp_tool_variant,
            "--include-skill-archive", include_skill_archive,
        ]
        skill_store_dir = str(inputs.get("skill_store_dir", "")).strip()
        if skill_store_dir:
            cmd.extend(["--skill-store-dir", skill_store_dir])
        if obs_error_markers:
            cmd.extend(["--obs-error-markers", *[str(x) for x in obs_error_markers]])
        _run_step(cmd, fail_fast=fail_fast, step_name="build_unified", output_path=out["unified_parquet"], resume=is_resume)

    # ── Step 2: 计算工具动态指标 ─────────────────────────────────────────────
    if pipeline.get("tool_dynamics", True):
        cmd = [
            python_bin,
            str(script_dir / "tool_dynamics_metrics.py"),
            "--unified-parquet", str(out["unified_parquet"]),
            "--output-dir", str(out["results_dir"]),
            "--top-k-tools", str(top_k_tools),
            "--rolling-window", str(rolling_window),
            "--top-k-invalid-reasons", str(top_k_invalid_reasons),
        ]
        _run_step(cmd, fail_fast=fail_fast, step_name="tool_dynamics", output_path=out["tool_dynamics_json"], resume=is_resume)

    # ── Step 3: 计算 skill 生命周期指标 ─────────────────────────────────────
    if pipeline.get("skill_lifecycle", True):
        cmd = [
            python_bin,
            str(script_dir / "skill_lifecycle_metrics.py"),
            "--unified-parquet", str(out["unified_parquet"]),
            "--output-dir", str(out["results_dir"]),
            "--uplift-window", str(uplift_window),
            "--reuse-window", str(reuse_window),
            "--min-calls-for-skill-stats", str(min_calls_for_skill_stats),
        ]
        _run_step(cmd, fail_fast=fail_fast, step_name="skill_lifecycle", output_path=out["skill_lifecycle_csv"], resume=is_resume)

    # ── Step 4: skill 语义分类 ───────────────────────────────────────────────
    if pipeline.get("skill_semantic_clustering", True):
        cmd = [
            python_bin,
            str(script_dir / "skill_semantic_clustering_llm.py"),
            "--unified-parquet", str(out["unified_parquet"]),
            "--output-dir", str(out["results_dir"]),
            "--mode", llm_mode,
            "--max-skills", str(max_skills),
            "--min-calls", str(min_calls),
        ]
        _append_llm_args(cmd, config)  # 如果启用 LLM，追加 base_url / model / api_key
        _run_step(cmd, fail_fast=fail_fast, step_name="skill_semantic_clustering", resume=is_resume)

    # ── Step 5: 挖掘"惊喜发现" ──────────────────────────────────────────────
    if pipeline.get("surprise_miner", True):
        cmd = [
            python_bin,
            str(script_dir / "surprise_miner_llm.py"),
            "--unified-parquet", str(out["unified_parquet"]),
            "--output-dir", str(out["results_dir"]),
            "--mode", llm_mode,
            "--min-effect-size", str(surprise_min_effect_size),
            "--min-calls", str(surprise_min_calls),
            "--max-findings", str(surprise_max_findings),
        ]
        _append_llm_args(cmd, config)
        _run_step(cmd, fail_fast=fail_fast, step_name="surprise_miner", output_path=out["surprises_json"], resume=is_resume)

    # ── Step 6: 打包生成论文资产 ─────────────────────────────────────────────
    if pipeline.get("paper_bundle", True):
        cmd = [
            python_bin,
            str(script_dir / "make_paper_bundle.py"),
            "--unified-parquet", str(out["unified_parquet"]),
            "--tool-dynamics-json", str(out["tool_dynamics_json"]),
            "--skill-lifecycle-csv", str(out["skill_lifecycle_csv"]),
            "--surprises-json", str(out["surprises_json"]),
            "--output-dir", str(out["paper_bundle_dir"]),
            "--top-k-plot-items", str(paper_top_k_plot_items),
        ]
        _run_step(cmd, fail_fast=fail_fast, step_name="paper_bundle", output_path=out["paper_bundle_dir"], resume=is_resume)

    # ── Step 7: 统计各数据集 create_skill 调用量 ────────────────────────────
    if pipeline.get("create_skill_by_datasource", True):
        cmd = [
            python_bin,
            str(script_dir / "create_skill_by_datasource.py"),
            "--tool_log_dir", str(tool_log_dir),
            "--output", str(out["results_dir"]),
            "--top_n", str(create_skill_top_n),
        ]
        _run_step(cmd, fail_fast=fail_fast, step_name="create_skill_by_datasource", resume=is_resume)

    print("[pipeline] completed")
    print(f"[pipeline] root_dir={out['root_dir']}")
    print(f"[pipeline] analysis_cache_dir={out['analysis_cache_dir']}")
    print(f"[pipeline] results_dir={out['results_dir']}")
    print(f"[pipeline] paper_bundle_dir={out['paper_bundle_dir']}")


if __name__ == "__main__":
    main()
