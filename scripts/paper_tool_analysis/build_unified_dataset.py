#!/usr/bin/env python3
"""
【第一步：构建统一数据集】
把多种来源的训练/评测日志合并成一张大表（unified.parquet），供后续分析脚本使用。

数据来源：
  1. skills_*.json         → skill_snapshot
  2. tool_rollouts_*.json  → tool_summary / tool_rollout_example
  3. *.jsonl               → sample / tool_call / skill_new_event
  4. _skill_archive/_events.jsonl → skill_archive_event（可选）
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from common import (
    DEFAULT_OBS_ERROR_MARKERS,
    as_bool,
    as_float,
    as_int,
    detect_phase_from_data_sources,
    detect_tool_variant,
    ensure_dir,
    infer_skill_store_dir,
    infer_step_from_name,
    list_files_by_step,
    load_forever_tools,
    normalize_tool_kind,
    obs_has_error,
    parse_maybe_json_object,
    read_json,
    read_jsonl,
    skill_archive_events_path,
    write_json,
    write_jsonl,
)


SAMPLE_METRIC_MAP: Dict[str, str] = {
    "num_turns": "num_turns",
    "valid_traj": "valid_traj",
    "no_loss_on_traj": "no_loss_on_traj",
    "is_traj_finished": "is_traj_finished",
    "tool_call_success": "tool_call_success",
    "tool_processing_time_sec": "tool_processing_time_sec",
    "tool_queue_time_sec": "tool_queue_time_sec",
    "client_queued_time_sec": "client_queued_time_sec",
    "client_conn_create_time_sec": "client_conn_create_time_sec",
    "client_request_send_time_sec": "client_request_send_time_sec",
    "response_read_time_sec": "response_read_time_sec",
    "response_size_mb": "response_size_mb",
    "response_await_time_sec": "response_await_time_sec",
    "interact_time_time_sec": "interact_time_time_sec",
    "session_request_time": "session_request_time",
    "X-Process-Time_sec": "x_process_time_sec",
    "tool_penalty": "tool_penalty",
    "latency_penalty": "latency_penalty",
    "format_reward": "format_reward",
    "skill_reward": "skill_reward",
    "skill_valid_create_reward": "skill_valid_create_reward",
    "skill_reused_reward": "skill_reused_reward",
    "skill_valid_create_hit": "skill_valid_create_hit",
    "skill_reused_hit": "skill_reused_hit",
    "empty_responses": "empty_responses",
    "valid_action": "sample_valid_action",
    "per_action_length": "per_action_length",
    "per_obs_length": "per_obs_length",
    "per_action_logp": "per_action_logp",
    "per_reward_from_tool": "per_reward_from_tool",
    "traj_actions_length": "traj_actions_length",
    "traj_obs_length": "traj_obs_length",
    "format_message_turns": "format_message_turns",
    "generated_length": "generated_length",
    "agent_prompt_length_total": "agent_prompt_length_total",
    "agent_output_length_total": "agent_output_length_total",
    "retokenization_diff": "retokenization_diff",
    "close_traj_time": "close_traj_time",
}

NUMERIC_COLUMNS = {
    "step",
    "score",
    "is_correct",
    "accuracy",
    "total_reward",
    "latency_proxy_ms",
    "tool_used_count",
    "skill_used_times",
    "selected_skill_count",
    "tool_turn_index",
    "valid_action",
    "done",
    "finish",
    "success",
    "reward_from_tool",
    "processing_time_ms",
    "queue_time_ms",
    "client_queued_ms",
    "client_conn_create_ms",
    "client_request_send_ms",
    "response_read_ms",
    "response_size_mb",
    "response_await_ms",
    "interact_time_ms",
    "session_request_time",
    "x_process_time_sec",
    "tool_obs_has_error",
    "tool_effective_success",
    "tool_raw_success",
    "judge_correct_raw",
    "judge_allow_numeric_tolerance",
    "num_turns",
    "valid_traj",
    "no_loss_on_traj",
    "is_traj_finished",
    "tool_call_success",
    "tool_processing_time_sec",
    "tool_queue_time_sec",
    "client_queued_time_sec",
    "client_conn_create_time_sec",
    "client_request_send_time_sec",
    "response_read_time_sec",
    "response_await_time_sec",
    "tool_penalty",
    "latency_penalty",
    "format_reward",
    "skill_reward",
    "skill_valid_create_reward",
    "skill_reused_reward",
    "skill_valid_create_hit",
    "skill_reused_hit",
    "empty_responses",
    "sample_valid_action",
    "per_action_length",
    "per_obs_length",
    "per_action_logp",
    "per_reward_from_tool",
    "traj_actions_length",
    "traj_obs_length",
    "format_message_turns",
    "generated_length",
    "agent_prompt_length_total",
    "agent_output_length_total",
    "retokenization_diff",
    "close_traj_time",
    "skill_archive_global_step",
    "skill_archive_snapshot_copied",
}

SCHEMA_DEFAULTS: Dict[str, Any] = {
    "step": float("nan"),
    "row_type": "",
    "uid": "",
    "request_id": "",
    "trajectory_key": "",
    "data_source": "",
    "phase": "",
    "score": float("nan"),
    "is_correct": float("nan"),
    "accuracy": float("nan"),
    "total_reward": float("nan"),
    "latency_proxy_ms": float("nan"),
    "input_text": "",
    "output_text": "",
    "ground_truth": "",
    "tool_name": "",
    "tool_kind": "",
    "invalid_reason": "",
    "tool_model": "",
    "tool_prompt": "",
    "tool_obs": "",
    "tool_used_count": float("nan"),
    "tool_turn_index": float("nan"),
    "trajectory_id": "",
    "action": "",
    "valid_action": float("nan"),
    "done": float("nan"),
    "finish": float("nan"),
    "success": float("nan"),
    "reward_from_tool": float("nan"),
    "processing_time_ms": float("nan"),
    "queue_time_ms": float("nan"),
    "client_queued_ms": float("nan"),
    "client_conn_create_ms": float("nan"),
    "client_request_send_ms": float("nan"),
    "response_read_ms": float("nan"),
    "response_size_mb": float("nan"),
    "response_await_ms": float("nan"),
    "interact_time_ms": float("nan"),
    "session_request_time": float("nan"),
    "x_process_time_sec": float("nan"),
    "skill_path": "",
    "created_trajectory_id": "",
    "created_turn": "",
    "tool_obs_has_error": float("nan"),
    "tool_effective_success": float("nan"),
    "tool_raw_success": float("nan"),
    "judge_correct_raw": float("nan"),
    "judge_final_source": "",
    "judge_reason": "",
    "judge_llm_reason": "",
    "judge_allow_numeric_tolerance": float("nan"),
    "judge_parse_status": "",
    "judge_api_status": "",
    "judge_raw_response": "",
    "judge_numeric_extraction_answer": "",
    "judge_numeric_extraction_reason": "",
    "judge_numeric_extraction_parse_status": "",
    "judge_numeric_extraction_api_status": "",
    "judge_numeric_extraction_raw_response": "",
    "skill_name": "",
    "skill_status": "",
    "skill_used_times": float("nan"),
    "skill_description": "",
    "selection_stats_json": "",
    "skill_monitoring_metrics_json": "",
    "selected_skill_count": float("nan"),
    "selected_skill_names_json": "",
    "traj_stop_reason": "",
    "verl_tool_metrics_json": "",
    "num_turns": float("nan"),
    "valid_traj": float("nan"),
    "no_loss_on_traj": float("nan"),
    "is_traj_finished": float("nan"),
    "tool_call_success": float("nan"),
    "tool_processing_time_sec": float("nan"),
    "tool_queue_time_sec": float("nan"),
    "client_queued_time_sec": float("nan"),
    "client_conn_create_time_sec": float("nan"),
    "client_request_send_time_sec": float("nan"),
    "response_read_time_sec": float("nan"),
    "response_await_time_sec": float("nan"),
    "tool_penalty": float("nan"),
    "latency_penalty": float("nan"),
    "format_reward": float("nan"),
    "skill_reward": float("nan"),
    "skill_valid_create_reward": float("nan"),
    "skill_reused_reward": float("nan"),
    "skill_valid_create_hit": float("nan"),
    "skill_reused_hit": float("nan"),
    "empty_responses": float("nan"),
    "sample_valid_action": float("nan"),
    "per_action_length": float("nan"),
    "per_obs_length": float("nan"),
    "per_action_logp": float("nan"),
    "per_reward_from_tool": float("nan"),
    "traj_actions_length": float("nan"),
    "traj_obs_length": float("nan"),
    "format_message_turns": float("nan"),
    "generated_length": float("nan"),
    "agent_prompt_length_total": float("nan"),
    "agent_output_length_total": float("nan"),
    "retokenization_diff": float("nan"),
    "close_traj_time": float("nan"),
    "skill_archive_event": "",
    "skill_archive_reason": "",
    "skill_archive_timestamp": "",
    "skill_archive_global_step": float("nan"),
    "skill_archive_snapshot_copied": float("nan"),
    "skill_archive_source_path": "",
    "skill_archive_runtime_path": "",
    "skill_archive_path": "",
    "archive_metadata_json": "",
    "log_ref": "",
}


def _to_text(value: Any) -> str:
    """把任意值转成字符串；dict/list 先序列化成 JSON 字符串，避免列类型不一致。"""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _json_text(value: Any) -> str:
    """统一把复杂对象转成 JSON 字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _numeric(value: Any) -> float:
    return as_float(value, default=float("nan"))


def _bool_num(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return 1.0 if as_bool(value) else 0.0


def _trajectory_key(uid: Any, request_id: Any) -> str:
    request = _to_text(request_id).strip()
    if request:
        return request
    return _to_text(uid).strip()


def _base_row(*, step: Any, row_type: str, log_ref: str = "") -> Dict[str, Any]:
    row = dict(SCHEMA_DEFAULTS)
    row["step"] = _numeric(step)
    row["row_type"] = row_type
    row["log_ref"] = log_ref
    return row


def _extract_latency_proxy_ms(tool_interact_info: Any) -> float:
    """
    从 tool_interact_info 列表中提取工具调用延迟（毫秒）。
    """
    if not isinstance(tool_interact_info, list):
        return float("nan")
    total = 0.0
    has_value = False
    for item in tool_interact_info:
        if not isinstance(item, dict):
            continue
        for key in ("processing_time_ms", "latency", "latency_ms", "interact_time_ms"):
            if key in item:
                val = _numeric(item.get(key))
                if pd.notna(val):
                    total += val
                    has_value = True
                    break
    return total if has_value else float("nan")


def _tool_effective_success(
    *,
    invalid_reason: Any,
    tool_obs: Any,
    explicit_value: Any,
    obs_error_markers: Iterable[str],
) -> float:
    if explicit_value is not None and explicit_value != "":
        return 1.0 if as_bool(explicit_value) else 0.0
    raw_success = str(invalid_reason or "").strip() == ""
    has_obs_error = obs_has_error(tool_obs, markers=obs_error_markers)
    return 1.0 if (raw_success and not has_obs_error) else 0.0


def _fill_sample_metrics(row: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    row["verl_tool_metrics_json"] = _json_text(metrics)
    for src_key, dst_col in SAMPLE_METRIC_MAP.items():
        if src_key in metrics:
            row[dst_col] = _numeric(metrics.get(src_key))


def load_skill_logs(tool_log_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]], bool]:
    """
    读取所有 skills_*.json 文件，解析每个 step 被选中的 skill 列表。

    返回：
      rows      — skill_snapshot 行
      step_meta — 每步的 selected_skill_count / selection_stats / monitoring metrics
      found_any — 是否发现了至少一个 skill 日志文件
    """
    rows: List[Dict[str, Any]] = []
    step_meta: Dict[int, Dict[str, Any]] = {}
    files = list_files_by_step(tool_log_dir, "skills_*.json", prefix="skills_")
    for path in files:
        payload = read_json(path)
        step = int(payload.get("step") or infer_step_from_name(path, prefix="skills_") or -1)
        skills = payload.get("skills", [])
        if not isinstance(skills, list):
            skills = []
        selected_names = [
            str(s.get("name", "")).strip()
            for s in skills
            if isinstance(s, dict) and str(s.get("name", "")).strip()
        ]
        step_meta[step] = {
            "selected_skill_count": len(selected_names),
            "selected_skill_names": selected_names,
            "selection_stats_json": _json_text(payload.get("selection_stats", {})),
            "skill_monitoring_metrics_json": _json_text(payload.get("skill_monitoring_metrics", {})),
            "skill_log_ref": str(path),
        }

        for skill in skills:
            if not isinstance(skill, dict):
                continue
            row = _base_row(step=step, row_type="skill_snapshot", log_ref=f"{path}#skill:{skill.get('name', '')}")
            row.update(
                {
                    "skill_name": str(skill.get("name", "")),
                    "skill_status": "active",
                    "skill_used_times": _numeric(skill.get("used_times")),
                    "skill_description": str(skill.get("description", "")),
                    "skill_path": str(skill.get("path", "")),
                    "selection_stats_json": _json_text(payload.get("selection_stats", {})),
                    "skill_monitoring_metrics_json": _json_text(payload.get("skill_monitoring_metrics", {})),
                    "selected_skill_count": float(len(selected_names)),
                    "selected_skill_names_json": _json_text(selected_names),
                }
            )
            rows.append(row)
    return rows, step_meta, bool(files)


def load_tool_rollout_logs(tool_log_dir: Path, obs_error_markers: Iterable[str]) -> Tuple[List[Dict[str, Any]], bool]:
    """
    读取所有 tool_rollouts_*.json 文件，解析每个 step 各工具的汇总统计和具体样例。
    """
    rows: List[Dict[str, Any]] = []
    files = list_files_by_step(tool_log_dir, "tool_rollouts_*.json", prefix="tool_rollouts_")
    for path in files:
        payload = read_json(path)
        step = int(payload.get("step") or infer_step_from_name(path, prefix="tool_rollouts_") or -1)
        tools = payload.get("tools", {})
        if not isinstance(tools, dict):
            continue
        for tool_name, info in tools.items():
            if not isinstance(info, dict):
                continue
            tname = str(info.get("tool_name", tool_name))
            tkind = normalize_tool_kind(tname, info.get("tool_kind"))
            summary = _base_row(step=step, row_type="tool_summary", log_ref=f"{path}#tool:{tname}")
            summary.update(
                {
                    "tool_name": tname,
                    "tool_kind": tkind,
                    "tool_used_count": _numeric(info.get("used_count")),
                }
            )
            rows.append(summary)

            rollouts = info.get("rollouts", [])
            if not isinstance(rollouts, list):
                continue
            for i, rollout in enumerate(rollouts):
                if not isinstance(rollout, dict):
                    continue
                tool_obs = rollout.get("tool_obs")
                tool_obs_error = rollout.get("tool_obs_has_error")
                if tool_obs_error is None or tool_obs_error == "":
                    tool_obs_error = obs_has_error(tool_obs, markers=obs_error_markers)
                example = _base_row(
                    step=step,
                    row_type="tool_rollout_example",
                    log_ref=f"{path}#tool:{tname}:rollout:{i}",
                )
                example.update(
                    {
                        "uid": str(rollout.get("uid", "")),
                        "trajectory_key": _trajectory_key(rollout.get("uid", ""), ""),
                        "data_source": str(rollout.get("data_source", "")),
                        "is_correct": _bool_num(rollout.get("is_correct")),
                        "input_text": str(rollout.get("prompt_text_decoded", "")),
                        "output_text": str(rollout.get("response_text_decoded", "")),
                        "ground_truth": _to_text(rollout.get("ground_truth")),
                        "format_reward": _numeric(rollout.get("format_reward")),
                        "tool_name": tname,
                        "tool_kind": normalize_tool_kind(tname, rollout.get("tool_kind", tkind)),
                        "invalid_reason": _to_text(rollout.get("invalid_reason")),
                        "tool_model": _to_text(rollout.get("tool_model")),
                        "tool_prompt": _to_text(rollout.get("tool_prompt")),
                        "tool_obs": _to_text(tool_obs),
                        "tool_used_count": 1.0,
                        "skill_name": _to_text(rollout.get("skill_name")),
                        "tool_obs_has_error": 1.0 if as_bool(tool_obs_error) else 0.0,
                        "tool_effective_success": _tool_effective_success(
                            invalid_reason=rollout.get("invalid_reason"),
                            tool_obs=tool_obs,
                            explicit_value=rollout.get("tool_effective_success"),
                            obs_error_markers=obs_error_markers,
                        ),
                        "tool_raw_success": 1.0 if str(rollout.get("invalid_reason", "")).strip() == "" else 0.0,
                        "judge_correct_raw": _bool_num(rollout.get("judge_correct_raw")),
                        "judge_final_source": _to_text(rollout.get("judge_final_source")),
                        "judge_reason": _to_text(rollout.get("judge_reason")),
                        "judge_llm_reason": _to_text(rollout.get("judge_llm_reason")),
                        "judge_allow_numeric_tolerance": _bool_num(rollout.get("judge_allow_numeric_tolerance")),
                        "judge_parse_status": _to_text(rollout.get("judge_parse_status")),
                        "judge_api_status": _to_text(rollout.get("judge_api_status")),
                        "judge_raw_response": _to_text(rollout.get("judge_raw_response")),
                        "judge_numeric_extraction_answer": _to_text(rollout.get("judge_numeric_extraction_answer")),
                        "judge_numeric_extraction_reason": _to_text(rollout.get("judge_numeric_extraction_reason")),
                        "judge_numeric_extraction_parse_status": _to_text(
                            rollout.get("judge_numeric_extraction_parse_status")
                        ),
                        "judge_numeric_extraction_api_status": _to_text(
                            rollout.get("judge_numeric_extraction_api_status")
                        ),
                        "judge_numeric_extraction_raw_response": _to_text(
                            rollout.get("judge_numeric_extraction_raw_response")
                        ),
                    }
                )
                rows.append(example)
    return rows, bool(files)


def load_rollout_jsonl(
    rollout_data_dir: Path,
    *,
    max_files: int,
    obs_error_markers: Iterable[str],
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    读取 rollout/validation JSONL 文件，解析 sample / tool_call / skill_new_event。
    """
    rows: List[Dict[str, Any]] = []
    files = list_files_by_step(rollout_data_dir, "*.jsonl")
    if max_files > 0:
        files = files[-max_files:]
    for path in files:
        file_step = infer_step_from_name(path)
        for line_idx, record in enumerate(read_jsonl(path), start=1):
            if not isinstance(record, dict):
                continue
            step = int(record.get("step") or file_step or -1)
            uid = str(record.get("uid", ""))
            request_id = _to_text(record.get("request_id"))
            trajectory_key = _trajectory_key(uid, request_id)
            score = _numeric(record.get("score"))
            is_correct = _numeric(record.get("is_correct"))
            accuracy = _numeric(record.get("accuracy"))
            total_reward = _numeric(record.get("total_reward"))
            data_source = _to_text(record.get("data_source"))
            tool_interact_info = record.get("tool_interact_info")
            if isinstance(tool_interact_info, dict):
                tool_items = [tool_interact_info]
            elif isinstance(tool_interact_info, list):
                tool_items = tool_interact_info
            else:
                tool_items = []
            latency_proxy_ms = _extract_latency_proxy_ms(tool_items)

            sample = _base_row(step=step, row_type="sample", log_ref=f"{path}:{line_idx}")
            sample.update(
                {
                    "uid": uid,
                    "request_id": request_id,
                    "trajectory_key": trajectory_key,
                    "data_source": data_source,
                    "score": score,
                    "is_correct": is_correct,
                    "accuracy": accuracy,
                    "total_reward": total_reward,
                    "latency_proxy_ms": latency_proxy_ms,
                    "input_text": _to_text(record.get("input")),
                    "output_text": _to_text(record.get("output")),
                    "ground_truth": _to_text(record.get("gts")),
                    "traj_stop_reason": _to_text(record.get("traj_stop_reason")),
                }
            )
            metrics = parse_maybe_json_object(record.get("verl_tool_metrics"))
            if metrics:
                _fill_sample_metrics(sample, metrics)
            for key in (
                "tool_penalty",
                "latency_penalty",
                "format_reward",
                "skill_reward",
                "skill_valid_create_reward",
                "skill_reused_reward",
                "skill_valid_create_hit",
                "skill_reused_hit",
            ):
                if key in record:
                    sample[key] = _numeric(record.get(key))
            rows.append(sample)

            for turn_idx, item in enumerate(tool_items, start=1):
                if not isinstance(item, dict):
                    continue
                tool_name = _to_text(item.get("tool") or item.get("name"))
                if not tool_name:
                    continue
                tool_obs = item.get("obs")
                invalid_reason = _to_text(item.get("invalid_reason"))
                tool_obs_error = obs_has_error(tool_obs, markers=obs_error_markers)
                tool_call = _base_row(
                    step=step,
                    row_type="tool_call",
                    log_ref=f"{path}:{line_idx}:tool:{turn_idx}",
                )
                tool_call.update(
                    {
                        "uid": uid,
                        "request_id": request_id,
                        "trajectory_key": trajectory_key,
                        "data_source": data_source,
                        "score": score,
                        "is_correct": is_correct,
                        "accuracy": accuracy,
                        "total_reward": total_reward,
                        "latency_proxy_ms": latency_proxy_ms,
                        "input_text": _to_text(record.get("input")),
                        "output_text": _to_text(record.get("output")),
                        "ground_truth": _to_text(record.get("gts")),
                        "tool_name": tool_name,
                        "tool_kind": normalize_tool_kind(tool_name, _to_text(item.get("tool_kind"))),
                        "invalid_reason": invalid_reason,
                        "tool_model": _to_text(item.get("model")),
                        "tool_prompt": _to_text(item.get("prompt")),
                        "tool_obs": _to_text(tool_obs),
                        "tool_used_count": 1.0,
                        "tool_turn_index": float(turn_idx),
                        "trajectory_id": _to_text(item.get("trajectory_id")),
                        "action": _to_text(item.get("action")),
                        "valid_action": _bool_num(item.get("valid_action")),
                        "done": _bool_num(item.get("done")),
                        "finish": _bool_num(item.get("finish")),
                        "success": _bool_num(item.get("success")),
                        "reward_from_tool": _numeric(item.get("reward")),
                        "processing_time_ms": _numeric(item.get("processing_time_ms")),
                        "queue_time_ms": _numeric(item.get("queue_time_ms")),
                        "client_queued_ms": _numeric(item.get("client_queued_ms")),
                        "client_conn_create_ms": _numeric(item.get("client_conn_create_ms")),
                        "client_request_send_ms": _numeric(item.get("client_request_send_ms")),
                        "response_read_ms": _numeric(item.get("response_read_ms")),
                        "response_size_mb": _numeric(item.get("response_size_mb")),
                        "response_await_ms": _numeric(
                            item.get("time awaiting response_ms", item.get("response_await_ms"))
                        ),
                        "interact_time_ms": _numeric(item.get("interact_time_ms")),
                        "session_request_time": _numeric(item.get("session_request_time")),
                        "x_process_time_sec": _numeric(item.get("X-Process-Time")),
                        "skill_name": _to_text(item.get("skill_name")),
                        "skill_path": _to_text(item.get("skill_path")),
                        "created_trajectory_id": _to_text(item.get("created_trajectory_id")),
                        "created_turn": _to_text(item.get("created_turn")),
                        "tool_obs_has_error": 1.0 if tool_obs_error else 0.0,
                        "tool_effective_success": _tool_effective_success(
                            invalid_reason=invalid_reason,
                            tool_obs=tool_obs,
                            explicit_value=item.get("tool_effective_success"),
                            obs_error_markers=obs_error_markers,
                        ),
                        "tool_raw_success": 1.0 if invalid_reason.strip() == "" else 0.0,
                    }
                )
                rows.append(tool_call)

            batch_new_skills = record.get("batch_new_skills")
            if isinstance(batch_new_skills, dict):
                for key, info in batch_new_skills.items():
                    if not isinstance(info, dict):
                        continue
                    event = _base_row(
                        step=step,
                        row_type="skill_new_event",
                        log_ref=f"{path}:{line_idx}:new_skill:{key}",
                    )
                    event.update(
                        {
                            "uid": uid,
                            "request_id": request_id,
                            "trajectory_key": trajectory_key,
                            "data_source": data_source,
                            "score": score,
                            "is_correct": is_correct,
                            "accuracy": accuracy,
                            "total_reward": total_reward,
                            "latency_proxy_ms": latency_proxy_ms,
                            "input_text": _to_text(record.get("input")),
                            "output_text": _to_text(record.get("output")),
                            "ground_truth": _to_text(record.get("gts")),
                            "tool_name": "create_skill",
                            "tool_kind": "create_skill",
                            "tool_used_count": 1.0,
                            "skill_name": _to_text(info.get("skill_name")),
                            "skill_status": "new",
                            "skill_used_times": 1.0,
                            "skill_path": _to_text(info.get("path")),
                            "created_trajectory_id": _to_text(info.get("created_trajectory_id")),
                            "created_turn": _to_text(info.get("created_turn")),
                        }
                    )
                    rows.append(event)
    return rows, bool(files)


def load_skill_archive_events(skill_store_dir: Optional[Path]) -> Tuple[List[Dict[str, Any]], Optional[Path]]:
    """
    读取 _skill_archive/_events.jsonl，并展开成 skill_archive_event 行。
    """
    archive_path = skill_archive_events_path(skill_store_dir)
    if archive_path is None:
        return [], None
    rows: List[Dict[str, Any]] = []
    for idx, event in enumerate(read_jsonl(archive_path), start=1):
        if not isinstance(event, dict):
            continue
        step = event.get("global_step")
        row = _base_row(
            step=step if step is not None else float("nan"),
            row_type="skill_archive_event",
            log_ref=f"{archive_path}:{idx}",
        )
        row.update(
            {
                "skill_name": str(event.get("skill_name", "")),
                "skill_archive_event": str(event.get("event", "")),
                "skill_archive_reason": str(event.get("reason", "")),
                "skill_archive_timestamp": _to_text(event.get("timestamp")),
                "skill_archive_global_step": _numeric(event.get("global_step")),
                "skill_archive_snapshot_copied": _bool_num(event.get("snapshot_copied")),
                "skill_archive_source_path": _to_text(event.get("source_path")),
                "skill_archive_runtime_path": _to_text(event.get("runtime_path")),
                "skill_archive_path": _to_text(event.get("archive_path")),
                "archive_metadata_json": _json_text(event.get("metadata", {})),
            }
        )
        rows.append(row)
    return rows, archive_path


def attach_step_meta(df: pd.DataFrame, step_meta: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """
    把 step_meta（来自 skills_*.json）中的字段回填到 unified DataFrame 的对应行。
    """
    if df.empty or not step_meta:
        return df
    meta_df = pd.DataFrame(
        [
            {
                "step": float(step),
                "meta_selected_skill_count": float(meta.get("selected_skill_count", float("nan"))),
                "meta_selected_skill_names_json": _json_text(meta.get("selected_skill_names", [])),
                "meta_selection_stats_json": _json_text(meta.get("selection_stats_json", "")),
                "meta_skill_monitoring_metrics_json": _json_text(meta.get("skill_monitoring_metrics_json", "")),
            }
            for step, meta in step_meta.items()
        ]
    )
    if meta_df.empty:
        return df
    out = df.merge(meta_df, on="step", how="left")
    out["selected_skill_count"] = out["selected_skill_count"].where(
        out["selected_skill_count"].notna(),
        out["meta_selected_skill_count"],
    )
    out["selected_skill_names_json"] = out["selected_skill_names_json"].where(
        out["selected_skill_names_json"].astype(str).str.strip() != "",
        out["meta_selected_skill_names_json"].fillna(""),
    )
    out["selection_stats_json"] = out["selection_stats_json"].where(
        out["selection_stats_json"].astype(str).str.strip() != "",
        out["meta_selection_stats_json"].fillna(""),
    )
    out["skill_monitoring_metrics_json"] = out["skill_monitoring_metrics_json"].where(
        out["skill_monitoring_metrics_json"].astype(str).str.strip() != "",
        out["meta_skill_monitoring_metrics_json"].fillna(""),
    )
    return out.drop(
        columns=[
            "meta_selected_skill_count",
            "meta_selected_skill_names_json",
            "meta_selection_stats_json",
            "meta_skill_monitoring_metrics_json",
        ],
        errors="ignore",
    )


def _detect_and_print_meta(
    df: pd.DataFrame,
    *,
    phase_hint: str,
    variant_hint: str,
    tool_log_dir: Path,
    rollout_data_dir: Path,
    skill_store_dir: Optional[Path],
    skill_archive_path: Optional[Path],
    results_dir: Path,
) -> dict:
    """
    从 unified DataFrame 中自动检测 phase 和 tool_variant，
    打印检测结果并把元信息写入 results/meta.json。
    """
    tool_names_in_data = [str(n) for n in df.loc[df["tool_name"] != "", "tool_name"].unique()]
    if variant_hint == "auto":
        tool_variant = detect_tool_variant(tool_names_in_data)
        variant_source = "auto_detected"
    else:
        tool_variant = variant_hint
        variant_source = "config_explicit"
    effective_variant = tool_variant if tool_variant in ("id", "ood", "both") else "both"
    forever_tools_used = sorted(load_forever_tools(effective_variant))

    if phase_hint == "auto":
        data_sources = [str(ds) for ds in df.loc[df["data_source"] != "", "data_source"].unique()]
        phase = detect_phase_from_data_sources(data_sources)
        phase_source = "auto_detected"
        if phase == "unknown":
            print(
                "[meta] warning: could not detect phase from data_source values. "
                "Set experiment.phase explicitly in pipeline_config.yaml."
            )
    else:
        phase = phase_hint
        phase_source = "config_explicit"

    print(f"[meta] phase detected    : {phase} ({phase_source})")
    print(f"[meta] tool_variant      : {tool_variant} ({variant_source})")
    print(f"[meta] forever_tools ({len(forever_tools_used):2d}): {', '.join(forever_tools_used)}")
    if skill_store_dir:
        print(f"[meta] skill_store_dir   : {skill_store_dir}")
    if skill_archive_path:
        print(f"[meta] skill_archive     : {skill_archive_path}")
    if tool_variant == "unknown":
        print(
            "[meta] warning: no known forever tools found in data. "
            "Falling back to 'both' (id+ood). "
            "Set experiment.tool_variant explicitly if needed."
        )

    ensure_dir(results_dir)
    meta = {
        "phase": phase,
        "phase_source": phase_source,
        "tool_variant": tool_variant,
        "tool_variant_effective": effective_variant,
        "tool_variant_source": variant_source,
        "forever_tools": forever_tools_used,
        "tool_log_dir": str(tool_log_dir),
        "rollout_data_dir": str(rollout_data_dir),
        "skill_store_dir": str(skill_store_dir) if skill_store_dir else "",
        "skill_archive_path": str(skill_archive_path) if skill_archive_path else "",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    meta_path = results_dir / "meta.json"
    write_json(meta_path, meta)
    print(f"[meta] written to        : {meta_path}")
    return meta


def _resolve_skill_store_dir(explicit: str, tool_log_dir: Path, rollout_data_dir: Path) -> Optional[Path]:
    if explicit.strip():
        candidate = Path(explicit).expanduser().resolve()
        if candidate.exists():
            return candidate
        print(f"[build_unified] warning: skill_store_dir not found, skip archive: {candidate}")
        return None
    inferred = infer_skill_store_dir(tool_log_dir=tool_log_dir, rollout_data_dir=rollout_data_dir)
    if inferred is not None:
        print(f"[build_unified] auto-detected skill_store_dir: {inferred}")
    else:
        print("[build_unified] no skill_store_dir detected; archive ingestion disabled.")
    return inferred


def _coerce_schema_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in SCHEMA_DEFAULTS:
        if col not in df.columns:
            df[col] = SCHEMA_DEFAULTS[col]
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in df.columns:
        if col not in NUMERIC_COLUMNS:
            df[col] = df[col].fillna("").astype(str)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified parquet dataset from training/eval logs.")
    parser.add_argument("--tool-log-dir", required=True, help="Directory containing skills_*.json and tool_rollouts_*.json")
    parser.add_argument("--rollout-data-dir", required=True, help="Directory containing step-wise rollout/validation jsonl files")
    parser.add_argument("--skill-store-dir", default="", help="Optional skill store directory containing _skill_archive/_events.jsonl")
    parser.add_argument("--output-dir", default="scripts/paper_tool_analysis", help="Root output directory")
    parser.add_argument("--max-rollout-files", type=int, default=0, help="Use only last N rollout jsonl files (0 means all)")
    parser.add_argument(
        "--phase",
        default="auto",
        choices=["train", "eval", "auto"],
        help="Experiment phase (train/eval/auto). auto = infer from data_source values.",
    )
    parser.add_argument(
        "--tool-variant",
        default="auto",
        choices=["id", "ood", "auto"],
        help="Forever-tool set variant (id/ood/auto). auto = detect from tool names in data.",
    )
    parser.add_argument(
        "--include-skill-archive",
        default="auto",
        choices=["true", "false", "auto"],
        help="Whether to load skill archive events. auto = only if archive file exists.",
    )
    parser.add_argument(
        "--obs-error-markers",
        nargs="*",
        default=list(DEFAULT_OBS_ERROR_MARKERS),
        help="Markers used to flag tool obs errors, e.g. [stderr]",
    )
    args = parser.parse_args()

    tool_log_dir = Path(args.tool_log_dir).expanduser().resolve()
    rollout_data_dir = Path(args.rollout_data_dir).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()
    cache_dir = ensure_dir(out_root / "analysis_cache")
    results_dir = ensure_dir(out_root / "results")

    obs_error_markers = tuple(str(x) for x in (args.obs_error_markers or DEFAULT_OBS_ERROR_MARKERS))
    skill_store_dir = _resolve_skill_store_dir(args.skill_store_dir, tool_log_dir, rollout_data_dir)
    include_archive = args.include_skill_archive.lower()
    if include_archive == "false":
        skill_store_dir = None

    skill_rows, step_meta, has_skills_log = load_skill_logs(tool_log_dir)
    tool_rows, has_tool_rollouts = load_tool_rollout_logs(tool_log_dir, obs_error_markers)
    rollout_rows, has_sample_jsonl = load_rollout_jsonl(
        rollout_data_dir,
        max_files=args.max_rollout_files,
        obs_error_markers=obs_error_markers,
    )

    archive_rows: List[Dict[str, Any]] = []
    archive_path: Optional[Path] = None
    if include_archive != "false" and skill_store_dir is not None:
        archive_rows, archive_path = load_skill_archive_events(skill_store_dir)
        if include_archive == "true" and archive_path is None:
            print("[build_unified] warning: include_skill_archive=true but archive file not found.")
    has_skill_archive = len(archive_rows) > 0

    all_rows = skill_rows + tool_rows + rollout_rows + archive_rows
    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No rows loaded. Please check input directories.")

    df = attach_step_meta(df, step_meta)
    df = _coerce_schema_types(df)
    df = df.sort_values(["step", "row_type", "uid", "tool_name", "skill_name"], kind="stable").reset_index(drop=True)

    meta = _detect_and_print_meta(
        df=df,
        phase_hint=args.phase,
        variant_hint=args.tool_variant,
        tool_log_dir=tool_log_dir,
        rollout_data_dir=rollout_data_dir,
        skill_store_dir=skill_store_dir,
        skill_archive_path=archive_path,
        results_dir=results_dir,
    )
    df["phase"] = meta["phase"]

    parquet_path = cache_dir / "unified.parquet"
    jsonl_path = cache_dir / "unified.jsonl"
    summary_path = cache_dir / "build_summary.json"

    write_jsonl(jsonl_path, df.fillna("").to_dict(orient="records"))
    df.to_parquet(parquet_path, index=False)

    valid_steps = pd.to_numeric(df["step"], errors="coerce").dropna()
    summary = {
        "rows_total": int(len(df)),
        "rows_by_type": df["row_type"].value_counts(dropna=False).to_dict(),
        "steps_min": int(valid_steps.min()) if len(valid_steps) else None,
        "steps_max": int(valid_steps.max()) if len(valid_steps) else None,
        "phase": meta["phase"],
        "tool_variant": meta["tool_variant"],
        "tool_log_dir": str(tool_log_dir),
        "rollout_data_dir": str(rollout_data_dir),
        "skill_store_dir": str(skill_store_dir) if skill_store_dir else "",
        "skill_archive_path": str(archive_path) if archive_path else "",
        "has_skills_log": has_skills_log,
        "has_tool_rollouts": has_tool_rollouts,
        "has_sample_jsonl": has_sample_jsonl,
        "has_skill_archive": has_skill_archive,
        "output_parquet": str(parquet_path),
        "output_jsonl": str(jsonl_path),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
