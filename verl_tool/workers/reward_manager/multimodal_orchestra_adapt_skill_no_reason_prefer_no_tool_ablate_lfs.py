"""
Standalone copy-over no-reason variant of the MultimodalOrchestra reward manager.

This file is intended to be copied over the original Python module on a remote
server. It keeps the full reward-manager logic self-contained while removing
`reason` from LLM output schemas, and changes the reward shaping so that:
- simple problems can still prefer no-tool solutions
- correct tool usage retains a keepalive reward
- tool rescue is rewarded when no-tool fails
- latency only compares correct tool trajectories

This variant adds ablation switches for latency / format / skill rewards while
leaving accuracy reward, tool reward, and existing start-step behavior unchanged.
"""

import base64
import json
import math
import os
import re
import shutil
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
import urllib3
from verl import DataProto
from verl.trainer.ppo.tool_adaptation import archive_skill_event
from verl.workers.reward_manager import register
from verl_tool.workers.reward_manager.format_reward import compute_format_reward

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class ToolCallSummary:
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_counts: Dict[str, int] = field(default_factory=dict)
    total_latency: float = 0.0


@dataclass
class SampleInfo:
    uid: str
    data_source: str
    prompt_text_decoded: str
    response_text_decoded: str
    response_str: str
    ground_truth: Any
    tool_calls_info: ToolCallSummary
    is_correct: bool
    format_reward: float
    judge_correct_raw: bool = False
    judge_final_source: str = ""
    judge_reason: str = ""
    judge_llm_reason: str = ""
    judge_allow_numeric_tolerance: bool = False
    judge_parse_status: str = "not_requested"
    judge_api_status: str = "not_requested"
    judge_raw_response: str = ""
    judge_numeric_extraction_answer: str = ""
    judge_numeric_extraction_reason: str = ""
    judge_numeric_extraction_parse_status: str = "not_requested"
    judge_numeric_extraction_api_status: str = "not_requested"
    judge_numeric_extraction_raw_response: str = ""


@dataclass
class BatchStats:
    latency_min: Dict[str, float] = field(default_factory=dict)
    latency_max: Dict[str, float] = field(default_factory=dict)

FOREVER_TOOLS = [
    "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-32B-Instruct",
    "Qwen3-VL-235B-A22B-Instruct",
    "SAM3",
    "MinerU2.5",
    "PaddleOCR",
    "EasyOCR",
    "GroundingDINO",
    "Qwen-Image-Edit",
    "OpenCV",
]

CREATE_SKILL_TOOL_NAME = "create_skill"
RUN_SKILL_TOOL_NAME = "run_skill"
VALID_CREATE_REWARD = 0.1
REUSED_SKILL_REWARD = 0.1
DEFAULT_SKILL_REWARD_END_STEP = 10**18
RUN_SKILL_OBS_ERROR_MARKERS = (
    "[stderr]",
)
JUDGE_RAW_RESPONSE_LOG_MAX_CHARS = 6000


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _get_skill_store_dir() -> Path:
    env_dir = os.environ.get("VERL_SKILL_STORE_DIR")
    if env_dir:
        return Path(env_dir)
    return _repo_root() / "skills"


def _load_active_skill_names() -> set:
    store_dir = _get_skill_store_dir()
    active_path = store_dir / "active_skills.json"
    if active_path.exists():
        try:
            with active_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return {str(item) for item in data}
        except Exception:
            pass
    index_path = store_dir / "index.json"
    if index_path.exists():
        try:
            with index_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                names = set()
                for name, info in data.items():
                    if not isinstance(name, str):
                        continue
                    if not isinstance(info, dict):
                        names.add(name)
                        continue
                    status = str(info.get("status", "active")).strip().lower()
                    if status != "active":
                        continue
                    path_str = str(info.get("path", ""))
                    normalized_path = path_str.replace("\\", "/")
                    if "/_pending/" in normalized_path:
                        continue
                    names.add(name)
                return names
        except Exception:
            pass
    return set()


def _load_dynamic_skill_names_from_index() -> set:
    store_dir = _get_skill_store_dir()
    index_path = store_dir / "index.json"
    if not index_path.exists():
        return set()

    try:
        with index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()

    if not isinstance(data, dict):
        return set()

    names = set()
    for name, info in data.items():
        if not isinstance(name, str) or name in FOREVER_TOOLS:
            continue
        if not isinstance(info, dict):
            continue
        status = str(info.get("status", "active")).strip().lower()
        if status != "active":
            continue
        path_str = str(info.get("path", ""))
        normalized_path = path_str.replace("\\", "/")
        if not normalized_path or "/_pending/" in normalized_path:
            continue
        names.add(name)
    return names


@register("multimodal_orchestra_ablate_latency_format_skill")
class MultimodalOrchestraRewardManager:
    """
    MultimodalOrchestra-style Reward Manager.

    - Tool penalty: legacy field name for a composite tool/no-tool preference reward.
      Correct tool usage receives a small keepalive reward, correct tool rescue receives
      an extra reward when no-tool fails, and correct no-tool usage receives a mild
      efficiency reward when a correct tool alternative exists.
    - Latency penalty: legacy field name for a latency reward among correct tool trajectories.
    - Accuracy reward: compare parsed <answer> with ground truth, reward 1.0 if correct.
    """

    name = "multimodal_orchestra_ablate_latency_format_skill"

    def __init__(
        self,
        tokenizer,
        num_examine: int = 20,
        compute_score=None,
        reward_fn_key: str = "data_source",
        tool_penalty_coeff: float = 0.03,
        latency_penalty_coeff: float = 0.03,
        tool_penalty_start_step: int = 84,
        latency_penalty_start_step: int = 0,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.tool_penalty_coeff = tool_penalty_coeff
        self.latency_penalty_coeff = latency_penalty_coeff
        self.tool_keepalive_coeff = float(kwargs.get("tool_keepalive_coeff", self.tool_penalty_coeff / 3.0))
        self.tool_rescue_coeff = float(kwargs.get("tool_rescue_coeff", self.tool_penalty_coeff))
        self.no_tool_efficiency_coeff = float(
            kwargs.get("no_tool_efficiency_coeff", self.tool_penalty_coeff / 2.0)
        )
        self.tool_penalty_start_step = int(tool_penalty_start_step)
        self.latency_penalty_start_step = int(latency_penalty_start_step)
        self.visual_math_tolerance = float(kwargs.get("visual_math_tolerance", 1e-3))
        self.visual_math_relative_tolerance = float(kwargs.get("visual_math_relative_tolerance", 1e-2))
        self.enable_latency_reward = bool(kwargs.get("enable_latency_reward", True))
        self.enable_format_reward = bool(kwargs.get("enable_format_reward", True))
        self.skill_reward_enabled = bool(kwargs.get("skill_reward_enabled", True))
        self.skill_reward_end_step = int(kwargs.get("skill_reward_end_step", DEFAULT_SKILL_REWARD_END_STEP))
        self._llm_judge_base_url_cache: Optional[str] = None

        # Do not load skill names in __init__.
        # RewardManager is created once before training and will not be re-initialized.
        # Skill names can change per batch and must be loaded in __call__.

    def __call__(self, data: DataProto, return_dict: bool = False):
        # Prefer skill_names from batch.meta_info (injected by ray_trainer.py before compute_reward),
        # which guarantees exact consistency with the current injected batch skills.
        # Fallback to reading from skill storage if meta_info does not provide it.
        skill_names = data.meta_info.get("skill_names")
        if skill_names is None:
            skill_names = _load_active_skill_names()
        active_skill_names = set(skill_names)
        dynamic_skill_names = active_skill_names & _load_dynamic_skill_names_from_index()
        global_steps_raw = data.meta_info.get("global_steps", 0)
        try:
            global_steps = int(global_steps_raw)
        except (TypeError, ValueError):
            global_steps = 0

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_printed: Dict[str, int] = {}
        samples: List[SampleInfo] = []
        token_positions: List[Tuple[int, int]] = []

        for i in range(len(data)):
            sample, token_pos = self._extract_sample_info(data[i], i)
            samples.append(sample)
            token_positions.append(token_pos)

        # batch_stats:
        # - Trigger: runs every time a batch enters reward computation.
        # - Scope: only correct samples contribute to latency normalization.
        # - Meaning: aggregate min/max total_latency by uid for later latency reward computation.
        batch_stats = self._compute_batch_stats(samples)

        # batch_md_usage / batch_run_usage:
        # - Trigger: only tool calls from samples with sample.is_correct == True are counted.
        # - Filters: skip calls with invalid_reason; count only calls targeting active skill_names.
        # - Meaning:
        #   1) batch_md_usage: direct calls where tool_name is the skill name
        #      (excluding FOREVER_TOOLS and create_skill).
        #   2) batch_run_usage: calls with tool_name == run_skill and skill_name in active skill_names.
        batch_md_usage, batch_run_usage = self._compute_batch_skill_usage_stats(samples, active_skill_names)

        # batch_skill_usage:
        # - Trigger: computed after batch_md_usage / batch_run_usage.
        # - Meaning: merged per-skill counts from direct-skill calls and run_skill calls.
        batch_skill_usage = self._merge_skill_usage(batch_md_usage, batch_run_usage)

        # batch_new_skills:
        # - Trigger: count only calls with sample.is_correct == True and tool_name == create_skill.
        # - Filters: skip records with invalid_reason or missing skill_name/skill_path.
        # - Dedup key: created_trajectory_id + created_turn + skill_path.
        # - Meaning: metadata set of valid newly created skills within this batch.
        batch_new_skills = self._compute_batch_new_skill_info(samples)

        # batch_skill_monitoring_metrics:
        # - Scan all tool calls in this batch; no sample.is_correct filter is applied here.
        # - active_*: calls targeting active_skill_names (direct skill call or run_skill target).
        # - new_*: calls targeting new_skill_names, where new_skill_names comes from batch_new_skills.
        # - batch_new_skills is built from create_skill calls on correct samples only,
        #   and keeps only valid records (no invalid_reason + non-empty skill_name/skill_path).
        # - Therefore, batch_new_run_skill_success_calls + batch_new_run_skill_unsuccess_calls
        #   is a subset of run_skill usage, not the total run_skill call count.
        batch_skill_monitoring_metrics = self._compute_batch_skill_monitoring_metrics(
            samples=samples,
            active_skill_names=active_skill_names,
            batch_new_skills=batch_new_skills,
        )
        batch_tool_rollout_logs = self._compute_batch_tool_rollout_logs(samples, active_skill_names)

        for i, sample in enumerate(samples):
            (
                total_reward,
                accuracy_reward,
                tool_penalty,
                latency_penalty,
                format_reward,
                skill_reward,
                skill_valid_create_reward,
                skill_reused_reward,
                skill_valid_create_hit,
                skill_reused_hit,
            ) = self._compute_reward(
                sample,
                batch_stats,
                samples,
                dynamic_skill_names=dynamic_skill_names,
                global_steps=global_steps,
            )

            reward_extra_info["accuracy"].append(float(accuracy_reward))
            reward_extra_info["is_correct"].append(float(sample.is_correct))
            reward_extra_info["tool_penalty"].append(float(tool_penalty))
            reward_extra_info["latency_penalty"].append(float(latency_penalty))
            reward_extra_info["format_reward"].append(float(format_reward))
            reward_extra_info["skill_reward"].append(float(skill_reward))
            reward_extra_info["skill_valid_create_reward"].append(float(skill_valid_create_reward))
            reward_extra_info["skill_reused_reward"].append(float(skill_reused_reward))
            reward_extra_info["skill_valid_create_hit"].append(float(skill_valid_create_hit))
            reward_extra_info["skill_reused_hit"].append(float(skill_reused_hit))
            reward_extra_info["total_reward"].append(float(total_reward))
            reward_extra_info["uid"].append(sample.uid)

            if already_printed.get(sample.data_source, 0) < self.num_examine:
                already_printed[sample.data_source] = already_printed.get(sample.data_source, 0) + 1
                print(f"\n[Sample {i}] data_source={sample.data_source} uid={sample.uid}")
                print(f"[response] {sample.response_str[:400]}...")
                print(f"[ground_truth] {sample.ground_truth}")
                print(
                    f"[accuracy] {accuracy_reward:.4f}, [tool_penalty] {tool_penalty:.4f}, "
                    f"[latency_penalty] {latency_penalty:.4f}, "
                    f"[format_reward] {format_reward:.4f}, "
                    f"[skill_reward] {skill_reward:.4f}, "
                    f"[total] {total_reward:.4f}"
                )

        # Add batch-level skill usage metrics to reward_extra_info.
        reward_extra_info["batch_skill_usage"] = batch_skill_usage
        reward_extra_info["batch_skill_md_usage"] = batch_md_usage
        reward_extra_info["batch_skill_run_usage"] = batch_run_usage
        reward_extra_info["batch_new_skills"] = batch_new_skills
        reward_extra_info["batch_skill_monitoring_metrics"] = batch_skill_monitoring_metrics
        reward_extra_info["batch_tool_rollout_logs"] = batch_tool_rollout_logs
        # Keep six individual keys for downstream compatibility.
        for key, value in batch_skill_monitoring_metrics.items():
            reward_extra_info[key] = int(value)

        self._cleanup_invalid_created_skills(samples)

        for (idx, pos), reward in zip(token_positions, reward_extra_info["total_reward"]):
            reward_tensor[idx, pos] = reward

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(sorted(reward_extra_info.items()))}
        return reward_tensor

    def _extract_sample_info(self, data_item: DataProto, index: int) -> Tuple[SampleInfo, Tuple[int, int]]:
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        prompt_attention_mask = data_item.batch["attention_mask"][:prompt_length]
        valid_prompt_ids = prompt_ids[prompt_attention_mask == 1]
        prompt_text_decoded = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
        response_ids = data_item.batch["responses"]
        response_attention_mask = data_item.batch["attention_mask"][prompt_length:]
        valid_response_ids = response_ids[response_attention_mask == 1]
        valid_response_length = int((response_attention_mask == 1).sum())
        response_text_decoded = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

        data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")
        uid = str(data_item.non_tensor_batch.get("uid", f"sample-{index}"))

        tool_interact_info = data_item.non_tensor_batch.get("tool_interact_info", None)
        tool_calls_info = self._extract_tool_calls(tool_interact_info)
        format_messages = data_item.non_tensor_batch.get("format_messages", None)
        format_reward, _ = compute_format_reward(format_messages)

        answer_text = self._extract_answer_text(response_str)
        answer_judgment = self._evaluate_answer_correctness(answer_text, ground_truth, data_item)

        sample = SampleInfo(
            uid=uid,
            data_source=data_source,
            prompt_text_decoded=prompt_text_decoded,
            response_text_decoded=response_text_decoded,
            response_str=response_str,
            ground_truth=ground_truth,
            tool_calls_info=tool_calls_info,
            is_correct=bool(answer_judgment["is_correct"]),
            format_reward=float(format_reward),
            judge_correct_raw=bool(answer_judgment["judge_correct_raw"]),
            judge_final_source=str(answer_judgment["judge_final_source"]),
            judge_reason=str(answer_judgment["judge_reason"]),
            judge_llm_reason=str(answer_judgment["judge_llm_reason"]),
            judge_allow_numeric_tolerance=bool(answer_judgment["judge_allow_numeric_tolerance"]),
            judge_parse_status=str(answer_judgment["judge_parse_status"]),
            judge_api_status=str(answer_judgment["judge_api_status"]),
            judge_raw_response=str(answer_judgment["judge_raw_response"]),
            judge_numeric_extraction_answer=str(answer_judgment["judge_numeric_extraction_answer"]),
            judge_numeric_extraction_reason=str(answer_judgment["judge_numeric_extraction_reason"]),
            judge_numeric_extraction_parse_status=str(answer_judgment["judge_numeric_extraction_parse_status"]),
            judge_numeric_extraction_api_status=str(answer_judgment["judge_numeric_extraction_api_status"]),
            judge_numeric_extraction_raw_response=str(answer_judgment["judge_numeric_extraction_raw_response"]),
        )
        return sample, (index, int(valid_response_length - 1))

    def _compute_batch_stats(self, samples: List[SampleInfo]) -> BatchStats:
        grouped = self._group_samples_by_uid(samples)
        latency_min: Dict[str, float] = {}
        latency_max: Dict[str, float] = {}

        for uid, sample_list in grouped.items():
            correct_tool_samples = [s for s in sample_list if s.is_correct and s.tool_calls_info.tool_calls]
            latency_vals = [float(s.tool_calls_info.total_latency) for s in correct_tool_samples]
            latency_min[uid] = min(latency_vals) if latency_vals else 0.0
            latency_max[uid] = max(latency_vals) if latency_vals else 0.0

        return BatchStats(latency_min=latency_min, latency_max=latency_max)

    def _group_samples_by_uid(self, samples: List[SampleInfo]) -> Dict[str, List[SampleInfo]]:
        grouped: Dict[str, List[SampleInfo]] = defaultdict(list)
        for sample in samples:
            grouped[sample.uid].append(sample)
        return dict(grouped)

    def _compute_tool_penalty(
        self,
        samples: List[SampleInfo],
        sample: SampleInfo,
    ) -> float:
        """
        Compute tool reward under the legacy tool_penalty name.

        For samples with the same uid:
        1. Correct tool usage always receives a small keepalive reward.
        2. Correct tool usage receives an extra rescue reward when a no-tool
           trajectory for the same uid is incorrect.
        3. Correct no-tool usage receives a mild efficiency reward when a
           correct tool trajectory for the same uid also exists.
        4. Incorrect samples do not receive this reward.
        """
        uid = sample.uid
        uid_samples = [s for s in samples if s.uid == uid]
        used_tool = bool(sample.tool_calls_info.tool_calls)
        has_failed_no_tool_sample = any(
            (not s.tool_calls_info.tool_calls) and (not s.is_correct) for s in uid_samples
        )
        has_correct_tool_sample = any(bool(s.tool_calls_info.tool_calls) and s.is_correct for s in uid_samples)

        if not sample.is_correct:
            return 0.0

        if used_tool:
            reward = self.tool_keepalive_coeff
            if has_failed_no_tool_sample:
                reward += self.tool_rescue_coeff
            return reward

        if has_correct_tool_sample:
            return self.no_tool_efficiency_coeff

        return 0.0

    def _compute_latency_penalty(
        self,
        value: float,
        batch_stats: BatchStats,
        uid: str,
        is_correct: bool,
        used_tool: bool,
    ) -> float:
        """
        Compute latency reward under the legacy latency_penalty name.

        Only correct tool-usage trajectories receive a reward, normalized within
        the uid's correct-tool set so that lower latency gets a higher value.
        """
        if not is_correct or not used_tool:
            return 0.0

        min_val = batch_stats.latency_min.get(uid, 0.0)
        max_val = batch_stats.latency_max.get(uid, 0.0)

        if max_val <= min_val:
            return 0.0

        normalized = (max_val - value) / (max_val - min_val)
        return self.latency_penalty_coeff * normalized

    def _compute_accuracy_reward(self, is_correct: bool) -> float:
        """Compute accuracy reward. Return 1.0 if correct, otherwise 0.0."""
        return 1.0 if is_correct else 0.0

    def _compute_reward(
        self,
        sample: SampleInfo,
        batch_stats: BatchStats,
        samples: List[SampleInfo],
        dynamic_skill_names: set,
        global_steps: int,
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        """
        Compute total reward for one sample.

        Total reward = accuracy reward + tool penalty + latency penalty + format reward + skill reward

        Note: tool_penalty and latency_penalty keep their historical names but behave as rewards.
        """
        accuracy_reward = self._compute_accuracy_reward(sample.is_correct)
        tool_penalty = 0.0
        if global_steps >= self.tool_penalty_start_step:
            tool_penalty = self._compute_tool_penalty(samples, sample)
        latency_penalty = 0.0
        if self.enable_latency_reward and global_steps >= self.latency_penalty_start_step:
            latency_penalty = self._compute_latency_penalty(
                sample.tool_calls_info.total_latency,
                batch_stats,
                sample.uid,
                sample.is_correct,
                bool(sample.tool_calls_info.tool_calls),
            )
        format_reward = float(sample.format_reward) if self.enable_format_reward else 0.0
        (
            skill_reward,
            skill_valid_create_reward,
            skill_reused_reward,
            skill_valid_create_hit,
            skill_reused_hit,
        ) = self._compute_skill_reward(
            sample=sample,
            dynamic_skill_names=dynamic_skill_names,
            global_steps=global_steps,
        )

        total_reward = accuracy_reward + tool_penalty + latency_penalty + format_reward + skill_reward
        return (
            total_reward,
            accuracy_reward,
            tool_penalty,
            latency_penalty,
            format_reward,
            skill_reward,
            skill_valid_create_reward,
            skill_reused_reward,
            skill_valid_create_hit,
            skill_reused_hit,
        )

    def _skill_reward_is_active(self, global_steps: int) -> bool:
        return self.skill_reward_enabled and global_steps < self.skill_reward_end_step

    def _is_valid_create_tool_call(self, tool_call: Dict[str, Any]) -> bool:
        if str(tool_call.get("name", "")) != CREATE_SKILL_TOOL_NAME:
            return False
        if tool_call.get("invalid_reason"):
            return False
        skill_name = tool_call.get("skill_name")
        skill_path = tool_call.get("skill_path")
        return bool(skill_name and skill_path)

    def _tool_obs_has_error(self, obs: Any) -> bool:
        if obs is None:
            return False
        if isinstance(obs, str):
            obs_text = obs
        else:
            try:
                obs_text = json.dumps(obs, ensure_ascii=False)
            except Exception:
                obs_text = str(obs)
        lowered = obs_text.lower()
        return any(marker in lowered for marker in RUN_SKILL_OBS_ERROR_MARKERS)

    def _tool_call_effective_success(self, tool_call: Dict[str, Any]) -> bool:
        if tool_call.get("invalid_reason"):
            return False
        if self._tool_obs_has_error(tool_call.get("obs")):
            return False
        return True

    def _is_successful_dynamic_skill_reuse(
        self,
        tool_call: Dict[str, Any],
        dynamic_skill_names: set,
    ) -> bool:
        if not self._tool_call_effective_success(tool_call):
            return False

        tool_name = str(tool_call.get("name", "")).strip()
        if not tool_name:
            return False

        if tool_name == RUN_SKILL_TOOL_NAME:
            skill_name = str(tool_call.get("skill_name", "")).strip()
            return bool(skill_name and skill_name in dynamic_skill_names)

        if tool_name == CREATE_SKILL_TOOL_NAME or tool_name in FOREVER_TOOLS:
            return False

        return tool_name in dynamic_skill_names

    def _compute_skill_reward(
        self,
        sample: SampleInfo,
        dynamic_skill_names: set,
        global_steps: int,
    ) -> Tuple[float, float, float, float, float]:
        if not self._skill_reward_is_active(global_steps) or not sample.is_correct:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        valid_create_hit = 0.0
        reused_hit = 0.0

        for tool_call in sample.tool_calls_info.tool_calls:
            if self._is_valid_create_tool_call(tool_call):
                valid_create_hit = 1.0
            if self._is_successful_dynamic_skill_reuse(tool_call, dynamic_skill_names):
                reused_hit = 1.0
            if valid_create_hit and reused_hit:
                break

        valid_create_reward = VALID_CREATE_REWARD * valid_create_hit
        reused_reward = REUSED_SKILL_REWARD * reused_hit
        skill_reward = valid_create_reward + reused_reward
        return skill_reward, valid_create_reward, reused_reward, valid_create_hit, reused_hit

    def _extract_tool_calls(self, tool_interact_info: Any) -> ToolCallSummary:
        tool_items = self._normalize_tool_items(tool_interact_info)
        info = ToolCallSummary()

        for item in tool_items:
            tool_name = item.get("tool") or item.get("name") or ""
            tool_model = item.get("model")
            tool_prompt = item.get("prompt")
            tool_obs = item.get("obs")
            invalid_reason = item.get("invalid_reason")
            skill_path = item.get("skill_path")
            created_skill_name = item.get("skill_name")
            created_trajectory_id = item.get("created_trajectory_id")
            created_turn = item.get("created_turn")
            if tool_name:
                info.tool_calls.append(
                    {
                        "name": tool_name,
                        "model": tool_model,
                        "prompt": tool_prompt,
                        "obs": tool_obs,
                        "invalid_reason": invalid_reason,
                        "skill_path": skill_path,
                        "skill_name": created_skill_name,
                        "created_trajectory_id": created_trajectory_id,
                        "created_turn": created_turn,
                    }
                )
                info.tool_counts[tool_name] = info.tool_counts.get(tool_name, 0) + 1

            latency = item.get("latency")
            if isinstance(latency, (int, float)):
                info.total_latency += float(latency)

        return info

    def _compute_batch_skill_usage_stats(
        self, samples: List[SampleInfo], skill_names: set
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Count skill calls (direct skill calls + run_skill calls) on correct samples only."""
        batch_md_counts: Dict[str, int] = defaultdict(int)
        batch_run_counts: Dict[str, int] = defaultdict(int)

        for sample in samples:
            if not sample.is_correct:
                continue
            for tool_call in sample.tool_calls_info.tool_calls:
                tool_name = tool_call.get("name", "")
                invalid_reason = tool_call.get("invalid_reason")
                if invalid_reason:
                    continue

                if tool_name == RUN_SKILL_TOOL_NAME:
                    skill_name = tool_call.get("skill_name")
                    if skill_name and skill_name in skill_names:
                        batch_run_counts[str(skill_name)] += 1
                    continue

                if tool_name in FOREVER_TOOLS or tool_name == CREATE_SKILL_TOOL_NAME:
                    continue
                if tool_name in skill_names:
                    batch_md_counts[tool_name] += 1

        return dict(batch_md_counts), dict(batch_run_counts)

    def _merge_skill_usage(
        self, batch_md_usage: Dict[str, int], batch_run_usage: Dict[str, int]
    ) -> Dict[str, int]:
        merged: Dict[str, int] = {}
        for name, count in batch_md_usage.items():
            merged[name] = merged.get(name, 0) + int(count)
        for name, count in batch_run_usage.items():
            merged[name] = merged.get(name, 0) + int(count)
        return merged

    def _compute_batch_new_skill_info(self, samples: List[SampleInfo]) -> Dict[str, Dict[str, Any]]:
        """Collect metadata for newly created skills from create_skill (correct samples only, deduped by creation key)."""
        new_skill_map: Dict[str, Dict[str, Any]] = {}
        for sample in samples:
            if not sample.is_correct:
                continue
            for tool_call in sample.tool_calls_info.tool_calls:
                tool_name = tool_call.get("name", "")
                invalid_reason = tool_call.get("invalid_reason")
                if tool_name != CREATE_SKILL_TOOL_NAME:
                    continue
                if invalid_reason:
                    continue
                skill_name = tool_call.get("skill_name")
                skill_path = tool_call.get("skill_path")
                if not skill_name or not skill_path:
                    continue
                created_trajectory_id = tool_call.get("created_trajectory_id")
                created_turn = tool_call.get("created_turn")
                unique_key = f"{created_trajectory_id}:{created_turn}:{skill_path}"
                if unique_key in new_skill_map:
                    continue
                new_skill_map[unique_key] = {
                    "skill_name": str(skill_name),
                    "path": skill_path,
                    "created_trajectory_id": created_trajectory_id,
                    "created_turn": created_turn,
                }
        return new_skill_map

    def _compute_batch_skill_monitoring_metrics(
        self,
        samples: List[SampleInfo],
        active_skill_names: set,
        batch_new_skills: Dict[str, Dict[str, Any]],
    ) -> Dict[str, int]:
        """Compute per-batch monitoring metrics for active/new skill calls.

        Counting scope:
        - Scan every tool call in every sample; sample.is_correct is not used here.
        - active_* metrics count calls targeting active_skill_names:
          (1) direct skill call (tool_name in active_skill_names), or
          (2) run_skill where skill_name in active_skill_names.
        - new_* metrics count calls targeting new_skill_names.
        - new_skill_names is derived from batch_new_skills.
        - batch_new_skills is produced by _compute_batch_new_skill_info, which only
          keeps create_skill records from correct samples that are valid
          (no invalid_reason and with skill_name/skill_path present).

        Implication:
        - batch_new_run_skill_success_calls and batch_new_run_skill_unsuccess_calls
          are filtered subset counts of run_skill calls, not total run_skill usage.
        """
        new_skill_names = {
            str(info.get("skill_name"))
            for info in batch_new_skills.values()
            if isinstance(info, dict) and info.get("skill_name")
        }

        metrics = {
            "batch_active_skill_calls": 0,
            "batch_active_run_skill_success_calls": 0,
            "batch_active_run_skill_unsuccess_calls": 0,
            "batch_new_skill_calls": 0,
            "batch_new_run_skill_success_calls": 0,
            "batch_new_run_skill_unsuccess_calls": 0,
        }

        for sample in samples:
            for tool_call in sample.tool_calls_info.tool_calls:
                tool_name = str(tool_call.get("name", ""))
                invalid_reason = tool_call.get("invalid_reason")
                is_run = tool_name == RUN_SKILL_TOOL_NAME
                run_skill_name = str(tool_call.get("skill_name", "")) if is_run else ""
                is_success = self._tool_call_effective_success(tool_call) if is_run else not invalid_reason

                is_active_md = tool_name in active_skill_names
                is_active_run = is_run and run_skill_name in active_skill_names
                is_new_md = tool_name in new_skill_names
                is_new_run = is_run and run_skill_name in new_skill_names

                if is_active_md or is_active_run:
                    metrics["batch_active_skill_calls"] += 1
                if is_new_md or is_new_run:
                    metrics["batch_new_skill_calls"] += 1

                if is_active_run:
                    if is_success:
                        metrics["batch_active_run_skill_success_calls"] += 1
                    else:
                        metrics["batch_active_run_skill_unsuccess_calls"] += 1

                if is_new_run:
                    if is_success:
                        metrics["batch_new_run_skill_success_calls"] += 1
                    else:
                        metrics["batch_new_run_skill_unsuccess_calls"] += 1

        return metrics

    def _safe_json_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._safe_json_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._safe_json_value(v) for k, v in value.items()}
        return str(value)

    def _tool_kind(self, tool_name: str, skill_names: set) -> str:
        if tool_name in FOREVER_TOOLS:
            return "forever_tool"
        if tool_name == CREATE_SKILL_TOOL_NAME:
            return "create_skill"
        if tool_name == RUN_SKILL_TOOL_NAME:
            return "run_skill"
        if tool_name in skill_names:
            return "skill_direct"
        return "other_tool"

    def _compute_batch_tool_rollout_logs(
        self,
        samples: List[SampleInfo],
        skill_names: set,
    ) -> Dict[str, Dict[str, Any]]:
        per_tool_logs: Dict[str, Dict[str, Any]] = {}

        for sample in samples:
            for tool_call in sample.tool_calls_info.tool_calls:
                tool_name = str(tool_call.get("name", "")).strip()
                if not tool_name:
                    continue

                entry = per_tool_logs.setdefault(
                    tool_name,
                    {
                        "tool_name": tool_name,
                        "tool_kind": self._tool_kind(tool_name, skill_names),
                        "used_count": 0,
                        "rollouts": [],
                    },
                )
                entry["used_count"] += 1

                entry["rollouts"].append(
                    {
                        "uid": str(sample.uid),
                        "data_source": str(sample.data_source),
                        "is_correct": bool(sample.is_correct),
                        "prompt_text_decoded": sample.prompt_text_decoded,
                        "response_text_decoded": sample.response_text_decoded,
                        "ground_truth": self._safe_json_value(sample.ground_truth),
                        "format_reward": float(sample.format_reward),
                        "tool_name": tool_name,
                        "tool_kind": entry["tool_kind"],
                        "invalid_reason": self._safe_json_value(tool_call.get("invalid_reason")),
                        "tool_obs_has_error": self._tool_obs_has_error(tool_call.get("obs")),
                        "tool_effective_success": self._tool_call_effective_success(tool_call),
                        "tool_prompt": self._safe_json_value(tool_call.get("prompt")),
                        "tool_obs": self._safe_json_value(tool_call.get("obs")),
                        "tool_model": self._safe_json_value(tool_call.get("model")),
                        "judge_correct_raw": bool(sample.judge_correct_raw),
                        "judge_final_source": str(sample.judge_final_source),
                        "judge_reason": str(sample.judge_reason),
                        "judge_llm_reason": str(sample.judge_llm_reason),
                        "judge_allow_numeric_tolerance": bool(sample.judge_allow_numeric_tolerance),
                        "judge_parse_status": str(sample.judge_parse_status),
                        "judge_api_status": str(sample.judge_api_status),
                        "judge_raw_response": self._truncate_for_log(sample.judge_raw_response),
                        "judge_numeric_extraction_answer": str(sample.judge_numeric_extraction_answer),
                        "judge_numeric_extraction_reason": str(sample.judge_numeric_extraction_reason),
                        "judge_numeric_extraction_parse_status": str(sample.judge_numeric_extraction_parse_status),
                        "judge_numeric_extraction_api_status": str(sample.judge_numeric_extraction_api_status),
                        "judge_numeric_extraction_raw_response": self._truncate_for_log(
                            sample.judge_numeric_extraction_raw_response
                        ),
                    }
                )

        return per_tool_logs

    def _normalize_tool_items(self, tool_interact_info: Any) -> List[Dict[str, Any]]:
        if tool_interact_info is None:
            return []
        if isinstance(tool_interact_info, dict):
            items = [tool_interact_info]
        elif isinstance(tool_interact_info, (list, tuple)):
            items = list(tool_interact_info)
        else:
            try:
                items = list(tool_interact_info.tolist())
            except Exception:
                return []
        return [item for item in items if isinstance(item, dict)]

    def _cleanup_invalid_created_skills(self, samples: List[SampleInfo]) -> None:
        """Delete skills created by incorrect samples (under the pending directory)."""
        for sample in samples:
            if sample.is_correct:
                continue
            for tool_call in sample.tool_calls_info.tool_calls:
                if tool_call.get("name") != CREATE_SKILL_TOOL_NAME:
                    continue
                skill_path = tool_call.get("skill_path")
                if not skill_path:
                    continue
                try:
                    path = Path(str(skill_path))
                    if path.exists() and path.is_dir():
                        try:
                            archive_skill_event(
                                skill_path=path,
                                event="deleted",
                                reason="incorrect_trajectory",
                                store_dir=_get_skill_store_dir(),
                            )
                        except Exception:
                            pass
                        shutil.rmtree(path, ignore_errors=True)
                except Exception:
                    continue

    def _extract_answer_text(self, response_str: str) -> str:
        answer_match = re.search(r"<answer>(.*?)</answer>", response_str, re.DOTALL)
        if not answer_match:
            return ""
        return answer_match.group(1).strip()

    def _check_answer_correctness(self, answer_text: str, ground_truth: str, data_item: DataProto) -> bool:
        return bool(self._evaluate_answer_correctness(answer_text, ground_truth, data_item)["is_correct"])

    def _evaluate_answer_correctness(
        self,
        answer_text: str,
        ground_truth: str,
        data_item: DataProto,
    ) -> Dict[str, Any]:
        if not answer_text:
            return self._finalize_answer_judgment(
                is_correct=False,
                final_source="empty_answer",
                final_reason="",
                judge_result=self._empty_llm_judge_result(
                    reason="answer_text_empty",
                    parse_status="not_requested",
                    api_status="not_requested",
                    decision_source="not_requested",
                ),
                numeric_extraction_result=self._empty_numeric_extraction_result(),
            )

        # Extract images and question from extra_info.
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        images = extra_info.get("images", [])
        question = extra_info.get("question", "")

        # Filter out irrelevant fields and keep the rest as evidence.
        excluded_keys = {"split", "index", "qid"}
        extra_fields = {k: v for k, v in extra_info.items() if k not in excluded_keys}

        judge_result = self._llm_judge_correctness(answer_text, ground_truth, question, images, extra_fields)
        if bool(judge_result.get("correct", False)):
            return self._finalize_answer_judgment(
                is_correct=True,
                final_source="llm_correct",
                final_reason="",
                judge_result=judge_result,
                numeric_extraction_result=self._empty_numeric_extraction_result(),
            )

        numeric_extraction_result = self._empty_numeric_extraction_result()
        if self._is_visual_math_item(data_item):
            numeric_extraction_result = self._llm_extract_numeric_answer(answer_text, question)
            extracted_answer = str(numeric_extraction_result.get("extracted_answer", "") or "").strip()
            if extracted_answer and self._visual_math_numeric_match(extracted_answer, ground_truth):
                final_source = (
                    "visual_math_numeric_extracted_fallback"
                    if self._judge_result_is_available(judge_result)
                    else "judge_unavailable_visual_math_numeric_extracted_fallback"
                )
                return self._finalize_answer_judgment(
                    is_correct=True,
                    final_source=final_source,
                    final_reason="",
                    judge_result=judge_result,
                    numeric_extraction_result=numeric_extraction_result,
                )

            if self._numeric_extraction_is_available(numeric_extraction_result):
                final_source = "visual_math_numeric_extraction_no_match"
            else:
                final_source = "visual_math_numeric_extraction_unavailable"
            return self._finalize_answer_judgment(
                is_correct=False,
                final_source=final_source,
                final_reason="",
                judge_result=judge_result,
                numeric_extraction_result=numeric_extraction_result,
            )

        final_source = (
            "llm_incorrect_no_fallback"
            if self._judge_result_is_available(judge_result)
            else "judge_unavailable_no_fallback"
        )
        return self._finalize_answer_judgment(
            is_correct=False,
            final_source=final_source,
            final_reason="",
            judge_result=judge_result,
            numeric_extraction_result=numeric_extraction_result,
        )

    def _finalize_answer_judgment(
        self,
        is_correct: bool,
        final_source: str,
        final_reason: str,
        judge_result: Dict[str, Any],
        numeric_extraction_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "is_correct": bool(is_correct),
            "judge_correct_raw": bool(judge_result.get("correct", False)),
            "judge_final_source": str(final_source),
            "judge_reason": "",
            "judge_llm_reason": "",
            "judge_allow_numeric_tolerance": bool(judge_result.get("allow_numeric_tolerance", False)),
            "judge_parse_status": str(judge_result.get("parse_status", "not_requested")),
            "judge_api_status": str(judge_result.get("api_status", "not_requested")),
            "judge_raw_response": str(judge_result.get("raw_response", "") or ""),
            "judge_numeric_extraction_answer": str(numeric_extraction_result.get("extracted_answer", "") or ""),
            "judge_numeric_extraction_reason": "",
            "judge_numeric_extraction_parse_status": str(
                numeric_extraction_result.get("parse_status", "not_requested")
            ),
            "judge_numeric_extraction_api_status": str(
                numeric_extraction_result.get("api_status", "not_requested")
            ),
            "judge_numeric_extraction_raw_response": str(
                numeric_extraction_result.get("raw_response", "") or ""
            ),
        }

    def _judge_result_is_available(self, judge_result: Dict[str, Any]) -> bool:
        return (
            str(judge_result.get("api_status", "")) == "success"
            and str(judge_result.get("parse_status", "")) == "parsed"
        )

    def _numeric_extraction_is_available(self, extraction_result: Dict[str, Any]) -> bool:
        return (
            str(extraction_result.get("api_status", "")) == "success"
            and str(extraction_result.get("parse_status", "")) == "parsed"
        )

    def _compose_final_judge_reason(
        self,
        final_source: str,
        judge_result: Dict[str, Any],
        numeric_extraction_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        return ""

    def _is_visual_math_item(self, data_item: DataProto) -> bool:
        ability = data_item.non_tensor_batch.get("ability", "")
        if isinstance(ability, str):
            return ability.strip().lower() == "visual_math"
        return False

    def _visual_math_numeric_match(self, answer_text: str, ground_truth: str) -> bool:
        predicted_candidates = self._extract_number_candidates(answer_text)
        expected_candidates = self._extract_number_candidates(str(ground_truth))
        if not predicted_candidates or not expected_candidates:
            return False

        tol_abs = float(self.visual_math_tolerance)
        tol_rel = float(self.visual_math_relative_tolerance)

        predicted_final = self._extract_final_candidates(predicted_candidates)
        expected_final = self._extract_final_candidates(expected_candidates)

        # If both sides provide explicit final-answer candidates,
        # only compare those to avoid matching intermediate values.
        if predicted_final and expected_final:
            return self._has_close_candidate_pair(predicted_final, expected_final, tol_abs, tol_rel)

        # Fallback for loosely formatted answers where one side has no final marker.
        return self._has_close_candidate_pair(predicted_candidates, expected_candidates, tol_abs, tol_rel)

    def _extract_numbers(self, text: str) -> List[float]:
        return [val for val, _ in self._extract_number_candidates(text)]

    def _extract_number_candidates(self, text: str) -> List[Tuple[float, int]]:
        normalized = self._normalize_numeric_text(text)
        number_pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
        pi_token_pattern = r"(?<![0-9A-Za-z_])(?:pi|π)(?![0-9A-Za-z_])"
        candidates: List[Tuple[float, int]] = []
        expression_candidates: List[Tuple[int, int, float]] = []

        def add_candidate(raw: str, priority: int) -> None:
            try:
                val = float(raw)
            except (TypeError, ValueError):
                return
            add_value_candidate(val, priority)

        def add_value_candidate(val: float, priority: int) -> None:
            if math.isfinite(val):
                candidates.append((val, priority))

        def add_expression_candidate(end_idx: int, value: float, kind_priority: int) -> None:
            if math.isfinite(value):
                expression_candidates.append((end_idx, kind_priority, value))

        for match in re.finditer(rf"({number_pattern})\s*%", normalized):
            try:
                raw_val = float(match.group(1))
            except (TypeError, ValueError):
                continue
            add_value_candidate(raw_val, 3)
            add_value_candidate(raw_val / 100.0, 3)
            add_expression_candidate(match.end(), raw_val / 100.0, 3)

        for match in re.finditer(r"(?<![0-9A-Za-z_.])([-+]?\d+)\s*/\s*([-+]?\d+)(?![0-9A-Za-z_.])", normalized):
            try:
                numerator = float(match.group(1))
                denominator = float(match.group(2))
            except (TypeError, ValueError):
                continue
            if denominator != 0:
                value = numerator / denominator
                add_value_candidate(value, 3)
                add_expression_candidate(match.end(), value, 3)

        for match in re.finditer(rf"=\s*({number_pattern})(?!\s*[%/])\b", normalized):
            add_candidate(match.group(1), 1)
            try:
                rhs_val = float(match.group(1))
            except (TypeError, ValueError):
                continue
            add_expression_candidate(match.end(), rhs_val, 4)

        for match in re.finditer(r"=\s*([-+]?\d+)\s*/\s*([-+]?\d+)(?![0-9A-Za-z_.])", normalized):
            try:
                numerator = float(match.group(1))
                denominator = float(match.group(2))
            except (TypeError, ValueError):
                continue
            if denominator != 0:
                rhs_fraction = numerator / denominator
                add_value_candidate(rhs_fraction, 1)
                add_expression_candidate(match.end(), rhs_fraction, 4)

        for match in re.finditer(rf"=\s*({number_pattern})\s*%", normalized):
            try:
                rhs_percent = float(match.group(1)) / 100.0
            except (TypeError, ValueError):
                continue
            add_value_candidate(rhs_percent, 1)
            add_expression_candidate(match.end(), rhs_percent, 4)

        for match in re.finditer(rf"=\s*{pi_token_pattern}", normalized, flags=re.IGNORECASE):
            add_value_candidate(math.pi, 1)
            add_expression_candidate(match.end(), math.pi, 4)

        for match in re.finditer(pi_token_pattern, normalized, flags=re.IGNORECASE):
            add_value_candidate(math.pi, 4)
            add_expression_candidate(match.end(), math.pi, 2)

        plain_matches = list(re.finditer(number_pattern, normalized))
        for match in plain_matches:
            add_candidate(match.group(0), 4)
            try:
                plain_val = float(match.group(0))
            except (TypeError, ValueError):
                continue
            add_expression_candidate(match.end(), plain_val, 1)

        if expression_candidates:
            _, _, final_expr_value = max(expression_candidates, key=lambda item: (item[0], item[1]))
            add_value_candidate(final_expr_value, 0)

        return self._deduplicate_candidates(candidates)

    def _normalize_numeric_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", str(text))
        normalized = normalized.replace("−", "-").replace("–", "-").replace("—", "-").replace("﹣", "-")
        normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
        return normalized

    def _deduplicate_candidates(self, candidates: List[Tuple[float, int]]) -> List[Tuple[float, int]]:
        best_priority_by_value: Dict[float, int] = {}
        for value, priority in candidates:
            key = round(value, 12)
            prev = best_priority_by_value.get(key)
            if prev is None or priority < prev:
                best_priority_by_value[key] = priority

        deduped = [(value, priority) for value, priority in best_priority_by_value.items()]
        deduped.sort(key=lambda item: (item[1], item[0]))
        return deduped

    def _is_close_number(self, lhs: float, rhs: float, abs_tol: float, rel_tol: float) -> bool:
        diff = abs(lhs - rhs)
        if diff <= abs_tol:
            return True
        scale = max(abs(lhs), abs(rhs), 1.0)
        return diff <= rel_tol * scale

    def _extract_final_candidates(self, candidates: List[Tuple[float, int]]) -> List[Tuple[float, int]]:
        return [item for item in candidates if item[1] <= 1]

    def _has_close_candidate_pair(
        self,
        lhs_candidates: List[Tuple[float, int]],
        rhs_candidates: List[Tuple[float, int]],
        abs_tol: float,
        rel_tol: float,
    ) -> bool:
        for lhs, _ in lhs_candidates:
            for rhs, _ in rhs_candidates:
                if self._is_close_number(lhs, rhs, abs_tol, rel_tol):
                    return True
        return False

    def _llm_judge_correctness(
        self,
        answer_text: str,
        ground_truth: str,
        question: str,
        images: List[str],
        extra_info: Dict,
    ) -> Dict[str, Any]:
        """
        Use a multimodal LLM to judge answer correctness.

        The judgment considers image(s), question, model answer, and reference answer together.
        """
        base_url = self._load_llm_judge_base_url()
        if not base_url:
            return self._empty_llm_judge_result(
                reason="judge_base_url_missing",
                api_status="config_missing",
                decision_source="judge_unavailable",
            )

        # Build API endpoint.
        api_url = base_url.rstrip("/") + "/v1/chat/completions"

        # Build text prompt.
        extra_str = ""
        if extra_info:
            # Exclude images and question (already provided in dedicated fields).
            extra_for_display = {k: v for k, v in extra_info.items() if k not in ["images", "question"]}
            if extra_for_display:
                extra_str = "\nAdditional context: " + json.dumps(extra_for_display, ensure_ascii=False)

        prompt = f"""Model answer: {answer_text}
                    Reference answer: {ground_truth}
                    Question: {question}{extra_str}"""

        # Build system prompt.
        system_prompt = """You are an answer-judging expert. Evaluate whether the model answer is correct based on the model answer, the reference answer, the question, and the image(s).

## Important Context About Images

- The answering model may have used transformed or intermediate images during reasoning, such as cropped, zoomed, enhanced, rearranged, or otherwise processed versions of the original image.
- Those intermediate images are not saved and are NOT available to you during judging.
- Therefore, the image(s) you see now may not fully match the visual evidence that the answering model actually used when producing its answer.
- Treat the model answer as the primary evidence.
- Treat the provided image(s) only as auxiliary evidence.
- Do NOT mark an answer incorrect only because the current image(s) do not clearly show a detail mentioned in the answer.
- If the model answer matches the main information in the reference answer, prefer `correct = true`, even if some supporting visual details are not visible in the currently provided image(s).
- Only mark the answer incorrect when it clearly misses or contradicts the core information in the reference answer, or when it is in clear conflict with the visible evidence.

## Judgment Criteria

**Objective questions** (e.g., numerical computation, true/false with clear answers):
- The main criterion is whether the model answer captures the essential content of the reference answer.
- If the model answer gives the correct main conclusion or core fact, mark it correct even if it includes extra details not directly visible in the current image(s).
- Unit differences are acceptable only if the numeric value is equivalent.
- If the question is a numeric/computation/conversion/comparison task where small rounding error is acceptable, set `allow_numeric_tolerance` to true when the answer is numerically close.
- If the question requires exact factual value (e.g., exact year/date/count/identifier), set `allow_numeric_tolerance` to false.
- For image-based ranking/comparison questions, first judge whether the image actually supports a unique exact answer.
- If the image does not expose exact values and only shows coarse visual cues such as colors, shades, heatmaps, approximate bar heights, blurred labels, or low-resolution regions, do not require stricter certainty than the image supports.
- If the reference answer contains multiple acceptable candidates, and the model answer matches at least one of them, mark it correct when that candidate is consistent with the image and the image does not reliably distinguish a single unique winner.
- This relaxed judgment is appropriate only when ambiguity comes from the visual evidence itself, not when the model answer contradicts clearly visible evidence.
- Do not relax if the image provides enough detail to identify a unique answer.

**Multiple-choice questions**:
- If the question or additional context includes explicit answer options, use those options when judging answer equivalence.
- Treat an option label and the content of that option as equivalent answers.
- Apply this equivalence in both directions: if the model answer is the option label and the reference answer is the option content, or if the model answer is the option content and the reference answer is the option label, mark it correct when they map to the same option.
- Accept minor label-format variants such as `B`, `(B)`, `B.`, `Option B`, or `choice B` when they clearly refer to the same option.
- Mark it incorrect only when the chosen option maps to different content from the reference answer, or otherwise contradicts the question or visible evidence.

**Subjective questions** (descriptive or explanatory):
- Core criterion: whether the model answer correctly addresses the question.
- Core semantic agreement with the reference answer is sufficient.
- Extra wording in the reference answer should not affect correctness.
- The model answer may be shorter, as long as key information is correct.
- If the model answer captures the main meaning of the reference answer, prefer `correct = true` even when some visual support is unavailable in the currently provided image(s).

## Special Care For Ambiguous Visual QA

- Some map/chart questions ask for the state/region/category with the highest or lowest value, but the figure may encode value only by color rather than explicit numbers.
- In such cases, the reference answer may list several plausible answers because multiple regions look tied or near-tied from the image alone.
- If the model gives one plausible candidate that appears among the acceptable reference answers, and the image does not allow confident disambiguation to a single exact choice, prefer `correct = true`.
- If the model picks a candidate outside the acceptable reference answers, or the image clearly indicates a different unique choice, mark it incorrect.

## Output Format
Return strict JSON only:
{"correct": true, "allow_numeric_tolerance": false}
or
{"correct": false, "allow_numeric_tolerance": true}"""

        # Build multimodal content.
        content = []

        # Add images.
        for image_path in images:
            try:
                with open(image_path, "rb") as f:
                    image_base64 = base64.b64encode(f.read()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                })
            except Exception as e:
                print(f"[LLM Judge] Failed to load image {image_path}: {e}")

        # Add text prompt.
        content.append({
            "type": "text",
            "text": prompt
        })

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer token-abc123"
        }

        payload = {
            "model": "Qwen3-VL-235B-A22B-Instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            "max_tokens": 120,
            "temperature": 0.0,
            "stream": False
        }

        # Retry policy: at most 2 retries.
        max_retries = 2
        last_api_status = "not_attempted"
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=30,
                    verify=False
                )

                if response.status_code == 200:
                    last_api_status = "success"
                    data = response.json()
                    content_response = self._extract_chat_message_text(
                        data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    )

                    result = self._parse_llm_judge_response(content_response)
                    if result is not None:
                        result["raw_response"] = content_response
                        result["api_status"] = "success"
                        print(
                            f"[LLM Judge] Answer: '{answer_text[:50]}...', GT: '{ground_truth[:50]}...', "
                            f"Result: correct={result['correct']}, "
                            f"allow_numeric_tolerance={result['allow_numeric_tolerance']}"
                        )
                        return result

                    print(f"[LLM Judge] Failed to parse response: {content_response}")
                    return self._empty_llm_judge_result(
                        reason="llm_response_parse_failed",
                        raw_response=content_response,
                        parse_status="parse_failed",
                        api_status="success",
                        decision_source="parse_failed",
                    )
                else:
                    last_api_status = f"http_{response.status_code}"
                    print(f"[LLM Judge] API error (attempt {attempt + 1}/{max_retries + 1}): {response.status_code}")
            except requests.exceptions.Timeout:
                last_api_status = "timeout"
                print(f"[LLM Judge] Timeout (attempt {attempt + 1}/{max_retries + 1})")
            except Exception as e:
                last_api_status = "exception"
                print(f"[LLM Judge] Exception (attempt {attempt + 1}/{max_retries + 1}): {e}")

        # All retries failed; return conservative False.
        return self._empty_llm_judge_result(
            reason="llm_request_failed",
            api_status=last_api_status,
            decision_source="judge_unavailable",
        )

    def _empty_llm_judge_result(
        self,
        reason: str = "",
        raw_response: str = "",
        parse_status: str = "not_attempted",
        api_status: str = "not_attempted",
        decision_source: str = "judge_unavailable",
    ) -> Dict[str, Any]:
        return {
            "correct": False,
            "allow_numeric_tolerance": False,
            "reason": "",
            "raw_response": str(raw_response),
            "parse_status": str(parse_status),
            "api_status": str(api_status),
            "decision_source": str(decision_source),
        }

    def _llm_extract_numeric_answer(self, answer_text: str, question: str) -> Dict[str, Any]:
        base_url = self._load_llm_judge_base_url()
        if not base_url:
            return self._empty_numeric_extraction_result(
                reason="numeric_extraction_base_url_missing",
                api_status="config_missing",
            )

        api_url = base_url.rstrip("/") + "/v1/chat/completions"
        system_prompt = """You extract the final numeric answer that the model most likely intended to give.

Rules:
- Use the model answer as the primary evidence.
- Use the question only to identify which quantity the model is trying to answer.
- Do NOT use any reference answer.
- Do NOT infer a new value that is not supported by the model answer.
- If the answer contains multiple intermediate numbers, select the number, fraction, percentage, or symbolic numeric expression that is most likely intended as the final answer.
- Valid extracted answers include forms such as `12`, `0.5`, `1/2`, `50%`, `pi`, `3*pi/4`.
- If no clear final numeric answer can be extracted, return an empty string.

Return strict JSON only:
{"extracted_answer": "0.5"}"""

        prompt = f"""Question: {question}
Model answer: {answer_text}"""

        payload = {
            "model": "Qwen3-VL-235B-A22B-Instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 120,
            "temperature": 0.0,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer token-abc123",
        }

        max_retries = 2
        last_api_status = "not_attempted"
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=30,
                    verify=False,
                )

                if response.status_code == 200:
                    last_api_status = "success"
                    data = response.json()
                    content_response = self._extract_chat_message_text(
                        data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    )
                    result = self._parse_numeric_extraction_response(content_response)
                    if result is not None:
                        result["raw_response"] = content_response
                        result["api_status"] = "success"
                        print(
                            f"[LLM Numeric Extract] Answer: '{answer_text[:50]}...', "
                            f"Extracted: '{result['extracted_answer']}'"
                        )
                        return result

                    print(f"[LLM Numeric Extract] Failed to parse response: {content_response}")
                    return self._empty_numeric_extraction_result(
                        reason="numeric_extraction_parse_failed",
                        raw_response=content_response,
                        parse_status="parse_failed",
                        api_status="success",
                    )

                last_api_status = f"http_{response.status_code}"
                print(f"[LLM Numeric Extract] API error (attempt {attempt + 1}/{max_retries + 1}): {response.status_code}")
            except requests.exceptions.Timeout:
                last_api_status = "timeout"
                print(f"[LLM Numeric Extract] Timeout (attempt {attempt + 1}/{max_retries + 1})")
            except Exception as e:
                last_api_status = "exception"
                print(f"[LLM Numeric Extract] Exception (attempt {attempt + 1}/{max_retries + 1}): {e}")

        return self._empty_numeric_extraction_result(
            reason="numeric_extraction_request_failed",
            api_status=last_api_status,
        )

    def _empty_numeric_extraction_result(
        self,
        extracted_answer: str = "",
        reason: str = "",
        raw_response: str = "",
        parse_status: str = "not_requested",
        api_status: str = "not_requested",
    ) -> Dict[str, Any]:
        return {
            "extracted_answer": str(extracted_answer),
            "reason": "",
            "raw_response": str(raw_response),
            "parse_status": str(parse_status),
            "api_status": str(api_status),
        }

    def _load_llm_judge_base_url(self) -> str:
        if self._llm_judge_base_url_cache is not None:
            return self._llm_judge_base_url_cache

        url_json_path = Path(__file__).parent / "url.json"
        try:
            with open(url_json_path, "r", encoding="utf-8") as f:
                url_config = json.load(f)
        except Exception as e:
            print(f"[LLM Judge] Failed to load url.json: {e}")
            self._llm_judge_base_url_cache = ""
            return ""

        base_url = url_config.get("Qwen3-VL-235B-A22B-Instruct", "")
        if not base_url:
            print("[LLM Judge] Qwen3-VL-235B-A22B-Instruct not configured in url.json")
            self._llm_judge_base_url_cache = ""
            return ""

        self._llm_judge_base_url_cache = str(base_url)
        return self._llm_judge_base_url_cache

    def _extract_chat_message_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
            return "\n".join(part for part in text_parts if part)
        return str(content or "")

    def _truncate_for_log(self, text: Any, max_chars: int = JUDGE_RAW_RESPONSE_LOG_MAX_CHARS) -> str:
        text_str = str(text or "")
        if len(text_str) <= max_chars:
            return text_str
        return text_str[:max_chars] + "...[truncated]"

    def _parse_llm_judge_response(self, content_response: str) -> Optional[Dict[str, Any]]:
        for candidate in self._candidate_json_strings(content_response):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            result = self._normalize_llm_judge_result(parsed)
            if result is not None:
                return result
        return None

    def _parse_numeric_extraction_response(self, content_response: str) -> Optional[Dict[str, Any]]:
        for candidate in self._candidate_json_strings(content_response):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            result = self._normalize_numeric_extraction_result(parsed)
            if result is not None:
                return result
        return None

    def _candidate_json_strings(self, content_response: str) -> List[str]:
        content = str(content_response or "").strip()
        if not content:
            return []

        candidates = [content]
        fenced_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL | re.IGNORECASE)
        candidates.extend(fenced_matches)
        candidates.extend(match.group(0) for match in re.finditer(r"\{.*?\}", content, flags=re.DOTALL))

        deduped: List[str] = []
        seen = set()
        for candidate in candidates:
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _normalize_llm_judge_result(self, parsed: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(parsed, dict):
            return None

        correct = self._coerce_strict_bool(parsed.get("correct"))
        allow_numeric_tolerance = self._coerce_strict_bool(parsed.get("allow_numeric_tolerance"))
        if correct is None:
            return None
        if allow_numeric_tolerance is None:
            allow_numeric_tolerance = False
        return {
            "correct": correct,
            "allow_numeric_tolerance": allow_numeric_tolerance,
            "reason": "",
            "raw_response": "",
            "parse_status": "parsed",
            "api_status": "not_attempted",
            "decision_source": "llm_response",
        }

    def _normalize_numeric_extraction_result(self, parsed: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(parsed, dict):
            return None

        extracted_answer = parsed.get("extracted_answer", "")
        if extracted_answer is None:
            extracted_answer = ""
        if not isinstance(extracted_answer, str):
            extracted_answer = str(extracted_answer)
        return {
            "extracted_answer": extracted_answer.strip(),
            "reason": "",
            "raw_response": "",
            "parse_status": "parsed",
            "api_status": "not_attempted",
        }

    def _coerce_strict_bool(self, value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "true":
                return True
            if normalized == "false":
                return False
            return None
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        return None
