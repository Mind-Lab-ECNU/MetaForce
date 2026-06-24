"""
ChartOrchestra Reward Manager for verl-tool.

Implements ToolOrchestra-style preferences with example-level (uid-based) Z-score
normalization. Each unique uid (added in ray_trainer before batch repeat) is
treated as one example, so repeated rollouts share the same normalization stats.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
import urllib3
from verl import DataProto
from verl.workers.reward_manager import register

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class ToolCallSummary:
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_counts: Dict[str, int] = field(default_factory=dict)
    total_cost: float = 0.0
    total_latency: float = 0.0
    main_agent_cost: float = 0.0


@dataclass
class SampleInfo:
    uid: str
    data_source: str
    response_str: str
    ground_truth: Any
    pref_vec: Dict[str, float]
    tool_calls_info: ToolCallSummary
    is_correct: bool


@dataclass
class BatchStats:
    tool_counts_min: Dict[str, Dict[str, float]] = field(default_factory=dict)
    tool_counts_max: Dict[str, Dict[str, float]] = field(default_factory=dict)


@register("chart_orchestra")
class ChartOrchestraRewardManager:
    """
    ChartOrchestra 风格的 Reward Manager。

    - 工具偏好: 基于 pref_vec 和 min-max 归一化
    - 成本/延迟: 低成本低延迟更优 (使用负值归一化)
    - 准确性: 解析 <answer> 与 ground truth 比较
    - Z-score: 按 uid 分组，模拟 example 级别的归一化
    """

    name = "chart_orchestra"

    def __init__(
        self,
        tokenizer,
        num_examine: int = 20,
        compute_score=None,
        reward_fn_key: str = "data_source",
        tool_pricing_path: Optional[str] = None,
        tool_pricing: Optional[Dict[str, Any]] = None,
        apply_zscore: bool = True,
        zscore_clip: float = 3.0,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.apply_zscore = apply_zscore
        self.zscore_clip = zscore_clip

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        # breakpoint()
        already_printed: Dict[str, int] = {}
        samples: List[SampleInfo] = []
        token_positions: List[Tuple[int, int]] = []

        for i in range(len(data)):
            sample, token_pos = self._extract_sample_info(data[i], i)
            samples.append(sample)
            token_positions.append(token_pos)

        batch_stats = self._compute_batch_stats(samples)

        raw_rewards: List[float] = []
        uids: List[str] = []
        for i, sample in enumerate(samples):
            total_reward, accuracy_reward, tool_reward, cost_reward, latency_reward = (
                self._compute_reward(sample, batch_stats)
            )

            raw_rewards.append(float(total_reward))
            uids.append(sample.uid)

            reward_extra_info["accuracy"].append(float(accuracy_reward))
            reward_extra_info["is_correct"].append(float(sample.is_correct))
            reward_extra_info["tool_preference"].append(float(tool_reward))
            reward_extra_info["cost"].append(float(cost_reward))
            reward_extra_info["latency"].append(float(latency_reward))
            reward_extra_info["total_reward_raw"].append(float(total_reward))
            reward_extra_info["uid"].append(sample.uid)

            if already_printed.get(sample.data_source, 0) < self.num_examine:
                already_printed[sample.data_source] = already_printed.get(sample.data_source, 0) + 1
                print(f"\n[Sample {i}] data_source={sample.data_source} uid={sample.uid}")
                print(f"[response] {sample.response_str[:400]}...")
                print(f"[ground_truth] {sample.ground_truth}")
                print(
                    f"[accuracy] {accuracy_reward:.4f}, [tool] {tool_reward:.4f}, "
                    f"[cost] {cost_reward:.4f}, [latency] {latency_reward:.4f}, "
                    f"[raw_total] {total_reward:.4f}"
                )

        final_rewards = (
            self._apply_zscore_normalization(raw_rewards, uids) if self.apply_zscore else raw_rewards
        )

        for (idx, pos), reward in zip(token_positions, final_rewards):
            reward_tensor[idx, pos] = reward
            reward_extra_info["total_reward"].append(float(reward))

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(sorted(reward_extra_info.items()))}
        return reward_tensor

    def _extract_sample_info(self, data_item: DataProto, index: int) -> Tuple[SampleInfo, Tuple[int, int]]:
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data_item.batch["responses"]
        valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
        valid_response_ids = response_ids[:valid_response_length]
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = self._coerce_extra_info(data_item)
        pref_vec = extra_info.get("pref_vec", {})
        if not isinstance(pref_vec, dict):
            pref_vec = {}

        data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")
        uid = str(data_item.non_tensor_batch.get("uid", f"sample-{index}"))

        tool_interact_info = data_item.non_tensor_batch.get("tool_interact_info", None)
        tool_calls_info = self._extract_tool_calls(tool_interact_info, extra_info)

        answer_text = self._extract_answer_text(response_str)
        is_correct = self._check_answer_correctness(answer_text, ground_truth, data_source)

        sample = SampleInfo(
            uid=uid,
            data_source=data_source,
            response_str=response_str,
            ground_truth=ground_truth,
            pref_vec=pref_vec,
            tool_calls_info=tool_calls_info,
            is_correct=is_correct,
        )
        return sample, (index, int(valid_response_length - 1))

    def _coerce_extra_info(self, data_item: DataProto) -> Dict[str, Any]:
        extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}
        if not isinstance(extra_info, dict):
            extra_info = {}
        else:
            extra_info = dict(extra_info)

        if "agent_prompt_lengths" in data_item.non_tensor_batch:
            extra_info["agent_prompt_lengths"] = data_item.non_tensor_batch.get("agent_prompt_lengths")
        if "agent_output_lengths" in data_item.non_tensor_batch:
            extra_info["agent_output_lengths"] = data_item.non_tensor_batch.get("agent_output_lengths")
        return extra_info

    def _apply_zscore_normalization(self, rewards: List[float], uids: List[str]) -> List[float]:
        grouped = defaultdict(list)
        for reward, uid in zip(rewards, uids):
            grouped[uid].append(reward)

        stats = {}
        for uid, vals in grouped.items():
            mu = mean(vals)
            sigma = stdev(vals) if len(vals) > 1 else 0.0
            stats[uid] = (mu, sigma)

        normalized = []
        for reward, uid in zip(rewards, uids):
            mu, sigma = stats[uid]
            if sigma > 0:
                z = (reward - mu) / (sigma + 1e-6)
            else:
                z = 0.0
            z = max(-self.zscore_clip, min(self.zscore_clip, z))
            normalized.append(float(z))
        return normalized

    def _compute_batch_stats(self, samples: List[SampleInfo]) -> BatchStats:
        grouped = self._group_samples_by_uid(samples)
        return self._compute_min_max_stats(grouped)

    def _group_samples_by_uid(self, samples: List[SampleInfo]) -> Dict[str, List[SampleInfo]]:
        grouped: Dict[str, List[SampleInfo]] = defaultdict(list)
        for sample in samples:
            grouped[sample.uid].append(sample)
        return dict(grouped)

    def _compute_min_max_stats(self, grouped: Dict[str, List[SampleInfo]]) -> BatchStats:
        tool_counts_min: Dict[str, Dict[str, float]] = defaultdict(dict)
        tool_counts_max: Dict[str, Dict[str, float]] = defaultdict(dict)

        for uid, sample_list in grouped.items():
            tool_aggregate: Dict[str, List[float]] = defaultdict(list)
            cost_vals: List[float] = []
            latency_vals: List[float] = []
            accuracy_vals: List[float] = []

            for sample in sample_list:
                for tool_name, count in sample.tool_calls_info.tool_counts.items():
                    tool_aggregate[tool_name].append(float(count))

                cost_vals.append(float(sample.tool_calls_info.total_cost))
                latency_vals.append(float(sample.tool_calls_info.total_latency))
                accuracy_vals.append(1.0 if sample.is_correct else 0.0)

            for tool_name, values in tool_aggregate.items():
                tool_counts_min[uid][tool_name] = min(values)
                tool_counts_max[uid][tool_name] = max(values)

            acc_min, acc_max = self._min_max(accuracy_vals)
            cost_min, cost_max = self._min_max(cost_vals)
            latency_min, latency_max = self._min_max(latency_vals)

            tool_counts_min[uid]["accuracy"] = acc_min
            tool_counts_max[uid]["accuracy"] = acc_max
            tool_counts_min[uid]["cost"] = cost_min
            tool_counts_max[uid]["cost"] = cost_max
            tool_counts_min[uid]["latency"] = latency_min
            tool_counts_max[uid]["latency"] = latency_max

        return BatchStats(tool_counts_min=dict(tool_counts_min), tool_counts_max=dict(tool_counts_max))

    def _min_max(self, values: List[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 0.0
        return min(values), max(values)

    def _extract_tool_calls(self, tool_interact_info: Any, extra_info: Dict[str, Any]) -> ToolCallSummary:
        tool_items = self._normalize_tool_items(tool_interact_info)
        info = ToolCallSummary()

        for item in tool_items:
            tool_name = item.get("tool") or item.get("name") or ""
            model_variant = item.get("model_variant") or item.get("model") or tool_name
            arguments = item.get("arguments", {}) or {}

            info.tool_calls.append({"name": tool_name, "model": model_variant, "arguments": arguments})
            if model_variant:
                info.tool_counts[model_variant] = info.tool_counts.get(model_variant, 0) + 1

            cost = item.get("cost")
            if isinstance(cost, (int, float)):
                info.total_cost += float(cost)
            latency = item.get("latency")
            if isinstance(latency, (int, float)):
                info.total_latency += float(latency)

        info.main_agent_cost = self._estimate_main_agent_cost(tool_items, extra_info)
        info.total_cost += info.main_agent_cost
        return info

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

    def _estimate_main_agent_cost(self, tool_items: List[Dict[str, Any]], extra_info: Dict[str, Any]) -> float:
        pricing_map = extra_info.get("main_agent_tool_pricing") or {}
        if not isinstance(pricing_map, dict) or not pricing_map:
            return 0.0

        model_name = self._resolve_main_agent_model_name(extra_info, pricing_map)
        pricing = pricing_map.get(model_name) if model_name else None
        if pricing is None and len(pricing_map) == 1:
            pricing = next(iter(pricing_map.values()))
        if not isinstance(pricing, dict):
            return 0.0

        prompt_lengths = self._extract_agent_lengths(tool_items, "agent_prompt_length")
        output_lengths = self._extract_agent_lengths(tool_items, "agent_output_length")
        extra_prompt_lengths = self._normalize_agent_lengths(
            extra_info.get("agent_prompt_lengths", extra_info.get("agent_prompt_length"))
        )
        extra_output_lengths = self._normalize_agent_lengths(
            extra_info.get("agent_output_lengths", extra_info.get("agent_output_length"))
        )
        if len(extra_prompt_lengths) >= len(prompt_lengths):
            prompt_lengths = extra_prompt_lengths
        if len(extra_output_lengths) >= len(output_lengths):
            output_lengths = extra_output_lengths
        if not prompt_lengths and not output_lengths:
            return 0.0

        input_price = pricing.get("input_tokens_per_million", pricing.get("input_per_million", 0.0))
        output_price = pricing.get("output_tokens_per_million", pricing.get("output_per_million", 0.0))
        input_price = float(input_price or 0.0)
        output_price = float(output_price or 0.0)
        return (sum(prompt_lengths) * input_price + sum(output_lengths) * output_price)

    def _resolve_main_agent_model_name(
        self, extra_info: Dict[str, Any], pricing_map: Dict[str, Any]
    ) -> Optional[str]:
        candidate_keys = (
            "main_agent_model_name",
            "main_agent_model",
            "trained_model_type",
            "trained_model",
            "model_name",
            "model",
        )
        for key in candidate_keys:
            value = extra_info.get(key)
            if isinstance(value, str) and value:
                return value
        config = extra_info.get("config")
        if isinstance(config, dict):
            for key in candidate_keys:
                value = config.get(key)
                if isinstance(value, str) and value:
                    return value
        if len(pricing_map) == 1:
            return next(iter(pricing_map.keys()))
        return None

    def _extract_agent_lengths(self, tool_items: List[Dict[str, Any]], key: str) -> List[int]:
        best: List[int] = []
        for item in tool_items:
            val = item.get(key)
            if isinstance(val, list):
                if len(val) > len(best):
                    best = [int(v) for v in val if isinstance(v, (int, float))]
            elif isinstance(val, (int, float)):
                best = [int(val)]
        return best

    def _normalize_agent_lengths(self, value: Any) -> List[int]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value if isinstance(v, (int, float))]
        if isinstance(value, (int, float)):
            return [int(value)]
        return []

    def _extract_answer_text(self, response_str: str) -> str:
        answer_match = re.search(r"<answer>(.*?)</answer>", response_str, re.DOTALL)
        if not answer_match:
            return ""
        return answer_match.group(1).strip()

    def _check_answer_correctness(self, answer_text: str, ground_truth: str, data_source: str) -> bool:
        if not answer_text:
            return False
        source = (data_source or "").lower()
        if "figureqa" in source:
            result = self._check_figureqa_answer(answer_text, ground_truth)
        elif "chartqa" in source:
            result = self._check_chartqa_answer(answer_text, ground_truth)
        else:
            result = answer_text.strip() == str(ground_truth).strip()

        # 如果常规判断返回 False，使用 LLM 进行最终判断
        if not result:
            result = self._llm_judge_correctness(answer_text, ground_truth, data_source)
        return result

    def _llm_judge_correctness(self, answer_text: str, ground_truth: str, data_source: str) -> bool:
        """
        使用 LLM 对答案进行最终判断。

        当常规判断方法（精确匹配、数值匹配等）都失败时，
        调用 Qwen3-VL-32B-Instruct 模型进行语义级别的判断。
        """
        # 读取 url.json 获取 API 地址
        url_json_path = Path(__file__).parent / "url.json"
        try:
            with open(url_json_path, "r", encoding="utf-8") as f:
                url_config = json.load(f)
            base_url = url_config.get("Qwen3-VL-32B-Instruct", "")
            if not base_url:
                return False
        except Exception as e:
            print(f"[LLM Judge] Failed to load url.json: {e}")
            return False

        # 构造 API 端点
        api_url = base_url.rstrip("/") + "/v1/chat/completions"

        # 构造 prompt
        prompt = f"""请判断以下答案是否正确。

问题类别: {data_source}
模型答案: {answer_text}
标准答案: {ground_truth}

请只考虑语义是否正确，不考虑格式差异。如果答案的语义与标准答案一致或接近，则判定为正确。

请严格按照以下 JSON 格式返回，不要包含其他内容：
{{"correct": true}}
或
{{"correct": false}}"""

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer token-abc123"
        }

        payload = {
            "model": "Qwen3-VL-32B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0.0,
            "stream": False
        }

        # 重试机制：最多重试 2 次
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=10,
                    verify=False
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                    # 解析返回的 JSON
                    try:
                        result = json.loads(content.strip())
                        is_correct = result.get("correct", False)
                        print(f"[LLM Judge] Answer: '{answer_text[:50]}...', GT: '{ground_truth[:50]}...', Result: {is_correct}")
                        return bool(is_correct)
                    except json.JSONDecodeError:
                        # 尝试从返回内容中提取 JSON
                        json_match = re.search(r'\{[^}]*"correct"\s*:\s*(true|false)', content)
                        if json_match:
                            is_correct = json_match.group(1).lower() == "true"
                            print(f"[LLM Judge] Answer: '{answer_text[:50]}...', GT: '{ground_truth[:50]}...', Result: {is_correct}")
                            return is_correct
                        print(f"[LLM Judge] Failed to parse response: {content}")
                        return False
                else:
                    print(f"[LLM Judge] API error (attempt {attempt + 1}/{max_retries + 1}): {response.status_code}")
            except requests.exceptions.Timeout:
                print(f"[LLM Judge] Timeout (attempt {attempt + 1}/{max_retries + 1})")
            except Exception as e:
                print(f"[LLM Judge] Exception (attempt {attempt + 1}/{max_retries + 1}): {e}")

        # 所有重试都失败，返回保守值 False
        return False

    def _check_figureqa_answer(self, answer_text: str, ground_truth: str) -> bool:
        predicted = answer_text.strip()
        if predicted not in ("yes", "no"):
            return False
        expected = self._strip_trailing_punct(str(ground_truth).strip())
        return predicted == expected

    def _check_chartqa_answer(self, answer_text: str, ground_truth: str) -> bool:
        predicted = answer_text.strip()
        expected = str(ground_truth).strip()
        if predicted == expected:
            return True
        pred_norm = self._normalize_chartqa_text(predicted)
        exp_norm = self._normalize_chartqa_text(expected)
        if pred_norm and pred_norm == exp_norm:
            return True
        return self._numeric_match(predicted, expected)

    def _normalize_chartqa_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\s\.;:!?]+$", "", text)
        return text.lower()

    def _strip_trailing_punct(self, text: str) -> str:
        return re.sub(r"[\s\.;:!?]+$", "", text)

    def _extract_numbers(self, text: str) -> List[float]:
        cleaned = text.replace(",", "")
        return [float(m) for m in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)]

    def _numeric_match(self, predicted: str, expected: str) -> bool:
        predicted_nums = self._extract_numbers(predicted)
        expected_nums = self._extract_numbers(expected)
        if not predicted_nums or not expected_nums:
            return False
        pred_has_percent = "%" in predicted
        exp_has_percent = "%" in expected
        scales = [(1.0, 1.0)]
        if pred_has_percent != exp_has_percent:
            scales.extend([(100.0, 1.0), (1.0, 100.0)])
        for p_val in predicted_nums:
            for e_val in expected_nums:
                for p_scale, e_scale in scales:
                    pred_scaled = p_val * p_scale
                    exp_scaled = e_val * e_scale
                    tol = max(1e-3, 0.01 * abs(exp_scaled))
                    if abs(pred_scaled - exp_scaled) <= tol:
                        return True
        return False

    def _compute_reward(self, sample: SampleInfo, batch_stats: BatchStats) -> Tuple[float, float, float, float, float]:
        uid = sample.uid
        pref_vec = sample.pref_vec or {}

        min_stats = batch_stats.tool_counts_min.get(uid, {})
        max_stats = batch_stats.tool_counts_max.get(uid, {})

        accuracy_value = 1.0 if sample.is_correct else 0.0
        accuracy_reward = self._compute_accuracy_reward(accuracy_value, pref_vec, min_stats, max_stats)
        if not sample.is_correct:
            return 0.0, accuracy_reward, 0.0, 0.0, 0.0

        tool_total, tool_reward = self._compute_tool_preference_reward(
            sample.tool_calls_info.tool_counts, pref_vec, min_stats, max_stats
        )
        latency_reward = self._compute_penalty_reward(
            sample.tool_calls_info.total_latency, "latency", pref_vec, min_stats, max_stats
        )
        cost_reward = self._compute_penalty_reward(
            sample.tool_calls_info.total_cost, "cost", pref_vec, min_stats, max_stats
        )
        if latency_reward + cost_reward >= 0.8:
            total_reward = tool_total + accuracy_reward - 0.8
        else:
            total_reward = tool_total + accuracy_reward - latency_reward - cost_reward
        return total_reward, accuracy_reward, tool_reward, cost_reward, latency_reward

    def _compute_tool_preference_reward(
        self,
        tool_counts: Dict[str, int],
        pref_vec: Dict[str, Any],
        min_stats: Dict[str, float],
        max_stats: Dict[str, float],
    ) -> Tuple[float, float]:
        total_reward = 0.0
        tool_reward = 0.0

        for feature in min_stats.keys():
            if feature in ("accuracy", "latency", "cost"):
                continue
            min_val = min_stats.get(feature, 0.0)
            max_val = max_stats.get(feature, 0.0)
            if max_val <= min_val:
                continue
            weight = float(pref_vec.get(feature, 0.0) or 0.0)
            count_val = float(tool_counts.get(feature, 0.0))
            contrib = weight * (count_val - min_val) / (max_val - min_val)
            total_reward += contrib
            tool_reward += contrib

        return total_reward, tool_reward

    def _compute_accuracy_reward(
        self,
        accuracy_value: float,
        pref_vec: Dict[str, Any],
        min_stats: Dict[str, float],
        max_stats: Dict[str, float],
    ) -> float:
        weight = float(pref_vec.get("accuracy", 0.0) or 0.0)
        return weight * accuracy_value

    def _compute_penalty_reward(
        self,
        value: float,
        feature: str,
        pref_vec: Dict[str, Any],
        min_stats: Dict[str, float],
        max_stats: Dict[str, float],
    ) -> float:
        min_val = min_stats.get(feature, 0.0)
        max_val = max_stats.get(feature, 0.0)
        if max_val <= min_val:
            return 0.0
        weight = float(pref_vec.get(feature, 0.0) or 0.0)
        return weight * (value - min_val) / (max_val - min_val)
