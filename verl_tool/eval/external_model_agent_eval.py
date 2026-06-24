import argparse
import asyncio
from collections import Counter
import copy
from contextlib import AsyncExitStack
import json
import math
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import aiohttp
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.data import DataLoader

from verl.protocol import DataProto
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tracking import Tracking
from verl_tool.agent_loop.vision_utils import encode_image_url
from verl_tool.trainer.ppo.metric_util import process_validation_metrics
from verl_tool.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.tool_adaptation import ToolAdaptationManager, inject_tools_to_batch


DEFAULT_TOKENIZER_PATH = "/inspire/hdd/project/ai4education/public/Models/Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_TOOL_TYPE_BY_VARIANT = {
    "all": "multimodal_processor_tool_adapt_skill",
    "id": "multimodal_processor_tool_adapt_skill_id",
    "ood": "multimodal_processor_tool_adapt_skill_ood",
}
TRANSIENT_MODEL_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
TRANSIENT_MODEL_EXCEPTIONS = (
    asyncio.TimeoutError,
    aiohttp.ServerDisconnectedError,
    aiohttp.ClientConnectorError,
    aiohttp.ClientOSError,
    aiohttp.ClientPayloadError,
)


@dataclass
class EvalConfig:
    val_data_path: str
    tokenizer_path: str
    model_base_url: str
    model_name: str
    tool_server_url: str
    reward_manager: str
    train_data_path: Optional[str]
    run_id: str
    validation_data_dir: Optional[str]
    tool_log_dir: str
    skill_store_dir: str
    batch_size: int
    n: int
    max_prompt_length: int
    max_response_length: int
    max_action_length: int
    max_obs_length: int
    max_turns: int
    temperature: float
    top_p: float
    truncation: str
    filter_overlong_prompts_workers: int
    tool_call_timeout: float
    tool_call_max_retries: int
    max_concurrent_trajectories: int
    num_workers: int
    per_worker_max_concurrency: int
    action_stop_tokens: list[str]
    mtrl_role: str
    eval_tool_variant: str
    latency_penalty_start_step: int
    tool_penalty_start_step: int
    trust_remote_code: bool
    logger_backends: list[str]
    project_name: str
    experiment_name: str


@dataclass
class ModelResponse:
    text: str
    finish_reason: Optional[str]
    stop_reason: Optional[str] = None
    request_summary: Optional[dict[str, Any]] = None


class ExternalModelRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        elapsed_sec: float,
        request_summary: dict[str, Any],
        exception_class: str,
        reason_code: str,
        http_status: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.elapsed_sec = elapsed_sec
        self.request_summary = request_summary
        self.exception_class = exception_class
        self.reason_code = reason_code
        self.http_status = http_status


@dataclass
class TrajectoryResult:
    uid: str
    data_source: str
    reward_model: Any
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    num_turns: int
    tool_interact_info: list[dict[str, Any]]
    traj_stop_reason: str
    verl_tool_metrics: dict[str, Any]
    format_messages: list[dict[str, Any]]
    rollout_n: int = 0
    traj_id: str = ""
    failed: bool = False
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None
    failure_code: Optional[str] = None
    failure_attempts: int = 0
    failure_elapsed_sec: float = 0.0
    exception_class: Optional[str] = None
    request_summary: Optional[dict[str, Any]] = None
    timing_info: dict[str, Any] = field(default_factory=dict)
    cleanup_error: Optional[str] = None


@dataclass
class TrajectoryTiming:
    model_request_build_time_sec: float = 0.0
    image_encode_time_sec: float = 0.0
    model_http_time_sec: float = 0.0
    tool_http_time_sec: float = 0.0
    tokenize_obs_time_sec: float = 0.0
    trajectory_wall_time_sec: float = 0.0
    model_request_count: int = 0
    tool_request_count: int = 0
    image_encode_count: int = 0
    request_payload_bytes: int = 0
    request_image_count: int = 0
    request_prompt_text_chars: int = 0

    def on_model_request(self, request_summary: dict[str, Any]) -> None:
        self.model_request_count += 1
        self.request_payload_bytes += int(request_summary.get("request_payload_bytes", 0))
        self.request_image_count += int(request_summary.get("image_count", 0))
        self.request_prompt_text_chars += int(request_summary.get("prompt_text_chars", 0))

    def to_dict(self) -> dict[str, Any]:
        request_count = max(self.model_request_count, 1)
        return {
            "model_request_build_time_sec": self.model_request_build_time_sec,
            "image_encode_time_sec": self.image_encode_time_sec,
            "model_http_time_sec": self.model_http_time_sec,
            "tool_http_time_sec": self.tool_http_time_sec,
            "tokenize_obs_time_sec": self.tokenize_obs_time_sec,
            "trajectory_wall_time_sec": self.trajectory_wall_time_sec,
            "model_request_count": self.model_request_count,
            "tool_request_count": self.tool_request_count,
            "image_encode_count": self.image_encode_count,
            "request_payload_bytes": self.request_payload_bytes,
            "avg_request_payload_bytes": self.request_payload_bytes / request_count,
            "avg_request_image_count": self.request_image_count / request_count,
            "avg_request_prompt_text_chars": self.request_prompt_text_chars / request_count,
        }


class EncodedImageCache:
    def __init__(self) -> None:
        self._shared_cache: dict[str, str] = {}

    @staticmethod
    def _stable_key(image: Any) -> Optional[str]:
        if isinstance(image, str):
            return f"str::{image}"
        if isinstance(image, dict):
            if isinstance(image.get("image"), str):
                return f"image::{image['image']}"
            if isinstance(image.get("url"), str):
                return f"url::{image['url']}"
        return None

    def resolve(self, image: Any, timing: TrajectoryTiming, local_cache: dict[int, str]) -> str:
        if isinstance(image, str) and image.startswith("data:image/"):
            return image

        shared_key = self._stable_key(image)
        if shared_key is not None and shared_key in self._shared_cache:
            return self._shared_cache[shared_key]

        local_key = None if shared_key is not None else id(image)
        if local_key is not None and local_key in local_cache:
            return local_cache[local_key]

        start_time = time.perf_counter()
        encoded = encode_image_url(image)
        timing.image_encode_time_sec += time.perf_counter() - start_time
        timing.image_encode_count += 1

        if shared_key is not None:
            self._shared_cache[shared_key] = encoded
        elif local_key is not None:
            local_cache[local_key] = encoded
        return encoded


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone multimodal orchestra eval with an external model API.")
    parser.add_argument("--val-data-path", required=True)
    parser.add_argument("--train-data-path", default=None)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tool-server-url", required=True)
    parser.add_argument("--reward-manager", default="multimodal_orchestra")
    parser.add_argument("--run-id", default=os.environ.get("VERL_RUN_ID", f"external_eval_{uuid.uuid4().hex[:8]}"))
    parser.add_argument("--validation-data-dir", default=None)
    parser.add_argument("--tool-log-dir", required=True)
    parser.add_argument("--skill-store-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=8192 * 3)
    parser.add_argument("--max-response-length", type=int, default=8192 * 2 + 4096)
    parser.add_argument("--max-action-length", type=int, default=8192)
    parser.add_argument("--max-obs-length", type=int, default=8192)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--truncation", default="error")
    parser.add_argument("--filter-overlong-prompts-workers", type=int, default=32)
    parser.add_argument("--tool-call-timeout", type=float, default=100.0)
    parser.add_argument("--tool-call-max-retries", type=int, default=1)
    parser.add_argument("--max-concurrent-trajectories", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--per-worker-max-concurrency", type=int, default=None)
    parser.add_argument("--action-stop-token", dest="action_stop_tokens", action="append", default=["</tool_call>"])
    parser.add_argument("--mtrl-role", default="user")
    parser.add_argument("--eval-tool-variant", default="all", choices=sorted(DEFAULT_TOOL_TYPE_BY_VARIANT))
    parser.add_argument("--latency-penalty-start-step", type=int, default=0)
    parser.add_argument("--tool-penalty-start-step", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--logger-backends", default="console,swanlab")
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--experiment-name", default=None)
    return parser


def parse_args() -> EvalConfig:
    args = build_arg_parser().parse_args()
    action_stop_tokens = [token for token in args.action_stop_tokens if token]
    logger_backends = [item.strip() for item in str(args.logger_backends).split(",") if item.strip()]
    project_name = args.project_name or args.reward_manager
    experiment_name = args.experiment_name or args.run_id
    num_workers = max(int(args.num_workers), 1)
    total_concurrency = max(int(args.max_concurrent_trajectories), 1)
    per_worker_max_concurrency = (
        max(int(args.per_worker_max_concurrency), 1)
        if args.per_worker_max_concurrency is not None
        else max(1, math.ceil(total_concurrency / num_workers))
    )
    return EvalConfig(
        val_data_path=args.val_data_path,
        tokenizer_path=args.tokenizer_path,
        model_base_url=args.model_base_url,
        model_name=args.model_name,
        tool_server_url=args.tool_server_url,
        reward_manager=args.reward_manager,
        train_data_path=args.train_data_path,
        run_id=args.run_id,
        validation_data_dir=args.validation_data_dir,
        tool_log_dir=args.tool_log_dir,
        skill_store_dir=args.skill_store_dir,
        batch_size=args.batch_size,
        n=args.n,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        max_action_length=args.max_action_length,
        max_obs_length=args.max_obs_length,
        max_turns=args.max_turns,
        temperature=args.temperature,
        top_p=args.top_p,
        truncation=args.truncation,
        filter_overlong_prompts_workers=args.filter_overlong_prompts_workers,
        tool_call_timeout=args.tool_call_timeout,
        tool_call_max_retries=args.tool_call_max_retries,
        max_concurrent_trajectories=total_concurrency,
        num_workers=num_workers,
        per_worker_max_concurrency=per_worker_max_concurrency,
        action_stop_tokens=action_stop_tokens,
        mtrl_role=args.mtrl_role,
        eval_tool_variant=args.eval_tool_variant,
        latency_penalty_start_step=args.latency_penalty_start_step,
        tool_penalty_start_step=args.tool_penalty_start_step,
        trust_remote_code=args.trust_remote_code,
        logger_backends=logger_backends,
        project_name=project_name,
        experiment_name=experiment_name,
    )


def normalize_chat_endpoint(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    if stripped.endswith("/v1"):
        return f"{stripped}/chat/completions"
    return f"{stripped}/v1/chat/completions"


def ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def split_text_by_placeholders(text: str, placeholder: str) -> list[str]:
    if not text:
        return [""]
    parts = text.split(placeholder)
    if not parts:
        return [text]
    return parts


def make_local_content_from_text_and_images(text: str, num_images: int) -> Any:
    if num_images <= 0:
        return text
    normalized = text or ""
    image_tags = normalized.count("<image>")
    if image_tags < num_images:
        normalized += "<image>" * (num_images - image_tags)
    parts = split_text_by_placeholders(normalized, "<image>")
    content: list[dict[str, Any]] = []
    for idx, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if idx < len(parts) - 1:
            content.append({"type": "image"})
    return content


def tokenize_text(tokenizer, text: str, max_len: Optional[int] = None, truncation_side: str = "right") -> list[int]:
    token_ids = tokenizer.encode(text or "", add_special_tokens=False)
    if max_len is None or len(token_ids) <= max_len:
        return token_ids
    if truncation_side == "left":
        return token_ids[-max_len:]
    if truncation_side == "middle":
        left_half = max_len // 2
        right_half = max_len - left_half
        middle = tokenizer.encode("...(truncated)...", add_special_tokens=False)
        kept = token_ids[:left_half] + middle + token_ids[-right_half:]
        return kept[-max_len:]
    return token_ids[:max_len]


def sanitize_request(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, dict):
        return {sanitize_request(key): sanitize_request(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(sanitize_request(item) for item in obj)
    if isinstance(obj, str):
        return obj.replace("\x00", "")
    return obj


def extract_action_from_model_output(
    gen_text: str,
    action_stop_tokens: list[str],
    stop_reason: Optional[Any] = None,
) -> tuple[bool, str]:
    normalized_text = gen_text or ""
    normalized_stop_reason = None if stop_reason is None else str(stop_reason)

    for stop_token in action_stop_tokens:
        if stop_token and stop_token in normalized_text:
            return True, normalized_text.split(stop_token)[0] + stop_token

    for stop_token in action_stop_tokens:
        if stop_token and normalized_stop_reason and stop_token in normalized_stop_reason:
            return True, normalized_text.split(stop_token)[0] + stop_token

    if "<tool_call>" in normalized_text and "</tool_call>" not in normalized_text:
        prefix, _, suffix = normalized_text.partition("<tool_call>")
        return True, prefix + "<tool_call>" + suffix + "</tool_call>"

    return False, ""


def make_openai_content_from_text_and_image_urls(text: str, image_urls: list[str]) -> Any:
    if not image_urls:
        return text
    normalized = text or ""
    image_tags = normalized.count("<image>")
    if image_tags < len(image_urls):
        normalized += "<image>" * (len(image_urls) - image_tags)
    parts = split_text_by_placeholders(normalized, "<image>")
    content: list[dict[str, Any]] = []
    for idx, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if idx < len(parts) - 1:
            if idx < len(image_urls):
                content.append({"type": "image_url", "image_url": {"url": image_urls[idx]}})
            else:
                content.append({"type": "text", "text": "<image>"})
    return content


def build_openai_messages(
    local_messages: list[dict[str, Any]],
    running_image_data: list[Any],
    image_url_cache: Optional[EncodedImageCache] = None,
    timing: Optional[TrajectoryTiming] = None,
    local_image_cache: Optional[dict[int, str]] = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    image_idx = 0
    openai_messages: list[dict[str, Any]] = []
    running_image_urls: list[str] = []
    cache = image_url_cache or EncodedImageCache()
    trajectory_timing = timing or TrajectoryTiming()
    local_cache = local_image_cache if local_image_cache is not None else {}
    for message in local_messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            openai_messages.append({"role": role, "content": str(content)})
            continue
        converted: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                converted.append({"type": "text", "text": str(item)})
                continue
            item_type = item.get("type")
            if item_type == "text":
                converted.append({"type": "text", "text": str(item.get("text", ""))})
                continue
            if item_type == "image":
                if image_idx >= len(running_image_data):
                    raise ValueError("Message/image placeholder count exceeds available running_image_data.")
                image_source = running_image_data[image_idx]
                image_url = cache.resolve(image_source, trajectory_timing, local_cache)
                running_image_urls.append(image_url)
                converted.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    }
                )
                image_idx += 1
                continue
            raise NotImplementedError(f"Unsupported content block type for external model API: {item_type}")
        openai_messages.append({"role": role, "content": converted})
    if image_idx != len(running_image_data):
        raise ValueError(
            f"Unused images when building request: used={image_idx} total={len(running_image_data)}"
        )
    return openai_messages, running_image_urls


class ExternalModelClient:
    def __init__(self, config: EvalConfig):
        self.endpoint = normalize_chat_endpoint(config.model_base_url)
        self.model_name = config.model_name
        self.timeout_seconds = max(float(config.tool_call_timeout), 1.0)
        self.max_retries = max(int(config.tool_call_max_retries), 0)
        self.timeout = aiohttp.ClientTimeout(
            total=None,
            connect=min(self.timeout_seconds, 30.0),
            sock_connect=min(self.timeout_seconds, 30.0),
            sock_read=self.timeout_seconds,
        )
        self.api_key = (
            os.environ.get("MODEL_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("VLLM_API_KEY")
            or "EMPTY"
        )
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "ExternalModelClient":
        await self._recreate_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.session is not None and not self.session.closed:
            await self.session.close()
        self.session = None

    async def _recreate_session(self) -> aiohttp.ClientSession:
        if self.session is not None and not self.session.closed:
            await self.session.close()
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            return await self._recreate_session()
        return self.session

    def _build_request_summary(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        stop: list[str],
        request_payload_bytes: int = 0,
    ) -> dict[str, Any]:
        image_count = 0
        text_chars = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                text_chars += len(content)
                continue
            if not isinstance(content, list):
                text_chars += len(str(content))
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_count += 1
                elif isinstance(item, dict) and item.get("type") == "text":
                    text_chars += len(str(item.get("text", "")))
                else:
                    text_chars += len(str(item))
        return {
            "model_name": self.model_name,
            "message_count": len(messages),
            "image_count": image_count,
            "max_tokens": int(max_tokens),
            "stop_count": len(stop),
            "prompt_text_chars": text_chars,
            "request_payload_bytes": int(request_payload_bytes),
        }

    @staticmethod
    def _normalize_reason_code(exc: Exception, http_status: Optional[int] = None) -> str:
        if http_status is not None:
            return f"http_{http_status}"
        if isinstance(exc, asyncio.TimeoutError):
            return "timeout"
        if isinstance(exc, aiohttp.ServerDisconnectedError):
            return "server_disconnected"
        if isinstance(exc, aiohttp.ClientConnectorError):
            return "connection_error"
        if isinstance(exc, aiohttp.ClientOSError):
            return "client_os_error"
        if isinstance(exc, aiohttp.ClientPayloadError):
            return "payload_error"
        return exc.__class__.__name__.lower()

    @staticmethod
    def _is_retryable_exception(exc: Exception, http_status: Optional[int] = None) -> bool:
        if http_status is not None:
            return http_status in TRANSIENT_MODEL_STATUS_CODES
        return isinstance(exc, TRANSIENT_MODEL_EXCEPTIONS)

    def _build_final_request_error(
        self,
        exc: Exception,
        *,
        attempts: int,
        elapsed_sec: float,
        request_summary: dict[str, Any],
        http_status: Optional[int] = None,
    ) -> ExternalModelRequestError:
        if http_status is not None:
            message = f"External model API error {http_status}: {exc}"
        else:
            message = f"External model request failed after {attempts} attempts: {exc.__class__.__name__}: {exc}"
        return ExternalModelRequestError(
            message,
            attempts=attempts,
            elapsed_sec=elapsed_sec,
            request_summary=request_summary,
            exception_class=exc.__class__.__name__,
            reason_code=self._normalize_reason_code(exc, http_status=http_status),
            http_status=http_status,
        )

    async def generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: list[str],
    ) -> ModelResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            # "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop,
            "include_stop_str_in_output": True,
        }
        request_body = json.dumps(sanitize_request(payload), ensure_ascii=False)
        request_summary = self._build_request_summary(
            messages,
            max_tokens=max_tokens,
            stop=stop,
            request_payload_bytes=len(request_body.encode("utf-8")),
        )
        start_time = time.time()
        last_exc: Optional[Exception] = None
        last_http_status: Optional[int] = None

        for attempt in range(self.max_retries + 1):
            session = await self._get_session()
            try:
                async with session.post(self.endpoint, headers=headers, data=request_body) as resp:
                    if resp.status >= 400:
                        body_text = await resp.text()
                        print(
                            f"[external_model_http_error] status={resp.status} body={body_text}",
                            flush=True,
                        )
                        if resp.status == 400 and "include_stop_str_in_output" in body_text:
                            fallback_payload = dict(payload)
                            fallback_payload.pop("include_stop_str_in_output", None)
                            fallback_body = json.dumps(sanitize_request(fallback_payload), ensure_ascii=False)
                            async with session.post(self.endpoint, headers=headers, data=fallback_body) as retry_resp:
                                retry_resp.raise_for_status()
                                data = await retry_resp.json()
                                print(
                                    "[external_model_response]",
                                    json.dumps(data, ensure_ascii=False, indent=2),
                                    flush=True,
                                )
                                parsed = self._parse_response(data)
                                parsed.request_summary = request_summary
                                return parsed
                        last_http_status = resp.status
                        error = RuntimeError(body_text[:500] if body_text else f"HTTP {resp.status}")
                        if self._is_retryable_exception(error, http_status=resp.status) and attempt < self.max_retries:
                            await asyncio.sleep(min(8.0, 1.0 * (2**attempt)) + random.uniform(0.0, 0.25))
                            continue
                        raise self._build_final_request_error(
                            error,
                            attempts=attempt + 1,
                            elapsed_sec=time.time() - start_time,
                            request_summary=request_summary,
                            http_status=resp.status,
                        )
                    data = await resp.json()
                    print(
                        "[external_model_response]",
                        json.dumps(data, ensure_ascii=False, indent=2),
                        flush=True,
                    )
                    parsed = self._parse_response(data)
                    parsed.request_summary = request_summary
                    return parsed
            except ExternalModelRequestError:
                raise
            except Exception as exc:
                last_exc = exc
                if self._is_retryable_exception(exc) and attempt < self.max_retries:
                    if isinstance(
                        exc,
                        (
                            aiohttp.ServerDisconnectedError,
                            aiohttp.ClientConnectorError,
                            aiohttp.ClientOSError,
                            aiohttp.ClientPayloadError,
                        ),
                    ):
                        await self._recreate_session()
                    await asyncio.sleep(min(8.0, 1.0 * (2**attempt)) + random.uniform(0.0, 0.25))
                    continue
                raise self._build_final_request_error(
                    exc,
                    attempts=attempt + 1,
                    elapsed_sec=time.time() - start_time,
                    request_summary=request_summary,
                    http_status=last_http_status,
                ) from exc

        if last_exc is None:
            last_exc = RuntimeError("Unknown external model request failure.")
        raise self._build_final_request_error(
            last_exc,
            attempts=self.max_retries + 1,
            elapsed_sec=time.time() - start_time,
            request_summary=request_summary,
            http_status=last_http_status,
        )

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> ModelResponse:
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"No choices returned from external model API: {json.dumps(data)[:500]}")
        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            content = "".join(text_parts)
        return ModelResponse(
            text=content or "",
            finish_reason=choice.get("finish_reason"),
            stop_reason=choice.get("stop_reason"),
        )


class ToolServerClient:
    def __init__(self, config: EvalConfig):
        self.url = config.tool_server_url
        self.timeout_seconds = config.tool_call_timeout
        self.max_retries = config.tool_call_max_retries
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "ToolServerClient":
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.session is not None and not self.session.closed:
            await self.session.close()
        self.session = None

    async def interact(
        self,
        traj_id: str,
        action: str,
        do_action: bool,
        extra_fields: Optional[dict[str, Any]],
        is_last_step: bool,
    ) -> dict[str, Any]:
        if self.session is None:
            raise RuntimeError("ToolServerClient session is not initialized.")
        payload = {
            "trajectory_ids": [traj_id],
            "actions": [action],
            "finish": [not do_action],
            "is_last_step": [is_last_step],
        }
        if extra_fields is not None:
            payload["extra_fields"] = [extra_fields]

        safe_payload = sanitize_request(payload)
        last_error: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self.session.post(self.url, json=safe_payload) as resp:
                    resp.raise_for_status()
                    response = await resp.json()
                    break
            except Exception as exc:
                last_error = str(exc)
                if attempt >= self.max_retries:
                    response = {
                        "observations": [f"Tool call error: {exc}"],
                        "dones": [1],
                        "valids": [0],
                        "processing_time_ms": 0.0,
                        "success": False,
                    }
                    break
                await asyncio.sleep(1.0)

        obs = response["observations"][0]
        tool_interact_info: dict[str, Any]
        if isinstance(obs, str):
            tool_interact_info = {"obs": obs, "reward": None}
        elif isinstance(obs, dict):
            tool_interact_info = dict(obs)
            tool_interact_info.setdefault("obs", "")
            tool_interact_info.setdefault("reward", None)
        else:
            raise ValueError(f"Unsupported observation type from tool server: {type(obs)}")

        tool_interact_info["trajectory_id"] = traj_id
        tool_interact_info["action"] = action
        tool_interact_info["done"] = int(response["dones"][0])
        tool_interact_info["valid_action"] = int(response["valids"][0])
        tool_interact_info["finish"] = not do_action
        tool_interact_info["is_last_step"] = is_last_step
        tool_interact_info["processing_time_ms"] = float(response.get("processing_time_ms", 0.0))
        tool_interact_info["queue_time_ms"] = float(response.get("queue_time_ms", 0.0))
        tool_interact_info["success"] = bool(response.get("success", True))
        if last_error and not tool_interact_info["success"]:
            tool_interact_info["invalid_reason"] = last_error
        return tool_interact_info

    async def close(self, traj_id: str) -> Optional[str]:
        if self.session is None:
            return None
        payload = {
            "trajectory_ids": [traj_id],
            "actions": [""],
            "finish": [True],
            "is_last_step": [True],
        }
        try:
            async with self.session.post(self.url, json=payload):
                return None
        except Exception as exc:
            return str(exc)


class ExternalEvalWorker:
    def __init__(self, config: EvalConfig, worker_id: int):
        self.config = config
        self.worker_id = worker_id
        self.model_client = ExternalModelClient(config)
        self.tool_client = ToolServerClient(config)
        self.image_url_cache = EncodedImageCache()
        self.semaphore = asyncio.Semaphore(max(1, config.per_worker_max_concurrency))
        self._exit_stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "ExternalEvalWorker":
        self._exit_stack = AsyncExitStack()
        try:
            await self._exit_stack.enter_async_context(self.model_client)
            await self._exit_stack.enter_async_context(self.tool_client)
        except Exception:
            await self._exit_stack.aclose()
            self._exit_stack = None
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None

    async def run_trajectory(self, tokenizer, sample_item: Any, rollout_n: int) -> TrajectoryResult:
        async with self.semaphore:
            return await run_single_trajectory(
                config=self.config,
                tokenizer=tokenizer,
                sample_item=sample_item,
                rollout_n=rollout_n,
                model_client=self.model_client,
                tool_client=self.tool_client,
                image_url_cache=self.image_url_cache,
            )


class ExternalEvalWorkerPool:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.total_semaphore = asyncio.Semaphore(max(1, config.max_concurrent_trajectories))
        self.workers = [ExternalEvalWorker(config, worker_id=i) for i in range(max(1, config.num_workers))]
        self._exit_stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "ExternalEvalWorkerPool":
        self._exit_stack = AsyncExitStack()
        try:
            for worker in self.workers:
                await self._exit_stack.enter_async_context(worker)
        except Exception:
            await self._exit_stack.aclose()
            self._exit_stack = None
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None

    async def run_trajectory(self, worker_idx: int, tokenizer, sample_item: Any, rollout_n: int) -> TrajectoryResult:
        worker = self.workers[worker_idx % len(self.workers)]
        async with self.total_semaphore:
            return await worker.run_trajectory(tokenizer=tokenizer, sample_item=sample_item, rollout_n=rollout_n)


def build_dataset_config(config: EvalConfig) -> Any:
    return OmegaConf.create(
        {
            "prompt_key": "prompt",
            "image_key": "images",
            "video_key": "videos",
            "max_prompt_length": config.max_prompt_length,
            "return_raw_chat": True,
            "truncation": config.truncation,
            "filter_overlong_prompts": False,
            "filter_overlong_prompts_workers": config.filter_overlong_prompts_workers,
            "shuffle": False,
            "reward_fn_key": "data_source",
            "apply_chat_template_kwargs": {},
        }
    )


def build_reward_config(config: EvalConfig) -> Any:
    return OmegaConf.create(
        {
            "data": {"reward_fn_key": "data_source"},
            "reward_model": {
                "reward_manager": config.reward_manager,
                "sandbox_fusion": {},
            },
        }
    )


def ensure_uid(batch: DataProto) -> None:
    if "uid" in batch.non_tensor_batch:
        return
    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)


def repeat_sample_kwargs(sample_item: Any, rollout_n: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for key, value in sample_item.non_tensor_batch.items():
        kwargs[key] = value
    kwargs["rollout_n"] = rollout_n
    return kwargs


def build_extra_fields(sample_kwargs: dict[str, Any], running_image_urls: list[str]) -> dict[str, Any]:
    extra_fields = copy.deepcopy(sample_kwargs.get("extra_info", {})) if sample_kwargs.get("extra_info") is not None else {}
    if running_image_urls:
        extra_fields["images"] = list(running_image_urls)
    return extra_fields


def compute_tool_metrics(
    num_turns: int,
    action_lengths: list[int],
    obs_lengths: list[int],
    tool_interact_info: list[dict[str, Any]],
    format_messages: list[dict[str, Any]],
    response_mask: list[int],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "num_turns": num_turns,
        "per_action_length": float(np.mean(action_lengths)) if action_lengths else 0.0,
        "per_obs_length": float(np.mean(obs_lengths)) if obs_lengths else 0.0,
        "traj_actions_length": int(sum(action_lengths)),
        "traj_obs_length": int(sum(obs_lengths)),
        "tool_call_success": float(np.mean([info.get("success", 1.0) for info in tool_interact_info]))
        if tool_interact_info
        else 1.0,
        "format_message_turns": len(format_messages),
        "valid_traj": float(any(mask == 1 for mask in response_mask)),
        "tool_processing_time_sec": sum(float(info.get("processing_time_ms", 0.0)) for info in tool_interact_info)
        / 1000.0,
        "tool_queue_time_sec": sum(float(info.get("queue_time_ms", 0.0)) for info in tool_interact_info) / 1000.0,
    }
    return metrics


def build_trajectory_result(
    *,
    sample_kwargs: dict[str, Any],
    rollout_n: int,
    traj_id: str,
    prompt_ids: list[int],
    response_ids: list[int],
    response_mask: list[int],
    tool_interact_info: list[dict[str, Any]],
    traj_stop_reason: str,
    format_messages: list[dict[str, Any]],
    action_lengths: list[int],
    obs_lengths: list[int],
    failed: bool = False,
    failure_stage: Optional[str] = None,
    failure_reason: Optional[str] = None,
    failure_code: Optional[str] = None,
    failure_attempts: int = 0,
    failure_elapsed_sec: float = 0.0,
    exception_class: Optional[str] = None,
    request_summary: Optional[dict[str, Any]] = None,
    timing_info: Optional[dict[str, Any]] = None,
    cleanup_error: Optional[str] = None,
) -> TrajectoryResult:
    verl_tool_metrics = compute_tool_metrics(
        num_turns=len(format_messages),
        action_lengths=action_lengths,
        obs_lengths=obs_lengths,
        tool_interact_info=tool_interact_info,
        format_messages=format_messages,
        response_mask=response_mask,
    )
    if failed:
        verl_tool_metrics["failed"] = 1.0
    if timing_info:
        verl_tool_metrics.update(timing_info)
    return TrajectoryResult(
        uid=str(sample_kwargs["uid"]),
        data_source=str(sample_kwargs.get("data_source", "unknown")),
        reward_model=sample_kwargs["reward_model"],
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        num_turns=len(format_messages),
        tool_interact_info=tool_interact_info,
        traj_stop_reason=traj_stop_reason,
        verl_tool_metrics=verl_tool_metrics,
        format_messages=format_messages,
        rollout_n=rollout_n,
        traj_id=traj_id,
        failed=failed,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
        failure_code=failure_code,
        failure_attempts=failure_attempts,
        failure_elapsed_sec=failure_elapsed_sec,
        exception_class=exception_class,
        request_summary=request_summary,
        timing_info=timing_info or {},
        cleanup_error=cleanup_error,
    )


async def run_single_trajectory(
    config: EvalConfig,
    tokenizer,
    sample_item: Any,
    rollout_n: int,
    model_client: ExternalModelClient,
    tool_client: ToolServerClient,
    image_url_cache: Optional[EncodedImageCache] = None,
) -> TrajectoryResult:
    sample_kwargs = repeat_sample_kwargs(sample_item, rollout_n)
    prompt_ids = list(sample_kwargs["raw_prompt_ids"])
    running_prompt_ids = list(prompt_ids)
    initial_image_data = list((sample_kwargs.get("multi_modal_data") or {}).get("image", []))
    local_messages = copy.deepcopy(sample_kwargs["raw_prompt"])
    response_mask: list[int] = []
    response_ids: list[int] = []
    action_lengths: list[int] = []
    obs_lengths: list[int] = []
    format_messages: list[dict[str, Any]] = []
    tool_interact_info: list[dict[str, Any]] = []
    traj_id = uuid.uuid4().hex
    traj_stop_reason = "unknown"
    failure_stage: Optional[str] = None
    current_request_summary: Optional[dict[str, Any]] = None
    cleanup_error: Optional[str] = None
    timing = TrajectoryTiming()
    traj_start_time = time.perf_counter()
    local_image_cache: dict[int, str] = {}
    shared_image_cache = image_url_cache or EncodedImageCache()

    try:
        if isinstance(local_messages, np.ndarray):
            local_messages = local_messages.tolist()
        elif isinstance(local_messages, tuple):
            local_messages = list(local_messages)
        if not isinstance(local_messages, list):
            failure_stage = "prepare_trajectory"
            raise TypeError(f"raw_prompt must be a list of messages, got {type(local_messages)}")

        build_start = time.perf_counter()
        openai_messages, running_image_urls = build_openai_messages(
            local_messages,
            initial_image_data,
            image_url_cache=shared_image_cache,
            timing=timing,
            local_image_cache=local_image_cache,
        )
        timing.model_request_build_time_sec += time.perf_counter() - build_start

        for step in range(config.max_turns + 1):
            available_length = max(config.max_response_length - len(running_prompt_ids) + len(prompt_ids), 0)
            max_tokens = min(config.max_action_length, available_length)
            is_last_step = step == config.max_turns
            if max_tokens <= 0:
                traj_stop_reason = "max_model_len_exceeded"
                break

            failure_stage = "prepare_model_request"
            failure_stage = "model_generate"
            model_http_start = time.perf_counter()
            model_response = await model_client.generate(
                messages=openai_messages,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=max_tokens,
                stop=config.action_stop_tokens,
            )
            timing.model_http_time_sec += time.perf_counter() - model_http_start
            current_request_summary = model_response.request_summary
            if current_request_summary is not None:
                timing.on_model_request(current_request_summary)
            current_request_summary = None
            gen_text = model_response.text or ""
            do_action, action_text = extract_action_from_model_output(
                gen_text,
                config.action_stop_tokens,
                stop_reason=model_response.stop_reason,
            )
            if do_action and action_text:
                gen_text = action_text

            gen_ids = tokenize_text(tokenizer, gen_text)
            response_ids.extend(gen_ids)
            response_mask.extend([1] * len(gen_ids))
            action_lengths.append(len(gen_ids))
            format_messages.append({"role": "assistant", "content": gen_text})
            openai_messages.append({"role": "assistant", "content": gen_text})
            running_prompt_ids.extend(gen_ids)

            finish_reason = model_response.finish_reason or "stop"
            if do_action and not is_last_step:
                extra_fields = build_extra_fields(sample_kwargs, running_image_urls)
                failure_stage = "tool_interact"
                tool_http_start = time.perf_counter()
                tool_result = await tool_client.interact(
                    traj_id=traj_id,
                    action=action_text,
                    do_action=True,
                    extra_fields=extra_fields,
                    is_last_step=is_last_step,
                )
                timing.tool_http_time_sec += time.perf_counter() - tool_http_start
                timing.tool_request_count += 1
                tool_interact_info.append(tool_result)

                obs_text = str(tool_result.get("obs", ""))[: max(config.max_obs_length * 8, 0)]
                tool_images = tool_result.get("image")
                new_image_urls: list[str] = []
                if tool_images:
                    if not isinstance(tool_images, list):
                        tool_images = [tool_images]
                    for item in tool_images:
                        new_image_urls.append(shared_image_cache.resolve(item, timing, local_image_cache))
                    running_image_urls.extend(new_image_urls)

                obs_build_start = time.perf_counter()
                obs_content = make_openai_content_from_text_and_image_urls(obs_text, new_image_urls)
                openai_messages.append({"role": config.mtrl_role, "content": obs_content})
                timing.model_request_build_time_sec += time.perf_counter() - obs_build_start

                obs_token_text = obs_text
                if new_image_urls:
                    obs_token_text = obs_text + "<image>" * max(0, len(new_image_urls) - obs_text.count("<image>"))
                obs_tokenize_start = time.perf_counter()
                obs_token_ids = tokenize_text(
                    tokenizer,
                    obs_token_text,
                    max_len=config.max_obs_length,
                    truncation_side="left",
                )
                timing.tokenize_obs_time_sec += time.perf_counter() - obs_tokenize_start
                response_ids.extend(obs_token_ids)
                response_mask.extend([0] * len(obs_token_ids))
                obs_lengths.append(len(obs_token_ids))
                running_prompt_ids.extend(obs_token_ids)

                if tool_result.get("done"):
                    traj_stop_reason = "tool_signaled_done"
                    break
                continue

            if finish_reason == "length":
                traj_stop_reason = "max_response_length_reached"
            elif do_action and is_last_step:
                traj_stop_reason = "max_turns_reached"
            elif not gen_text.strip():
                traj_stop_reason = "model_chose_to_finish_with_empty_response"
            else:
                traj_stop_reason = "model_chose_to_finish"
            break
    except Exception as exc:
        timing.trajectory_wall_time_sec = time.perf_counter() - traj_start_time
        cleanup_error = await tool_client.close(traj_id)
        failure_reason = str(exc)
        failure_attempts = 1
        failure_elapsed_sec = 0.0
        failure_code = exc.__class__.__name__.lower()
        exception_class = exc.__class__.__name__
        request_summary = current_request_summary
        if isinstance(exc, ExternalModelRequestError):
            failure_attempts = exc.attempts
            failure_elapsed_sec = exc.elapsed_sec
            failure_code = exc.reason_code
            exception_class = exc.exception_class
            request_summary = exc.request_summary
        return build_trajectory_result(
            sample_kwargs=sample_kwargs,
            rollout_n=rollout_n,
            traj_id=traj_id,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            tool_interact_info=tool_interact_info,
            traj_stop_reason="model_request_failed" if failure_stage == "model_generate" else "trajectory_failed",
            format_messages=format_messages,
            action_lengths=action_lengths,
            obs_lengths=obs_lengths,
            failed=True,
            failure_stage=failure_stage or "unknown",
            failure_reason=failure_reason,
            failure_code=failure_code,
            failure_attempts=failure_attempts,
            failure_elapsed_sec=failure_elapsed_sec,
            exception_class=exception_class,
            request_summary=request_summary,
            timing_info=timing.to_dict(),
            cleanup_error=cleanup_error,
        )

    timing.trajectory_wall_time_sec = time.perf_counter() - traj_start_time
    cleanup_error = await tool_client.close(traj_id)
    return build_trajectory_result(
        sample_kwargs=sample_kwargs,
        rollout_n=rollout_n,
        traj_id=traj_id,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        tool_interact_info=tool_interact_info,
        traj_stop_reason=traj_stop_reason,
        format_messages=format_messages,
        action_lengths=action_lengths,
        obs_lengths=obs_lengths,
        timing_info=timing.to_dict(),
        cleanup_error=cleanup_error,
    )


def left_pad_sequences(sequences: list[list[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(1, max((len(seq) for seq in sequences), default=0))
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for idx, seq in enumerate(sequences):
        if not seq:
            continue
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        padded[idx, -len(seq) :] = seq_tensor
        mask[idx, -len(seq) :] = 1
    return padded, mask


def right_pad_sequences(sequences: list[list[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(1, max((len(seq) for seq in sequences), default=0))
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for idx, seq in enumerate(sequences):
        if not seq:
            continue
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        padded[idx, : len(seq)] = seq_tensor
        mask[idx, : len(seq)] = 1
    return padded, mask


def build_result_dataproto(tokenizer, results: list[TrajectoryResult], skill_names: set[str]) -> DataProto:
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompt_tensor, prompt_mask = left_pad_sequences([item.prompt_ids for item in results], pad_token_id)
    response_tensor, response_attn_mask = right_pad_sequences([item.response_ids for item in results], pad_token_id)
    response_mask_tensor, _ = right_pad_sequences([item.response_mask for item in results], 0)
    attention_mask = torch.cat([prompt_mask, response_attn_mask], dim=1)
    input_ids = torch.cat([prompt_tensor, response_tensor], dim=1)
    position_ids = compute_position_id_with_mask(attention_mask)

    batch = TensorDict(
        {
            "prompts": prompt_tensor,
            "responses": response_tensor,
            "response_mask": response_mask_tensor * response_attn_mask,
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            "position_ids": position_ids,
        },
        batch_size=[len(results)],
    )

    non_tensor_batch = {
        "uid": np.array([item.uid for item in results], dtype=object),
        "data_source": np.array([item.data_source for item in results], dtype=object),
        "reward_model": np.array([item.reward_model for item in results], dtype=object),
        "tool_interact_info": np.array([item.tool_interact_info for item in results], dtype=object),
        "traj_stop_reason": np.array([item.traj_stop_reason for item in results], dtype=object),
        "verl_tool_metrics": np.array([item.verl_tool_metrics for item in results], dtype=object),
        "format_messages": np.array([item.format_messages for item in results], dtype=object),
        "failed": np.array([item.failed for item in results], dtype=np.bool_),
        "failure_stage": np.array([item.failure_stage for item in results], dtype=object),
        "failure_reason": np.array([item.failure_reason for item in results], dtype=object),
        "failure_code": np.array([item.failure_code for item in results], dtype=object),
        "failure_attempts": np.array([item.failure_attempts for item in results], dtype=np.int32),
        "failure_elapsed_sec": np.array([item.failure_elapsed_sec for item in results], dtype=np.float32),
        "exception_class": np.array([item.exception_class for item in results], dtype=object),
        "request_summary": np.array([item.request_summary for item in results], dtype=object),
        "timing_info": np.array([item.timing_info for item in results], dtype=object),
        "cleanup_error": np.array([item.cleanup_error for item in results], dtype=object),
        "traj_id": np.array([item.traj_id for item in results], dtype=object),
        "rollout_n": np.array([item.rollout_n for item in results], dtype=np.int32),
        "__num_turns__": np.array([item.num_turns for item in results], dtype=np.int32),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"skill_names": skill_names, "global_steps": 0})


def dump_generations(
    dump_path: str,
    step: int,
    inputs: list[str],
    outputs: list[str],
    gts: list[Any],
    scores: list[float],
    reward_extra_infos_dict: dict[str, list[Any]],
) -> None:
    ensure_dir(dump_path)
    filename = os.path.join(dump_path, f"{step}.jsonl")
    base_data = {
        "input": inputs,
        "output": outputs,
        "gts": gts,
        "score": scores,
        "step": [step] * len(inputs),
    }
    for key, value in reward_extra_infos_dict.items():
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)) and len(value) == len(inputs):
            base_data[key] = list(value)

    lines = []
    for idx in range(len(inputs)):
        entry = {key: value[idx] for key, value in base_data.items()}
        lines.append(json.dumps(entry, ensure_ascii=False))

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Dumped generations to {filename}")


def dump_failed_trajectories(dump_dir: str, step: int, results: list[TrajectoryResult]) -> None:
    if not results:
        return
    ensure_dir(dump_dir)
    filename = os.path.join(dump_dir, "failures.jsonl")
    with open(filename, "a", encoding="utf-8") as f:
        for item in results:
            entry = {
                "step": step,
                "uid": item.uid,
                "data_source": item.data_source,
                "rollout_n": item.rollout_n,
                "traj_id": item.traj_id,
                "traj_stop_reason": item.traj_stop_reason,
                "failure_stage": item.failure_stage,
                "failure_reason": item.failure_reason,
                "failure_code": item.failure_code,
                "exception_class": item.exception_class,
                "attempt_count": item.failure_attempts,
                "elapsed_sec": item.failure_elapsed_sec,
                "request_summary": item.request_summary,
                "timing_info": item.timing_info,
                "num_turns_completed": item.num_turns,
                "tool_interact_info": item.tool_interact_info,
                "cleanup_error": item.cleanup_error,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Appended {len(results)} failed trajectories to {filename}")


def format_progress_bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        total = 1
    current = min(max(current, 0), total)
    filled = int(current * width / total)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def print_inner_progress(
    *,
    batch_idx: int,
    total_batches: Optional[int],
    completed_trajectories: int,
    total_trajectories_estimate: Optional[int],
    batch_success_count: int,
    batch_failure_count: int,
    cumulative_failure_count: int,
    top_failure_codes: list[tuple[str, int]],
) -> None:
    batch_total = total_batches if total_batches and total_batches > 0 else 1
    batch_percent = min(100.0, (batch_idx / batch_total) * 100.0)
    batch_progress = f"{batch_idx}/{total_batches}" if total_batches is not None else f"{batch_idx}/?"
    traj_progress = (
        f"{completed_trajectories}/{total_trajectories_estimate}"
        if total_trajectories_estimate is not None
        else f"{completed_trajectories}/?"
    )
    cumulative_failure_ratio = (
        cumulative_failure_count / completed_trajectories if completed_trajectories > 0 else 0.0
    )
    print(
        f"[INNER_PROGRESS] batches={batch_progress} percent={batch_percent:.1f}% "
        f"bar={format_progress_bar(batch_idx, batch_total)} trajectories={traj_progress} "
        f"batch_success={batch_success_count} batch_failure={batch_failure_count} "
        f"cumulative_failure_ratio={cumulative_failure_ratio:.3f} top_failure_codes={top_failure_codes}"
    )


def summarize_batch_timings(results: list[TrajectoryResult]) -> dict[str, float]:
    if not results:
        return {}

    keys = [
        "trajectory_wall_time_sec",
        "model_request_build_time_sec",
        "image_encode_time_sec",
        "model_http_time_sec",
        "tool_http_time_sec",
        "tokenize_obs_time_sec",
        "avg_request_payload_bytes",
        "avg_request_image_count",
        "avg_request_prompt_text_chars",
        "model_request_count",
        "tool_request_count",
        "image_encode_count",
    ]
    summary: dict[str, float] = {}
    for key in keys:
        values = [float(item.timing_info.get(key, 0.0)) for item in results]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_sum"] = float(np.sum(values))
    return summary


def print_batch_timing_summary(batch_idx: int, results: list[TrajectoryResult]) -> None:
    summary = summarize_batch_timings(results)
    if not summary:
        return

    print(
        f"[BATCH {batch_idx} TIMING] "
        f"traj_wall_mean={summary['trajectory_wall_time_sec_mean']:.3f}s "
        f"model_http_mean={summary['model_http_time_sec_mean']:.3f}s "
        f"tool_http_mean={summary['tool_http_time_sec_mean']:.3f}s "
        f"build_mean={summary['model_request_build_time_sec_mean']:.3f}s "
        f"image_encode_mean={summary['image_encode_time_sec_mean']:.3f}s "
        f"payload_mean={summary['avg_request_payload_bytes_mean']:.1f}B "
        f"req_count_mean={summary['model_request_count_mean']:.2f} "
        f"tool_count_mean={summary['tool_request_count_mean']:.2f}"
    )


async def run_batch_async(
    config: EvalConfig,
    tokenizer,
    batch: DataProto,
    model_client: Optional[ExternalModelClient] = None,
    tool_client: Optional[ToolServerClient] = None,
    worker_pool: Optional[ExternalEvalWorkerPool] = None,
) -> list[TrajectoryResult]:
    if worker_pool is not None:
        tasks = []
        task_idx = 0
        for sample_idx in range(len(batch)):
            sample_item = batch[sample_idx]
            for rollout_n in range(config.n):
                tasks.append(
                    asyncio.create_task(
                        worker_pool.run_trajectory(
                            worker_idx=task_idx,
                            tokenizer=tokenizer,
                            sample_item=sample_item,
                            rollout_n=rollout_n,
                        )
                    )
                )
                task_idx += 1
        return await asyncio.gather(*tasks)

    if model_client is None or tool_client is None:
        raise ValueError("run_batch_async requires either worker_pool or both model_client and tool_client.")

    semaphore = asyncio.Semaphore(max(1, config.max_concurrent_trajectories))

    async def wrapped(sample_item: Any, rollout_n: int) -> TrajectoryResult:
        async with semaphore:
            return await run_single_trajectory(
                config=config,
                tokenizer=tokenizer,
                sample_item=sample_item,
                rollout_n=rollout_n,
                model_client=model_client,
                tool_client=tool_client,
            )

    tasks = []
    for sample_idx in range(len(batch)):
        sample_item = batch[sample_idx]
        for rollout_n in range(config.n):
            tasks.append(asyncio.create_task(wrapped(sample_item, rollout_n)))
    return await asyncio.gather(*tasks)


def collect_outputs(tokenizer, batch: DataProto) -> tuple[list[str], list[str]]:
    prompt_len = batch.batch["prompts"].shape[1]
    prompt_attn = batch.batch["attention_mask"][:, :prompt_len]
    response_attn = batch.batch["attention_mask"][:, prompt_len:]
    inputs = [
        tokenizer.decode(batch.batch["prompts"][i][prompt_attn[i] == 1], skip_special_tokens=False)
        for i in range(len(batch))
    ]
    outputs = [
        tokenizer.decode(batch.batch["responses"][i][response_attn[i] == 1], skip_special_tokens=False)
        for i in range(len(batch))
    ]
    return inputs, outputs


def extend_reward_info(dest: dict[str, list[Any]], reward_extra_info: dict[str, Any], batch_size: int) -> None:
    for key, value in reward_extra_info.items():
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)) and len(value) == batch_size:
            dest.setdefault(key, []).extend(list(value))


def normalize_batch_field(value: Any, expected_len: int) -> list[Any] | None:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == expected_len:
        return value
    return None


def to_builtin_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_builtin_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin_jsonable(item) for item in value]
    return value


def build_metric_dict(
    data_source_lst: list[np.ndarray],
    sample_uids: list[str],
    reward_extra_infos_dict: dict[str, list[Any]],
    sample_turns: list[np.ndarray],
) -> dict[str, Any]:
    if not data_source_lst:
        return {}
    data_sources = np.concatenate(data_source_lst, axis=0)
    data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
    metric_dict: dict[str, Any] = {}
    for data_source, var2metric2val in data_src2var2metric2val.items():
        core_var = "acc" if "acc" in var2metric2val else "reward"
        for var_name, metric2val in var2metric2val.items():
            n_max = max(int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys())
            for metric_name, metric_val in metric2val.items():
                if (
                    (var_name == core_var)
                    and any(metric_name.startswith(prefix) for prefix in ["mean", "maj", "best"])
                    and (f"@{n_max}" in metric_name)
                ):
                    metric_sec = "val-core"
                else:
                    metric_sec = "val-aux"
                metric_dict[f"{metric_sec}/{data_source}/{var_name}/{metric_name}"] = to_builtin_jsonable(metric_val)
    if sample_turns:
        turns = np.concatenate(sample_turns)
        metric_dict["val-aux/num_turns/min"] = int(turns.min())
        metric_dict["val-aux/num_turns/max"] = int(turns.max())
        metric_dict["val-aux/num_turns/mean"] = float(turns.mean())
    return metric_dict


def build_failure_metric_dict(failed_results: list[TrajectoryResult], total_trajectories: int) -> dict[str, Any]:
    metric_dict: dict[str, Any] = {
        "val-failure/total": len(failed_results),
        "val-failure/ratio": (len(failed_results) / total_trajectories) if total_trajectories > 0 else 0.0,
    }
    if not failed_results:
        return metric_dict

    stage_counts = Counter(item.failure_stage or "unknown" for item in failed_results)
    code_counts = Counter(item.failure_code or "unknown" for item in failed_results)
    data_source_counts = Counter(item.data_source for item in failed_results)
    exception_counts = Counter(item.exception_class or "unknown" for item in failed_results)
    for stage, count in stage_counts.items():
        metric_dict[f"val-failure/by_stage/{stage}"] = count
    for code, count in code_counts.items():
        metric_dict[f"val-failure/by_code/{code}"] = count
    for data_source, count in data_source_counts.items():
        metric_dict[f"val-failure/by_data_source/{data_source}"] = count
    for exc_class, count in exception_counts.items():
        metric_dict[f"val-failure/by_exception/{exc_class}"] = count
    return metric_dict


def load_dataset_and_dataloader(config: EvalConfig, tokenizer, processor) -> DataLoader:
    dataset_config = build_dataset_config(config)
    dataset = RLHFDataset(
        data_files=config.val_data_path,
        tokenizer=tokenizer,
        processor=processor,
        config=dataset_config,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )


async def evaluate_async(config: EvalConfig) -> dict[str, Any]:
    os.environ["VERL_RUN_ID"] = config.run_id
    os.environ["VERL_SKILL_STORE_DIR"] = config.skill_store_dir
    ensure_dir(config.tool_log_dir)
    ensure_dir(config.skill_store_dir)
    ensure_dir(config.validation_data_dir)

    tokenizer = hf_tokenizer(config.tokenizer_path, trust_remote_code=config.trust_remote_code)
    processor = hf_processor(config.tokenizer_path, trust_remote_code=config.trust_remote_code, use_fast=True)
    reward_config = build_reward_config(config)
    reward_fn = load_reward_manager(
        reward_config,
        tokenizer,
        num_examine=1,
        latency_penalty_start_step=config.latency_penalty_start_step,
        tool_penalty_start_step=config.tool_penalty_start_step,
    )
    tool_adaptation_manager = ToolAdaptationManager(
        skill_store_dir=config.skill_store_dir,
        log_dir=config.tool_log_dir,
        max_skills=50,
    )
    dataloader = load_dataset_and_dataloader(config, tokenizer, processor)
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    dataset_size = len(dataloader.dataset) if hasattr(dataloader, "dataset") and hasattr(dataloader.dataset, "__len__") else None
    total_trajectories_estimate = (dataset_size * config.n) if dataset_size is not None else None

    data_source_lst: list[np.ndarray] = []
    reward_extra_infos_dict: dict[str, list[Any]] = {}
    sample_scores: list[float] = []
    sample_inputs: list[str] = []
    sample_outputs: list[str] = []
    sample_gts: list[Any] = []
    sample_uids: list[str] = []
    sample_turns: list[np.ndarray] = []
    failed_results_all: list[TrajectoryResult] = []
    total_trajectories = 0
    failure_dump_dir = config.validation_data_dir or config.tool_log_dir

    async with ExternalEvalWorkerPool(config) as worker_pool:
        for batch_idx, batch_dict in enumerate(dataloader, start=1):
            batch = DataProto.from_single_dict(batch_dict)
            ensure_uid(batch)
            tools_prompt = tool_adaptation_manager.on_batch_start()
            if tools_prompt:
                batch = inject_tools_to_batch(
                    batch=batch,
                    tools_prompt=tools_prompt,
                    tokenizer=tokenizer,
                    processor=processor,
                    max_prompt_length=config.max_prompt_length,
                    truncation=config.truncation,
                    apply_chat_template_kwargs={},
                )

            results = await run_batch_async(
                config=config,
                tokenizer=tokenizer,
                batch=batch,
                worker_pool=worker_pool,
            )
            total_trajectories += len(results)
            failed_results = [item for item in results if item.failed]
            success_results = [item for item in results if not item.failed]
            failed_results_all.extend(failed_results)

            top_failure_reasons = Counter(item.failure_code or "unknown" for item in failed_results).most_common(3)
            print(
                f"[BATCH {batch_idx}] success_count={len(success_results)} failure_count={len(failed_results)} "
                f"top_failure_codes={top_failure_reasons}"
            )
            print_batch_timing_summary(batch_idx, results)
            print_inner_progress(
                batch_idx=batch_idx,
                total_batches=total_batches,
                completed_trajectories=total_trajectories,
                total_trajectories_estimate=total_trajectories_estimate,
                batch_success_count=len(success_results),
                batch_failure_count=len(failed_results),
                cumulative_failure_count=len(failed_results_all),
                top_failure_codes=top_failure_reasons,
            )
            if failed_results:
                dump_failed_trajectories(failure_dump_dir, batch_idx, failed_results)

            if not success_results:
                continue

            result_batch = build_result_dataproto(
                tokenizer=tokenizer,
                results=success_results,
                skill_names=tool_adaptation_manager.get_current_skill_names(),
            )

            inputs, outputs = collect_outputs(tokenizer, result_batch)
            scores_result = reward_fn(result_batch, return_dict=True)
            reward_tensor = scores_result["reward_tensor"]
            reward_extra_info = scores_result.get("reward_extra_info", {})
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            sample_inputs.extend(inputs)
            sample_outputs.extend(outputs)
            sample_gts.extend([item.reward_model.get("ground_truth") for item in success_results])
            sample_uids.extend([item.uid for item in success_results])
            sample_turns.append(result_batch.non_tensor_batch["__num_turns__"])
            data_source_lst.append(result_batch.non_tensor_batch["data_source"])
            extend_reward_info(reward_extra_infos_dict, reward_extra_info, batch_size=len(success_results))

            val_skill_step = batch_idx
            tool_adaptation_manager.process_reward_extra_info(
                reward_extra_info=reward_extra_info,
                global_step=val_skill_step,
                write_immediately=True,
            )

            if config.validation_data_dir:
                batch_dump_extra: dict[str, list[Any]] = {}
                extend_reward_info(batch_dump_extra, reward_extra_info, batch_size=len(success_results))
                data_sources = normalize_batch_field(result_batch.non_tensor_batch.get("data_source"), len(success_results))
                if data_sources is not None:
                    batch_dump_extra.setdefault("data_source", data_sources)
                dump_generations(
                    dump_path=config.validation_data_dir,
                    step=batch_idx,
                    inputs=inputs,
                    outputs=outputs,
                    gts=[item.reward_model.get("ground_truth") for item in success_results],
                    scores=scores,
                    reward_extra_infos_dict=batch_dump_extra,
                )

    metric_dict = build_metric_dict(data_source_lst, sample_uids, reward_extra_infos_dict, sample_turns)
    metric_dict.update(build_failure_metric_dict(failed_results_all, total_trajectories))
    print(json.dumps(metric_dict, ensure_ascii=False, indent=2, sort_keys=True))
    return metric_dict


def main() -> None:
    config = parse_args()
    print(json.dumps(config.__dict__, ensure_ascii=False, indent=2, sort_keys=True))
    tracking_config = OmegaConf.create(
        {
            "trainer": {
                "project_name": config.project_name,
                "experiment_name": config.experiment_name,
                "logger": config.logger_backends,
            },
            "eval": {
                "val_data_path": config.val_data_path,
                "model_base_url": config.model_base_url,
                "model_name": config.model_name,
                "reward_manager": config.reward_manager,
                "run_id": config.run_id,
                "eval_tool_variant": config.eval_tool_variant,
            },
        }
    )
    logger = Tracking(
        project_name=config.project_name,
        experiment_name=config.experiment_name,
        default_backend=config.logger_backends,
        config=OmegaConf.to_container(tracking_config, resolve=True),
    )
    metric_dict = asyncio.run(evaluate_async(config))
    logger.log(data=metric_dict, step=0)


if __name__ == "__main__":
    main()
