import argparse
import asyncio
import copy
import heapq
import json
import math
import os
import random
import time
import uuid
import warnings
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.utils import hf_processor, hf_tokenizer
from verl_tool.agent_loop.vision_utils import decode_image_url, process_image
from verl_tool.eval import external_model_agent_eval as ext
from verl_tool.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.tool_adaptation import ToolAdaptationManager, inject_tools_to_batch


@dataclass
class TokenEvalConfig:
    val_data_path: str
    tokenizer_path: str
    model_path: str
    model_family: str
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
    tensor_model_parallel_size: int
    gpu_memory_utilization: float
    rollout_max_num_seqs: int
    rollout_max_num_batched_tokens: int
    n_gpus_per_node: int
    rollout_name: str


@dataclass
class TokenModelResponse:
    token_ids: list[int]
    text: str
    finish_reason: Optional[str]
    stop_reason: Optional[str]
    request_summary: Optional[dict[str, Any]] = None


@dataclass
class PromptGenerationInput:
    prompt_ids: Optional[list[int]]
    prompt_text: Optional[str]
    image_data: list[Any]
    prompt_length_estimate: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run standalone multimodal orchestra eval with a local token-in/token-out rollout backend."
    )
    parser.add_argument("--val-data-path", required=True)
    parser.add_argument("--train-data-path", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-family", default="auto", choices=["auto", "interns1", "internvl", "qwen_vl", "generic"])
    parser.add_argument("--tool-server-url", required=True)
    parser.add_argument("--reward-manager", default="multimodal_orchestra")
    parser.add_argument("--run-id", default=os.environ.get("VERL_RUN_ID", f"token_eval_{uuid.uuid4().hex[:8]}"))
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
    parser.add_argument("--eval-tool-variant", default="all", choices=sorted(ext.DEFAULT_TOOL_TYPE_BY_VARIANT))
    parser.add_argument("--latency-penalty-start-step", type=int, default=0)
    parser.add_argument("--tool-penalty-start-step", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--logger-backends", default="console,swanlab")
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--rollout-max-num-seqs", type=int, default=512)
    parser.add_argument("--rollout-max-num-batched-tokens", type=int, default=10000)
    parser.add_argument("--n-gpus-per-node", type=int, default=None)
    parser.add_argument("--rollout-name", default="vllm", choices=["vllm"])
    return parser


def parse_args() -> TokenEvalConfig:
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
    model_path = args.model_path
    tokenizer_path = args.tokenizer_path or model_path
    n_gpus_per_node = args.n_gpus_per_node or max(torch.cuda.device_count(), 1)
    return TokenEvalConfig(
        val_data_path=args.val_data_path,
        tokenizer_path=tokenizer_path,
        model_path=model_path,
        model_family=args.model_family,
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
        tensor_model_parallel_size=max(int(args.tensor_model_parallel_size), 1),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        rollout_max_num_seqs=max(int(args.rollout_max_num_seqs), 1),
        rollout_max_num_batched_tokens=max(int(args.rollout_max_num_batched_tokens), 1),
        n_gpus_per_node=n_gpus_per_node,
        rollout_name=args.rollout_name,
    )


def load_model_config(config: TokenEvalConfig) -> Any:
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code,
        )
    except Exception as exc:
        warnings.warn(
            f"Failed to load model config from {config.model_path}: {exc}. Falling back to processor/path heuristics.",
            stacklevel=1,
        )
        return None


def resolve_model_family(configured: str, processor: Any, model_config: Any = None, *paths: Optional[str]) -> str:
    if configured != "auto":
        return configured
    architectures = [str(item).lower() for item in (getattr(model_config, "architectures", None) or [])]
    model_type = str(getattr(model_config, "model_type", "") or "").lower()
    if model_type == "interns1" or any("interns1" in item for item in architectures):
        return "interns1"
    if "internvl" in model_type or any("internvl" in item for item in architectures):
        return "internvl"
    if processor is None:
        for path in paths:
            lowered = (path or "").lower()
            if "interns1" in lowered:
                return "interns1"
            if "internvl" in lowered:
                return "internvl"
        return "generic"
    processor_name = processor.__class__.__name__
    if "InternS1" in processor_name:
        return "interns1"
    if "InternVL" in processor_name:
        return "internvl"
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and "Qwen2VLImageProcessor" in image_processor.__class__.__name__:
        return "qwen_vl"
    return "generic"


def looks_like_intern_family(model_family: str, *paths: Optional[str]) -> bool:
    if (model_family or "").lower() in {"internvl", "interns1"}:
        return True
    for path in paths:
        if path and "intern" in path.lower():
            return True
    return False


def load_eval_processor(config: TokenEvalConfig, tokenizer) -> tuple[Any, str]:
    model_config = load_model_config(config)
    processor = hf_processor(config.tokenizer_path, trust_remote_code=config.trust_remote_code, use_fast=True)
    model_family = resolve_model_family(
        config.model_family,
        processor,
        model_config,
        config.model_path,
        config.tokenizer_path,
    )
    if model_family == "internvl":
        slow_processor = hf_processor(config.tokenizer_path, trust_remote_code=config.trust_remote_code, use_fast=False)
        if slow_processor is not None:
            processor = slow_processor
            model_family = resolve_model_family(
                config.model_family,
                processor,
                model_config,
                config.model_path,
                config.tokenizer_path,
            )
    elif processor is None and looks_like_intern_family(config.model_family, config.model_path, config.tokenizer_path):
        processor = hf_processor(config.tokenizer_path, trust_remote_code=config.trust_remote_code, use_fast=False)
        model_family = resolve_model_family(
            config.model_family,
            processor,
            model_config,
            config.model_path,
            config.tokenizer_path,
        )
    return processor, model_family


def common_prefix_len(lhs: list[int], rhs: list[int]) -> int:
    size = min(len(lhs), len(rhs))
    idx = 0
    while idx < size and lhs[idx] == rhs[idx]:
        idx += 1
    return idx


def decode_tool_images(tool_images: Any) -> list[Any]:
    if not tool_images:
        return []
    if not isinstance(tool_images, list):
        tool_images = [tool_images]
    decoded: list[Any] = []
    for item in tool_images:
        if isinstance(item, str):
            decoded.append(decode_image_url(item))
        else:
            decoded.append(process_image(item))
    return decoded


def build_standalone_ray_env(config: TokenEvalConfig) -> dict[str, str]:
    env_vars = {
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "true"),
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
        "VLLM_LOGGING_LEVEL": os.environ.get("VLLM_LOGGING_LEVEL", "WARN"),
        "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "1"),
        # vLLM TP ranks expect a consistent global CUDA_VISIBLE_DEVICES view. If Ray rewrites
        # visible devices per actor, vLLM V1 symmetric-memory rendezvous can see overlapping
        # device allocations across ranks and fail during engine startup.
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": os.environ.get(
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1"
        ),
    }
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_visible_devices and config.tensor_model_parallel_size > 1 and config.n_gpus_per_node > 1:
        cuda_visible_devices = ",".join(str(idx) for idx in range(config.n_gpus_per_node))
    if cuda_visible_devices:
        env_vars["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env_vars


def uses_text_prompt_path(model_family: str) -> bool:
    return (model_family or "").lower() == "interns1"


class RebuildingPromptState:
    def __init__(
        self,
        *,
        tokenizer,
        processor,
        config: TokenEvalConfig,
        sample_kwargs: dict[str, Any],
        model_family: str,
        image_url_cache: ext.EncodedImageCache,
        timing: ext.TrajectoryTiming,
    ) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.model_family = model_family
        self.local_messages = copy.deepcopy(sample_kwargs["raw_prompt"])
        self.running_image_data = list((sample_kwargs.get("multi_modal_data") or {}).get("image", []))
        self.image_url_cache = image_url_cache
        self.local_image_cache: dict[int, str] = {}
        self.timing = timing
        self.use_text_prompt = uses_text_prompt_path(model_family)
        self.initial_prompt_ids = self.build_prompt_ids(add_generation_prompt=True)

    def build_tool_extra_fields(self, sample_kwargs: dict[str, Any]) -> dict[str, Any]:
        extra_fields = copy.deepcopy(sample_kwargs.get("extra_info", {})) if sample_kwargs.get("extra_info") else {}
        if self.running_image_data:
            extra_fields["images"] = [
                self.image_url_cache.resolve(image, self.timing, self.local_image_cache)
                for image in self.running_image_data
            ]
        return extra_fields

    def build_prompt_text_for_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

    def build_prompt_ids_for_messages(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Any],
        *,
        add_generation_prompt: bool,
    ) -> list[int]:
        if self.use_text_prompt:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
            )
        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
            model_inputs = self.processor(
                text=[raw_prompt],
                images=image_data or None,
                return_tensors="pt",
            )
            return model_inputs["input_ids"].squeeze(0).tolist()
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )

    def build_prompt_ids(self, *, add_generation_prompt: bool) -> list[int]:
        return self.build_prompt_ids_for_messages(
            list(self.local_messages),
            list(self.running_image_data),
            add_generation_prompt=add_generation_prompt,
        )

    def get_generation_input(self) -> PromptGenerationInput:
        prompt_ids = self.build_prompt_ids(add_generation_prompt=True)
        if self.use_text_prompt:
            prompt_text = self.build_prompt_text_for_messages(
                list(self.local_messages),
                add_generation_prompt=True,
            )
            return PromptGenerationInput(
                prompt_ids=None,
                prompt_text=prompt_text,
                image_data=list(self.running_image_data),
                prompt_length_estimate=len(prompt_ids),
            )
        return PromptGenerationInput(
            prompt_ids=prompt_ids,
            prompt_text=None,
            image_data=list(self.running_image_data),
            prompt_length_estimate=len(prompt_ids),
        )

    def append_assistant_text(self, gen_text: str) -> None:
        self.local_messages.append({"role": "assistant", "content": gen_text})

    def append_observation(self, obs_text: str, tool_images: Any) -> list[int]:
        before_ids = self.build_prompt_ids(add_generation_prompt=False)
        decoded_images = decode_tool_images(tool_images)
        if decoded_images:
            self.running_image_data.extend(decoded_images)
            content = ext.make_local_content_from_text_and_images(obs_text, len(decoded_images))
        else:
            content = obs_text
        self.local_messages.append({"role": self.config.mtrl_role, "content": content})

        after_ids = self.build_prompt_ids(add_generation_prompt=False)
        prefix_len = common_prefix_len(before_ids, after_ids)
        if prefix_len < len(before_ids):
            fallback_messages = [{"role": self.config.mtrl_role, "content": content}]
            return self.build_prompt_ids_for_messages(
                fallback_messages,
                decoded_images,
                add_generation_prompt=False,
            )
        return after_ids[len(before_ids) :]


class StandaloneAsyncLLMServerManager:
    def __init__(self, server_handles: list[Any], max_cache_size: int = 10000):
        self.server_handles = list(server_handles)
        random.shuffle(self.server_handles)
        self.weighted_servers = [[0, (hash(server), server)] for server in self.server_handles]
        heapq.heapify(self.weighted_servers)
        self.request_id_to_server: dict[str, Any] = {}
        self.max_cache_size = max_cache_size

    def _choose_server(self, request_id: str) -> Any:
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]
        server = self.weighted_servers[0][1][1]
        self.weighted_servers[0][0] += 1
        heapq.heapreplace(self.weighted_servers, self.weighted_servers[0])
        if len(self.request_id_to_server) >= self.max_cache_size:
            self.request_id_to_server.pop(next(iter(self.request_id_to_server)))
        self.request_id_to_server[request_id] = server
        return server

    async def generate(
        self,
        request_id: str,
        *,
        sampling_params: dict[str, Any],
        prompt_ids: Optional[list[int]] = None,
        prompt_text: Optional[str] = None,
        image_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
    ) -> Any:
        server = self._choose_server(request_id)
        return await server.generate.remote(
            request_id=str(uuid.uuid4()),
            prompt_ids=prompt_ids,
            prompt_text=prompt_text,
            sampling_params=sampling_params,
            image_data=image_data,
            audio_data=audio_data,
        )


def _qwen2_5_dedup_multimodal_tokens_local(prompt_ids: list[int], processor):
    if processor is None:
        return prompt_ids

    is_qwen2_5_omni = "Qwen2_5OmniProcessor" in processor.__class__.__name__
    image_processor = getattr(processor, "image_processor", None)
    is_qwen2vl = (
        not is_qwen2_5_omni
        and image_processor is not None
        and "Qwen2VLImageProcessor" in image_processor.__class__.__name__
    )

    if not (is_qwen2vl or is_qwen2_5_omni):
        return prompt_ids

    prompt_array = np.array(prompt_ids)
    mask = np.ones(len(prompt_array), dtype=bool)

    if is_qwen2vl:
        image_token_id = processor.image_token_id
    else:
        image_token_id = processor.tokenizer.image_token_id

    is_image_token = prompt_array == image_token_id
    mask[1:] &= ~(is_image_token[1:] & is_image_token[:-1])

    if is_qwen2_5_omni:
        audio_token_id = processor.tokenizer.audio_token_id
        is_audio_token = prompt_array == audio_token_id
        mask[1:] &= ~(is_audio_token[1:] & is_audio_token[:-1])

    return prompt_array[mask].tolist()


def get_standalone_rollout_parallelism(config: TokenEvalConfig) -> tuple[int, int]:
    rollout_world_size = max(int(config.tensor_model_parallel_size), 1)
    total_gpus = max(int(config.n_gpus_per_node), 1)
    if total_gpus < rollout_world_size:
        return 1, rollout_world_size
    if total_gpus % rollout_world_size != 0:
        raise ValueError(
            "Standalone rollout requires n_gpus_per_node to be divisible by tensor_model_parallel_size. "
            f"Got n_gpus_per_node={total_gpus}, tensor_model_parallel_size={rollout_world_size}."
        )
    return total_gpus // rollout_world_size, rollout_world_size


def get_standalone_server_class():
    cached = getattr(get_standalone_server_class, "_cached", None)
    if cached is not None:
        return cached

    import ray
    from vllm.lora.request import LoRARequest
    from vllm.outputs import RequestOutput
    from verl.workers.rollout.vllm_rollout.utils import (
        VLLM_LORA_INT_ID,
        VLLM_LORA_NAME,
        VLLM_LORA_PATH,
    )
    from verl.workers.rollout.vllm_rollout.vllm_async_server import (
        ActorHandle,
        HFModelConfig,
        RewardModelConfig,
        RolloutConfig,
        RolloutMode,
        SamplingParams,
        TokensPrompt,
        vLLMHttpServerBase,
    )
    from verl_tool.workers.rollout.replica import VerlToolTokenOutput

    @ray.remote(num_cpus=1)
    class StandalonePromptTextvLLMHttpServer(vLLMHttpServerBase):
        def __init__(
            self,
            config: RolloutConfig | RewardModelConfig,
            model_config: HFModelConfig,
            rollout_mode: RolloutMode,
            workers: list[ActorHandle],
            replica_rank: int,
            node_rank: int,
            gpus_per_node: int,
            nnodes: int,
        ):
            original_max_model_len = config.max_model_len
            super().__init__(config, model_config, rollout_mode, workers, replica_rank, node_rank, gpus_per_node, nnodes)
            self.config.max_model_len = (
                max(original_max_model_len, self.config.max_model_len)
                if original_max_model_len is not None
                else self.config.max_model_len
            )

        async def generate(
            self,
            sampling_params: dict[str, Any],
            request_id: str,
            prompt_ids: Optional[list[int]] = None,
            prompt_text: Optional[str] = None,
            image_data: Optional[list[Any]] = None,
            audio_data: Optional[list[Any]] = None,
        ) -> VerlToolTokenOutput:
            if prompt_ids is None and prompt_text is None:
                raise ValueError("Either prompt_ids or prompt_text must be provided.")
            if prompt_ids is not None and prompt_text is not None:
                raise ValueError("prompt_ids and prompt_text are mutually exclusive.")

            if prompt_ids is not None:
                max_tokens = min(
                    self.config.max_model_len - len(prompt_ids),
                    sampling_params.get("max_tokens", self.config.response_length),
                )
            else:
                max_tokens = min(
                    sampling_params.get("max_tokens", self.config.response_length),
                    self.config.response_length,
                )
            sampling_params["max_tokens"] = max_tokens
            sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
            sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
            sampling_params = SamplingParams(**sampling_params)

            multi_modal_data: dict[str, Any] = {}
            if image_data:
                multi_modal_data["image"] = image_data
            if audio_data:
                multi_modal_data["audio"] = audio_data

            if prompt_ids is not None:
                prompt_ids = _qwen2_5_dedup_multimodal_tokens_local(prompt_ids, self.model_config.processor)
                prompt = TokensPrompt(
                    prompt_token_ids=prompt_ids,
                    multi_modal_data=multi_modal_data or None,
                )
            else:
                prompt = {
                    "prompt": prompt_text or "",
                    "multi_modal_data": multi_modal_data or None,
                }

            lora_request = None
            if self.model_config.lora_rank > 0:
                lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
                if lora_loaded:
                    lora_request = LoRARequest(
                        lora_name=VLLM_LORA_NAME,
                        lora_int_id=VLLM_LORA_INT_ID,
                        lora_path=VLLM_LORA_PATH,
                    )

            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            )

            final_res: Optional[RequestOutput] = None
            async for output in generator:
                final_res = output
            assert final_res is not None

            token_ids = final_res.outputs[0].token_ids
            log_probs = None
            if sampling_params.logprobs is not None:
                log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(final_res.outputs[0].logprobs)]
            finish_reason = final_res.outputs[0].finish_reason
            stop_reason = final_res.outputs[0].stop_reason
            text = final_res.outputs[0].text
            finished = final_res.finished
            return VerlToolTokenOutput(
                token_ids=token_ids,
                log_probs=log_probs,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
                text=text,
                finished=finished,
            )

    get_standalone_server_class._cached = StandalonePromptTextvLLMHttpServer
    return StandalonePromptTextvLLMHttpServer


class StandaloneTokenModelBackend:
    def __init__(self, config: TokenEvalConfig, tokenizer, model_family: str) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.model_family = model_family
        self.manager: Any = None
        self.replicas: list[Any] = []
        self.rollout_config = None
        self._owns_ray = False
        self._server_handles = []
        self._ray = None
        self.num_replicas = 0

    async def __aenter__(self) -> "StandaloneTokenModelBackend":
        import ray
        from verl.workers.rollout.replica import get_rollout_replica_class

        self._ray = ray
        ray_env = build_standalone_ray_env(self.config)
        os.environ.update(ray_env)
        if not ray.is_initialized():
            ray.init(
                runtime_env={"env_vars": ray_env},
                ignore_reinit_error=True,
            )
            self._owns_ray = True

        config_dir = Path(__file__).resolve().parents[1] / "trainer" / "config"
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            trainer_config = compose(config_name="ppo_trainer")

        trainer_config.trainer.n_gpus_per_node = self.config.n_gpus_per_node
        trainer_config.trainer.nnodes = 1
        trainer_config.data.trust_remote_code = self.config.trust_remote_code
        trainer_config.actor_rollout_ref.model.path = self.config.model_path
        trainer_config.actor_rollout_ref.model.trust_remote_code = self.config.trust_remote_code
        trainer_config.actor_rollout_ref.rollout.name = self.config.rollout_name
        trainer_config.actor_rollout_ref.rollout.mode = "async"
        trainer_config.actor_rollout_ref.rollout.tensor_model_parallel_size = self.config.tensor_model_parallel_size
        trainer_config.actor_rollout_ref.rollout.data_parallel_size = 1
        trainer_config.actor_rollout_ref.rollout.pipeline_model_parallel_size = 1
        trainer_config.actor_rollout_ref.rollout.gpu_memory_utilization = self.config.gpu_memory_utilization
        trainer_config.actor_rollout_ref.rollout.max_num_seqs = self.config.rollout_max_num_seqs
        trainer_config.actor_rollout_ref.rollout.max_num_batched_tokens = self.config.rollout_max_num_batched_tokens
        trainer_config.actor_rollout_ref.rollout.enforce_eager = True
        trainer_config.actor_rollout_ref.rollout.free_cache_engine = True
        trainer_config.actor_rollout_ref.rollout.skip_tokenizer_init = False
        trainer_config.actor_rollout_ref.rollout.prompt_length = self.config.max_prompt_length
        trainer_config.actor_rollout_ref.rollout.response_length = self.config.max_action_length
        if looks_like_intern_family(self.model_family, self.config.model_path, self.config.tokenizer_path):
            trainer_config.actor_rollout_ref.rollout.engine_kwargs = {
                "vllm": {
                    # Intern-family models can fail during vLLM startup profiling when the
                    # multimodal dummy input shape inferred by the engine does not match the
                    # processor's expected image resolution. Skip startup MM profiling and let
                    # real requests drive multimodal preprocessing.
                    "skip_mm_profiling": True,
                }
            }
        self.rollout_config = trainer_config

        num_replicas, rollout_world_size = get_standalone_rollout_parallelism(self.config)
        self.num_replicas = num_replicas
        rollout_replica_class = get_rollout_replica_class(self.config.rollout_name)
        server_class = get_standalone_server_class()
        self.replicas = [
            rollout_replica_class(
                replica_rank=replica_rank,
                config=trainer_config.actor_rollout_ref.rollout,
                model_config=trainer_config.actor_rollout_ref.model,
                gpus_per_node=self.config.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]
        for replica in self.replicas:
            replica.server_class = server_class
        await asyncio.gather(*[replica.init_standalone() for replica in self.replicas])
        self._server_handles = [replica.server_handle for replica in self.replicas]
        if len(self._server_handles) != num_replicas:
            raise RuntimeError(
                f"Expected {num_replicas} rollout replicas for world_size={rollout_world_size}, "
                f"but initialized {len(self._server_handles)} server handles."
            )
        self.manager = StandaloneAsyncLLMServerManager(self._server_handles)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.replicas:
            await asyncio.gather(
                *[replica.sleep() for replica in self.replicas],
                return_exceptions=True,
            )
        if self._owns_ray and self._ray is not None and self._ray.is_initialized():
            self._ray.shutdown()
        self.manager = None
        self.replicas = []
        self.num_replicas = 0
        self._ray = None

    async def generate(
        self,
        *,
        request_id: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        prompt_ids: Optional[list[int]] = None,
        prompt_text: Optional[str] = None,
        image_data: Optional[list[Any]] = None,
    ) -> TokenModelResponse:
        if self.manager is None:
            raise RuntimeError("StandaloneTokenModelBackend has not been initialized.")
        if prompt_ids is None and prompt_text is None:
            raise ValueError("Either prompt_ids or prompt_text must be provided.")
        if prompt_ids is not None and prompt_text is not None:
            raise ValueError("prompt_ids and prompt_text are mutually exclusive.")
        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": list(self.config.action_stop_tokens),
            "include_stop_str_in_output": True,
        }
        output = await self.manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            prompt_text=prompt_text,
            sampling_params=sampling_params,
            image_data=image_data,
        )
        token_ids = list(output.token_ids)
        text = getattr(output, "text", None)
        if text is None:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        stop_reason = getattr(output, "stop_reason", None)
        if isinstance(stop_reason, int):
            stop_reason = self.tokenizer.decode([stop_reason], skip_special_tokens=False)
        return TokenModelResponse(
            token_ids=token_ids,
            text=text or "",
            finish_reason=getattr(output, "finish_reason", None),
            stop_reason=stop_reason,
            request_summary={
                "prompt_token_count": len(prompt_ids or []),
                "prompt_text_chars": len(prompt_text or ""),
                "image_count": len(image_data or []),
                "max_tokens": max_tokens,
            },
        )


class TokenEvalWorker:
    def __init__(self, config: TokenEvalConfig, model_backend: StandaloneTokenModelBackend, worker_id: int):
        self.config = config
        self.worker_id = worker_id
        self.model_backend = model_backend
        self.tool_client = ext.ToolServerClient(config)  # type: ignore[arg-type]
        self.image_url_cache = ext.EncodedImageCache()
        self.semaphore = asyncio.Semaphore(max(1, config.per_worker_max_concurrency))
        self._exit_stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "TokenEvalWorker":
        self._exit_stack = AsyncExitStack()
        try:
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

    async def run_trajectory(self, tokenizer, processor, model_family: str, sample_item: Any, rollout_n: int) -> ext.TrajectoryResult:
        async with self.semaphore:
            return await run_single_trajectory(
                config=self.config,
                tokenizer=tokenizer,
                processor=processor,
                model_family=model_family,
                sample_item=sample_item,
                rollout_n=rollout_n,
                model_backend=self.model_backend,
                tool_client=self.tool_client,
                image_url_cache=self.image_url_cache,
            )


class TokenEvalWorkerPool:
    def __init__(self, config: TokenEvalConfig, model_backend: StandaloneTokenModelBackend):
        self.config = config
        self.total_semaphore = asyncio.Semaphore(max(1, config.max_concurrent_trajectories))
        self.workers = [TokenEvalWorker(config, model_backend=model_backend, worker_id=i) for i in range(max(1, config.num_workers))]
        self._exit_stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "TokenEvalWorkerPool":
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

    async def run_trajectory(self, worker_idx: int, tokenizer, processor, model_family: str, sample_item: Any, rollout_n: int) -> ext.TrajectoryResult:
        worker = self.workers[worker_idx % len(self.workers)]
        async with self.total_semaphore:
            return await worker.run_trajectory(tokenizer, processor, model_family, sample_item, rollout_n)


async def run_single_trajectory(
    config: TokenEvalConfig,
    tokenizer,
    processor,
    model_family: str,
    sample_item: Any,
    rollout_n: int,
    model_backend: StandaloneTokenModelBackend,
    tool_client: ext.ToolServerClient,
    image_url_cache: Optional[ext.EncodedImageCache] = None,
) -> ext.TrajectoryResult:
    sample_kwargs = ext.repeat_sample_kwargs(sample_item, rollout_n)
    if isinstance(sample_kwargs.get("raw_prompt"), np.ndarray):
        sample_kwargs["raw_prompt"] = sample_kwargs["raw_prompt"].tolist()
    prompt_ids: list[int] = []
    response_ids: list[int] = []
    response_mask: list[int] = []
    action_lengths: list[int] = []
    obs_lengths: list[int] = []
    format_messages: list[dict[str, Any]] = []
    tool_interact_info: list[dict[str, Any]] = []
    traj_id = uuid.uuid4().hex
    traj_stop_reason = "unknown"
    failure_stage: Optional[str] = None
    cleanup_error: Optional[str] = None
    timing = ext.TrajectoryTiming()
    traj_start_time = time.perf_counter()
    request_summary: Optional[dict[str, Any]] = None
    model_family_resolved = model_family
    try:
        build_start = time.perf_counter()
        state = RebuildingPromptState(
            tokenizer=tokenizer,
            processor=processor,
            config=config,
            sample_kwargs=sample_kwargs,
            model_family=model_family_resolved,
            image_url_cache=image_url_cache or ext.EncodedImageCache(),
            timing=timing,
        )
        timing.model_request_build_time_sec += time.perf_counter() - build_start
        prompt_ids = list(state.initial_prompt_ids)

        for step in range(config.max_turns + 1):
            generation_input = state.get_generation_input()
            available_length = max(
                config.max_response_length - generation_input.prompt_length_estimate + len(prompt_ids),
                0,
            )
            max_tokens = min(config.max_action_length, available_length)
            is_last_step = step == config.max_turns
            if max_tokens <= 0:
                traj_stop_reason = "max_model_len_exceeded"
                break

            failure_stage = "model_generate"
            model_start = time.perf_counter()
            model_response = await model_backend.generate(
                request_id=traj_id,
                prompt_ids=generation_input.prompt_ids,
                prompt_text=generation_input.prompt_text,
                image_data=generation_input.image_data,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=max_tokens,
            )
            timing.model_http_time_sec += time.perf_counter() - model_start
            timing.model_request_count += 1
            request_summary = model_response.request_summary

            gen_ids = list(model_response.token_ids)
            gen_text = model_response.text or ""
            response_ids.extend(gen_ids)
            response_mask.extend([1] * len(gen_ids))
            action_lengths.append(len(gen_ids))
            format_messages.append({"role": "assistant", "content": gen_text})
            state.append_assistant_text(gen_text)

            do_action = False
            action_text = ""
            for stop_token in config.action_stop_tokens:
                if stop_token in gen_text:
                    do_action = True
                    action_text = gen_text.split(stop_token)[0] + stop_token
                    break

            finish_reason = model_response.finish_reason or "stop"
            if do_action and not is_last_step:
                extra_fields = state.build_tool_extra_fields(sample_kwargs)
                failure_stage = "tool_interact"
                tool_start = time.perf_counter()
                tool_result = await tool_client.interact(
                    traj_id=traj_id,
                    action=action_text,
                    do_action=True,
                    extra_fields=extra_fields,
                    is_last_step=is_last_step,
                )
                timing.tool_http_time_sec += time.perf_counter() - tool_start
                timing.tool_request_count += 1
                tool_interact_info.append(tool_result)

                obs_text = str(tool_result.get("obs", ""))[: max(config.max_obs_length * 8, 0)]
                obs_build_start = time.perf_counter()
                obs_token_ids = state.append_observation(obs_text, tool_result.get("image"))
                timing.model_request_build_time_sec += time.perf_counter() - obs_build_start
                if len(obs_token_ids) > config.max_obs_length:
                    obs_token_ids = obs_token_ids[-config.max_obs_length :]
                response_ids.extend(obs_token_ids)
                response_mask.extend([0] * len(obs_token_ids))
                obs_lengths.append(len(obs_token_ids))
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
        return ext.build_trajectory_result(
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
            failure_reason=str(exc),
            failure_code=exc.__class__.__name__.lower(),
            failure_attempts=1,
            failure_elapsed_sec=0.0,
            exception_class=exc.__class__.__name__,
            request_summary=request_summary,
            timing_info=timing.to_dict(),
            cleanup_error=cleanup_error,
        )

    timing.trajectory_wall_time_sec = time.perf_counter() - traj_start_time
    cleanup_error = await tool_client.close(traj_id)
    return ext.build_trajectory_result(
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
        request_summary=request_summary,
        timing_info=timing.to_dict(),
        cleanup_error=cleanup_error,
    )


async def run_batch_async(
    config: TokenEvalConfig,
    tokenizer,
    processor,
    model_family: str,
    batch: DataProto,
    worker_pool: TokenEvalWorkerPool,
) -> list[ext.TrajectoryResult]:
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
                        processor=processor,
                        model_family=model_family,
                        sample_item=sample_item,
                        rollout_n=rollout_n,
                    )
                )
            )
            task_idx += 1
    return await asyncio.gather(*tasks)


async def evaluate_async(config: TokenEvalConfig) -> dict[str, Any]:
    os.environ["VERL_RUN_ID"] = config.run_id
    os.environ["VERL_SKILL_STORE_DIR"] = config.skill_store_dir
    ext.ensure_dir(config.tool_log_dir)
    ext.ensure_dir(config.skill_store_dir)
    ext.ensure_dir(config.validation_data_dir)

    tokenizer = hf_tokenizer(config.tokenizer_path, trust_remote_code=config.trust_remote_code)
    processor, model_family = load_eval_processor(config, tokenizer)

    reward_config = ext.build_reward_config(config)  # type: ignore[arg-type]
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
    dataloader = ext.load_dataset_and_dataloader(config, tokenizer, processor)  # type: ignore[arg-type]
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
    failed_results_all: list[ext.TrajectoryResult] = []
    total_trajectories = 0
    failure_dump_dir = config.validation_data_dir or config.tool_log_dir

    async with StandaloneTokenModelBackend(config, tokenizer, model_family=model_family) as model_backend:
        async with TokenEvalWorkerPool(config, model_backend=model_backend) as worker_pool:
            for batch_idx, batch_dict in enumerate(dataloader, start=1):
                batch = DataProto.from_single_dict(batch_dict)
                ext.ensure_uid(batch)
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
                    processor=processor,
                    model_family=model_family,
                    batch=batch,
                    worker_pool=worker_pool,
                )
                total_trajectories += len(results)
                failed_results = [item for item in results if item.failed]
                success_results = [item for item in results if not item.failed]
                failed_results_all.extend(failed_results)

                top_failure_reasons = ext.Counter(item.failure_code or "unknown" for item in failed_results).most_common(3)
                print(
                    f"[BATCH {batch_idx}] success_count={len(success_results)} failure_count={len(failed_results)} "
                    f"top_failure_codes={top_failure_reasons}"
                )
                ext.print_batch_timing_summary(batch_idx, results)
                ext.print_inner_progress(
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
                    ext.dump_failed_trajectories(failure_dump_dir, batch_idx, failed_results)

                if not success_results:
                    continue

                result_batch = ext.build_result_dataproto(
                    tokenizer=tokenizer,
                    results=success_results,
                    skill_names=tool_adaptation_manager.get_current_skill_names(),
                )

                inputs, outputs = ext.collect_outputs(tokenizer, result_batch)
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
                ext.extend_reward_info(reward_extra_infos_dict, reward_extra_info, batch_size=len(success_results))

                tool_adaptation_manager.process_reward_extra_info(
                    reward_extra_info=reward_extra_info,
                    global_step=batch_idx,
                    write_immediately=True,
                )

                if config.validation_data_dir:
                    batch_dump_extra: dict[str, list[Any]] = {}
                    ext.extend_reward_info(batch_dump_extra, reward_extra_info, batch_size=len(success_results))
                    data_sources = ext.normalize_batch_field(result_batch.non_tensor_batch.get("data_source"), len(success_results))
                    if data_sources is not None:
                        batch_dump_extra.setdefault("data_source", data_sources)
                    ext.dump_generations(
                        dump_path=config.validation_data_dir,
                        step=batch_idx,
                        inputs=inputs,
                        outputs=outputs,
                        gts=[item.reward_model.get("ground_truth") for item in success_results],
                        scores=scores,
                        reward_extra_infos_dict=batch_dump_extra,
                    )

    metric_dict = ext.build_metric_dict(data_source_lst, sample_uids, reward_extra_infos_dict, sample_turns)
    metric_dict.update(ext.build_failure_metric_dict(failed_results_all, total_trajectories))
    metric_dict["eval/model_family"] = model_family
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
                "model_path": config.model_path,
                "model_family": config.model_family,
                "reward_manager": config.reward_manager,
                "run_id": config.run_id,
                "eval_tool_variant": config.eval_tool_variant,
            },
        }
    )
    logger = ext.Tracking(
        project_name=config.project_name,
        experiment_name=config.experiment_name,
        default_backend=config.logger_backends,
        config=OmegaConf.to_container(tracking_config, resolve=True),
    )
    metric_dict = asyncio.run(evaluate_async(config))
    logger.log(data=metric_dict, step=0)


if __name__ == "__main__":
    main()
