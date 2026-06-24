import copy
import warnings
from dataclasses import dataclass
from typing import Any, Optional

from verl_tool.agent_loop.vision_utils import decode_image_url, process_image


def load_model_config(model_path: str, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        warnings.warn(
            f"Failed to load model config from {model_path}: {exc}. Falling back to processor/path heuristics.",
            stacklevel=1,
        )
        return None


def resolve_model_family(
    configured: str,
    processor: Any,
    model_config: Any = None,
    *paths: Optional[str],
) -> str:
    if configured != "auto":
        return configured

    architectures = [str(item).lower() for item in (getattr(model_config, "architectures", None) or [])]
    model_type = str(getattr(model_config, "model_type", "") or "").lower()
    if model_type == "interns1" or any("interns1" in item for item in architectures):
        return "interns1"
    if "internvl" in model_type or any("internvl" in item for item in architectures):
        return "internvl"

    image_processor = getattr(processor, "image_processor", None) if processor is not None else None
    processor_name = processor.__class__.__name__ if processor is not None else ""
    if "InternS1" in processor_name:
        return "interns1"
    if "InternVL" in processor_name:
        return "internvl"
    if image_processor is not None and "Qwen2VLImageProcessor" in image_processor.__class__.__name__:
        return "qwen_vl"

    for path in paths:
        lowered = (path or "").lower()
        if "interns1" in lowered:
            return "interns1"
        if "internvl" in lowered:
            return "internvl"
        if "qwen" in lowered and "vl" in lowered:
            return "qwen_vl"

    return "generic"


def is_intern_family(model_family: str) -> bool:
    return (model_family or "").lower() in {"internvl", "interns1"}


def looks_like_intern_family(model_family: str, *paths: Optional[str]) -> bool:
    if is_intern_family(model_family):
        return True
    for path in paths:
        lowered = (path or "").lower()
        if "internvl" in lowered or "interns1" in lowered:
            return True
    return False


def load_processor_for_family(local_path: str, trust_remote_code: bool, model_family: str):
    from verl.utils import hf_processor

    if looks_like_intern_family(model_family, local_path):
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=False)
        if processor is not None:
            return processor
    return hf_processor(local_path, trust_remote_code=trust_remote_code)


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
    decoded = []
    for item in tool_images:
        if isinstance(item, str):
            decoded.append(decode_image_url(item))
        else:
            decoded.append(process_image(item))
    return decoded


def make_local_content_from_text_and_images(text: str, num_images: int) -> list[dict[str, str]]:
    content = []
    if text:
        segments = text.split("<image>")
    else:
        segments = [""]
    for idx, segment in enumerate(segments):
        if segment:
            content.append({"type": "text", "text": segment})
        if idx < len(segments) - 1:
            content.append({"type": "image"})
    existing_images = sum(1 for item in content if item.get("type") == "image")
    for _ in range(max(num_images - existing_images, 0)):
        content.append({"type": "image"})
    if not content:
        content.append({"type": "text", "text": ""})
    return content


@dataclass
class PromptGenerationInput:
    prompt_ids: Optional[list[int]]
    prompt_text: Optional[str]
    image_data: list[Any]
    prompt_length_estimate: int


class InternVLPromptState:
    def __init__(
        self,
        *,
        tokenizer,
        processor,
        mtrl_role: str,
        sample_kwargs: dict[str, Any],
    ) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.mtrl_role = mtrl_role
        raw_prompt = sample_kwargs["raw_prompt"]
        if hasattr(raw_prompt, "tolist"):
            raw_prompt = raw_prompt.tolist()
        self.local_messages = copy.deepcopy(raw_prompt)
        self.running_image_data = list((sample_kwargs.get("multi_modal_data") or {}).get("image", []))

    def build_prompt_text(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
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
        prompt_text = self.build_prompt_text(messages, add_generation_prompt=add_generation_prompt)
        model_inputs = self.processor(
            text=[prompt_text],
            images=image_data or None,
            return_tensors="pt",
        )
        return model_inputs["input_ids"].squeeze(0).tolist()

    def build_prompt_ids(self, *, add_generation_prompt: bool) -> list[int]:
        return self.build_prompt_ids_for_messages(
            list(self.local_messages),
            list(self.running_image_data),
            add_generation_prompt=add_generation_prompt,
        )

    def get_generation_input(self) -> PromptGenerationInput:
        prompt_ids = self.build_prompt_ids(add_generation_prompt=True)
        prompt_text = self.build_prompt_text(
            list(self.local_messages),
            add_generation_prompt=True,
        )
        return PromptGenerationInput(
            prompt_ids=prompt_ids,
            prompt_text=prompt_text,
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
            content = make_local_content_from_text_and_images(obs_text, len(decoded_images))
        else:
            content = obs_text
        self.local_messages.append({"role": self.mtrl_role, "content": content})

        after_ids = self.build_prompt_ids(add_generation_prompt=True)
        prefix_len = common_prefix_len(before_ids, after_ids)
        if prefix_len < len(before_ids):
            fallback_messages = [{"role": self.mtrl_role, "content": content}]
            return self.build_prompt_ids_for_messages(
                fallback_messages,
                decoded_images,
                add_generation_prompt=True,
            )
        return after_ids[prefix_len:]

    def build_rebuild_payload(self) -> dict[str, Any]:
        return {
            "messages": copy.deepcopy(self.local_messages),
            "images": list(self.running_image_data),
        }
import copy
import warnings
from dataclasses import dataclass
from typing import Any, Optional

from verl_tool.agent_loop.vision_utils import decode_image_url, process_image


def load_model_config(model_path: str, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        warnings.warn(
            f"Failed to load model config from {model_path}: {exc}. Falling back to processor/path heuristics.",
            stacklevel=1,
        )
        return None


def resolve_model_family(
    configured: str,
    processor: Any,
    model_config: Any = None,
    *paths: Optional[str],
) -> str:
    if configured != "auto":
        return configured

    architectures = [str(item).lower() for item in (getattr(model_config, "architectures", None) or [])]
    model_type = str(getattr(model_config, "model_type", "") or "").lower()
    if model_type == "interns1" or any("interns1" in item for item in architectures):
        return "interns1"
    if "internvl" in model_type or any("internvl" in item for item in architectures):
        return "internvl"

    image_processor = getattr(processor, "image_processor", None) if processor is not None else None
    processor_name = processor.__class__.__name__ if processor is not None else ""
    if "InternS1" in processor_name:
        return "interns1"
    if "InternVL" in processor_name:
        return "internvl"
    if image_processor is not None and "Qwen2VLImageProcessor" in image_processor.__class__.__name__:
        return "qwen_vl"

    for path in paths:
        lowered = (path or "").lower()
        if "interns1" in lowered:
            return "interns1"
        if "internvl" in lowered:
            return "internvl"
        if "qwen" in lowered and "vl" in lowered:
            return "qwen_vl"

    return "generic"


def is_intern_family(model_family: str) -> bool:
    return (model_family or "").lower() in {"internvl", "interns1"}


def looks_like_intern_family(model_family: str, *paths: Optional[str]) -> bool:
    if is_intern_family(model_family):
        return True
    for path in paths:
        lowered = (path or "").lower()
        if "internvl" in lowered or "interns1" in lowered:
            return True
    return False


def load_processor_for_family(local_path: str, trust_remote_code: bool, model_family: str):
    from verl.utils import hf_processor

    if looks_like_intern_family(model_family, local_path):
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=False)
        if processor is not None:
            return processor
    return hf_processor(local_path, trust_remote_code=trust_remote_code)


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
    decoded = []
    for item in tool_images:
        if isinstance(item, str):
            decoded.append(decode_image_url(item))
        else:
            decoded.append(process_image(item))
    return decoded


def make_local_content_from_text_and_images(text: str, num_images: int) -> list[dict[str, str]]:
    content = []
    if text:
        segments = text.split("<image>")
    else:
        segments = [""]
    for idx, segment in enumerate(segments):
        if segment:
            content.append({"type": "text", "text": segment})
        if idx < len(segments) - 1:
            content.append({"type": "image"})
    existing_images = sum(1 for item in content if item.get("type") == "image")
    for _ in range(max(num_images - existing_images, 0)):
        content.append({"type": "image"})
    if not content:
        content.append({"type": "text", "text": ""})
    return content


@dataclass
class PromptGenerationInput:
    prompt_ids: Optional[list[int]]
    prompt_text: Optional[str]
    image_data: list[Any]
    prompt_length_estimate: int


class InternVLPromptState:
    def __init__(
        self,
        *,
        tokenizer,
        processor,
        mtrl_role: str,
        sample_kwargs: dict[str, Any],
    ) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.mtrl_role = mtrl_role
        raw_prompt = sample_kwargs["raw_prompt"]
        if hasattr(raw_prompt, "tolist"):
            raw_prompt = raw_prompt.tolist()
        self.local_messages = copy.deepcopy(raw_prompt)
        self.running_image_data = list((sample_kwargs.get("multi_modal_data") or {}).get("image", []))

    def build_prompt_text(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
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
        prompt_text = self.build_prompt_text(messages, add_generation_prompt=add_generation_prompt)
        model_inputs = self.processor(
            text=[prompt_text],
            images=image_data or None,
            return_tensors="pt",
        )
        return model_inputs["input_ids"].squeeze(0).tolist()

    def build_prompt_ids(self, *, add_generation_prompt: bool) -> list[int]:
        return self.build_prompt_ids_for_messages(
            list(self.local_messages),
            list(self.running_image_data),
            add_generation_prompt=add_generation_prompt,
        )

    def get_generation_input(self) -> PromptGenerationInput:
        prompt_ids = self.build_prompt_ids(add_generation_prompt=True)
        prompt_text = self.build_prompt_text(
            list(self.local_messages),
            add_generation_prompt=True,
        )
        return PromptGenerationInput(
            prompt_ids=prompt_ids,
            prompt_text=prompt_text,
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
            content = make_local_content_from_text_and_images(obs_text, len(decoded_images))
        else:
            content = obs_text
        self.local_messages.append({"role": self.mtrl_role, "content": content})

        after_ids = self.build_prompt_ids(add_generation_prompt=True)
        prefix_len = common_prefix_len(before_ids, after_ids)
        if prefix_len < len(before_ids):
            fallback_messages = [{"role": self.mtrl_role, "content": content}]
            return self.build_prompt_ids_for_messages(
                fallback_messages,
                decoded_images,
                add_generation_prompt=True,
            )
        return after_ids[prefix_len:]

    def build_rebuild_payload(self) -> dict[str, Any]:
        return {
            "messages": copy.deepcopy(self.local_messages),
            "images": list(self.running_image_data),
        }
