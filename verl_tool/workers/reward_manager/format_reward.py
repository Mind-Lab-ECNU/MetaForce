# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any

import numpy as np

__all__ = ["compute_format_reward"]

FORMAT_REWARD_VALUE = 0.2
TRAIN_THINK_CONTENT_PATTERN = r"(?:(?!</?(?:thinking|think)>|<answer>|<tool_call>).)*?"
INSTRUCT_THINK_CONTENT_PATTERN = r"(?:(?!</?(?:thinking|think)>|<answer>|<tool_call>).)*?"
TRAIN_THINK_PATTERN = rf"(?:<think>)?{TRAIN_THINK_CONTENT_PATTERN}</think>"
INSTRUCT_THINK_PATTERN = rf"<thinking>{INSTRUCT_THINK_CONTENT_PATTERN}</thinking>"
THINK_PREFIX_PATTERN = rf"(?:{TRAIN_THINK_PATTERN}|{INSTRUCT_THINK_PATTERN})"
ANSWER_INNER_PATTERN = r"(?:(?!</?(?:thinking|think)>|<answer>).)*?"


def _extract_messages(messages_obj: Any) -> list[dict[str, Any]] | None:
    """Normalize messages to a list of dicts."""
    if messages_obj is None:
        return None
    if isinstance(messages_obj, np.ndarray):
        try:
            messages_obj = messages_obj.item()
        except ValueError:
            messages_obj = messages_obj.tolist()
    if isinstance(messages_obj, list):
        return messages_obj
    return None


def compute_format_reward(messages_obj: Any) -> tuple[float, dict[str, Any]]:
    """Compute a format reward (+0.2 / -0.2) without attaching verbose metadata."""
    messages = _extract_messages(messages_obj)
    if messages is None:
        return -FORMAT_REWARD_VALUE, {}

    assistant_contents = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            assistant_contents.append(str(msg.get("content", "")))

    if not assistant_contents:
        return -FORMAT_REWARD_VALUE, {}

    # format_messages 记录的是模型实际输出。
    # 因此当前兼容三种前缀：
    # - train 旧格式： “思考正文...</think>”
    # - train 显式格式： “<think>思考正文...</think>”
    # - instruct 格式： “<thinking>...</thinking>”
    # 这些前缀与后续标签之间都允许 0/1/2 个换行。
    tool_pattern = re.compile(
        rf"^{THINK_PREFIX_PATTERN}\n{{0,2}}<tool_call>.*?</tool_call>$",
        re.DOTALL,
    )
    # 最后一轮 answer 可带或不带前置 thinking，
    # 但必须以单个顶层 <answer>...</answer> 结束，answer 内不能再出现 think 标签或嵌套 answer。
    answer_pattern = re.compile(
        rf"^(?:{THINK_PREFIX_PATTERN}\n{{0,2}})?<answer>{ANSWER_INNER_PATTERN}</answer>$",
        re.DOTALL,
    )

    for idx, content in enumerate(assistant_contents):
        is_last = idx == len(assistant_contents) - 1
        expected_pattern = answer_pattern if is_last else tool_pattern
        if not expected_pattern.match(content):
            return -FORMAT_REWARD_VALUE, {}

    return FORMAT_REWARD_VALUE, {}