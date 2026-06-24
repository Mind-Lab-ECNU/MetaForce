#!/usr/bin/env python3
"""
【公共工具库】
所有分析脚本都会从这里 import 函数，避免重复写相同的代码。
包含：文件读写、路径推断、数值转换、信息熵计算、工具类型归一化、LLM 调用。
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional

import requests


STEP_RE = re.compile(r"(\d+)")
DEFAULT_OBS_ERROR_MARKERS: tuple[str, ...] = ("[stderr]",)

# common.py 在 scripts/paper_tool_analysis/，往上两级是 workspace 根目录
_WORKSPACE_ROOT: Path = Path(__file__).resolve().parents[2]
_TOOL_FILE_ID: Path = (
    _WORKSPACE_ROOT / "verl_tool" / "servers" / "tools"
    / "multimodal_processor_tool_adapt_skill_id.py"
)
_TOOL_FILE_OOD: Path = (
    _WORKSPACE_ROOT / "verl_tool" / "servers" / "tools"
    / "multimodal_processor_tool_adapt_skill_ood.py"
)

_FOREVER_TOOLS_CACHE: Dict[str, FrozenSet[str]] = {}


def _parse_valid_mcp_func_names(path: Path) -> List[str]:
    """
    用 AST（不 import）从工具文件中提取 valid_mcp_func_names 列表。
    避免触发 torch / vllm 等重依赖，analysis 脚本可在轻量环境运行。
    文件不存在或解析失败时返回空列表并打印 warning。
    """
    if not path.exists():
        print(f"[common] warning: tool file not found: {path}")
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        print(f"[common] warning: failed to parse {path}: {exc}")
        return []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "valid_mcp_func_names":
                if isinstance(node.value, ast.List):
                    return [
                        elt.value
                        for elt in node.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
    print(f"[common] warning: valid_mcp_func_names not found in {path}")
    return []


def load_forever_tools(variant: str = "both") -> FrozenSet[str]:
    """
    加载 forever tools 集合（带模块级缓存，只解析一次）。
    variant 说明：
      "id"   → 仅 ID 实验工具
      "ood"  → 仅 OOD 实验工具
      "both" → 两者合并（默认，最保守）
    返回小写 frozenset，与 normalize_tool_kind 的 name.lower() 保持一致。
    """
    if "id" not in _FOREVER_TOOLS_CACHE:
        id_tools: FrozenSet[str] = frozenset(
            n.lower() for n in _parse_valid_mcp_func_names(_TOOL_FILE_ID)
        )
        ood_tools: FrozenSet[str] = frozenset(
            n.lower() for n in _parse_valid_mcp_func_names(_TOOL_FILE_OOD)
        )
        _FOREVER_TOOLS_CACHE["id"] = id_tools
        _FOREVER_TOOLS_CACHE["ood"] = ood_tools
        _FOREVER_TOOLS_CACHE["both"] = id_tools | ood_tools
    return _FOREVER_TOOLS_CACHE.get(variant, _FOREVER_TOOLS_CACHE["both"])


def detect_tool_variant(tool_names: Iterable[str]) -> str:
    """
    根据 rollout 数据中实际出现的工具名，自动推断是 ID 还是 OOD 工具集。
    """
    id_set = load_forever_tools("id")
    ood_set = load_forever_tools("ood")
    id_hits = 0
    ood_hits = 0
    for name in tool_names:
        low = (name or "").strip().lower()
        if low in id_set:
            id_hits += 1
        if low in ood_set:
            ood_hits += 1
    if id_hits == 0 and ood_hits == 0:
        return "unknown"
    if id_hits > 0 and ood_hits > 0:
        return "both"
    return "id" if id_hits >= ood_hits else "ood"


def detect_phase_from_data_sources(data_sources: Iterable[str]) -> str:
    """
    从 data_source 字段值中推断是 train 还是 eval 阶段。
    含 '_test' / '_eval' / '_val' 关键词的数据集归属于 eval。
    """
    eval_kw = ("_test", "_eval", "_val")
    train_kw = ("_train",)
    train_count = 0
    eval_count = 0
    for ds in data_sources:
        s = (ds or "").strip().lower()
        if any(k in s for k in eval_kw):
            eval_count += 1
        elif any(k in s for k in train_kw):
            train_count += 1
    if train_count == 0 and eval_count == 0:
        return "unknown"
    return "eval" if eval_count >= train_count else "train"


FOREVER_TOOLS: FrozenSet[str] = load_forever_tools("both")


def ensure_dir(path: str | Path) -> Path:
    """创建目录（如果已存在则不报错），并返回 Path 对象。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Dict[str, Any]:
    """读取 JSON 文件，要求顶层必须是字典。"""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def read_jsonl(path: str | Path) -> List[Any]:
    """读取 JSONL 文件，忽略空行和无法解析的行。"""
    rows: List[Any] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def write_json(path: str | Path, data: Any) -> None:
    """把任意数据写成 JSON 文件，中文不转义，缩进 2 格便于阅读。"""
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    """把多条字典写成 JSONL 文件（每行一个 JSON）。"""
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def infer_step_from_name(path: str | Path, prefix: str = "") -> Optional[int]:
    """
    从文件名推断训练 step 编号。
    例如：skills_10.json → 10，tool_rollouts_4.json → 4。
    """
    name = Path(path).name
    if prefix and name.startswith(prefix):
        name = name[len(prefix):]
    matches = STEP_RE.findall(name)
    if not matches:
        return None
    return int(matches[-1])


def list_files_by_step(root: str | Path, pattern: str, prefix: str = "") -> List[Path]:
    """
    列出目录下符合 pattern 的所有文件，并按 step 编号从小到大排序。
    """
    files = list(Path(root).glob(pattern))
    files.sort(key=lambda p: (infer_step_from_name(p, prefix=prefix) or -1, p.name))
    return files


def as_float(value: Any, default: float = float("nan")) -> float:
    """安全地把任意值转成 float，转换失败时返回 default（默认 NaN）。"""
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def as_int(value: Any, default: int = 0) -> int:
    """安全地把任意值转成 int。"""
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def as_bool(value: Any) -> bool:
    """
    把任意值转成布尔值，支持字符串 "true"/"yes"/"1" 等写法。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def json_object_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    从文本中提取 JSON 对象，专门用于解析 LLM 的返回内容。
    """
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*", "", cleaned).strip()
        cleaned = cleaned.rstrip("`").strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def parse_maybe_json_object(value: Any) -> Dict[str, Any]:
    """把 dict 或 JSON 字符串解析成字典；失败时返回空字典。"""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def parse_maybe_json_list(value: Any) -> List[Any]:
    """把 list 或 JSON 字符串解析成列表；失败时返回空列表。"""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def truncate_text(text: Any, limit: int = 300) -> str:
    """
    把文本截断到 limit 个字符，超出部分用 "..." 替代。
    同时把换行符替换成空格，便于单行展示。
    """
    if text is None:
        return ""
    s = str(text).replace("\n", " ").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def entropy(counts: List[float]) -> float:
    """
    计算信息熵 H = -Σ p·log(p)，用于衡量分布的多样性。
    """
    arr = [c for c in counts if c > 0]
    total = sum(arr)
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in arr:
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent


def normalize_tool_kind(
    tool_name: str,
    tool_kind: Optional[str] = None,
    variant: str = "both",
) -> str:
    """
    把工具名归一化为标准的工具类型（tool_kind）。
    """
    if isinstance(tool_kind, str) and tool_kind.strip():
        return tool_kind.strip()
    name = (tool_name or "").strip().lower()
    if name == "create_skill":
        return "create_skill"
    if name == "run_skill":
        return "run_skill"
    if name in load_forever_tools(variant if variant in ("id", "ood", "both") else "both"):
        return "forever_tool"
    return "other_tool"


def obs_has_error(obs: Any, markers: Iterable[str] = DEFAULT_OBS_ERROR_MARKERS) -> bool:
    """
    用字符串 marker 检查工具 observation 是否带错误痕迹。
    默认沿用 run_skill 的 "[stderr]" 规则。
    """
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
    for marker in markers:
        marker_text = str(marker or "").strip().lower()
        if marker_text and marker_text in lowered:
            return True
    return False


def heuristic_skill_category(name: str, description: str, samples: str = "") -> str:
    """
    用关键词规则给 skill 打类别标签，不需要 LLM，完全离线运行。
    """
    text = " ".join([name or "", description or "", samples or ""]).lower()
    rules = [
        ("ocr_document", ["ocr", "document", "pdf", "mineru", "paddle"]),
        ("chart_table", ["chart", "table", "series", "trend", "plot", "csv"]),
        ("geometry_math", ["geometry", "math", "equation", "compute", "fraction"]),
        ("code_execution", ["python", "script", "code", "program", "bash"]),
        ("image_editing", ["image", "edit", "segmentation", "sam", "crop"]),
        ("retrieval_search", ["search", "retrieve", "web", "browser", "bing", "google"]),
        ("reasoning_postprocess", ["summarize", "report", "brief", "extract", "analyze"]),
    ]
    for cat, keys in rules:
        if any(k in text for k in keys):
            return cat
    return "other"


def normalize_archive_reason_group(reason: Any) -> str:
    """
    把 skill archive 的 reason 归并成论文里更稳定的死亡/清理类别。
    """
    text = str(reason or "").strip().lower()
    if not text:
        return "unknown"
    if text == "validation_cleanup":
        return "validation_cleanup"
    if text == "incorrect_trajectory":
        return "incorrect_trajectory"
    if text == "duplicate_existing":
        return "duplicate_existing"
    if text == "fit_exit":
        return "fit_exit"
    if text == "snapshot_disappearance":
        return "snapshot_disappearance"
    if text in {"periodic_step_cleanup", "step_end", "unit_test"}:
        return "reset"
    if "dedup" in text:
        return "dedup"
    return "unknown"


def _swap_anchor(path: Path, src_anchor: str, dst_anchor: str) -> Optional[Path]:
    parts = list(path.expanduser().resolve().parts)
    lowered = [part.lower() for part in parts]
    try:
        idx = lowered.index(src_anchor.lower())
    except ValueError:
        return None
    new_parts = parts[:idx] + [dst_anchor] + parts[idx + 1 :]
    return Path(*new_parts)


def infer_skill_store_dir(
    *,
    tool_log_dir: str | Path | None = None,
    rollout_data_dir: str | Path | None = None,
) -> Optional[Path]:
    """
    从 tool_log_dir 或 rollout/validation_data 目录推断 skills/<phase>/<run_id> 目录。
    只返回已存在的目录；推断不到时返回 None。
    """
    candidates: List[Path] = []
    for raw in (tool_log_dir, rollout_data_dir):
        if not raw:
            continue
        path = Path(raw).expanduser().resolve()
        for anchor in ("tool_logs", "rollout", "validation_data"):
            swapped = _swap_anchor(path, anchor, "skills")
            if swapped is not None:
                candidates.append(swapped)
    seen: set[str] = set()
    for candidate in candidates:
        norm = str(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        if candidate.exists():
            return candidate
    return None


def skill_archive_events_path(skill_store_dir: str | Path | None) -> Optional[Path]:
    """返回 `_skill_archive/_events.jsonl` 的路径；不存在时返回 None。"""
    if not skill_store_dir:
        return None
    store_dir = Path(skill_store_dir).expanduser().resolve()
    path = store_dir / "_skill_archive" / "_events.jsonl"
    if path.exists():
        return path
    return None


def call_llm_json(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: Optional[str] = None,
    timeout: int = 60,
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> Optional[Dict[str, Any]]:
    """
    调用 OpenAI 兼容接口，要求模型返回 JSON 格式的结果。
    """
    if not base_url:
        return None
    api_url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key or os.environ.get('OPENAI_API_KEY', '')}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return json_object_from_text(str(content))
    except Exception:
        return None
