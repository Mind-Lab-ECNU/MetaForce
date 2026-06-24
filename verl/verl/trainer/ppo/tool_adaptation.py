# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Skill Adaptation Module for Multimodal Tool-Augmented RL Training.

This module implements dynamic skill adaptation:
- Load skills at batch start and inject into prompts
- Collect skill usage statistics after reward computation
- Select next round skills based on usage (IQR)
- Persist active skills and usage metadata
"""

import copy
from datetime import datetime
import hashlib
import json
import os
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import yaml
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl import DataProto
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask


MAX_SKILLS = 50
TOOL_SELECTION_SEED = 42
SEMANTIC_DEDUP_ENABLED = True
SEMANTIC_DEDUP_TIMEOUT_SEC = 10
SEMANTIC_DEDUP_MAX_RETRIES_PER_PAIR = 3
SEMANTIC_DEDUP_MODEL = "Qwen3-VL-235B-A22B-Instruct"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _get_skill_store_dir(skill_store_dir: Optional[str] = None) -> Path:
    if skill_store_dir:
        return Path(skill_store_dir)
    env_dir = os.environ.get("VERL_SKILL_STORE_DIR")
    if env_dir:
        return Path(env_dir)
    return _repo_root() / "skills"


def _parse_skill_frontmatter(text: str) -> Dict[str, Any]:
    if not isinstance(text, str) or not text.startswith("---"):
        return {}
    try:
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}
        return yaml.safe_load(parts[1]) or {}
    except Exception:
        return {}


def _sanitize_skill_name(name: str) -> str:
    if not isinstance(name, str):
        name = ""
    value = name.strip().lower()
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "auto-skill"


def _load_skill_index(store_dir: Path) -> Dict[str, Dict[str, Any]]:
    index_path = store_dir / "index.json"
    if not index_path.exists():
        return {}
    try:
        with index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_skill_index(store_dir: Path, index: Dict[str, Dict[str, Any]]) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    index_path = store_dir / "index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def _load_active_skills(store_dir: Path) -> List[str]:
    active_path = store_dir / "active_skills.json"
    if active_path.exists():
        try:
            with active_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(item) for item in data]
        except Exception:
            pass
    return []


def _save_active_skills(store_dir: Path, names: List[str]) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    active_path = store_dir / "active_skills.json"
    with active_path.open("w", encoding="utf-8") as f:
        json.dump(list(names), f, ensure_ascii=False, indent=2)


def _scan_skill_dirs(store_dir: Path) -> List[Dict[str, Any]]:
    skills: List[Dict[str, Any]] = []
    if not store_dir.exists():
        return skills
    index = _load_skill_index(store_dir)
    for child in store_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
            continue
        skill_md = child / "SKILL.md"
        if not skill_md.exists():
            continue
        content = skill_md.read_text(encoding="utf-8")
        meta = _parse_skill_frontmatter(content)
        # Use directory name as canonical runtime skill name to keep
        # consistency with active_skills.json and promoted target directory.
        name = _sanitize_skill_name(str(child.name))
        description = str(meta.get("description") or index.get(name, {}).get("description", "")).strip()
        used_times = int(index.get(name, {}).get("used_times", 0))
        skills.append(
            {
                "name": name,
                "description": description,
                "path": str(child),
                "used_times": used_times,
                "status": index.get(name, {}).get("status", "active"),
            }
        )
    return skills


def _get_skill_pending_dir(store_dir: Path) -> Path:
    return store_dir / "_pending"


def _get_skill_archive_dir(store_dir: Path) -> Path:
    return store_dir / "_skill_archive"


def _workspace_root() -> Path:
    return _repo_root()


def _rewrite_skill_md_name(skill_md_path: Path, canonical_name: str) -> None:
    """Rewrite frontmatter `name` in SKILL.md to canonical runtime name."""
    if not skill_md_path.exists():
        return

    try:
        content = skill_md_path.read_text(encoding="utf-8")
    except Exception:
        return

    if not isinstance(content, str) or not content.startswith("---"):
        return

    parts = content.split("---", 2)
    if len(parts) < 3:
        return

    try:
        meta = yaml.safe_load(parts[1]) or {}
    except Exception:
        return
    if not isinstance(meta, dict):
        return

    target_name = _sanitize_skill_name(canonical_name)
    if _sanitize_skill_name(str(meta.get("name", ""))) == target_name:
        return

    meta["name"] = target_name
    rewritten_frontmatter = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
    rewritten_content = f"---\n{rewritten_frontmatter}\n---{parts[2]}"
    try:
        skill_md_path.write_text(rewritten_content, encoding="utf-8")
    except Exception:
        return


def _normalize_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _optional_normalize_path(path: Optional[Path | str]) -> str:
    if path in {None, ""}:
        return ""
    try:
        return _normalize_path(Path(path))
    except Exception:
        return str(path)


def _read_skill_name_from_dir(skill_dir: Path) -> str:
    skill_md = skill_dir / "SKILL.md"
    if skill_md.exists():
        try:
            content = skill_md.read_text(encoding="utf-8")
        except Exception:
            content = ""
        meta = _parse_skill_frontmatter(content)
        name = meta.get("name") or skill_dir.name
    else:
        name = skill_dir.name
    return _sanitize_skill_name(str(name))


def append_skill_archive_event(
    *,
    store_dir: Path,
    event: str,
    reason: str,
    skill_name: str = "",
    source_path: Optional[Path | str] = None,
    runtime_path: Optional[Path | str] = None,
    archive_path: Optional[Path | str] = None,
    global_step: Optional[int] = None,
    snapshot_copied: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        archive_dir = _get_skill_archive_dir(store_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "timestamp": datetime.utcnow().isoformat(timespec="microseconds"),
            "event": str(event).strip(),
            "reason": str(reason).strip(),
            "skill_name": _sanitize_skill_name(skill_name) if skill_name else "",
            "source_path": _optional_normalize_path(source_path),
            "runtime_path": _optional_normalize_path(runtime_path),
            "archive_path": _optional_normalize_path(archive_path),
            "global_step": global_step,
            "snapshot_copied": bool(snapshot_copied),
            "metadata": metadata if isinstance(metadata, dict) else {},
        }

        events_path = archive_dir / "_events.jsonl"
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        print(
            f"[ToolAdaptation] Failed to append skill archive event "
            f"event={event} reason={reason} skill_name={skill_name}: {exc}"
        )


def archive_skill_event(
    *,
    skill_path: Path | str,
    event: str,
    reason: str,
    store_dir: Path,
    global_step: Optional[int] = None,
    source_path: Optional[Path | str] = None,
    runtime_path: Optional[Path | str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    archive_path = ""
    snapshot_copied = False
    skill_dir = Path(skill_path)
    skill_name = _read_skill_name_from_dir(skill_dir)

    if skill_dir.exists() and skill_dir.is_dir():
        try:
            archive_dir = _get_skill_archive_dir(store_dir)
            archive_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            safe_event = _sanitize_skill_name(event)
            safe_reason = _sanitize_skill_name(reason)
            archive_name = f"{skill_name}__{safe_event}__{safe_reason}__{stamp}"
            archive_target = archive_dir / archive_name
            suffix = 1
            while archive_target.exists():
                archive_target = archive_dir / f"{archive_name}_{suffix}"
                suffix += 1
            shutil.copytree(skill_dir, archive_target)
            archive_path = str(archive_target)
            snapshot_copied = True
        except Exception as exc:
            print(
                f"[ToolAdaptation] Failed to snapshot archived skill "
                f"event={event} reason={reason} path={skill_dir}: {exc}"
            )

    append_skill_archive_event(
        store_dir=store_dir,
        event=event,
        reason=reason,
        skill_name=skill_name,
        source_path=source_path if source_path is not None else skill_dir,
        runtime_path=runtime_path if runtime_path is not None else skill_dir,
        archive_path=archive_path,
        global_step=global_step,
        snapshot_copied=snapshot_copied,
        metadata=metadata,
    )


def _iter_pending_skill_dirs(store_dir: Path) -> List[Path]:
    pending_root = _get_skill_pending_dir(store_dir)
    skill_dirs: List[Path] = []
    if not pending_root.exists():
        return skill_dirs

    for trajectory_dir in pending_root.iterdir():
        if not trajectory_dir.is_dir():
            continue
        for child in trajectory_dir.iterdir():
            if child.is_dir():
                skill_dirs.append(child)
    return skill_dirs


def _compute_skill_content_hash(skill_dir: Path) -> str:
    if not skill_dir.exists() or not skill_dir.is_dir():
        return ""

    files = sorted(
        [p for p in skill_dir.rglob("*") if p.is_file()],
        key=lambda p: str(p.relative_to(skill_dir)).replace("\\", "/"),
    )
    if not files:
        return ""

    hasher = hashlib.sha256()
    for file_path in files:
        rel_path = str(file_path.relative_to(skill_dir)).replace("\\", "/")
        hasher.update(rel_path.encode("utf-8"))
        hasher.update(b"\0")
        try:
            hasher.update(file_path.read_bytes())
        except Exception:
            hasher.update(b"<read-error>")
        hasher.update(b"\0")
    return hasher.hexdigest()


def _parse_json_object_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*", "", cleaned).strip()
        cleaned = cleaned.rstrip("`").strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
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


def _load_semantic_dedup_base_url() -> Optional[str]:
    url_json_path = _workspace_root() / "verl_tool" / "servers" / "tools" / "url.json"
    if not url_json_path.exists():
        return None
    try:
        with url_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        base_url = str(data.get(SEMANTIC_DEDUP_MODEL, "")).strip()
        return base_url or None
    except Exception:
        return None


def _extract_pending_skill_semantic_info(skill_dir: Path) -> Dict[str, Any]:
    skill_md_path = skill_dir / "SKILL.md"
    skill_md_content = ""
    if skill_md_path.exists():
        try:
            skill_md_content = skill_md_path.read_text(encoding="utf-8")
        except Exception:
            skill_md_content = ""

    frontmatter = _parse_skill_frontmatter(skill_md_content)
    skill_name = _sanitize_skill_name(str(frontmatter.get("name") or skill_dir.name))
    skill_description = str(frontmatter.get("description", "")).strip()

    spec_path = skill_dir / "SKILL_SPEC.json"
    spec_data: Dict[str, Any] = {}
    if spec_path.exists():
        try:
            data = json.loads(spec_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                spec_data = data
        except Exception:
            spec_data = {}

    raw_scripts = spec_data.get("scripts")
    scripts_summary: List[Dict[str, Any]] = []
    if isinstance(raw_scripts, list):
        for item in raw_scripts:
            if not isinstance(item, dict):
                continue
            script_path = str(item.get("path", "")).strip()
            params = item.get("params")
            params_summary: List[Dict[str, Any]] = []
            if isinstance(params, list):
                for param in params:
                    if not isinstance(param, dict):
                        continue
                    params_summary.append(
                        {
                            "name": str(param.get("name", "")).strip(),
                            "type": str(param.get("type", "")).strip(),
                            "required": bool(param.get("required", False)),
                        }
                    )
            scripts_summary.append(
                {
                    "path": script_path,
                    "params": sorted(params_summary, key=lambda p: (p["name"], p["type"], p["required"])),
                }
            )
    scripts_summary = sorted(scripts_summary, key=lambda s: s.get("path", ""))

    semantic_payload = {
        "name": skill_name,
        "description": skill_description,
        "requires_image": spec_data.get("requires_image", None),
        "scripts": scripts_summary,
    }
    semantic_summary = json.dumps(semantic_payload, ensure_ascii=False, sort_keys=True)

    return {
        "semantic_summary": semantic_summary,
    }


def _collect_pending_skills(store_dir: Path) -> List[Dict[str, Any]]:
    pending_root = _get_skill_pending_dir(store_dir)
    if not pending_root.exists():
        return []

    pending_items: List[Dict[str, Any]] = []
    trajectory_dirs = sorted([p for p in pending_root.iterdir() if p.is_dir()], key=lambda p: str(p))
    for trajectory_dir in trajectory_dirs:
        skill_dirs = sorted([p for p in trajectory_dir.iterdir() if p.is_dir()], key=lambda p: str(p))
        for skill_dir in skill_dirs:
            semantic_info = _extract_pending_skill_semantic_info(skill_dir)
            pending_items.append(
                {
                    "name": _read_skill_name_from_dir(skill_dir),
                    "path": _normalize_path(skill_dir),
                    "trajectory_id": trajectory_dir.name,
                    "content_hash": _compute_skill_content_hash(skill_dir),
                    "semantic_summary": semantic_info["semantic_summary"],
                }
            )
    return pending_items


def _semantic_llm_judge_duplicate(
    base_url: str,
    lhs: Dict[str, Any],
    rhs: Dict[str, Any],
    timeout_sec: int,
) -> Tuple[bool, str, str]:
    api_url = base_url.rstrip("/") + "/v1/chat/completions"
    api_key = os.environ.get("VERL_SEMANTIC_DEDUP_API_KEY", "token-abc123")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": SEMANTIC_DEDUP_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a skill deduplication judge. Decide whether two skills are functionally equivalent and which one is better. "
                    "Return JSON only: {\"duplicate\": bool, \"winner\": \"a\"|\"b\", \"reason\": string}."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Decision criterion: if two skills are functionally equivalent, treat them as duplicates "
                    "(same task intent, key inputs/outputs, and core workflow).\n"
                    "Skill A summary:\n"
                    f"{lhs.get('semantic_summary', '')}\n\n"
                    "Skill B summary:\n"
                    f"{rhs.get('semantic_summary', '')}\n"
                ),
            },
        ],
        "max_tokens": 8192,
        "temperature": 0.0,
        "stream": False,
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=timeout_sec, verify=False)
    response.raise_for_status()
    body = response.json()
    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed = _parse_json_object_from_text(content)
    if not parsed:
        raise ValueError(f"invalid semantic dedup response: {content}")
    duplicate = bool(parsed.get("duplicate", False))
    winner = str(parsed.get("winner", "")).strip().lower()
    reason = str(parsed.get("reason", "")).strip()
    return duplicate, winner, reason


def _semantic_group_pending_skills(
    base_url: str,
    pending_items: List[Dict[str, Any]],
    timeout_sec: int,
) -> List[List[str]]:
    api_url = base_url.rstrip("/") + "/v1/chat/completions"
    api_key = os.environ.get("VERL_SEMANTIC_DEDUP_API_KEY", "token-abc123")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    candidates: List[Dict[str, str]] = []
    valid_ids: List[str] = []
    id_to_path: Dict[str, str] = {}
    for idx, item in enumerate(pending_items, start=1):
        path = str(item.get("path", "")).strip()
        if not path:
            continue
        skill_id = f"s{idx}"
        valid_ids.append(skill_id)
        id_to_path[skill_id] = path
        candidates.append(
            {
                "id": skill_id,
                "semantic_summary": str(item.get("semantic_summary", "")),
            }
        )

    if not candidates:
        return []

    payload = {
        "model": SEMANTIC_DEDUP_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a skill deduplication grouper. Group skills by high semantic overlap and same core task intent. "
                    "Group when skills solve the same user-facing task with substantially overlapping inputs/outputs and workflow, "
                    "even if implementation details differ. "
                    "Containment/subsumption counts when one skill clearly includes most capabilities of the other. "
                    "If one skill is a practical variant or refactor of another for the same task, prefer grouping them together. "
                    "Return JSON only using candidate ids: {\"groups\": [[\"s1\"], [\"s2\",\"s3\"]]}."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Rules:\n"
                    "1) Every id must appear exactly once across groups.\n"
                    "2) Singleton groups are allowed.\n"
                    "3) Place items in the same group when they share the same task intent and most key "
                    "inputs/outputs/workflow overlap, including implementation variants.\n"
                    "4) Keep items separate only when task intent or expected outputs are materially different.\n"
                    "Candidates:\n"
                    f"{json.dumps(candidates, ensure_ascii=False)}"
                ),
            },
        ],
        "max_tokens": 8192,
        "temperature": 0.0,
        "stream": False,
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=timeout_sec, verify=False)
    response.raise_for_status()
    body = response.json()
    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed = _parse_json_object_from_text(content)
    if not parsed:
        raise ValueError(f"invalid semantic grouping response: {content}")

    raw_groups = parsed.get("groups")
    if not isinstance(raw_groups, list):
        raise ValueError(f"invalid semantic grouping groups field: {content}")

    id_set = set(valid_ids)
    seen: set[str] = set()
    groups: List[List[str]] = []
    for group in raw_groups:
        if not isinstance(group, list):
            raise ValueError(f"invalid semantic grouping item: {content}")
        normalized_group: List[str] = []
        for item in group:
            skill_id = str(item).strip()
            if not skill_id or skill_id not in id_set:
                raise ValueError(f"invalid semantic grouping id: {skill_id}")
            if skill_id in seen:
                raise ValueError(f"duplicate id in semantic grouping: {skill_id}")
            seen.add(skill_id)
            normalized_group.append(id_to_path[skill_id])
        if not normalized_group:
            raise ValueError("empty group in semantic grouping response")
        groups.append(normalized_group)

    if seen != id_set:
        missing = sorted(id_set - seen)
        raise ValueError(f"semantic grouping does not cover all ids, missing={missing}")
    return groups


def _semantic_pick_group_winner(
    base_url: str,
    group_paths: List[str],
    timeout_sec: int,
) -> Tuple[str, str]:
    api_url = base_url.rstrip("/") + "/v1/chat/completions"
    api_key = os.environ.get("VERL_SEMANTIC_DEDUP_API_KEY", "token-abc123")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    candidates: List[Dict[str, str]] = []
    valid_ids: List[str] = []
    id_to_path: Dict[str, str] = {}
    for idx, path in enumerate(group_paths, start=1):
        skill_id = f"s{idx}"
        valid_ids.append(skill_id)
        id_to_path[skill_id] = path
        skill_dir = Path(path)
        skill_md_path = skill_dir / "SKILL.md"
        spec_path = skill_dir / "SKILL_SPEC.json"
        try:
            skill_md = skill_md_path.read_text(encoding="utf-8") if skill_md_path.exists() else ""
        except Exception:
            skill_md = ""
        try:
            skill_spec = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
        except Exception:
            skill_spec = ""
        candidates.append(
            {
                "id": skill_id,
                "skill_md": skill_md,
                "skill_spec_json": skill_spec,
            }
        )

    payload = {
        "model": SEMANTIC_DEDUP_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a skill deduplication judge. Pick exactly one winner id from the candidate ids. "
                    "Prefer the more complete and broadly covering skill when duplicates overlap, including containment cases. "
                    "Return JSON only: {\"winner\": \"s1\", \"reason\": \"short reason\"}."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Choose one winner id from this list only:\n"
                    f"{json.dumps(valid_ids, ensure_ascii=False)}\n\n"
                    "Candidates (full skill content):\n"
                    f"{json.dumps(candidates, ensure_ascii=False)}"
                ),
            },
        ],
        "max_tokens": 8192,
        "temperature": 0.0,
        "stream": False,
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=timeout_sec, verify=False)
    response.raise_for_status()
    body = response.json()
    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed = _parse_json_object_from_text(content)
    if not parsed:
        raise ValueError(f"invalid semantic winner response: {content}")
    winner_id = str(parsed.get("winner", "")).strip()
    reason = str(parsed.get("reason", "")).strip()
    if winner_id not in set(valid_ids):
        raise ValueError(f"semantic winner id not in group: {winner_id}")
    return id_to_path[winner_id], reason


def _resolve_redirect(path: str, redirects: Dict[str, str]) -> str:
    seen = set()
    target = path
    while target in redirects and target not in seen:
        seen.add(target)
        target = redirects[target]
    return target


def _collapse_redirects(redirects: Dict[str, str]) -> Dict[str, str]:
    collapsed: Dict[str, str] = {}
    for src, dst in redirects.items():
        collapsed[src] = _resolve_redirect(dst, redirects)
    return collapsed


def _deduplicate_pending_skills_by_hash(store_dir: Path) -> Dict[str, str]:
    redirects: Dict[str, str] = {}
    keep_by_hash: Dict[str, Dict[str, Any]] = {}
    items = _collect_pending_skills(store_dir)

    for item in items:
        content_hash = str(item.get("content_hash", "")).strip()
        if not content_hash:
            continue
        item_path = str(item.get("path", ""))
        keeper = keep_by_hash.get(content_hash)
        if keeper is None:
            keep_by_hash[content_hash] = item
            continue

        keep_path = str(keeper.get("path", ""))
        if not item_path or not keep_path or item_path == keep_path:
            continue

        try:
            archive_skill_event(
                skill_path=item_path,
                event="deleted",
                reason="hash_dedup",
                store_dir=store_dir,
                metadata={"keep_path": _optional_normalize_path(keep_path)},
            )
            shutil.rmtree(item_path, ignore_errors=True)
            redirects[item_path] = keep_path
            print(
                f"[ToolAdaptation] Deduplicated pending skill by hash "
                f"hash={content_hash[:8]} drop={item_path} keep={keep_path}"
            )
        except Exception as exc:
            print(f"[ToolAdaptation] Failed to deduplicate pending skill by hash at {item_path}: {exc}")

    return redirects


def _deduplicate_pending_skills(store_dir: Path) -> Dict[str, str]:
    pending_items = _collect_pending_skills(store_dir)
    pending_total = len(pending_items)
    redirects: Dict[str, str] = {}
    hash_deduped = 0
    semantic_checked = 0
    semantic_deduped = 0
    llm_failures = 0
    fallback_used = False

    if pending_total <= 1:
        print(
            "[ToolAdaptation] Pending dedup summary: "
            f"pending_total={pending_total} hash_deduped=0 semantic_checked=0 "
            "semantic_deduped=0 llm_failures=0 fallback_used=False"
        )
        return redirects

    global_hash_fallback = False
    if not SEMANTIC_DEDUP_ENABLED:
        global_hash_fallback = True
        fallback_used = True
    else:
        base_url = _load_semantic_dedup_base_url()
        if not base_url:
            global_hash_fallback = True
            fallback_used = True
            print("[ToolAdaptation] Semantic dedup URL missing, fallback to hash dedup.")
        else:
            alive_by_path: Dict[str, Dict[str, Any]] = {
                str(item.get("path", "")): item for item in pending_items if str(item.get("path", ""))
            }
            groups: Optional[List[List[str]]] = None
            stage1_exc: Optional[Exception] = None
            for attempt in range(1, SEMANTIC_DEDUP_MAX_RETRIES_PER_PAIR + 1):
                try:
                    groups = _semantic_group_pending_skills(
                        base_url=base_url,
                        pending_items=pending_items,
                        timeout_sec=SEMANTIC_DEDUP_TIMEOUT_SEC,
                    )
                    stage1_exc = None
                    break
                except Exception as exc:
                    stage1_exc = exc
                    if attempt < SEMANTIC_DEDUP_MAX_RETRIES_PER_PAIR:
                        print(
                            "[ToolAdaptation] Semantic grouping retry "
                            f"{attempt}/{SEMANTIC_DEDUP_MAX_RETRIES_PER_PAIR} failed: {exc}"
                        )
                        continue
                    llm_failures += 1
                    fallback_used = True
                    global_hash_fallback = True
                    print(
                        "[ToolAdaptation] Semantic grouping failed after retries: "
                        f"{exc}. Fallback to hash dedup."
                    )

            if stage1_exc is None and groups is not None:
                print(
                    "[ToolAdaptation] Semantic grouping complete "
                    f"group_count={len(groups)}"
                )
                for group in groups:
                    active_group: List[str] = []
                    for path in group:
                        if path in alive_by_path and path not in active_group:
                            active_group.append(path)
                    if len(active_group) <= 1:
                        continue

                    winner_path = ""
                    winner_reason = ""
                    stage2_exc: Optional[Exception] = None
                    for attempt in range(1, SEMANTIC_DEDUP_MAX_RETRIES_PER_PAIR + 1):
                        try:
                            winner_path, winner_reason = _semantic_pick_group_winner(
                                base_url=base_url,
                                group_paths=active_group,
                                timeout_sec=SEMANTIC_DEDUP_TIMEOUT_SEC,
                            )
                            stage2_exc = None
                            break
                        except Exception as exc:
                            stage2_exc = exc
                            if attempt < SEMANTIC_DEDUP_MAX_RETRIES_PER_PAIR:
                                print(
                                    "[ToolAdaptation] Semantic winner retry "
                                    f"{attempt}/{SEMANTIC_DEDUP_MAX_RETRIES_PER_PAIR} failed: {exc}"
                                )
                                continue
                            llm_failures += 1
                            fallback_used = True
                            print(
                                "[ToolAdaptation] Semantic winner failed after retries: "
                                f"{exc}. Fallback to group hash dedup."
                            )

                    if stage2_exc is not None:
                        keep_by_hash: Dict[str, str] = {}
                        for path in active_group:
                            item = alive_by_path.get(path)
                            if item is None:
                                continue
                            content_hash = str(item.get("content_hash", "")).strip()
                            if not content_hash:
                                continue
                            keep_path = keep_by_hash.get(content_hash)
                            if keep_path is None:
                                keep_by_hash[content_hash] = path
                                continue
                            if path == keep_path:
                                continue
                            drop_path = path
                            if drop_path not in alive_by_path or keep_path not in alive_by_path:
                                continue
                            try:
                                archive_skill_event(
                                    skill_path=drop_path,
                                    event="deleted",
                                    reason="semantic_dedup_fallback",
                                    store_dir=store_dir,
                                    metadata={"keep_path": _optional_normalize_path(keep_path)},
                                )
                                shutil.rmtree(drop_path, ignore_errors=True)
                                redirects[drop_path] = keep_path
                                alive_by_path.pop(drop_path, None)
                                hash_deduped += 1
                                print(
                                    "[ToolAdaptation] Deduplicated pending skill by group_hash_fallback "
                                    f"drop={drop_path} keep={keep_path}"
                                )
                            except Exception as exc:
                                print(
                                    f"[ToolAdaptation] Failed to deduplicate pending skill by group hash at {drop_path}: {exc}"
                                )
                        continue

                    semantic_checked += 1
                    keep_path = winner_path
                    if not keep_path or keep_path not in alive_by_path:
                        continue
                    for path in active_group:
                        if path == keep_path or path not in alive_by_path:
                            continue
                        drop_path = path
                        try:
                            archive_skill_event(
                                skill_path=drop_path,
                                event="deleted",
                                reason="semantic_dedup",
                                store_dir=store_dir,
                                metadata={
                                    "keep_path": _optional_normalize_path(keep_path),
                                    "winner_reason": winner_reason,
                                },
                            )
                            shutil.rmtree(drop_path, ignore_errors=True)
                            redirects[drop_path] = keep_path
                            alive_by_path.pop(drop_path, None)
                            semantic_deduped += 1
                            print(
                                "[ToolAdaptation] Deduplicated pending skill by semantic_group_winner "
                                f"drop={drop_path} keep={keep_path} reason={winner_reason}"
                            )
                        except Exception as exc:
                            print(f"[ToolAdaptation] Failed to deduplicate pending skill by semantic at {drop_path}: {exc}")

    if global_hash_fallback:
        hash_redirects = _deduplicate_pending_skills_by_hash(store_dir)
        hash_deduped += len(hash_redirects)
        redirects.update(hash_redirects)

    redirects = _collapse_redirects(redirects)
    print(
        "[ToolAdaptation] Pending dedup summary: "
        f"pending_total={pending_total} hash_deduped={hash_deduped} "
        f"semantic_checked={semantic_checked} semantic_deduped={semantic_deduped} "
        f"llm_failures={llm_failures} fallback_used={fallback_used}"
    )
    return redirects


def _cleanup_empty_pending_dirs(store_dir: Path) -> None:
    pending_root = _get_skill_pending_dir(store_dir)
    if not pending_root.exists():
        return
    for trajectory_dir in pending_root.iterdir():
        if not trajectory_dir.is_dir():
            continue
        try:
            next(trajectory_dir.iterdir())
        except StopIteration:
            try:
                trajectory_dir.rmdir()
            except Exception:
                continue


def _reserve_unique_skill_name(base_name: str, store_dir: Path, reserved_names: set[str]) -> str:
    sanitized_base = _sanitize_skill_name(base_name)
    candidate = sanitized_base
    idx = 2
    while candidate in reserved_names or (store_dir / candidate).exists():
        candidate = _sanitize_skill_name(f"{sanitized_base}-{idx}")
        idx += 1
    reserved_names.add(candidate)
    return candidate


def format_tools_for_prompt(skills: List[Dict]) -> str:
    """Format dynamic skills into OpenAI function calling format."""
    formatted: List[Dict[str, Any]] = []
    for skill in skills:
        name = skill.get("name")
        description = skill.get("description", "")
        if not name:
            continue
        formatted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        )

    if not formatted:
        return ""

    return json.dumps(formatted, ensure_ascii=False, indent=2)


@dataclass
class ToolAdaptationState:
    """Skill adaptation state management."""

    current_skills: List[Dict] = field(default_factory=list)  # Skills for current round (A)
    next_skills: List[Dict] = field(default_factory=list)  # Skills for next round (B)
    skill_store_dir: str = ""
    has_pending_update: bool = False
    pending_index: Optional[Dict[str, Dict[str, Any]]] = None




def _build_tools_block(tools_prompt: str) -> str:
    return f"\n<tools>\n{tools_prompt}\n</tools>"


def _inject_tools_into_text(content: str, tools_prompt: str) -> str:
    """Inject tools into plain text content."""
    if "</tools>" in content:
        # Find the last </tools> tag
        last_idx = content.rfind("</tools>")
        # Insert before </tools>, adding comma to connect with existing tools.
        # Add a marker sentence so the model knows following injected entries are skills.
        injection = (
            ",\n"
            "/* Skills Notice: the following function signatures are dynamic skills. */\n"
            + tools_prompt
            + "\n"
        )
        return content[:last_idx] + injection + content[last_idx:]

    # No tools tag found, create new tools block
    return content + _build_tools_block(tools_prompt)


def _inject_tools_into_content(content: Any, tools_prompt: str) -> Any:
    """
    Inject tools into system message content for both text and multimodal formats.
    
    Args:
        content: Original content, either str or multimodal list[dict]
        tools_prompt: Formatted tools JSON string to inject (without <tools> tags)
    
    Returns:
        Modified content with tools injected, preserving original content shape
    """
    if isinstance(content, str):
        return _inject_tools_into_text(content, tools_prompt)

    if isinstance(content, list):
        # In multimodal format, text chunks are stored as:
        # {"type": "text", "text": "..."}
        for idx in range(len(content) - 1, -1, -1):
            block = content[idx]
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if not isinstance(text, str):
                continue
            if "</tools>" in text:
                block["text"] = _inject_tools_into_text(text, tools_prompt)
                return content

        # No </tools> found in any writable text block, append a new text block.
        content.append({"type": "text", "text": _build_tools_block(tools_prompt)})
        return content

    raise TypeError(
        f"[ToolAdaptation] system message content must be str or list, got {type(content).__name__}"
    )


def _modify_raw_prompt_messages(
    messages: List[Dict],
    tools_prompt: str,
) -> List[Dict]:
    """
    Modify messages list to inject tools into system message.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        tools_prompt: Formatted tools JSON string to inject
    
    Returns:
        Modified messages list (deep copied)
    """
    messages = copy.deepcopy(messages)
    for msg in messages:
        if isinstance(msg, dict) and msg.get('role') == 'system':
            content = msg.get('content', '')
            msg['content'] = _inject_tools_into_content(content, tools_prompt)
            break  # Only modify first system message
    return messages


def inject_tools_to_batch(
    batch: DataProto,
    tools_prompt: str,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin] = None,
    max_prompt_length: int = 4096,
    truncation: str = "left",
    apply_chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> DataProto:
    """
    Inject skills into batch's system prompts and re-tokenize.

    This function:
    1. Modifies raw_prompt to inject tools before the </tools> tag
    2. Re-tokenizes the modified prompt using tokenizer/processor
    3. Updates input_ids, attention_mask, position_ids in the batch

    Args:
        batch: DataProto batch containing samples
        tools_prompt: Formatted tools JSON string to inject (content only, no <tools> tags)
        tokenizer: Tokenizer for re-encoding
        processor: Optional processor for multimodal data
        max_prompt_length: Maximum prompt length for truncation
        truncation: Truncation strategy ('left', 'right', 'middle', 'error')
        apply_chat_template_kwargs: Extra kwargs for apply_chat_template

    Returns:
        Modified DataProto batch with injected tools and re-tokenized data
    """
    if not tools_prompt:
        return batch

    if not hasattr(batch, "non_tensor_batch") or "raw_prompt" not in batch.non_tensor_batch:
        raise ValueError(
            "[ToolAdaptation] raw_prompt not found in batch. "
            "Please enable data.return_raw_chat=True so raw_prompt (messages) is included."
        )

    batch_size = len(batch)
    print(f"[ToolAdaptation] Re-tokenizing {batch_size} samples with injected tools...")
    apply_chat_template_kwargs = apply_chat_template_kwargs or {}

    # Collect new tokenized data
    modified_raw_prompts = list(batch.non_tensor_batch["raw_prompt"])
    new_input_ids_list = []
    new_attention_mask_list = []
    new_position_ids_list = []
    new_raw_prompt_ids_list = []

    for i in range(batch_size):
        raw_prompt = batch.non_tensor_batch['raw_prompt'][i]
        
        # Get multimodal data if available
        multi_modal_data = None
        if 'multi_modal_data' in batch.non_tensor_batch:
            multi_modal_data = batch.non_tensor_batch['multi_modal_data'][i]
        
        # Modify raw_prompt to inject tools
        if isinstance(raw_prompt, list):
            # raw_prompt is messages format (List[Dict])
            modified_messages = _modify_raw_prompt_messages(raw_prompt, tools_prompt)
            modified_raw_prompts[i] = modified_messages
            messages = modified_messages
        else:
            raise TypeError(
                f"[ToolAdaptation] raw_prompt must be a list of messages, got {type(raw_prompt)} at index {i}. "
                "Please ensure data.return_raw_chat=True so raw_prompt is emitted as messages."
            )

        # Re-tokenize the modified prompt
        if processor is not None:
            # Use processor for multimodal data (image-only).
            raw_prompt_str = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **apply_chat_template_kwargs
            )
            if multi_modal_data is None or not isinstance(multi_modal_data, dict):
                raise ValueError(
                    f"[ToolAdaptation] multi_modal_data must be a dict when processor is used, got "
                    f"{type(multi_modal_data)} at index {i}."
                )
            images = multi_modal_data.get("image", None)
            model_inputs = processor(text=[raw_prompt_str], images=images, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
        else:
            # Use tokenizer with chat template
            raw_prompt_str = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **apply_chat_template_kwargs
            )
            model_inputs = tokenizer(raw_prompt_str, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        # Post-process (padding, truncation)
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
            truncation=truncation,
        )

        # Compute position_ids
        # Handle special cases for Qwen2VL/Qwen3VL
        if processor is not None and hasattr(processor, "image_processor"):
            image_processor_name = processor.image_processor.__class__.__name__
            if "Qwen2VLImageProcessor" in image_processor_name:
                # Qwen2VL/Qwen3VL uses mrope
                if "Qwen3VLProcessor" in processor.__class__.__name__:
                    from verl.models.transformers.qwen3_vl import get_rope_index
                else:
                    from verl.models.transformers.qwen2_vl import get_rope_index

                vision_position_ids = get_rope_index(
                    processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
                valid_mask = attention_mask[0].bool()
                text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0).unsqueeze(0)
            elif "Glm4vImageProcessor" in image_processor_name:
                from verl.models.transformers.glm4v import get_rope_index

                vision_position_ids = get_rope_index(
                    processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    attention_mask=attention_mask[0],
                )
                valid_mask = attention_mask[0].bool()
                text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0).unsqueeze(0)
            else:
                position_ids = compute_position_id_with_mask(attention_mask)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        # Compute raw_prompt_ids
        raw_prompt_ids = tokenizer.encode(raw_prompt_str, add_special_tokens=False)
        if len(raw_prompt_ids) > max_prompt_length:
            if truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-max_prompt_length:]
            elif truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:max_prompt_length]
            elif truncation == "middle":
                left_half = max_prompt_length // 2
                right_half = max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {max_prompt_length}.")

        new_input_ids_list.append(input_ids[0])
        new_attention_mask_list.append(attention_mask[0])
        new_position_ids_list.append(position_ids[0])
        new_raw_prompt_ids_list.append(raw_prompt_ids)
    print("the_last_updated_messages: ", modified_messages)
    # Stack new tensors and update batch
    # Need to handle potential different lengths by padding
    max_len = max(t.shape[-1] for t in new_input_ids_list)
    
    padded_input_ids = []
    padded_attention_mask = []
    padded_position_ids = []
    
    for i in range(batch_size):
        input_ids = new_input_ids_list[i]
        attention_mask = new_attention_mask_list[i]
        position_ids = new_position_ids_list[i]
        
        curr_len = input_ids.shape[-1]
        pad_len = max_len - curr_len
        
        if pad_len > 0:
            # Left padding
            input_ids = torch.cat([
                torch.full((pad_len,), tokenizer.pad_token_id, dtype=input_ids.dtype),
                input_ids
            ])
            attention_mask = torch.cat([
                torch.zeros(pad_len, dtype=attention_mask.dtype),
                attention_mask
            ])
            # Handle position_ids which may have multiple dimensions for mrope
            if position_ids.dim() == 1:
                position_ids = torch.cat([
                    torch.zeros(pad_len, dtype=position_ids.dtype),
                    position_ids
                ])
            else:
                # For mrope position_ids with shape (4, seq_len)
                pad_pos = torch.zeros((position_ids.shape[0], pad_len), dtype=position_ids.dtype)
                position_ids = torch.cat([pad_pos, position_ids], dim=-1)
        
        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)
        padded_position_ids.append(position_ids)

    # Commit updates atomically after all samples succeed.
    batch.batch["input_ids"] = torch.stack(padded_input_ids)
    batch.batch["attention_mask"] = torch.stack(padded_attention_mask)
    batch.batch["position_ids"] = torch.stack(padded_position_ids)
    batch.non_tensor_batch["raw_prompt"] = np.array(modified_raw_prompts, dtype=object)

    # Update raw_prompt_ids in non_tensor_batch
    if "raw_prompt_ids" in batch.non_tensor_batch:
        batch.non_tensor_batch["raw_prompt_ids"] = np.array(new_raw_prompt_ids_list, dtype=object)

    print(f"[ToolAdaptation] Re-tokenization complete. New max sequence length: {max_len}")

    return batch


def update_skill_usage(skills: List[Dict], usage: Dict[str, int]) -> List[Dict]:
    """Update skill usage counts based on batch statistics."""
    skills = copy.deepcopy(skills)
    for skill in skills:
        name = skill.get("name")
        if name in usage:
            skill["used_times"] = int(skill.get("used_times", 0)) + int(usage[name])
    return skills


def _skill_selection_key(skill: Dict) -> float:
    return float(skill.get("used_times", 0))


def select_skills_with_iqr(
    existing_skills: List[Dict],
    new_skills: List[Dict],
    max_skills: int = MAX_SKILLS,
) -> tuple[List[Dict], Dict[str, Any]]:
    """Select skills using IQR strategy."""
    total_count = len(existing_skills) + len(new_skills)
    stats: Dict[str, Any] = {
        "exceeded": total_count > max_skills,
        "old_count": len(existing_skills),
        "new_count": len(new_skills),
        "replaced": False,
        "max_skills": max_skills,
    }

    if max_skills <= 0:
        selected_all = existing_skills + new_skills
        stats["exceeded"] = False
        stats["final_count"] = len(selected_all)
        return selected_all, stats

    if total_count <= max_skills:
        selected_all = existing_skills + new_skills
        stats["final_count"] = len(selected_all)
        return selected_all, stats

    used_times = [float(skill.get("used_times", 0)) for skill in existing_skills]
    if not used_times:
        selected = new_skills[:max_skills]
        stats.update({"replaced": True, "final_count": len(selected), "selected_new": len(selected)})
        return selected, stats

    q1 = float(np.percentile(used_times, 25))
    q3 = float(np.percentile(used_times, 75))
    iqr = max(q3 - q1, 1.0)
    threshold = q3 + iqr

    keep_existing = [s for s in existing_skills if s.get("used_times", 0) >= threshold]
    remaining_existing = [s for s in existing_skills if s.get("used_times", 0) < threshold]

    if len(keep_existing) > max_skills:
        keep_existing = sorted(keep_existing, key=_skill_selection_key, reverse=True)[:max_skills]

    slots = max_skills - len(keep_existing)
    rng = random.Random(TOOL_SELECTION_SEED)
    selected_new = rng.sample(new_skills, k=min(slots, len(new_skills))) if slots > 0 else []
    slots -= len(selected_new)

    extra_existing: List[Dict] = []
    if slots > 0 and remaining_existing:
        extra_existing = sorted(remaining_existing, key=_skill_selection_key, reverse=True)[:slots]

    selected_skills = keep_existing + selected_new + extra_existing
    stats.update(
        {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "threshold": threshold,
            "selected_existing": len(keep_existing) + len(extra_existing),
            "selected_new": len(selected_new),
            "final_count": len(selected_skills),
        }
    )
    stats["replaced"] = len(selected_skills) < total_count
    return selected_skills, stats


def log_selected_skills(
    skills: List[Dict],
    global_step: int,
    log_dir: str,
    selection_stats: Optional[Dict[str, Any]] = None,
    skill_monitoring_metrics: Optional[Dict[str, int]] = None,
) -> None:
    """Log selected skills to a file."""
    if not log_dir:
        return
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"skills_{global_step}.json")
    payload = {
        "step": global_step,
        "total_selected": len(skills),
        "selection_stats": selection_stats or {},
        "skill_monitoring_metrics": skill_monitoring_metrics or {},
        "skills": [
            {
                "name": s.get("name"),
                "description": s.get("description"),
                "used_times": s.get("used_times", 0),
                "path": s.get("path"),
            }
            for s in skills
        ],
    }
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[ToolAdaptation] Logged skill selection to {log_path}")
    except Exception as e:
        print(f"[ToolAdaptation] Failed to log skill selection: {e}")


def log_tool_rollouts(
    tool_rollout_logs: Dict[str, Any],
    global_step: int,
    log_dir: str,
) -> None:
    """Log raw per-tool rollout samples to a separate file.

    `tool_rollout_logs` is produced by the reward manager as per-tool_name
    usage data for the current batch. Each entry's `used_count` is the raw call
    count for that tool_name (unfiltered by sample correctness, active/new skill
    scope, or invalid_reason).

    Because of this, `used_count` should not be directly compared against
    filtered monitoring metrics such as batch_new_run_skill_success_calls or
    batch_new_run_skill_unsuccess_calls.
    """
    if not log_dir:
        return
    if not isinstance(tool_rollout_logs, dict) or not tool_rollout_logs:
        return

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"tool_rollouts_{global_step}.json")
    payload = {
        "step": global_step,
        "tools": tool_rollout_logs,
    }
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[ToolAdaptation] Logged tool rollouts to {log_path}")
    except Exception as e:
        print(f"[ToolAdaptation] Failed to log tool rollouts: {e}")


class ToolAdaptationManager:
    """
    Skill Adaptation Manager for integrating with ray_trainer.

    This manager handles the lifecycle of skill adaptation:
    1. on_batch_start: Load skills and prepare prompt injection
    2. on_reward_computed: Process skill usage and generate next round skills
    3. get_tools_prompt_for_validation: Return current skills for validation
    4. on_validation_end: Persist next round active skills
    """

    def __init__(
        self,
        skill_store_dir: Optional[str] = None,
        log_dir: str = "tool_logs",
        max_skills: int = MAX_SKILLS,
    ):
        """
        Initialize the ToolAdaptationManager.

        Args:
            skill_store_dir: Directory containing skills
            log_dir: Directory for skill selection logs
            max_skills: Max skills to select per round
        """
        store_dir = _get_skill_store_dir(skill_store_dir)
        self.state = ToolAdaptationState(skill_store_dir=str(store_dir))
        self.log_dir = log_dir
        self.max_skills = min(max_skills, MAX_SKILLS)

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        print(f"[ToolAdaptation] Initialized with skill_store_dir={store_dir}, log_dir={log_dir}")

    def on_batch_start(self) -> str:
        """
        Called at the start of each batch.

        Loads skills from store and returns formatted prompt for injection.

        Returns:
            Formatted tools prompt string (empty if no tools)
        """
        store_dir = Path(self.state.skill_store_dir)
        all_skills = _scan_skill_dirs(store_dir)
        active_names = _load_active_skills(store_dir)
        if active_names:
            active_set = set(active_names)
            current_skills = [s for s in all_skills if s.get("name") in active_set]
        else:
            current_skills = all_skills

        # Reset pending update flag
        self.state.has_pending_update = False
        self.state.next_skills = []
        self.state.pending_index = None
        self.state.current_skills = current_skills

        # Return formatted prompt for injection
        tools_prompt = format_tools_for_prompt(self.state.current_skills)

        if tools_prompt:
            print(f"[ToolAdaptation] Batch start: {len(self.state.current_skills)} skills loaded")
        else:
            print("[ToolAdaptation] Batch start: No skills to inject")

        return tools_prompt

    def on_reward_computed(
        self,
        batch_skill_usage: Dict[str, int],
        batch_new_skills: Dict[str, Dict[str, Any]],
        global_step: int,
        batch_skill_monitoring_metrics: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Called after reward computation.

        Processes skill usage statistics and generates next round skills.

        Args:
            batch_skill_usage: Dict mapping skill names to usage counts
            batch_new_skills: Dict mapping create-instance ids to new skill info
            batch_skill_monitoring_metrics: Additional per-batch monitoring metrics for logging
            global_step: Current training step
        """
        print(f"[ToolAdaptation] Processing reward results at step {global_step}")
        print(f"[ToolAdaptation] Skill usage: {batch_skill_usage}")
        print(f"[ToolAdaptation] New skills: {batch_new_skills}")
        print(f"[ToolAdaptation] Skill monitoring metrics: {batch_skill_monitoring_metrics or {}}")

        store_dir = Path(self.state.skill_store_dir)
        pending_redirects = _deduplicate_pending_skills(store_dir)
        index = _load_skill_index(store_dir)
        reserved_names: set[str] = set(index.keys())
        active_hash_by_name: Dict[str, str] = {}

        # Step 1: Update usage counts for existing skills
        updated_skills = update_skill_usage(self.state.current_skills, batch_skill_usage)
        for skill in updated_skills:
            name = skill.get("name")
            if not name:
                continue
            entry = index.get(name, {})
            content_hash = str(entry.get("content_hash", "")).strip()
            skill_path = Path(str(skill.get("path", "")))
            if not content_hash and skill_path.exists() and skill_path.is_dir():
                content_hash = _compute_skill_content_hash(skill_path)
            entry.update(
                {
                    "name": name,
                    "description": skill.get("description", ""),
                    "path": skill.get("path", ""),
                    "used_times": int(skill.get("used_times", 0)),
                    "status": "active",
                    "content_hash": content_hash,
                }
            )
            index[name] = entry
            reserved_names.add(name)
            active_hash_by_name[name] = content_hash

        # Step 2: Promote new skills
        new_skill_entries: List[Dict] = []
        for _, info in batch_new_skills.items():
            raw_name = _sanitize_skill_name(str(info.get("skill_name", "")))
            path_str = info.get("path")
            if not raw_name or not path_str:
                continue
            src_path = Path(path_str)
            canonical_path_str = pending_redirects.get(_normalize_path(src_path), _normalize_path(src_path))
            src_path = Path(canonical_path_str)
            if not src_path.exists():
                continue
            content_hash = _compute_skill_content_hash(src_path)
            existing_hash = active_hash_by_name.get(raw_name, "")

            if existing_hash and existing_hash == content_hash:
                try:
                    archive_skill_event(
                        skill_path=src_path,
                        event="deleted",
                        reason="duplicate_existing",
                        store_dir=store_dir,
                        global_step=global_step,
                        metadata={"existing_skill_name": raw_name},
                    )
                except Exception as exc:
                    print(f"[ToolAdaptation] Failed to archive duplicate new skill {src_path}: {exc}")
                shutil.rmtree(src_path, ignore_errors=True)
                print(
                    f"[ToolAdaptation] Skip duplicate new skill name={raw_name} "
                    f"hash={content_hash[:8]} path={src_path}"
                )
                continue

            if existing_hash and existing_hash != content_hash:
                hash_suffix = f"step{global_step}"
                desired_name = _sanitize_skill_name(f"{raw_name}-{hash_suffix}")
                final_name = _reserve_unique_skill_name(desired_name, store_dir, reserved_names)
                print(
                    f"[ToolAdaptation] Rename conflicting new skill old_name={raw_name} "
                    f"new_name={final_name} old_hash={existing_hash[:8]} new_hash={content_hash[:8]}"
                )
            else:
                final_name = _reserve_unique_skill_name(raw_name, store_dir, reserved_names)

            target_dir = store_dir / final_name
            try:
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_path), str(target_dir))
            except Exception as exc:
                print(f"[ToolAdaptation] Failed to promote new skill {raw_name} from {src_path}: {exc}")
                continue
            skill_md = target_dir / "SKILL.md"
            _rewrite_skill_md_name(skill_md, final_name)
            description = ""
            if skill_md.exists():
                meta = _parse_skill_frontmatter(skill_md.read_text(encoding="utf-8"))
                description = str(meta.get("description", "")).strip()
            new_entry = {
                "name": final_name,
                "description": description,
                "path": str(target_dir),
                "used_times": 1,
                "status": "active",
                "content_hash": content_hash,
                "created_trajectory_id": info.get("created_trajectory_id"),
                "created_turn": info.get("created_turn"),
            }
            index[final_name] = new_entry
            new_skill_entries.append(new_entry)
            active_hash_by_name[final_name] = content_hash
            try:
                archive_skill_event(
                    skill_path=target_dir,
                    event="created",
                    reason="promoted",
                    store_dir=store_dir,
                    global_step=global_step,
                    source_path=src_path,
                    runtime_path=target_dir,
                    metadata={"original_skill_name": raw_name},
                )
            except Exception as exc:
                print(f"[ToolAdaptation] Failed to archive promoted skill {target_dir}: {exc}")

        _cleanup_empty_pending_dirs(store_dir)

        # Step 3: Select next skills (IQR)
        selected_skills, stats = select_skills_with_iqr(
            existing_skills=updated_skills,
            new_skills=new_skill_entries,
            max_skills=self.max_skills,
        )

        # Step 4: Log selection and store for later write
        log_selected_skills(
            selected_skills,
            global_step,
            self.log_dir,
            stats,
            skill_monitoring_metrics=batch_skill_monitoring_metrics or {},
        )
        self.state.next_skills = selected_skills
        self.state.has_pending_update = True
        self.state.pending_index = index

        print(f"[ToolAdaptation] Generated {len(selected_skills)} skills for next round")

    def process_reward_extra_info(
        self,
        reward_extra_info: Optional[Dict[str, Any]],
        global_step: int,
        write_immediately: bool = False,
    ) -> None:
        """
        Process reward_extra_info with the same adaptation logic used in training.

        This helper keeps train/validation code paths consistent by extracting
        skill-related payload once and routing to on_reward_computed.

        Args:
            reward_extra_info: reward_extra_info dict returned by reward manager.
            global_step: Step identifier used for skill selection logging.
            write_immediately: If True, persist active skills right away.
        """
        if not isinstance(reward_extra_info, dict):
            return

        batch_skill_usage = reward_extra_info.get("batch_skill_usage", {})
        if not isinstance(batch_skill_usage, dict):
            batch_skill_usage = {}

        batch_new_skills = reward_extra_info.get("batch_new_skills", {})
        if not isinstance(batch_new_skills, dict):
            batch_new_skills = {}

        batch_skill_monitoring_metrics = reward_extra_info.get("batch_skill_monitoring_metrics", {})
        if not isinstance(batch_skill_monitoring_metrics, dict):
            batch_skill_monitoring_metrics = {}

        batch_tool_rollout_logs = reward_extra_info.get("batch_tool_rollout_logs", {})
        if not isinstance(batch_tool_rollout_logs, dict):
            batch_tool_rollout_logs = {}

        log_tool_rollouts(
            tool_rollout_logs=batch_tool_rollout_logs,
            global_step=global_step,
            log_dir=self.log_dir,
        )

        self.on_reward_computed(
            batch_skill_usage=batch_skill_usage,
            batch_new_skills=batch_new_skills,
            batch_skill_monitoring_metrics=batch_skill_monitoring_metrics,
            global_step=global_step,
        )

        if write_immediately:
            self.write_pending_update()

    def get_tools_prompt_for_validation(self) -> str:
        """
        Get tools prompt for validation.

        Returns the current round skills (A), not the next round skills (B).

        Returns:
            Formatted tools prompt string
        """
        return format_tools_for_prompt(self.state.current_skills)

    def on_validation_end(self) -> None:
        """
        Called after validation ends.

        Writes the next round skills (B) to active_skills.json.
        """
        self.write_pending_update()

    def write_pending_update(self) -> None:
        """
        Write pending skills to active_skills.json, if any.
        """
        if not self.state.has_pending_update:
            print("[ToolAdaptation] No pending update to write")
            return

        store_dir = Path(self.state.skill_store_dir)
        if self.state.pending_index is not None:
            _save_skill_index(store_dir, self.state.pending_index)
            self.state.pending_index = None
        active_names = [s.get("name") for s in self.state.next_skills if s.get("name")]
        _save_active_skills(store_dir, active_names)
        self.state.has_pending_update = False
        self.state.current_skills = list(self.state.next_skills)
        print(f"[ToolAdaptation] Updated active_skills.json with {len(active_names)} skills")

    def get_current_skill_names(self) -> set:
        """Return the set of current skill names."""
        names = set()
        for skill in self.state.current_skills:
            name = skill.get("name")
            if name:
                names.add(name)
        return names

    def reset_skill_store(self, reason: str, global_step: Optional[int] = None) -> bool:
        """Clear the current run skill store and reset in-memory adaptation state."""
        store_dir = Path(self.state.skill_store_dir)
        resolved_store_dir = Path(os.path.realpath(store_dir))

        if str(resolved_store_dir) in {"", os.path.sep}:
            print(
                f"[ToolAdaptation] Skip skill store reset reason={reason} step={global_step} "
                f"because store_dir is unsafe: {resolved_store_dir}"
            )
            return False

        try:
            resolved_store_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(
                f"[ToolAdaptation] Failed to prepare skill store reset reason={reason} "
                f"step={global_step} path={resolved_store_dir}: {exc}"
            )
            return False

        removed_entries = 0
        failed_entries = 0
        for child in list(resolved_store_dir.iterdir()):
            child_real = Path(os.path.realpath(child))
            try:
                common = os.path.commonpath([str(resolved_store_dir), str(child_real)])
            except ValueError:
                failed_entries += 1
                print(
                    f"[ToolAdaptation] Skip reset child reason={reason} step={global_step} "
                    f"because path is invalid: {child_real}"
                )
                continue

            if common != str(resolved_store_dir):
                failed_entries += 1
                print(
                    f"[ToolAdaptation] Skip reset child reason={reason} step={global_step} "
                    f"because target is outside store_dir: {child_real}"
                )
                continue

            try:
                if child_real.name == "_skill_archive":
                    continue
                if child_real.is_dir():
                    if child_real.name == "_pending":
                        for pending_skill_dir in _iter_pending_skill_dirs(resolved_store_dir):
                            try:
                                archive_skill_event(
                                    skill_path=pending_skill_dir,
                                    event="deleted",
                                    reason=reason,
                                    store_dir=resolved_store_dir,
                                    global_step=global_step,
                                )
                            except Exception as exc:
                                print(
                                    f"[ToolAdaptation] Failed to archive pending skill during reset "
                                    f"reason={reason} step={global_step} path={pending_skill_dir}: {exc}"
                                )
                    else:
                        try:
                            archive_skill_event(
                                skill_path=child_real,
                                event="deleted",
                                reason=reason,
                                store_dir=resolved_store_dir,
                                global_step=global_step,
                            )
                        except Exception as exc:
                            print(
                                f"[ToolAdaptation] Failed to archive skill during reset "
                                f"reason={reason} step={global_step} path={child_real}: {exc}"
                            )
                    shutil.rmtree(child_real)
                else:
                    child_real.unlink()
                removed_entries += 1
            except FileNotFoundError:
                continue
            except Exception as exc:
                failed_entries += 1
                print(
                    f"[ToolAdaptation] Failed to remove child during reset reason={reason} "
                    f"step={global_step} path={child_real}: {exc}"
                )

        _save_skill_index(resolved_store_dir, {})
        _save_active_skills(resolved_store_dir, [])
        _get_skill_pending_dir(resolved_store_dir).mkdir(parents=True, exist_ok=True)

        self.state.current_skills = []
        self.state.next_skills = []
        self.state.pending_index = None
        self.state.has_pending_update = False

        print(
            f"[ToolAdaptation] Reset skill store reason={reason} step={global_step} "
            f"path={resolved_store_dir} removed_entries={removed_entries} failed_entries={failed_entries}"
        )
        return failed_entries == 0
