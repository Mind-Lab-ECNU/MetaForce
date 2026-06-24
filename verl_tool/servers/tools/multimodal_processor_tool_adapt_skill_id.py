import asyncio
import ast
import base64
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import yaml

from .base import BaseTool, register_tool

RUN_SKILL_TIMEOUT = 10
SKILL_MAX_NAME_LEN = 64
SKILL_SCRIPT_MAX_SYNTAX_RETRIES = 10
SKILL_SYNTAX_ERROR_MAX_CHARS = 1200
SKILL_SCRIPT_SMOKE_TEST_TIMEOUT = 5
SKILL_ALLOWED_THIRD_PARTY_IMPORTS = {"requests", "yaml"}
SKILL_CAPABILITY_IMPORT_CANDIDATES = (
    "requests",
    "yaml",
    "numpy",
    "PIL",
    "pandas",
    "bs4",
    "lxml",
    "torch",
    "tensorflow",
    "sentence_transformers",
    "transformers",
)
SKILL_RESULT_START_MARKER = "===SKILL_RESULT_START==="
SKILL_RESULT_END_MARKER = "===SKILL_RESULT_END==="
SKILL_TEXT_ARTIFACT_SUFFIXES = {
    ".csv",
    ".html",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".svg",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
SKILL_RUNTIME_CAPABILITY_CACHE: Optional[Dict[str, Any]] = None
SKILL_BANNED_IMPORT_PREFIXES = (
    "anthropic",
    "cv2",
    "diffusers",
    "google.generativeai",
    "langchain",
    "llama_index",
    "litellm",
    "modelscope",
    "ollama",
    "openai",
    "paddleocr",
    "vllm",
)
SKILL_BANNED_MODEL_USAGE_PATTERNS = (
    ".from_pretrained(",
    "from_pretrained(",
    "snapshot_download(",
    "hf_hub_download(",
    "huggingface_hub",
    "torch.hub.load(",
    "hub.load(",
    "SentenceTransformer(",
    "AutoTokenizer.from_pretrained",
    "AutoModel.from_pretrained",
    "AutoProcessor.from_pretrained",
    "AutoConfig.from_pretrained",
    "TFAutoModel.from_pretrained",
    "pipeline(",
)
SKILL_BANNED_SHELL_PATTERNS = (
    "pip install",
    "python -m pip install",
    "uv pip install",
    "poetry add",
    "conda install",
    "mamba install",
)


def _looks_like_base64(value: str) -> bool:
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if len(candidate) < 32:
        return False
    if len(candidate) % 4 != 0:
        return False
    return re.fullmatch(r"[A-Za-z0-9+/=]+", candidate) is not None


def _safe_path_exists(path_like: Union[str, Path]) -> bool:
    try:
        path_obj = path_like if isinstance(path_like, Path) else Path(path_like)
        return path_obj.exists()
    except (OSError, ValueError, TypeError):
        return False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _get_skill_store_dir() -> Path:
    env_dir = os.environ.get("VERL_SKILL_STORE_DIR")
    if env_dir:
        return Path(env_dir)
    return _repo_root() / "skills"


def _get_skill_pending_dir() -> Path:
    return _get_skill_store_dir() / "_pending"


def _load_skill_index() -> Dict[str, Dict[str, Any]]:
    index_path = _get_skill_store_dir() / "index.json"
    if not index_path.exists():
        return {}
    try:
        with index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _load_active_skill_names() -> Optional[set[str]]:
    store_dir = _get_skill_store_dir()
    active_path = store_dir / "active_skills.json"
    if active_path.exists():
        try:
            with active_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return {_sanitize_skill_name(str(item)) for item in data if str(item).strip()}
        except Exception:
            pass

    index = _load_skill_index()
    if index:
        names: set[str] = set()
        for name, info in index.items():
            if not isinstance(name, str):
                continue
            if not isinstance(info, dict):
                names.add(_sanitize_skill_name(name))
                continue
            status = str(info.get("status", "active")).strip().lower()
            if status != "active":
                continue
            path_str = str(info.get("path", "")).replace("\\", "/")
            if "/_pending/" in path_str:
                continue
            names.add(_sanitize_skill_name(name))
        return names

    return None


def _parse_skill_frontmatter(text: str) -> Dict[str, Any]:
    if not text.startswith("---"):
        return {}
    try:
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}
        frontmatter = parts[1]
        data = yaml.safe_load(frontmatter) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_skill_md(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _sanitize_skill_name(name: str) -> str:
    if not isinstance(name, str):
        name = ""
    value = name.strip().lower()
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "auto-skill"

def _ensure_unique_dir(base_dir: Path, name: str) -> Path:
    candidate = base_dir / name
    if not candidate.exists():
        return candidate
    for idx in range(2, 1000):
        candidate = base_dir / f"{name}-{idx}"
        if not candidate.exists():
            return candidate
    return base_dir / f"{name}-{uuid.uuid4().hex[:8]}"


def _sanitize_env(env: Dict[str, str]) -> Dict[str, str]:
    safe_env = {}
    sensitive_pattern = re.compile(r"(token|secret|key|password|auth)", re.IGNORECASE)
    for k, v in env.items():
        if sensitive_pattern.search(k):
            continue
        safe_env[k] = v
    return safe_env


def _build_canonical_skill_md(
    skill_name: str,
    skill_description: str,
    description: str,
    requires_image: Optional[bool],
    overview_md: Optional[str] = None,
) -> str:
    usage_body = (overview_md or "").strip() or description.strip()
    if requires_image is True:
        image_contract = "- This skill requires `args.image_index` in `run_skill` if you want to call a script that needs image input.\n"
    elif requires_image is False:
        image_contract = "- This skill does not require image input; do not pass `image_index` to `run_skill`.\n"
    else:
        image_contract = ""

    return (
        "---\n"
        f"name: {skill_name}\n"
        f"description: {skill_description}\n"
        "---\n\n"
        "## Usage\n"
        f"{usage_body}\n\n"
        "## Image Input Contract\n"
        f"{image_contract}"
    )


def _normalize_rel_script_path(path_str: str) -> Optional[str]:
    raw = str(path_str or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute() or ".." in p.parts:
        return None
    return p.as_posix()


def _load_stage1_spec_from_file(skill_dir: Path) -> Optional[Dict[str, Any]]:
    spec_path = skill_dir / "SKILL_SPEC.json"
    if not spec_path.exists():
        return None
    try:
        data = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _entrypoint_requires_image(stage1_spec: Optional[Dict[str, Any]], entrypoint: str) -> Optional[bool]:
    if not isinstance(stage1_spec, dict):
        return None
    scripts = stage1_spec.get("scripts")
    if not isinstance(scripts, list):
        return None
    normalized_entrypoint = _normalize_rel_script_path(entrypoint)
    if not normalized_entrypoint:
        return None
    for item in scripts:
        if not isinstance(item, dict):
            continue
        script_path = _normalize_rel_script_path(item.get("path", ""))
        if script_path != normalized_entrypoint:
            continue
        params = item.get("params")
        if not isinstance(params, list):
            return False
        for param in params:
            if not isinstance(param, dict):
                continue
            if str(param.get("name", "")).strip() != "image_index":
                continue
            return bool(param.get("required", False))
        return False
    return None


def _normalize_script_specs(raw_scripts: Any) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    if not isinstance(raw_scripts, list):
        return specs

    for item in raw_scripts:
        if not isinstance(item, dict):
            continue

        raw_path = str(item.get("path", "")).strip()
        if not raw_path:
            continue
        path_obj = Path(raw_path)
        suffix = path_obj.suffix.lower()
        if suffix not in {".py", ".sh"}:
            continue
        if path_obj.is_absolute() or ".." in path_obj.parts:
            continue

        purpose = str(item.get("purpose", "")).strip() or str(item.get("notes", "")).strip()
        notes = str(item.get("notes", "")).strip()
        raw_params = item.get("params")
        params: List[Dict[str, Any]] = []
        if isinstance(raw_params, list):
            for p in raw_params:
                if not isinstance(p, dict):
                    continue
                name = str(p.get("name", "")).strip()
                if not name:
                    continue
                params.append(
                    {
                        "name": name,
                        "type": str(p.get("type", "string")).strip() or "string",
                        "required": bool(p.get("required", False)),
                        "description": str(p.get("description", "")).strip(),
                    }
                )

        specs.append(
            {
                "path": raw_path,
                "purpose": purpose,
                "params": params,
                "notes": notes,
            }
        )

    return specs


def _get_skill_python_bin() -> str:
    return shutil.which("python3") or shutil.which("python") or "python"


def _get_skill_runtime_capabilities() -> Dict[str, Any]:
    global SKILL_RUNTIME_CAPABILITY_CACHE
    if SKILL_RUNTIME_CAPABILITY_CACHE is not None:
        return {
            "python_bin": SKILL_RUNTIME_CAPABILITY_CACHE["python_bin"],
            "python_version": SKILL_RUNTIME_CAPABILITY_CACHE["python_version"],
            "available_imports": list(SKILL_RUNTIME_CAPABILITY_CACHE["available_imports"]),
        }

    python_bin = _get_skill_python_bin()
    version_cmd = [
        python_bin,
        "-c",
        "import platform; print(platform.python_version())",
    ]
    try:
        version_result = subprocess.run(
            version_cmd,
            text=True,
            capture_output=True,
            timeout=SKILL_SCRIPT_SMOKE_TEST_TIMEOUT,
            check=False,
        )
        python_version = (version_result.stdout or "").strip() or "unknown"
    except Exception:
        python_version = "unknown"

    available_imports: List[str] = []
    candidate_imports = sorted(set(SKILL_ALLOWED_THIRD_PARTY_IMPORTS) | set(SKILL_CAPABILITY_IMPORT_CANDIDATES))
    for module_name in candidate_imports:
        try:
            result = subprocess.run(
                [python_bin, "-c", f"import {module_name}"],
                text=True,
                capture_output=True,
                timeout=SKILL_SCRIPT_SMOKE_TEST_TIMEOUT,
                check=False,
            )
        except Exception:
            continue
        if result.returncode == 0:
            available_imports.append(module_name)

    SKILL_RUNTIME_CAPABILITY_CACHE = {
        "python_bin": python_bin,
        "python_version": python_version,
        "available_imports": available_imports,
    }
    return {
        "python_bin": python_bin,
        "python_version": python_version,
        "available_imports": list(available_imports),
    }


def _format_skill_runtime_capabilities_for_prompt(runtime_caps: Dict[str, Any]) -> str:
    available_imports = runtime_caps.get("available_imports") or []
    available_text = ", ".join(str(name) for name in available_imports) if available_imports else "(none)"
    return (
        "Runtime capabilities (exactly what the current environment supports):\n"
        f"- Python executable: {runtime_caps.get('python_bin', 'python')}\n"
        f"- Python version: {runtime_caps.get('python_version', 'unknown')}\n"
        f"- Available third-party imports you MAY use: {available_text}\n"
        "- Do not use any third-party import that is not explicitly listed above.\n"
    )


def _build_skill_stdout_contract(script_spec: Dict[str, Any]) -> Dict[str, Any]:
    params = script_spec.get("params") if isinstance(script_spec.get("params"), list) else []
    artifact_params: List[str] = []
    for param in params:
        if not isinstance(param, dict):
            continue
        name = str(param.get("name", "")).strip()
        lowered = name.lower()
        if not name:
            continue
        if (
            lowered.startswith("output_")
            or lowered.endswith("_output")
            or lowered in {"output", "output_path", "output_file", "outfile", "out_file", "save_path"}
        ):
            artifact_params.append(name)
    return {
        "result_start_marker": SKILL_RESULT_START_MARKER,
        "result_end_marker": SKILL_RESULT_END_MARKER,
        "artifact_params": artifact_params,
    }


def _format_stdout_contract_for_prompt(stdout_contract: Dict[str, Any]) -> str:
    artifact_params = stdout_contract.get("artifact_params") or []
    artifact_text = ", ".join(str(name) for name in artifact_params) if artifact_params else "(none)"
    return (
        "Output contract (must pass create_skill validation):\n"
        f"- On success, print the final consumable result between `{SKILL_RESULT_START_MARKER}` and `{SKILL_RESULT_END_MARKER}`.\n"
        "- The text between those markers must be non-empty.\n"
        f"- Artifact output params for this script: {artifact_text}.\n"
        "- If you save an artifact to a file, do not print only the file path; print the actual textual content or a meaningful textual summary between the markers.\n"
        "- On fatal errors, write the error to stderr and exit non-zero.\n"
    )


def _guess_output_suffix(param_name: str) -> str:
    lowered = str(param_name or "").lower()
    if "html" in lowered:
        return ".html"
    if "markdown" in lowered or lowered.endswith("_md") or lowered.endswith("md"):
        return ".md"
    if "json" in lowered:
        return ".json"
    if "svg" in lowered:
        return ".svg"
    if "yaml" in lowered or lowered.endswith("_yml"):
        return ".yaml"
    if "csv" in lowered:
        return ".csv"
    if "xml" in lowered:
        return ".xml"
    if "code" in lowered:
        return ".txt"
    return ".txt"


def _write_skill_sample_file(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = {
            "title": "sample",
            "items": [1, 2, 3],
            "text": "sample input text",
            "summary": "sample summary",
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if suffix == ".csv":
        path.write_text("name,value\nalpha,1\nbeta,2\n", encoding="utf-8")
        return
    if suffix in {".yaml", ".yml"}:
        path.write_text("title: sample\nitems:\n  - alpha\n  - beta\n", encoding="utf-8")
        return
    if suffix == ".html":
        path.write_text("<html><body><h1>sample</h1><p>content</p></body></html>\n", encoding="utf-8")
        return
    if suffix == ".svg":
        path.write_text(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\"><rect width=\"10\" height=\"10\" fill=\"red\"/></svg>\n",
            encoding="utf-8",
        )
        return
    path.write_text("sample input text\nline two\n", encoding="utf-8")


def _write_dummy_image(path: Path) -> None:
    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0n0AAAAASUVORK5CYII="
    )
    path.write_bytes(base64.b64decode(png_base64))


def _build_skill_validation_args(
    script_spec: Dict[str, Any],
    tmp_root: Path,
) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    args: Dict[str, Any] = {}
    expected_outputs: Dict[str, Path] = {}
    params = script_spec.get("params") if isinstance(script_spec.get("params"), list) else []
    for param in params:
        if not isinstance(param, dict):
            continue
        name = str(param.get("name", "")).strip()
        if not name or name == "image_index":
            continue
        lowered = name.lower()
        value_type = str(param.get("type", "string")).strip().lower()

        if (
            lowered.startswith("output_")
            or lowered.endswith("_output")
            or lowered in {"output", "output_path", "output_file", "outfile", "out_file", "save_path"}
        ):
            output_path = tmp_root / f"{lowered}{_guess_output_suffix(lowered)}"
            expected_outputs[name] = output_path
            args[name] = str(output_path)
            continue

        if (
            lowered.startswith("input_")
            or lowered.endswith("_input")
            or lowered in {"input", "input_path", "input_file", "file", "file_path", "path"}
        ):
            input_path = tmp_root / f"{lowered}{_guess_output_suffix(lowered)}"
            _write_skill_sample_file(input_path)
            args[name] = str(input_path)
            continue

        if lowered in {"text", "query", "prompt", "caption", "description", "style", "focus", "title"}:
            args[name] = "sample generated content"
            continue

        if value_type in {"int", "integer"}:
            args[name] = 1
            continue
        if value_type in {"float", "number"}:
            args[name] = 1.0
            continue
        if value_type in {"bool", "boolean"}:
            args[name] = True
            continue
        if value_type in {"list", "array"}:
            args[name] = ["alpha", "beta"]
            continue
        if value_type in {"dict", "object", "json"}:
            args[name] = {"sample": True, "value": 1}
            continue

        args[name] = "sample"

    return args, expected_outputs


def _extract_stdout_between_markers(stdout: str, stdout_contract: Dict[str, Any]) -> str:
    if not isinstance(stdout, str):
        return ""
    start_marker = str(stdout_contract.get("result_start_marker", SKILL_RESULT_START_MARKER))
    end_marker = str(stdout_contract.get("result_end_marker", SKILL_RESULT_END_MARKER))
    start = stdout.find(start_marker)
    if start < 0:
        return ""
    end = stdout.find(end_marker, start + len(start_marker))
    if end < 0:
        return ""
    return stdout[start + len(start_marker) : end].strip()


def _stdout_satisfies_contract(
    stdout: str,
    stdout_contract: Dict[str, Any],
    expected_outputs: Dict[str, Path],
) -> Tuple[bool, str]:
    body = _extract_stdout_between_markers(stdout, stdout_contract)
    if not body:
        return False, "missing required stdout result markers or empty result body"

    normalized = body.strip().strip("'\"")
    lowered = normalized.lower()
    if lowered.startswith(("saved to ", "written to ", "output saved", "file saved", "stored at ")):
        return False, "result body only reports a saved path instead of content"

    if expected_outputs:
        output_paths = {str(path) for path in expected_outputs.values()}
        output_names = {path.name for path in expected_outputs.values()}
        if normalized in output_paths or normalized in output_names:
            return False, "result body only contains an output file path"
        if len(normalized) < 20:
            return False, "result body is too short to be a meaningful artifact preview"

    return True, ""


def _validate_generated_script_contract(
    path: str,
    content: str,
    script_spec: Dict[str, Any],
    allowed_imports: Optional[set[str]] = None,
) -> Tuple[bool, str]:
    runtime_ok, runtime_error = _validate_generated_script_runtime(path, content, allowed_imports=allowed_imports)
    if not runtime_ok:
        return False, runtime_error

    target_rel_path = Path(path or "generated.py")
    stdout_contract = _build_skill_stdout_contract(script_spec)
    try:
        with tempfile.TemporaryDirectory(prefix="skill_contract_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            tmp_script = tmp_root / target_rel_path
            tmp_script.parent.mkdir(parents=True, exist_ok=True)
            tmp_script.write_text(content, encoding="utf-8")
            validation_args, expected_outputs = _build_skill_validation_args(script_spec, tmp_root)
            env = _sanitize_env(os.environ.copy())
            env["SKILL_DIR"] = str(tmp_root)
            if _entrypoint_requires_image({"scripts": [script_spec]}, str(target_rel_path)) is True:
                image_path = tmp_root / "skill_input.png"
                _write_dummy_image(image_path)
                env["SKILL_IMAGE_PATH"] = str(image_path)
            cmd = _build_skill_command(tmp_script, validation_args)
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=SKILL_SCRIPT_SMOKE_TEST_TIMEOUT,
                cwd=str(tmp_root),
                env=env,
                check=False,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()
                if len(detail) > SKILL_SYNTAX_ERROR_MAX_CHARS:
                    detail = detail[:SKILL_SYNTAX_ERROR_MAX_CHARS] + "..."
                return False, f"contract dry-run failed for `{path}`: {detail or 'non-zero exit'}"

            for output_path in expected_outputs.values():
                if not output_path.exists():
                    return (
                        False,
                        f"contract dry-run failed for `{path}`: expected output file `{output_path.name}` was not created",
                    )

            stdout_ok, stdout_error = _stdout_satisfies_contract(result.stdout or "", stdout_contract, expected_outputs)
            if not stdout_ok:
                return False, stdout_error
    except Exception as exc:
        return False, f"contract validation failed: {exc}"
    return True, ""


def _fallback_script_content(script_spec: Dict[str, Any]) -> str:
    path = str(script_spec.get("path", "")).strip()
    suffix = Path(path).suffix.lower()
    purpose = str(script_spec.get("purpose", "")).strip() or "Script generated by fallback."
    params = script_spec.get("params") if isinstance(script_spec.get("params"), list) else []

    needs_image = any(
        isinstance(p, dict)
        and str(p.get("name", "")).strip() == "image_index"
        and bool(p.get("required", False))
        for p in params
    )

    if suffix == ".sh":
        image_contract_part = (
            "if [[ -n \"${SKILL_IMAGE_PATH:-}\" ]]; then\n"
            "  echo \"IMAGE_REQUIRED=true\"\n"
            "  echo \"SKILL_IMAGE_PATH=${SKILL_IMAGE_PATH}\"\n"
            "elif [[ -n \"${SKILL_IMAGE_DATA_URL:-}\" ]]; then\n"
            "  echo \"IMAGE_REQUIRED=true\"\n"
            "  echo \"SKILL_IMAGE_DATA_URL=present\"\n"
            "else\n"
            "  echo \"ERROR: missing required image input (SKILL_IMAGE_PATH/SKILL_IMAGE_DATA_URL)\" >&2\n"
            "  exit 1\n"
            "fi\n"
            if needs_image
            else ""
        )
        return (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            "fail() {\n"
            "  echo \"$1\" >&2\n"
            "  exit 1\n"
            "}\n\n"
            f"# TODO: {purpose}\n"
            f"{image_contract_part}"
            f"echo \"TODO: implement {path}\"\n"
        )

    param_lines = []
    for p in params:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name", "")).strip()
        if name == "image_index":
            # image_index is consumed by runtime and not forwarded to script CLI.
            continue
        if not name:
            continue
        required = bool(p.get("required", False))
        desc = str(p.get("description", "")).strip()
        help_text = desc or f"Argument: {name}"
        param_lines.append(
            f"parser.add_argument('--{name}', required={str(required)}, help={json.dumps(help_text, ensure_ascii=False)})"
        )
    param_block = "\n".join(param_lines) if param_lines else "# No parameters declared."

    image_contract_block = (
        "image_path = os.getenv(\"SKILL_IMAGE_PATH\")\n"
        "image_data_url = os.getenv(\"SKILL_IMAGE_DATA_URL\")\n"
        "print(\"IMAGE_REQUIRED=True\")\n"
        "if image_path:\n"
        "    print(f\"SKILL_IMAGE_PATH_SET={bool(image_path)}\")\n"
        "    print(f\"SKILL_IMAGE_DATA_URL_SET={bool(image_data_url)}\")\n"
        "    print(f\"SKILL_IMAGE_PATH={image_path}\")\n"
        "elif image_data_url:\n"
        "    print(f\"SKILL_IMAGE_PATH_SET={bool(image_path)}\")\n"
        "    print(f\"SKILL_IMAGE_DATA_URL_SET={bool(image_data_url)}\")\n"
        "    print(\"SKILL_IMAGE_DATA_URL=present\")\n\n"
        "else:\n"
        "    fail(\"ERROR: missing required image input (SKILL_IMAGE_PATH/SKILL_IMAGE_DATA_URL)\")\n\n"
        if needs_image
        else ""
    )

    return (
        "import argparse\n\n"
        "import os\n\n"
        "import sys\n\n"
        "def fail(message):\n"
        "    print(message, file=sys.stderr)\n"
        "    raise SystemExit(1)\n\n"
        "parser = argparse.ArgumentParser()\n"
        f"{param_block}\n"
        "args = parser.parse_args()\n"
        f"{image_contract_block}"
        f"print({json.dumps('TODO: ' + purpose, ensure_ascii=False)})\n"
    )


def _validate_generated_script_syntax(path: str, content: str) -> Tuple[bool, str]:
    suffix = Path(path or "").suffix.lower()
    if suffix == ".py":
        try:
            compile(content, path or "<generated>", "exec")
            return True, ""
        except SyntaxError as exc:
            return False, f"{exc.msg} (line {exc.lineno}, offset {exc.offset})"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    if suffix == ".sh":
        bash_bin = shutil.which("bash") or "/bin/bash"
        try:
            result = subprocess.run(
                [bash_bin, "-n"],
                input=content,
                text=True,
                capture_output=True,
                check=False,
            )
        except Exception as exc:
            return False, f"bash -n failed to run: {exc}"
        if result.returncode == 0:
            return True, ""
        err = (result.stderr or result.stdout or "").strip()
        if not err:
            err = f"bash -n exited with code {result.returncode}"
        return False, err

    return True, ""


def _validate_generated_script_dependencies(
    path: str,
    content: str,
    allowed_imports: Optional[set[str]] = None,
) -> Tuple[bool, str]:
    suffix = Path(path or "").suffix.lower()
    if suffix == ".py":
        deps_ok, deps_error = _validate_python_script_dependencies(content, allowed_imports=allowed_imports)
        if not deps_ok:
            return False, deps_error
        return _validate_python_script_model_usage(content)
    if suffix == ".sh":
        return _validate_shell_script_dependencies(content)
    return True, ""


def _validate_python_script_dependencies(
    content: str,
    allowed_imports: Optional[set[str]] = None,
) -> Tuple[bool, str]:
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        return False, f"syntax error while checking imports: {exc.msg}"

    stdlib_modules = set(getattr(sys, "stdlib_module_names", set()))
    for node in ast.walk(tree):
        module_name = None
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                ok, reason = _check_python_import_name(module_name, stdlib_modules, allowed_imports=allowed_imports)
                if not ok:
                    return False, reason
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if node.level:
                continue
            ok, reason = _check_python_import_name(module_name, stdlib_modules, allowed_imports=allowed_imports)
            if not ok:
                return False, reason
    return True, ""


def _extract_python_import_names(content: str) -> Tuple[Optional[List[str]], str]:
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        return None, f"syntax error while checking imports: {exc.msg}"

    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = str(alias.name or "").strip()
                if module_name:
                    imports.append(module_name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            module_name = str(node.module or "").strip()
            if module_name:
                imports.append(module_name)
    return imports, ""


def _check_python_import_name(
    module_name: str,
    stdlib_modules: set[str],
    allowed_imports: Optional[set[str]] = None,
) -> Tuple[bool, str]:
    root = str(module_name or "").strip().split(".", 1)[0]
    if not root:
        return True, ""
    lowered_full = str(module_name or "").strip().lower()
    lowered_root = root.lower()
    for banned in SKILL_BANNED_IMPORT_PREFIXES:
        if lowered_full == banned or lowered_full.startswith(f"{banned}."):
            return (
                False,
                f"disallowed dependency `{module_name}`: generated scripts must avoid external model/toolkit libraries",
            )
    normalized_allowed = {
        str(name or "").strip().split(".", 1)[0].lower()
        for name in (allowed_imports if allowed_imports is not None else SKILL_ALLOWED_THIRD_PARTY_IMPORTS)
        if str(name or "").strip()
    }
    if lowered_root in stdlib_modules or lowered_root in normalized_allowed:
        return True, ""
    return (
        False,
        f"unsupported dependency `{module_name}`: use Python standard library or the exact allowed third-party imports only",
    )


def _validate_shell_script_dependencies(content: str) -> Tuple[bool, str]:
    lowered = content.lower()
    for pattern in SKILL_BANNED_SHELL_PATTERNS:
        if pattern in lowered:
            return False, f"disallowed shell dependency installer `{pattern}` in generated script"
    for pattern in ("huggingface-cli download", "hf download", "python -m huggingface_hub"):
        if pattern in lowered:
            return False, f"disallowed model download command `{pattern}` in generated script"
    return True, ""


def _validate_python_script_model_usage(content: str) -> Tuple[bool, str]:
    for pattern in SKILL_BANNED_MODEL_USAGE_PATTERNS:
        if pattern in content:
            return False, f"disallowed model/tokenizer loading pattern `{pattern}` in generated script"
    return True, ""


def _validate_generated_script_runtime(
    path: str,
    content: str,
    allowed_imports: Optional[set[str]] = None,
) -> Tuple[bool, str]:
    suffix = Path(path or "").suffix.lower()
    if suffix == ".py":
        return _validate_python_script_runtime(path, content, allowed_imports=allowed_imports)
    return True, ""


def _validate_python_script_runtime(
    path: str,
    content: str,
    allowed_imports: Optional[set[str]] = None,
) -> Tuple[bool, str]:
    python_bin = _get_skill_python_bin()
    import_names, import_error = _extract_python_import_names(content)
    if import_names is None:
        return False, import_error

    normalized_imports: List[str] = []
    seen_imports: set[str] = set()
    stdlib_modules = set(getattr(sys, "stdlib_module_names", set()))
    for module_name in import_names:
        root = module_name.split(".", 1)[0].strip()
        if not root or root in seen_imports:
            continue
        ok, reason = _check_python_import_name(root, stdlib_modules, allowed_imports=allowed_imports)
        if not ok:
            return False, reason
        seen_imports.add(root)
        normalized_imports.append(root)

    for module_name in normalized_imports:
        import_cmd = [python_bin, "-c", f"import {module_name}"]
        try:
            result = subprocess.run(
                import_cmd,
                text=True,
                capture_output=True,
                timeout=SKILL_SCRIPT_SMOKE_TEST_TIMEOUT,
                check=False,
            )
        except Exception as exc:
            return False, f"runtime import check failed for `{module_name}`: {exc}"
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            if len(detail) > SKILL_SYNTAX_ERROR_MAX_CHARS:
                detail = detail[:SKILL_SYNTAX_ERROR_MAX_CHARS] + "..."
            return False, f"runtime import check failed for `{module_name}`: {detail or 'non-zero exit'}"

    target_rel_path = Path(path or "generated.py")
    try:
        with tempfile.TemporaryDirectory(prefix="skill_smoke_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            tmp_script = tmp_root / target_rel_path
            tmp_script.parent.mkdir(parents=True, exist_ok=True)
            tmp_script.write_text(content, encoding="utf-8")
            result = subprocess.run(
                [python_bin, str(tmp_script), "--help"],
                text=True,
                capture_output=True,
                timeout=SKILL_SCRIPT_SMOKE_TEST_TIMEOUT,
                cwd=str(tmp_root),
                check=False,
            )
    except Exception as exc:
        return False, f"runtime smoke test failed: {exc}"

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if len(detail) > SKILL_SYNTAX_ERROR_MAX_CHARS:
            detail = detail[:SKILL_SYNTAX_ERROR_MAX_CHARS] + "..."
        return False, f"runtime smoke test failed for `{path}` with `--help`: {detail or 'non-zero exit'}"
    return True, ""


def _build_stage2_retry_prompt(validation_error: str, attempt: int, max_attempts: int) -> str:
    compact_error = (validation_error or "").strip()
    if len(compact_error) > SKILL_SYNTAX_ERROR_MAX_CHARS:
        compact_error = compact_error[:SKILL_SYNTAX_ERROR_MAX_CHARS] + "..."
    return (
        "Previous attempt failed validation. Regenerate the same target script and fix the issue.\n"
        f"Retry attempt: {attempt + 1}/{max_attempts}\n"
        "Rules:\n"
        "- Keep `path` exactly the target path.\n"
        "- `content` must be non-empty.\n"
        "- Fix syntax/format issues reported below.\n"
        f"Validation error:\n{compact_error or 'unknown validation error'}\n"
    )


def _build_skill_command(entrypoint: Path, args: Optional[Dict[str, Any]]) -> List[str]:
    if entrypoint.suffix.lower() == ".py":
        python_bin = shutil.which("python3") or shutil.which("python") or "python"
        cmd = [python_bin, str(entrypoint)]
    elif entrypoint.suffix.lower() == ".sh":
        bash_bin = shutil.which("bash") or "/bin/bash"
        cmd = [bash_bin, str(entrypoint)]
    else:
        raise ValueError("Unsupported entrypoint extension; only .py and .sh are allowed.")

    if args and isinstance(args, dict):
        for key, value in args.items():
            flag = f"--{key}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
                else:
                    cmd.extend([flag, "false"])
            elif value is None:
                cmd.extend([flag, "null"])
            elif isinstance(value, (dict, list)):
                cmd.extend([flag, json.dumps(value, ensure_ascii=False)])
            else:
                cmd.extend([flag, str(value)])
    return cmd


def _load_url_mapping() -> Dict[str, str]:
    """Load the mapping from model name to URL from url.json in the same directory."""
    url_path = Path(__file__).with_name("url.json")
    if not url_path.exists():
        return {}
    try:
        return json.loads(url_path.read_text())
    except Exception:
        return {}


def _to_data_url(image_path: Path) -> str:
    """Encode a local image path into a data URL."""
    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{encoded}"


def _image_to_base64(image: Union[str, Path]) -> str:
    """Convert an image path or data URL into a base64 string."""
    if isinstance(image, Path):
        path = image
    elif isinstance(image, str):
        if image.startswith("data:image") and "base64," in image:
            return image.split("base64,", 1)[1]
        path = Path(image)
        try:
            if not path.exists():
                return image
        except OSError:
            return image
    else:
        raise ValueError("Unsupported image type for base64 conversion.")

    from PIL import Image

    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _base64_to_mask(mask_b64: str) -> "Any":
    """Convert a base64 mask into a NumPy array."""
    from PIL import Image
    import numpy as np

    mask_data = base64.b64decode(mask_b64)
    mask_image = Image.open(BytesIO(mask_data))
    return np.array(mask_image)


def _image_to_data_url(image: Any) -> str:
    """Convert an image object or base64 string into a data URL."""
    from PIL import Image

    if isinstance(image, str):
        if image.startswith("data:image") and "base64," in image:
            return image
        if _looks_like_base64(image):
            return f"data:image/png;base64,{image.strip()}"
        path = Path(image)
        try:
            if path.exists():
                image = Image.open(path)
            else:
                return f"data:image/png;base64,{image}"
        except OSError:
            return f"data:image/png;base64,{image.strip()}"

    if isinstance(image, Path):
        image = Image.open(image)

    if isinstance(image, bytes):
        encoded = base64.b64encode(image).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    if isinstance(image, Image.Image):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    raise ValueError("Unsupported image type for data URL conversion.")


def _load_pil_image(image_obj: Any) -> "Any":
    """Convert an image object from the environment into a PIL Image."""
    from PIL import Image

    if isinstance(image_obj, Image.Image):
        image = image_obj
    elif isinstance(image_obj, bytes):
        image = Image.open(BytesIO(image_obj))
    elif isinstance(image_obj, Path):
        image = Image.open(image_obj)
    elif isinstance(image_obj, str):
        if image_obj.startswith("data:image") and "base64," in image_obj:
            image_bytes = base64.b64decode(image_obj.split("base64,", 1)[1])
            image = Image.open(BytesIO(image_bytes))
        else:
            if _looks_like_base64(image_obj):
                image_bytes = base64.b64decode(image_obj.strip())
                image = Image.open(BytesIO(image_bytes))
            else:
                try:
                    path = Path(image_obj)
                    if _safe_path_exists(path):
                        image = Image.open(path)
                    else:
                        raise ValueError("Unsupported image string format for PIL conversion.")
                except OSError as exc:
                    raise ValueError("Unsupported image string format for PIL conversion.") from exc
    else:
        raise ValueError("Unsupported image type for PIL conversion.")

    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return image


class OpenCVProcessor:
    """OpenCV image processor supporting basic geometric and enhancement operations."""

    @staticmethod
    def process(pil_image: "Any", operation: str, **args) -> "Any":
        import cv2
        import numpy as np
        from PIL import Image

        img = np.array(pil_image)
        height, width = img.shape[:2]
        op = (operation or "").lower()

        if op == "crop":
            x = int(args["x"])
            y = int(args["y"])
            w = int(args["w"])
            h = int(args["h"])
            if x < 0 or y < 0:
                raise ValueError("crop coordinates cannot be negative.")
            if w <= 0 or h <= 0:
                raise ValueError("crop dimensions must be positive.")
            if x + w > width or y + h > height:
                raise ValueError("crop region exceeds image bounds.")
            out = img[y : y + h, x : x + w]

        elif op == "resize":
            target_w = int(args["width"])
            target_h = int(args["height"])
            if target_w <= 0 or target_h <= 0:
                raise ValueError("resize dimensions must be positive.")
            interpolation = str(args.get("interpolation", "LINEAR")).upper()
            interp_map = {
                "NEAREST": cv2.INTER_NEAREST,
                "LINEAR": cv2.INTER_LINEAR,
                "AREA": cv2.INTER_AREA,
                "CUBIC": cv2.INTER_CUBIC,
            }
            out = cv2.resize(img, (target_w, target_h), interpolation=interp_map.get(interpolation, cv2.INTER_LINEAR))

        elif op == "rotate":
            angle = float(args["angle"])
            scale = float(args.get("scale", 1.0))
            center = args.get("center")
            if center is None:
                center = (width // 2, height // 2)
            else:
                if not isinstance(center, (list, tuple)) or len(center) != 2:
                    raise ValueError("center must be a tuple of (x, y).")
                center = (float(center[0]), float(center[1]))
            if scale <= 0:
                raise ValueError("scale must be positive.")
            matrix = cv2.getRotationMatrix2D(center, angle, scale)
            out = cv2.warpAffine(img, matrix, (width, height))

        elif op == "flip":
            flip_code = int(args["flip_code"])
            if flip_code not in {-1, 0, 1}:
                raise ValueError("flip_code must be -1, 0, or 1.")
            out = cv2.flip(img, flip_code)

        elif op == "grayscale":
            out = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

        elif op == "blur":
            ksize = int(args["ksize"])
            if ksize <= 0:
                raise ValueError("ksize must be positive.")
            if ksize % 2 == 0:
                ksize += 1
            out = cv2.GaussianBlur(img, (ksize, ksize), 0)

        elif op == "threshold":
            thresh = args.get("thresh")
            maxval = int(args.get("maxval", 255))
            if not (0 <= maxval <= 255):
                raise ValueError("maxval must be in [0, 255].")
            thresh_type = str(args.get("type", "BINARY")).upper()
            if thresh_type not in {"BINARY", "BINARY_INV", "OTSU"}:
                thresh_type = "BINARY"

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
            if thresh_type == "OTSU":
                _, out = cv2.threshold(gray, 0, maxval, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            else:
                if thresh is None:
                    raise ValueError("thresh is required for non-OTSU threshold.")
                thresh_val = int(thresh)
                if not (0 <= thresh_val <= 255):
                    raise ValueError("thresh must be in [0, 255].")
                if thresh_type == "BINARY_INV":
                    _, out = cv2.threshold(gray, thresh_val, maxval, cv2.THRESH_BINARY_INV)
                else:
                    _, out = cv2.threshold(gray, thresh_val, maxval, cv2.THRESH_BINARY)

        elif op == "canny":
            threshold1 = int(args["threshold1"])
            threshold2 = int(args["threshold2"])
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
            out = cv2.Canny(gray, threshold1, threshold2)

        else:
            raise ValueError(f"Unsupported operation: {operation}")

        if out.ndim == 2:
            return Image.fromarray(out, mode="L")
        return Image.fromarray(out, mode="RGB")


def _visualize_sam3_overlay(
    image: Union[str, Path, "Any"],
    masks: List["Any"],
    boxes: Optional[List[List[float]]] = None,
    scores: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
    alpha: float = 0.5,
    show_boxes: bool = True,
) -> "Any":
    """Visualize SAM3 segmentation results.

    Args:
        image: original image
        masks: list of masks (NumPy arrays)
        boxes: list of bounding boxes
        scores: list of confidence scores
        labels: list of labels
        alpha: mask transparency
        show_boxes: whether to render bounding boxes

    Returns:
        PIL Image: visualization result
    """
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    import colorsys

    if isinstance(image, (str, Path)):
        image = Image.open(image)

    image = image.convert("RGBA")

    n_masks = len(masks)
    if n_masks == 0:
        return image

    # Generate colors.
    colors = []
    for i in range(n_masks):
        hue = i / max(n_masks, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(tuple(int(c * 255) for c in rgb))

    # Overlay masks.
    for mask, color in zip(masks, colors):
        if mask.ndim == 3:
            mask = mask.squeeze()

        mask_uint8 = mask.astype(np.uint8)
        if mask_uint8.max() <= 1:
            mask_uint8 = mask_uint8 * 255

        mask_image = Image.fromarray(mask_uint8)
        if mask_image.size != image.size:
            mask_image = mask_image.resize(image.size, Image.NEAREST)

        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha_mask = mask_image.point(lambda v: int(v / 255 * alpha * 255))
        overlay.putalpha(alpha_mask)
        image = Image.alpha_composite(image, overlay)

    # Draw bounding boxes and labels.
    if show_boxes and boxes and len(boxes) > 0:
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        for i, (box, color) in enumerate(zip(boxes, colors)):
            if len(box) >= 4:
                x1, y1, x2, y2 = box[:4]
                draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)

                # Label text.
                label_parts = []
                if labels and i < len(labels):
                    label_parts.append(labels[i])
                if scores and i < len(scores):
                    label_parts.append(f"{scores[i]:.2f}")

                if label_parts:
                    label_text = " ".join(label_parts)
                    draw.text((x1 + 5, y1 + 5), label_text, fill=(255, 255, 255, 255), font=font)

    return image


def _visualize_sam3(
    image_obj: Union[str, Path, "Any"],
    masks_base64: List[str],
    boxes: Optional[List[List[float]]] = None,
    scores: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
) -> "Any":
    """Generate a visualization image using SAM3 overlay helpers."""
    from PIL import Image

    if isinstance(image_obj, str) and image_obj.startswith("data:image"):
        image_bytes = base64.b64decode(image_obj.split("base64,", 1)[1])
        image_obj = Image.open(BytesIO(image_bytes))

    masks = [_base64_to_mask(mask) for mask in masks_base64]
    return _visualize_sam3_overlay(image_obj, masks, boxes, scores, labels=labels)


def _parse_tool_call(action: str) -> Tuple[Dict[str, Any], bool]:
    """Parse JSON wrapped by <tool_call>...</tool_call>."""
    try:
        payload_str = action
        if "<tool_call>" in action and "</tool_call>" in action:
            payload_str = action.split("<tool_call>")[1].split("</tool_call>")[0]
        payload = json.loads(payload_str)
        if not isinstance(payload, dict):
            return {}, False
        return payload, True
    except Exception:
        return {}, False


def _name_is_create_skill(name: Any) -> bool:
    """Check whether name corresponds to create_skill."""
    return isinstance(name, str) and name.strip().lower() == "create_skill"


PYTHON_CODE_TIMEOUT = 10
PYTHON_CODE_MAX_OBS_LENGTH = 100000
PYTHON_CODE_PRE_IMPORT_LIBS = (
    "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
    "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\n"
    "from random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\n"
    "from typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\n"
    "import copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\n"
    "import io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n\n"
)

firejail_command_exists = shutil.which("firejail") is not None


def _check_forbidden_imports(code: str) -> bool:
    """Check if code contains forbidden imports or dangerous patterns."""
    forbidden_modules = [
        "subprocess",
        "multiprocessing",
        "threading",
        "socket",
        "psutil",
        "resource",
        "ctypes",
    ]
    for module in forbidden_modules:
        if f"import {module}" in code or f"from {module}" in code:
            return True

    dangerous_patterns = [
        "os.system",
        "os.popen",
        "os.spawn",
        "os.fork",
        "os.exec",
        "sys.exit",
        "os._exit",
        "os.kill",
    ]
    for pattern in dangerous_patterns:
        if pattern in code:
            return True
    return False


def _wrap_python_code_blocks(code: Union[str, List[str]]) -> str:
    """Wrap code blocks with error handling and auto-print for the last expression."""
    wrapped_code = ""
    if isinstance(code, str):
        code = [code]

    wrapped_code += "import sys, os, io, ast\n\n"

    wrapped_code += """
def parse_and_exec_salvageable(code_string):
    lines = code_string.splitlines()
    current_block = ""
    local_namespace = {}

    for line in lines:
        if current_block:
            current_block += "\\n" + line
        else:
            current_block = line

        if not line.strip() or line.strip().startswith('#'):
            continue

        try:
            ast.parse(current_block)
            try:
                exec(current_block, globals(), local_namespace)
                current_block = ""
            except Exception as e:
                print(f"Runtime error in block: {e}")
                current_block = ""
        except SyntaxError:
            pass

    return local_namespace

"""

    wrapped_code += """
def execute_with_last_expr_capture(code_string):
    import sys
    import io
    from contextlib import redirect_stdout, redirect_stderr

    stdout_capture = io.StringIO()

    lines = code_string.strip().split('\\n')
    if not lines:
        return "", None

    if len(lines) == 1:
        setup_code = ""
        last_line = lines[0]
    else:
        setup_code = '\\n'.join(lines[:-1])
        last_line = lines[-1].strip()

    namespace = {}
    if setup_code:
        with redirect_stdout(stdout_capture):
            try:
                exec(setup_code, globals(), namespace)
            except Exception as e:
                print(f"Error in setup: {e}", file=sys.stderr)
                return stdout_capture.getvalue(), None

    globals().update(namespace)

    last_value = None
    try:
        compile(last_line, '<string>', 'eval')
        with redirect_stdout(stdout_capture):
            last_value = eval(last_line, globals(), namespace)
    except SyntaxError:
        with redirect_stdout(stdout_capture):
            try:
                exec(last_line, globals(), namespace)
            except Exception as e:
                print(f"Error in last line: {e}", file=sys.stderr)
    except Exception as e:
        with redirect_stdout(stdout_capture):
            print(f"Error evaluating expression: {e}", file=sys.stderr)

    return stdout_capture.getvalue(), last_value

"""

    for i, block in enumerate(code):
        is_last_block = i == len(code) - 1

        if not is_last_block:
            wrapped_block = (
                f"\n# Code block {i+1} (previous)\n"
                f"original_stdout, original_stderr = sys.stdout, sys.stderr\n"
                f"sys.stdout, sys.stderr = io.StringIO(), io.StringIO()\n"
                f"try:\n"
                f"    exported_vars = parse_and_exec_salvageable('''{block}''')\n"
                f"finally:\n"
                f"    sys.stdout, sys.stderr = original_stdout, original_stderr\n\n"
                f"    for name, value in exported_vars.items():\n"
                f"        globals()[name] = value\n"
            )
        else:
            wrapped_block = f"""
# Code block {i+1} (current - with auto-print)
_stdout_output, _last_expr_value = execute_with_last_expr_capture('''{block}''')

if _stdout_output:
    print(_stdout_output, end='')

if _last_expr_value is not None:
    _value_str = str(_last_expr_value)
    if _value_str not in _stdout_output:
        print(_last_expr_value)
"""

        wrapped_code += wrapped_block

    return wrapped_code


def _clean_traceback(text: str, base_path: str) -> str:
    pattern = re.compile(re.escape('File "' + base_path + "/"))
    return pattern.sub('File "', text)


def _set_python_code_limits() -> None:
    try:
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, resource.RLIM_INFINITY))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (PYTHON_CODE_TIMEOUT, resource.RLIM_INFINITY))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (500 * 1024 * 1024, 500 * 1024 * 1024))
    except Exception:
        pass


def _execute_python_code_sync(
    code: Union[str, List[str]],
    timeout: int = PYTHON_CODE_TIMEOUT,
    stdin: Optional[str] = None,
    python_path: Optional[str] = None,
    pre_import_lib: bool = False,
    use_firejail: bool = False,
) -> Tuple[str, str, bool, float]:
    """Execute Python code in a Firejail sandbox with a timeout."""
    start_time = time.time()
    code_text = code if isinstance(code, str) else "\n".join(code)
    if _check_forbidden_imports(code_text):
        latency = time.time() - start_time
        return "", "Execution blocked: Code contains potentially dangerous operations or imports.", True, latency

    original_env = os.environ.copy()
    cwd = os.path.join(os.getcwd(), "tmp/firejail", str(uuid.uuid4().hex))
    if not os.path.exists(cwd):
        os.makedirs(cwd, exist_ok=True)

    file_name = "main.py"
    file_path = os.path.join(cwd, file_name)
    code_wrapped = _wrap_python_code_blocks(code)
    if pre_import_lib:
        code_wrapped = PYTHON_CODE_PRE_IMPORT_LIBS + code_wrapped
    with open(file_path, "w") as f:
        f.write(code_wrapped)

    if not python_path:
        python_path = "python3"
    else:
        assert os.path.exists(python_path), f"Python path {python_path} does not exist."

    if use_firejail and firejail_command_exists:
        env = {}
        essential_vars = [
            "PATH",
            "HOME",
            "USER",
            "SHELL",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TERM",
            "PYTHONIOENCODING",
            "PYTHONUNBUFFERED",
            "PYTHONHASHSEED",
            "PYTHONDONTWRITEBYTECODE",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "TMPDIR",
            "TEMP",
            "TMP",
            "DISPLAY",
            "XAUTHORITY",
        ]
        for var in essential_vars:
            if var in original_env:
                env[var] = original_env[var]
        env["OPENBLAS_NUM_THREADS"] = "1"
        if "PYTHONPATH" in env:
            del env["PYTHONPATH"]

        command = [
            "firejail",
            "--quiet",
            "--seccomp=socket",
            "--noprofile",
            "--rlimit-nproc=32",
            "--rlimit-nofile=32",
            "--rlimit-fsize=2m",
            "--rlimit-as=1096m",
        ]
        command.extend([python_path, file_path])
        subprocess_cwd = cwd
    else:
        env = original_env
        command = [python_path, file_name]
        subprocess_cwd = cwd

    has_error = False
    try:
        result = subprocess.run(
            command,
            input=stdin if stdin else None,
            env=env,
            text=True,
            capture_output=True,
            preexec_fn=_set_python_code_limits,
            timeout=timeout,
            cwd=subprocess_cwd,
        )
        stdout = _clean_traceback(result.stdout, cwd)
        stderr = _clean_traceback(result.stderr, cwd)
        stderr = stderr if stderr else ""
        if stderr:
            has_error = True
    except subprocess.TimeoutExpired as exc:
        has_error = True
        stdout = exc.stdout if exc.stdout else ""
        stderr = exc.stderr if exc.stderr else ""
        stdout = stdout.decode("utf-8") if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode("utf-8") if isinstance(stderr, bytes) else stderr
        stderr += f"Execution timed out after {timeout} seconds.\n"

    try:
        if os.path.exists(cwd):
            shutil.rmtree(cwd)
    except Exception:
        pass

    assert isinstance(stdout, str), f"Expected stdout to be a string, got {type(stdout)}"
    assert isinstance(stderr, str), f"Expected stderr to be a string, got {type(stderr)}"
    latency = time.time() - start_time
    return stdout, stderr, has_error, latency


def _extract_python_code_argument(arguments: Dict[str, Any]) -> Optional[str]:
    if not isinstance(arguments, dict):
        return None
    code = arguments.get("code")
    if code is None:
        return None
    if not isinstance(code, str):
        code = str(code)

    code = code.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
    code = re.sub(r"^\s*```\w*\s*\n?", "", code)
    code = re.sub(r"\n?\s*```\s*$", "", code)
    return code.strip()


@register_tool
class MultimodalProcessorTool(BaseTool):
    """
    Multimodal model invocation tool with an internal async event loop.
    """

    tool_type = "multimodal_processor_tool_adapt_skill_id"

    # Registered sub-tools. Use backend model names directly.
    valid_mcp_func_names = [
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

    sam3_tool_name = "SAM3"
    sam3_model_name = "sam3"
    mineru_tool_name = "MinerU2.5"
    mineru_model_name = "MinerU2.5-2509-1.2B"
    paddle_tool_name = "PaddleOCR"
    paddle_model_name = "PaddleOCR-VL"
    easyocr_tool_name = "EasyOCR"
    easyocr_model_name = "EasyOCR"
    groundingdino_tool_name = "GroundingDINO"
    groundingdino_model_name = "GroundingDINO"
    # base_tool/tool_name for image edit; this is what tool_call should use.
    image_edit_tool_name = "Qwen-Image-Edit"
    # Actual backend model name for image edit; obs should report this full name.
    image_edit_model_name = "Qwen-Image-Edit-2511"
    opencv_tool_name = "OpenCV"
    create_skill_tool_name = "create_skill"
    run_skill_tool_name = "run_skill"
    skill_gen_model_name = "Qwen3-VL-235B-A22B-Instruct-skill-id"

    def __init__(self, num_workers: int = 1, request_timeout: int = 150):
        super().__init__(num_workers)
        self.request_timeout = request_timeout
        self.url_mapping = _load_url_mapping()

    def _load_skill_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Load existing skills (excluding pending skills)."""
        store_dir = _get_skill_store_dir()
        index = _load_skill_index()
        active_skill_names = _load_active_skill_names()
        skills: Dict[str, Dict[str, Any]] = {}
        if store_dir.exists():
            for child in store_dir.iterdir():
                if not child.is_dir():
                    continue
                if child.name.startswith("_"):
                    continue
                skill_md = child / "SKILL.md"
                if not skill_md.exists():
                    continue
                content = _read_skill_md(skill_md)
                meta = _parse_skill_frontmatter(content)
                # Use directory name as canonical runtime skill name to stay
                # consistent with active_skills.json entries.
                name = _sanitize_skill_name(str(child.name))
                if active_skill_names is not None and name not in active_skill_names:
                    continue
                description = meta.get("description") or index.get(name, {}).get("description", "")
                skills[name] = {
                    "name": name,
                    "description": str(description) if description else "",
                    "path": str(child),
                    "skill_md": content,
                    "status": index.get(name, {}).get("status", "active"),
                }
        return skills

    def _load_pending_skills(self, trajectory_id: str) -> Dict[str, Dict[str, Any]]:
        pending_dir = _get_skill_pending_dir() / trajectory_id
        skills: Dict[str, Dict[str, Any]] = {}
        if not pending_dir.exists():
            return skills
        for child in pending_dir.iterdir():
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.exists():
                continue
            content = _read_skill_md(skill_md)
            meta = _parse_skill_frontmatter(content)
            name = meta.get("name") or child.name
            name = _sanitize_skill_name(str(name))
            description = meta.get("description", "")
            skills[name] = {
                "name": name,
                "description": str(description) if description else "",
                "path": str(child),
                "skill_md": content,
                "status": "pending",
            }
        return skills

    def _resolve_skill(self, trajectory_id: str, skill_name: str) -> Optional[Dict[str, Any]]:
        skill_name = _sanitize_skill_name(skill_name)
        catalog = self._load_skill_catalog()
        if skill_name in catalog:
            return catalog[skill_name]
        pending = self._load_pending_skills(trajectory_id)
        return pending.get(skill_name)

    def get_usage_inst(self) -> str:
        return (
            "multimodal_processor_tool_adapt_skill_id:\n"
            "Multimodal model call example (Qwen3-VL-8B/32B/235B): <tool_call>{\"name\": \"Qwen3-VL-8B-Instruct\", "
            "\"arguments\": {\"prompt\": \"Describe the image\", \"image_index\": 1}}</tool_call>\n"
            "Qwen3-VL-235B example: <tool_call>{\"name\": \"Qwen3-VL-235B-A22B-Instruct\", "
            "\"arguments\": {\"prompt\": \"Describe the image\", \"image_index\": 1}}</tool_call>\n"
            "SAM3 example: <tool_call>{\"name\": \"SAM3\", "
            "\"arguments\": {\"segment_type\": \"text\", \"text_prompt\": \"person\", \"image_index\": 1}}</tool_call>\n"
            "MinerU2.5 example: <tool_call>{\"name\": \"MinerU2.5\", "
            "\"arguments\": {\"extract_type\": \"content\", \"content_type\": \"text\", \"image_index\": 1}}</tool_call>\n"
            "PaddleOCR example: <tool_call>{\"name\": \"PaddleOCR\", "
            "\"arguments\": {\"image_index\": 1}}</tool_call>\n"
            "EasyOCR example: <tool_call>{\"name\": \"EasyOCR\", "
            "\"arguments\": {\"image_index\": 1}}</tool_call>\n"
            "GroundingDINO example: <tool_call>{\"name\": \"GroundingDINO\", "
            "\"arguments\": {\"image_index\": 1, \"prompt\": \"a person.\"}}</tool_call>\n"
            "Qwen-Image-Edit example: <tool_call>{\"name\": \"Qwen-Image-Edit\", "
            "\"arguments\": {\"prompt\": \"turn the girl into a beautiful woman\", \"image_index\": 1}}</tool_call>\n"
            "OpenCV example: <tool_call>{\"name\": \"OpenCV\", "
            "\"arguments\": {\"operation\": \"crop\", \"x\": 100, \"y\": 100, \"w\": 200, \"h\": 200, "
            "\"image_index\": 1}}</tool_call>\n"
            "Skill call example: <tool_call>{\"name\": \"chart_analysis\", \"arguments\": null}</tool_call>\n"
            "create_skill example: <tool_call>{\"name\": \"create_skill\", "
            "\"arguments\": {\"description\": \"Extract key data from bar charts and compute summary stats\"}}</tool_call>\n"
            "run_skill example: <tool_call>{\"name\": \"run_skill\", "
            "\"arguments\": {\"skill_name\": \"chart_analysis\", "
            "\"entrypoint\": \"scripts/run.py\", \"args\": {\"image_index\": 1, \"...\": \"...\"}}}</tool_call>\n"
            "run_skill constraints: validate against target entrypoint in SKILL_SPEC.json; "
            "if entrypoint params declare image_index(required=true), args.image_index is required; "
            "otherwise image_index should not be passed."
            "\nRuntime constraint: args.image_index is used only by runtime for image selection and for injecting "
            "SKILL_IMAGE_PATH/SKILL_IMAGE_DATA_URL. Scripts should not depend on --image_index."
        )

    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        # 1. Parse normal JSON wrapped in `...`
        payload, ok = _parse_tool_call(action)
        if not ok:
            return {}, False

        name = payload.get("name")
        raw_arguments = payload.get("arguments")
        if raw_arguments is None:
            arguments = None
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            return {}, False

        # 2. Handle create_skill tool_call
        if _name_is_create_skill(name):
            description = arguments.get("description") if isinstance(arguments, dict) else None
            return {
                "name": "__create_skill__",
                "arguments": {
                    "description": description,
                },
                "raw_action": action,
            }, True

        if isinstance(name, str) and name.strip().lower() == self.run_skill_tool_name:
            return {
                "name": "__run_skill__",
                "arguments": arguments or {},
                "raw_action": action,
            }, True

        # 3. Check if tool is in valid_mcp_func_names (forever tools)
        if name in self.valid_mcp_func_names:
            payload["arguments"] = arguments or {}
            return payload, True
        payload["arguments"] = arguments or {}
        return payload, True

    def load_env(self, trajectory_id: str) -> Dict[str, Any]:
        """Load or initialize environment state with cross-call image persistence."""
        env = self.env_cache.get(trajectory_id)
        if env is None:
            env = {
                "trajectory_id": trajectory_id,
                "metadata": {"turns": 0},
                "previous_obs": [],
                "images": None,
                "temporary_images": [],
            }
        return env

    def save_image_to_env(self, trajectory_id: str, image: Union[str, Path]) -> Union[str, Path]:
        """Save a new image into the environment for future tool calls."""
        env = self.load_env(trajectory_id)
        env["temporary_images"].append(image)
        return image

    def update_env(
        self,
        trajectory_id: str,
        env: Dict[str, Any],
        action: Any,
        is_valid: bool,
        extra_field: Dict[str, Any],
        observation: Any,
        **kwargs,
    ) -> None:
        """Update environment state and persist generated images when present."""
        if isinstance(observation, dict) and "image" in observation:
            if env.get("images") is None:
                env["images"] = []
            if isinstance(observation["image"], list):
                env["images"].extend(
                    [self.save_image_to_env(trajectory_id, img) for img in observation["image"]]
                )
            else:
                env["images"].append(self.save_image_to_env(trajectory_id, observation["image"]))

        super().update_env(trajectory_id, env, action, is_valid, extra_field, observation, **kwargs)

    def delete_env(self, trajectory_id: str) -> None:
        """Delete environment state and clean up temporary image paths."""
        env = self.env_cache.pop(trajectory_id, None)
        if not env:
            return
        for image in env.get("temporary_images", []):
            path = None
            if isinstance(image, Path):
                path = image
            elif isinstance(image, str) and not image.startswith("data:image") and _safe_path_exists(image):
                path = Path(image)
            if path and _safe_path_exists(path):
                try:
                    path.unlink()
                except Exception:
                    continue

    def _resolve_image_to_data_url(self, image_obj: Union[str, Path]) -> Optional[str]:
        """Convert an environment image object to a data URL."""
        if isinstance(image_obj, Path):
            if _safe_path_exists(image_obj):
                return _to_data_url(image_obj)
            return None
        if isinstance(image_obj, str):
            if image_obj.startswith("data:image") and "base64," in image_obj:
                return image_obj
            if _looks_like_base64(image_obj):
                return f"data:image/png;base64,{image_obj.strip()}"
            if _safe_path_exists(image_obj):
                path_obj = Path(image_obj)
                return _to_data_url(path_obj)
        return None

    def _prepare_images(self, env: Dict[str, Any], extra_field: Dict[str, Any]) -> None:
        """Initialize env['images'] from extra_field only on first access."""
        if env["images"] is not None:
            return
        images = extra_field.get("images") or []
        if isinstance(images, list):
            env["images"] = images.copy()
        else:
            env["images"] = []

    async def _call_model_async(
        self,
        model_name: str,
        image_data_url: str,
        prompt: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Call the model HTTP endpoint asynchronously."""
        base_url = self.url_mapping.get(model_name)
        if not base_url:
            raise ValueError(f"Model URL for {model_name} not found.")

        url = f"{base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer token-abc123"}
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 8192,
        }

        loop = asyncio.get_event_loop()

        def _post():
            start = time.time()
            resp = requests.post(url, headers=headers, json=payload, timeout=self.request_timeout)
            latency = time.time() - start
            resp.raise_for_status()
            return resp.json(), latency

        data, latency = await loop.run_in_executor(None, _post)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {}) or {}
        metadata = {
            "latency": latency,
            "usage": usage,
            "model": model_name,
        }
        return content, metadata

    async def _call_text_model_async(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> Tuple[str, Dict[str, Any]]:
        base_url = self.url_mapping.get(model_name)
        if not base_url:
            raise ValueError(f"Model URL for {model_name} not found.")

        url = f"{base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer token-abc123"}
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 8192,
        }

        loop = asyncio.get_event_loop()

        def _post():
            start = time.time()
            resp = requests.post(url, headers=headers, json=payload, timeout=self.request_timeout)
            latency = time.time() - start
            resp.raise_for_status()
            return resp.json(), latency

        data, latency = await loop.run_in_executor(None, _post)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        metadata = {
            "latency": latency,
            "model": model_name,
            "usage": data.get("usage", {}),
        }
        return content, metadata

    def _parse_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        if not isinstance(text, str):
            return None
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*", "", cleaned).strip()
            cleaned = cleaned.rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None

    async def _repair_generated_script_output(
        self,
        target_path: str,
        script_spec: Dict[str, Any],
        script_content: str,
        validation_error: str,
        runtime_caps: Dict[str, Any],
    ) -> str:
        stdout_contract = _build_skill_stdout_contract(script_spec)
        repair_prompt = (
            "Repair exactly one existing script with minimal edits.\n"
            f"Target script path: {target_path}\n"
            "Keep the same script path and preserve the original core logic.\n"
            "Do not add any dependency outside the exact runtime capability list below.\n"
            f"{_format_skill_runtime_capabilities_for_prompt(runtime_caps)}\n"
            f"{_format_stdout_contract_for_prompt(stdout_contract)}\n"
            "Repair goal:\n"
            "- Keep the script runnable from CLI.\n"
            "- Keep argparse / existing params compatible.\n"
            "- Add or fix stdout printing so create_skill validation passes.\n"
            "- Do not print only a saved file path.\n"
            "- Do not add fake logic or simulated results.\n"
            "- Keep fatal errors on stderr with non-zero exit.\n"
            f"Validation failure:\n{validation_error}\n\n"
            "Original script:\n"
            f"```{Path(target_path).suffix.lstrip('.') or 'text'}\n{script_content}\n```\n\n"
            "Return JSON only in this shape:\n"
            "{\"path\": \"<target path>\", \"content\": \"<repaired script content>\"}\n"
        )
        repaired_text, _ = await self._call_text_model_async(
            self.skill_gen_model_name,
            "You repair one small script with minimal safe edits. Return JSON only.",
            repair_prompt,
        )
        repaired_data = self._parse_json_from_text(repaired_text) or {}
        repaired_path = str(repaired_data.get("path", "")).strip() if isinstance(repaired_data, dict) else ""
        repaired_content = str(repaired_data.get("content", "")) if isinstance(repaired_data, dict) else ""
        if repaired_path != target_path or not repaired_content.strip():
            return ""
        return repaired_content

    async def _generate_skill_artifacts(
        self,
        description: str,
        existing_skills: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        runtime_caps = _get_skill_runtime_capabilities()
        runtime_caps_text = _format_skill_runtime_capabilities_for_prompt(runtime_caps)
        allowed_imports = {str(name) for name in runtime_caps.get("available_imports") or []}
        existing_brief = "\n".join(
            f"- {s.get('name')}: {s.get('description', '')}" for s in existing_skills if s.get("name")
        )
        system_prompt = (
            "You are a skill designer. Create a concise Claude-style skill based on the description. "
            "Return JSON only."
        )
        user_prompt = (
            "Skill description:\n"
            f"{description}\n\n"
            "Existing skills (avoid duplication):\n"
            f"{existing_brief or '- none'}\n\n"
            f"{runtime_caps_text}\n"
            "Stage 1 (planning only): Return JSON with:\n"
            "- skill_name (lowercase, dash-separated)\n"
            "- skill_description (short)\n"
            "- requires_image (boolean)\n"
            "- skill_overview_md (markdown overview of how to use this skill)\n"
            "- scripts: list of script specs (optional). Each item must be:\n"
            "  {path, purpose, params, notes}\n"
            "  where params is a list of {name, type, required, description}\n"
            "Rules:\n"
            "- Do NOT write any code in Stage 1.\n"
            "- Do NOT include script content/source/code fields.\n"
            "- You MUST compare against the provided Existing skills before proposing a new skill.\n"
            "- `skill_name` must not exactly match, trivially rename, or be confusingly similar to any existing skill name.\n"
            "- Do NOT propose a skill whose purpose, workflow, scripts, outputs, or core capability is identical to or substantially overlaps with an existing skill.\n"
            "- If the requested capability is close to an existing skill, differentiate it clearly with a narrower/specific scope and make that distinction explicit in `skill_description` and `skill_overview_md`.\n"
            "- skill_overview_md should explain: what each script does, key parameters, and when to call each script/model.\n"
            "- `requires_image` must be either true or false (no optional/ambiguous state).\n"
            "- Set `requires_image=true` if ANY script in this skill needs image input.\n"
            "- Set `requires_image=false` only when ALL scripts are text-only and never need image input.\n"
            "- If `requires_image=true`, include `image_index` in script params with `required=true` if any script needs image input.\n"
            "- If `requires_image=false`, no script may include `image_index` in params.\n"
            "- Do not add image contract boilerplate to skill_overview_md; it is appended by runtime canonicalization.\n"
            "- Do NOT generate multimodal outputs.\n"
            "- IMPORTANT: scripts will be executed via CLI argv, not function calls.\n"
            "- IMPORTANT: run_skill args are passed as `--key value` (or `--flag` for true boolean).\n"
            "- IMPORTANT: prefer explicit, CLI-friendly params and avoid hidden in-process assumptions.\n"
            "- IMPORTANT: in script notes/overview, emphasize that scripts should print key result variables for downstream model consumption, not only file paths.\n"
            "- IMPORTANT: plan scripts so they rely on Python standard library first, plus only the exact allowed third-party imports shown in Runtime capabilities.\n"
            "- IMPORTANT: do not propose hard-to-obtain, niche, imaginary, obscure, or environment-specific dependencies.\n"
            "- IMPORTANT: if a dependency is not in Python standard library or the exact allowed import list shown in Runtime capabilities, do not use it.\n"
            "- IMPORTANT: do not rely on external model SDKs/clients or ML frameworks inside generated scripts.\n"
            "- IMPORTANT: local imports such as `torch`, `tensorflow`, `transformers`, or `sentence_transformers` are allowed only if they are already available in Runtime capabilities.\n"
            "- IMPORTANT: even when those libraries are available, do NOT download models or tokenizers and do NOT call APIs such as `from_pretrained`, `snapshot_download`, `hf_hub_download`, `torch.hub.load`, or `pipeline(...)`.\n"
            "- IMPORTANT: still avoid remote/model-serving / OCR / heavyweight stacks such as diffusers, openai, anthropic, vllm, langchain, tesseract, pytesseract, easyocr, paddleocr unless explicitly allowed by Runtime capabilities and still without model downloading.\n"
            "- IMPORTANT: on success, print useful results to stdout.\n"
            "- IMPORTANT: on fatal errors, print a concise message to stderr and exit with a non-zero status instead of printing error text to stdout.\n"
            "- IMPORTANT: every script path must end with .py or .sh.\n"
            "- If no scripts are needed, return an empty scripts list.\n"
            "Image skill example (multiple related scripts):\n"
            "{\n"
            "  \"skill_name\": \"chart-analysis\",\n"
            "  \"skill_description\": \"Analyze chart data and summarize trends\",\n"
            "  \"requires_image\": true,\n"
            "  \"skill_overview_md\": \"### Scripts\\n- `scripts/extract_series.py`: extract chart series from image.\\n- `scripts/compute_metrics.py`: compute trends and growth rates from extracted series JSON.\\n- `scripts/summarize_report.py`: generate concise final report from metrics JSON.\\n\\n### Parameters\\n- `run_skill.args.image_index` (required for extract_series): image selector for caller\\n- `--output_json` (required): output file path for intermediate/final data\\n- `--input_json` (required for downstream scripts): input file path from previous step\\n- `--focus` (optional): target metric focus for summary\\n\\n### Workflow\\n1. Run `extract_series.py` first to produce structured data.\\n2. Run `compute_metrics.py` on the extracted data.\\n3. Run `summarize_report.py` to get final textual summary.\\n\\n### Call Contract\\n- run_skill.args.image_index is required only for image-reading scripts.\\n\\n### When to call\\n- Use this skill for chart understanding tasks that need extraction + calculation + final summary.\",\n"
            "  \"scripts\": [\n"
            "    {\n"
            "      \"path\": \"scripts/extract_series.py\",\n"
            "      \"purpose\": \"Read chart image and output structured series values\",\n"
            "      \"params\": [\n"
            "        {\"name\": \"image_index\", \"type\": \"integer\", \"required\": true, \"description\": \"The index of the image to analyze. Index starts from 1, maximum value is the number of images in the current environment. Choose 1 to operate on the first image.\"},\n"
            "        {\"name\": \"output_json\", \"type\": \"string\", \"required\": true, \"description\": \"path to save extracted series JSON\"}\n"
            "      ],\n"
            "      \"notes\": \"Use argparse and print concise output\"\n"
            "    },\n"
            "    {\n"
            "      \"path\": \"scripts/compute_metrics.py\",\n"
            "      \"purpose\": \"Compute trend metrics from extracted series JSON\",\n"
            "      \"params\": [\n"
            "        {\"name\": \"input_json\", \"type\": \"string\", \"required\": true, \"description\": \"path to extracted series JSON\"},\n"
            "        {\"name\": \"output_json\", \"type\": \"string\", \"required\": true, \"description\": \"path to save computed metrics JSON\"}\n"
            "      ],\n"
            "      \"notes\": \"Use argparse and print concise output\"\n"
            "    },\n"
            "    {\n"
            "      \"path\": \"scripts/summarize_report.py\",\n"
            "      \"purpose\": \"Generate final natural language report from metrics\",\n"
            "      \"params\": [\n"
            "        {\"name\": \"input_json\", \"type\": \"string\", \"required\": true, \"description\": \"path to computed metrics JSON\"},\n"
            "        {\"name\": \"focus\", \"type\": \"string\", \"required\": false, \"description\": \"optional focus such as growth, volatility, ranking\"}\n"
            "      ],\n"
            "      \"notes\": \"Use argparse and print concise output\"\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Non-image skill example (multiple related scripts):\n"
            "{\n"
            "  \"skill_name\": \"table-summary\",\n"
            "  \"skill_description\": \"Summarize parsed table text\",\n"
            "  \"requires_image\": false,\n"
            "  \"skill_overview_md\": \"### Scripts\\n- `scripts/clean_table_text.py`: normalize noisy table text.\\n- `scripts/extract_table_facts.py`: extract key-value facts from cleaned text.\\n- `scripts/write_brief.py`: create concise brief from extracted facts.\\n\\n### Parameters\\n- `--text` (required): raw or cleaned table text\\n- `--input_json` (required for downstream scripts): facts/metrics JSON input\\n- `--style` (optional): output style for final brief\\n\\n### Workflow\\n1. Run `clean_table_text.py` to normalize raw text.\\n2. Run `extract_table_facts.py` to structure facts.\\n3. Run `write_brief.py` for final summary.\\n\\n### When to call\\n- Use this skill for text-only table post-processing with deterministic intermediate steps.\",\n"
            "  \"scripts\": [\n"
            "    {\n"
            "      \"path\": \"scripts/clean_table_text.py\",\n"
            "      \"purpose\": \"Normalize raw table text\",\n"
            "      \"params\": [\n"
            "        {\"name\": \"text\", \"type\": \"string\", \"required\": true, \"description\": \"input text\"}\n"
            "      ],\n"
            "      \"notes\": \"Use argparse and print concise output\"\n"
            "    },\n"
            "    {\n"
            "      \"path\": \"scripts/extract_table_facts.py\",\n"
            "      \"purpose\": \"Extract structured facts from cleaned table text\",\n"
            "      \"params\": [\n"
            "        {\"name\": \"text\", \"type\": \"string\", \"required\": true, \"description\": \"cleaned table text\"},\n"
            "        {\"name\": \"output_json\", \"type\": \"string\", \"required\": true, \"description\": \"path to save extracted facts JSON\"}\n"
            "      ],\n"
            "      \"notes\": \"Use argparse and print concise output\"\n"
            "    },\n"
            "    {\n"
            "      \"path\": \"scripts/write_brief.py\",\n"
            "      \"purpose\": \"Generate concise brief from extracted facts\",\n"
            "      \"params\": [\n"
            "        {\"name\": \"input_json\", \"type\": \"string\", \"required\": true, \"description\": \"path to extracted facts JSON\"},\n"
            "        {\"name\": \"style\", \"type\": \"string\", \"required\": false, \"description\": \"brief style, e.g. executive or technical\"}\n"
            "      ],\n"
            "      \"notes\": \"Use argparse and print concise output\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )

        content, _ = await self._call_text_model_async(self.skill_gen_model_name, system_prompt, user_prompt)
        data = self._parse_json_from_text(content) or {}
        skill_name = _sanitize_skill_name(str(data.get("skill_name", "")))
        skill_description = str(data.get("skill_description", "")).strip()
        raw_requires_image = data.get("requires_image", None)
        requires_image: Optional[bool] = raw_requires_image if isinstance(raw_requires_image, bool) else None
        skill_overview_md = str(data.get("skill_overview_md", "")).strip()
        script_specs = _normalize_script_specs(data.get("scripts"))
        stage1_spec = {
            "requires_image": requires_image if isinstance(requires_image, bool) else None,
            "scripts": script_specs,
        }

        if not skill_name:
            skill_name = _sanitize_skill_name("auto-skill")
        if not skill_description:
            skill_description = description.strip()[:200]
        skill_md = _build_canonical_skill_md(
            skill_name=skill_name,
            skill_description=skill_description,
            description=description,
            requires_image=requires_image,
            overview_md=skill_overview_md,
        )

        files: List[Dict[str, Any]] = []
        generation_warnings: List[str] = []
        if script_specs:
            full_specs_json = json.dumps(script_specs, ensure_ascii=False, indent=2)
            max_attempts = SKILL_SCRIPT_MAX_SYNTAX_RETRIES + 1
            for script_spec in script_specs:
                target_path = script_spec["path"]
                stdout_contract = _build_skill_stdout_contract(script_spec)
                base_stage2_user_prompt = (
                    "Stage 2 (single script generation): Generate exactly one script.\n"
                    f"Target script path: {target_path}\n"
                    "You must generate only the target script for this call.\n"
                    "Scripts must be .py or .sh only.\n"
                    "Runtime contract (must follow):\n"
                    "- Scripts are launched as external commands with argv.\n"
                    "- Args are passed as `--key value` pairs.\n"
                    "- `args.image_index` is consumed by runtime for image selection and is not forwarded "
                    "to script CLI args.\n"
                    "- Do not declare `--image_index` in the script; use image env vars instead.\n"
                    "- If the Stage-1 script spec includes `image_index`, mean treat it as caller-side contract "
                    "metadata only (do not parse it as a CLI argument).\n"
                    "- Boolean true may be sent as `--key` only; boolean false as `--key false`.\n"
                    "- None may be sent as `--key null`.\n"
                    "- dict/list args are JSON strings; parse them before use.\n"
                    "- For Python scripts, use argparse to parse CLI args robustly.\n"
                    "- For Python scripts, put executable code under `if __name__ == '__main__':` and make sure `python <script> --help` succeeds in a clean sandbox.\n"
                    f"{runtime_caps_text}"
                    "- Dependency rule: use Python standard library first; only use third-party imports explicitly listed in Runtime capabilities.\n"
                    "- Dependency rule: do not import or depend on hard-to-install, niche, imaginary, obscure, or environment-specific libraries.\n"
                    "- Dependency rule: do not call external model SDKs/clients/frameworks inside the script.\n"
                    "- Dependency rule: local libraries such as `torch`, `tensorflow`, `transformers`, and `sentence_transformers` may be imported only when they appear in Runtime capabilities.\n"
                    "- Dependency rule: even when imported, do NOT download models/tokenizers and do NOT use patterns such as `from_pretrained`, `snapshot_download`, `hf_hub_download`, `torch.hub.load`, or `pipeline(...)`.\n"
                    "- Dependency rule: do not write shell code that installs packages at runtime (`pip install`, `conda install`, etc.).\n"
                    "- Validation rule: generated Python scripts will be checked by actually importing their dependencies and running `python <script> --help` in the sandbox. If that would fail, do not generate that script.\n"
                    f"{_format_stdout_contract_for_prompt(stdout_contract)}"
                    "- Error channel contract: on fatal errors, write the message to stderr and exit non-zero; do not print fatal error text only to stdout.\n"
                    "- Error channel example (Python): `print(message, file=sys.stderr); raise SystemExit(1)`.\n"
                    "- Error channel example (shell): `echo \"message\" >&2; exit 1`.\n"
                    "- Diversity requirement: the script's output must vary based on the actual input content, files, and observations from this run. Do NOT output generic/static placeholder descriptions, boilerplate summaries, or simulated results that could be reused unchanged across different inputs.\n"
                    "  - BAD: 'header: Top navigation bar with logo and menu items' (same for every image)\n"
                    "  - GOOD: 'header: Purple gradient background, white logo left-aligned, 3 menu items: Home, About, Contact (right-aligned)'\n"
                    "- Diversity requirement: never fabricate a default template just to satisfy the output format. If the real input does not support a claimed detail, inspect more, infer conservatively, or report uncertainty explicitly.\n"
                    "- For image handling, follow this rule per target script: "
                    "if the script needs image input, it must read image data from "
                    "`SKILL_IMAGE_PATH` / `SKILL_IMAGE_DATA_URL` env vars; "
                    "otherwise it must not rely on `SKILL_IMAGE_*` env vars.\n"
                    "- Never use `SKILL_IMAGE_INDEX` as an env var in scripts.\n"
                    "- Runtime does not inject `SKILL_IMAGE_INDEX`; only `SKILL_IMAGE_PATH` / "
                    "`SKILL_IMAGE_DATA_URL` are available for image access.\n"
                    "- Image env semantics:\n"
                    "  - `SKILL_IMAGE_PATH`: local filesystem path to selected image "
                    "(example: `/tmp/skill_img_abc123.png`).\n"
                    "  - `SKILL_IMAGE_DATA_URL`: data URL for selected image "
                    "(example prefix: `data:image/png;base64,....`).\n"
                    "  - Prefer `SKILL_IMAGE_PATH` when present; otherwise fallback to "
                    "`SKILL_IMAGE_DATA_URL`.\n"
                    "Return JSON only in this shape:\n"
                    "{\"path\": \"<target path>\", \"content\": \"<script content>\"}\n"
                    f"Skill description:\n{description}\n\n"
                    f"Skill name: {skill_name}\n"
                    f"Skill short description: {skill_description}\n\n"
                    f"Skill overview markdown (from Stage 1):\n{skill_overview_md or '(empty)'}\n\n"
                    f"All script specs:\n{full_specs_json}\n\n"
                    f"Current target script spec:\n{json.dumps(script_spec, ensure_ascii=False, indent=2)}\n"
                )
                selected_content = ""
                last_error = ""
                for attempt in range(max_attempts):
                    stage2_user_prompt = (
                        base_stage2_user_prompt
                        if attempt == 0
                        else (
                            f"{base_stage2_user_prompt}\n\n"
                            f"{_build_stage2_retry_prompt(last_error, attempt, max_attempts)}"
                        )
                    )
                    content2, _ = await self._call_text_model_async(
                        self.skill_gen_model_name,
                        "You are a helpful assistant that writes one small, safe script per request. Return JSON only.",
                        stage2_user_prompt,
                    )
                    data2 = self._parse_json_from_text(content2) or {}
                    returned_path = str(data2.get("path", "")).strip() if isinstance(data2, dict) else ""
                    returned_content = str(data2.get("content", "")) if isinstance(data2, dict) else ""

                    if returned_path != target_path or not returned_content.strip():
                        last_error = "invalid model output path/content."
                        generation_warnings.append(
                            f"Script generation retry {attempt + 1}/{max_attempts} for {target_path}: {last_error}"
                        )
                        continue

                    syntax_ok, syntax_error = _validate_generated_script_syntax(target_path, returned_content)
                    if not syntax_ok:
                        last_error = f"syntax error: {syntax_error}"
                        generation_warnings.append(
                            f"Script generation retry {attempt + 1}/{max_attempts} for {target_path}: {last_error}"
                        )
                        continue

                    deps_ok, deps_error = _validate_generated_script_dependencies(
                        target_path,
                        returned_content,
                        allowed_imports=allowed_imports,
                    )
                    if not deps_ok:
                        last_error = f"dependency violation: {deps_error}"
                        generation_warnings.append(
                            f"Script generation retry {attempt + 1}/{max_attempts} for {target_path}: {last_error}"
                        )
                        continue

                    contract_ok, contract_error = _validate_generated_script_contract(
                        target_path,
                        returned_content,
                        script_spec,
                        allowed_imports=allowed_imports,
                    )
                    if not contract_ok:
                        repaired_content = await self._repair_generated_script_output(
                            target_path=target_path,
                            script_spec=script_spec,
                            script_content=returned_content,
                            validation_error=contract_error,
                            runtime_caps=runtime_caps,
                        )
                        if repaired_content:
                            syntax_ok, syntax_error = _validate_generated_script_syntax(target_path, repaired_content)
                            if syntax_ok:
                                deps_ok, deps_error = _validate_generated_script_dependencies(
                                    target_path,
                                    repaired_content,
                                    allowed_imports=allowed_imports,
                                )
                                if deps_ok:
                                    contract_ok, contract_error = _validate_generated_script_contract(
                                        target_path,
                                        repaired_content,
                                        script_spec,
                                        allowed_imports=allowed_imports,
                                    )
                                    if contract_ok:
                                        returned_content = repaired_content
                        if not contract_ok:
                            last_error = f"contract validation: {contract_error}"
                            generation_warnings.append(
                                f"Script generation retry {attempt + 1}/{max_attempts} for {target_path}: {last_error}"
                            )
                            continue

                    runtime_ok, runtime_error = _validate_generated_script_runtime(
                        target_path,
                        returned_content,
                        allowed_imports=allowed_imports,
                    )
                    if not runtime_ok:
                        last_error = f"runtime validation: {runtime_error}"
                        generation_warnings.append(
                            f"Script generation retry {attempt + 1}/{max_attempts} for {target_path}: {last_error}"
                        )
                        continue

                    selected_content = returned_content
                    break

                if not selected_content:
                    raise ValueError(
                        f"failed to generate a validated script for {target_path}: {last_error or 'retries exhausted'}"
                    )

                files.append({"path": target_path, "content": selected_content})

        return {
            "name": skill_name,
            "description": skill_description,
            "skill_md": skill_md,
            "stage1_spec": stage1_spec,
            "files": files,
            "generation_warnings": generation_warnings,
        }

    async def _call_sam3_text_async(
        self,
        image_base64: str,
        text_prompt: str,
        threshold: float = 0.5,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Call the SAM3 text-based segmentation endpoint."""
        base_url = self.url_mapping.get(self.sam3_model_name)
        if not base_url:
            raise ValueError(f"Model URL for {self.sam3_model_name} not found.")

        url = f"{base_url}/segment/text"
        payload = {
            "image_base64": image_base64,
            "text_prompt": text_prompt,
            "threshold": threshold,
        }

        loop = asyncio.get_event_loop()

        def _post():
            start = time.time()
            resp = requests.post(url, json=payload, timeout=self.request_timeout)
            latency = time.time() - start
            resp.raise_for_status()
            return resp.json(), latency

        data, latency = await loop.run_in_executor(None, _post)
        metadata = {
            "latency": latency,
            "model": self.sam3_model_name,
        }
        return data, metadata

    async def _call_sam3_box_async(
        self,
        image_base64: str,
        boxes: List[List[float]],
        labels: List[bool],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Call the SAM3 box-based segmentation endpoint."""
        base_url = self.url_mapping.get(self.sam3_model_name)
        if not base_url:
            raise ValueError(f"Model URL for {self.sam3_model_name} not found.")

        url = f"{base_url}/segment/box"
        payload = {
            "image_base64": image_base64,
            "boxes": boxes,
            "labels": labels,
        }

        loop = asyncio.get_event_loop()

        def _post():
            start = time.time()
            resp = requests.post(url, json=payload, timeout=self.request_timeout)
            latency = time.time() - start
            resp.raise_for_status()
            return resp.json(), latency

        data, latency = await loop.run_in_executor(None, _post)
        metadata = {
            "latency": latency,
            "model": self.sam3_model_name,
        }
        return data, metadata

    async def _call_mineru_async(
        self,
        image_base64: str,
        extract_type: str,
        content_type: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Call the MinerU document extraction endpoint."""
        base_url = self.url_mapping.get(self.mineru_model_name)
        if not base_url:
            raise ValueError(f"Model URL for {self.mineru_model_name} not found.")

        url = f"{base_url}/extract"
        payload = {
            "image_base64": image_base64,
            "extract_type": extract_type,
        }
        if content_type:
            payload["content_type"] = content_type

        loop = asyncio.get_event_loop()

        def _post():
            start = time.time()
            resp = requests.post(url, json=payload, timeout=self.request_timeout)
            latency = time.time() - start
            resp.raise_for_status()
            return resp.json(), latency

        data, latency = await loop.run_in_executor(None, _post)
        metadata = {
            "latency": latency,
            "model": self.mineru_model_name,
        }
        return data, metadata

    async def _call_paddle_async(
        self,
        file_path: Path,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Call the PaddleOCR file endpoint."""
        base_url = self.url_mapping.get(self.paddle_model_name)
        if not base_url:
            raise ValueError(f"Model URL for {self.paddle_model_name} not found.")

        url = f"{base_url}/ocr"
        loop = asyncio.get_event_loop()

        def _post():
            start = time.time()
            with file_path.open("rb") as f:
                files = {"file": f}
                data = {"task": "ocr"}
                resp = requests.post(url, files=files, data=data, timeout=self.request_timeout)
            latency = time.time() - start
            resp.raise_for_status()
            return resp.json(), latency

        data, latency = await loop.run_in_executor(None, _post)
        metadata = {
            "latency": latency,
            "model": self.paddle_model_name,
        }
        return data, metadata

    async def _call_image_edit_async(
        self,
        images_base64: List[str],
        prompt: str,
        negative_prompt: str = " ",
        num_inference_steps: int = 40,
        true_cfg_scale: float = 4.0,
        guidance_scale: float = 1.0,
        seed: int = 0,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Call the Qwen image editing endpoint."""
        base_url = self.url_mapping.get(self.image_edit_model_name)
        if not base_url:
            raise ValueError(f"Model URL for {self.image_edit_model_name} not found.")

        url = f"{base_url}/edit_base64"
        payload = {
            "prompt": prompt,
            "images_base64": json.dumps(images_base64),
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": true_cfg_scale,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }

        loop = asyncio.get_event_loop()

        def _post():
            start = time.time()
            resp = requests.post(url, data=payload, timeout=self.request_timeout)
            latency = time.time() - start
            resp.raise_for_status()
            return resp.json(), latency

        data, latency = await loop.run_in_executor(None, _post)
        metadata = {
            "latency": latency,
            "model": self.image_edit_model_name,
        }
        return data, metadata

    async def _call_easyocr_async(
        self,
        image_base64: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        base_url = self.url_mapping.get(self.easyocr_model_name)
        if not base_url:
            raise ValueError(f"Model URL for {self.easyocr_model_name} not found.")

        url = f"{base_url}/infer_base64"
        payload = {
            "image_base64": image_base64,
            "langs": "en",
        }

        loop = asyncio.get_event_loop()

        def _post():
            start = time.time()
            resp = requests.post(url, data=payload, timeout=self.request_timeout)
            latency = time.time() - start
            resp.raise_for_status()
            return resp.json(), latency

        data, latency = await loop.run_in_executor(None, _post)
        metadata = {
            "latency": latency,
            "model": self.easyocr_model_name,
        }
        return data, metadata

    async def _call_groundingdino_async(
        self,
        image_base64: str,
        prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        base_url = self.url_mapping.get(self.groundingdino_model_name)
        if not base_url:
            raise ValueError(f"Model URL for {self.groundingdino_model_name} not found.")

        url = f"{base_url}/infer_base64"
        payload = {
            "image_base64": image_base64,
            "prompt": prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
        }

        loop = asyncio.get_event_loop()

        def _post():
            start = time.time()
            resp = requests.post(url, data=payload, timeout=self.request_timeout)
            latency = time.time() - start
            resp.raise_for_status()
            return resp.json(), latency

        data, latency = await loop.run_in_executor(None, _post)
        metadata = {
            "latency": latency,
            "model": self.groundingdino_model_name,
        }
        return data, metadata

    async def _conduct_action_async(
        self, trajectory_id: str, action: str, extra_field: Dict[str, Any]
    ) -> Tuple[Any, bool, bool]:
        parsed, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)

        def _infer_observation_tool_name(action_payload: Optional[Dict[str, Any]]) -> Optional[str]:
            if not isinstance(action_payload, dict):
                return None
            payload_name = action_payload.get("name")
            if not isinstance(payload_name, str) or not payload_name:
                return None
            if payload_name == "__create_skill__":
                return self.create_skill_tool_name
            if payload_name == "__run_skill__":
                return self.run_skill_tool_name
            return payload_name

        def _finalize(
            observation: Any,
            done: bool,
            valid: bool,
            action_payload: Optional[Dict[str, Any]] = None,
            is_valid_flag: Optional[bool] = None,
        ) -> Tuple[Any, bool, bool]:
            payload = parsed if action_payload is None else action_payload
            ok = is_valid if is_valid_flag is None else is_valid_flag
            if (
                isinstance(observation, dict)
                and not observation.get("tool")
                and (observation.get("invalid_reason") or not valid or not ok)
            ):
                inferred_tool_name = _infer_observation_tool_name(payload)
                if inferred_tool_name:
                    observation["tool"] = inferred_tool_name
            self.update_env(trajectory_id, env, payload, ok, extra_field, observation)
            self.save_env(trajectory_id, env)
            return observation, done, valid

        if not is_valid:
            observation = {
                "obs": "Invalid tool_call format for multimodal_processor_tool.",
                "invalid_reason": "parse_failed",
            }
            return _finalize(observation, False, False)

        arguments = parsed.get("arguments", {})
        tool_name = parsed["name"]

        # Handle <create_skill> case
        if tool_name == "__create_skill__":
            create_args = parsed.get("arguments", {})
            description = create_args.get("description", "")

            if not description or not isinstance(description, str):
                observation = {
                    "obs": "Invalid create_skill tool_call format: description is required.",
                    "invalid_reason": "missing_parameters",
                }
                return _finalize(observation, False, False)

            create_turn = int(env.get("metadata", {}).get("turns", 0)) + 1
            trajectory_id_str = str(trajectory_id)

            try:
                existing_skills = list(self._load_skill_catalog().values())
                artifacts = await self._generate_skill_artifacts(description, existing_skills)
                skill_name = artifacts["name"]
                skill_description = artifacts["description"]
                skill_md = artifacts["skill_md"]
                stage1_spec = artifacts.get("stage1_spec")
                files = artifacts["files"]

                pending_root = _get_skill_pending_dir() / trajectory_id_str
                pending_root.mkdir(parents=True, exist_ok=True)
                skill_dir = _ensure_unique_dir(pending_root, skill_name)
                skill_dir.mkdir(parents=True, exist_ok=True)

                skill_md_path = skill_dir / "SKILL.md"
                skill_md_path.write_text(skill_md, encoding="utf-8")
                if isinstance(stage1_spec, dict):
                    skill_spec_path = skill_dir / "SKILL_SPEC.json"
                    skill_spec_path.write_text(
                        json.dumps(stage1_spec, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                for file_item in files:
                    if not isinstance(file_item, dict):
                        continue
                    rel_path = str(file_item.get("path", "")).strip()
                    content = file_item.get("content", "")
                    if not rel_path:
                        continue
                    target_path = (skill_dir / rel_path).resolve()
                    if skill_dir not in target_path.parents and target_path != skill_dir:
                        continue
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        target_path.write_text(str(content), encoding="utf-8")
                    except Exception:
                        continue

                observation = {
                    "obs": f"[Skill: {skill_name}] created. SKILL.md:\n{skill_md}",
                    "tool": self.create_skill_tool_name,
                    "skill_name": skill_name,
                    "skill_path": str(skill_dir),
                    "description": skill_description,
                    "created_trajectory_id": trajectory_id_str,
                    "created_turn": create_turn,
                }
                return _finalize(observation, False, True)
            except Exception as exc:
                observation = {
                    "obs": f"create_skill failed: {exc}",
                    "invalid_reason": "request_failed",
                }
                return _finalize(observation, False, False)

        if tool_name == "__run_skill__":
            run_args = parsed.get("arguments", {})
            skill_name = run_args.get("skill_name")
            entrypoint = run_args.get("entrypoint")
            args = run_args.get("args", {})

            if not skill_name or not entrypoint:
                observation = {
                    "obs": "Invalid run_skill tool_call format: skill_name and entrypoint are required.",
                    "invalid_reason": "missing_parameters",
                }
                return _finalize(observation, False, False)

            skill = self._resolve_skill(trajectory_id, str(skill_name))
            if not skill:
                observation = {
                    "obs": f"Unknown skill: {skill_name}",
                    "invalid_reason": "unknown_skill",
                }
                return _finalize(observation, False, False)

            skill_dir = Path(skill.get("path", ""))
            try:
                skill_dir_resolved = skill_dir.resolve()
            except Exception:
                skill_dir_resolved = skill_dir
            if not skill_dir_resolved.exists():
                observation = {
                    "obs": f"Skill path not found: {skill_dir_resolved}",
                    "invalid_reason": "missing_skill_path",
                }
                return _finalize(observation, False, False)

            entrypoint_path = Path(entrypoint)
            if entrypoint_path.is_absolute():
                observation = {
                    "obs": "Entrypoint must be a relative path within the skill directory.",
                    "invalid_reason": "invalid_entrypoint",
                }
                return _finalize(observation, False, False)

            resolved_entrypoint = (skill_dir_resolved / entrypoint_path).resolve()
            if (
                skill_dir_resolved not in resolved_entrypoint.parents
                and resolved_entrypoint != skill_dir_resolved
            ):
                observation = {
                    "obs": "Entrypoint path escapes skill directory.",
                    "invalid_reason": "invalid_entrypoint",
                }
                return _finalize(observation, False, False)

            if not resolved_entrypoint.exists() or not resolved_entrypoint.is_file():
                observation = {
                    "obs": f"Entrypoint not found: {entrypoint}",
                    "invalid_reason": "missing_entrypoint",
                }
                return _finalize(observation, False, False)

            if resolved_entrypoint.suffix.lower() not in {".py", ".sh"}:
                observation = {
                    "obs": "Unsupported entrypoint extension; only .py and .sh are allowed.",
                    "invalid_reason": "invalid_entrypoint",
                }
                return _finalize(observation, False, False)

            if args is None:
                args = {}
            if not isinstance(args, dict):
                observation = {
                    "obs": "Invalid args format; must be a JSON object.",
                    "invalid_reason": "invalid_arguments",
                }
                return _finalize(observation, False, False)

            stage1_spec = _load_stage1_spec_from_file(skill_dir_resolved)
            entrypoint_requires_image = _entrypoint_requires_image(stage1_spec, str(entrypoint))

            self._prepare_images(env, extra_field)
            image_index = args.get("image_index")
            if entrypoint_requires_image is True and image_index is None:
                observation = {
                    "obs": (
                        "Missing required parameter: image_index. "
                        "This entrypoint requires image input; pass `args.image_index` in run_skill."
                    ),
                    "invalid_reason": "missing_image_index",
                }
                return _finalize(observation, False, False)

            if entrypoint_requires_image is True and not env.get("images"):
                observation = {
                    "obs": "No images available for this entrypoint. Provide images in extra_fields and pass args.image_index.",
                    "invalid_reason": "invalid_image_index",
                }
                return _finalize(observation, False, False)
            image_path = None
            if image_index is not None:
                try:
                    image_index = int(image_index)
                except Exception:
                    observation = {
                        "obs": "Invalid image_index format. It should be an integer.",
                        "invalid_reason": "invalid_image_index",
                    }
                    return _finalize(observation, False, False)
                if image_index <= 0 or image_index > len(env["images"]):
                    observation = {
                        "obs": f"Invalid image_index. It should be between 1 and {len(env['images'])}.",
                        "invalid_reason": "invalid_image_index",
                    }
                    return _finalize(observation, False, False)
                image_obj = env["images"][image_index - 1]

                if isinstance(image_obj, Path):
                    if _safe_path_exists(image_obj):
                        image_path = str(image_obj)
                elif isinstance(image_obj, str):
                    # Strings that are not data URL/base64 are treated as file paths only when safely resolvable.
                    if (
                        not image_obj.startswith("data:image")
                        and not _looks_like_base64(image_obj)
                        and _safe_path_exists(image_obj)
                    ):
                        image_path = image_obj

                if not image_path:
                    image_data_url = self._resolve_image_to_data_url(image_obj)
                    try:
                        if image_data_url and image_data_url.startswith("data:image") and "base64," in image_data_url:
                            fd, tmp_path = tempfile.mkstemp(prefix="skill_img_", suffix=".png")
                            os.close(fd)
                            b64 = image_data_url.split("base64,", 1)[1]
                            with open(tmp_path, "wb") as f:
                                f.write(base64.b64decode(b64))
                            image_path = tmp_path
                            env["temporary_images"].append(Path(tmp_path))
                    except Exception:
                        image_path = None
                if not image_path:
                    observation = {
                        "obs": (
                            "Failed to resolve image input for run_skill. "
                            "Expected a valid local image path, data URL, or raw base64 image."
                        ),
                        "invalid_reason": "invalid_image_input",
                    }
                    return _finalize(observation, False, False)

            cmd_args = dict(args or {})
            cmd_args.pop("image_index", None)
            try:
                cmd = _build_skill_command(resolved_entrypoint, cmd_args)
            except Exception as exc:
                observation = {
                    "obs": f"Failed to build run_skill command: {exc}",
                    "invalid_reason": "invalid_entrypoint",
                }
                return _finalize(observation, False, False)

            env_vars = _sanitize_env(os.environ.copy())
            env_vars["SKILL_DIR"] = str(skill_dir_resolved)
            if image_path:
                env_vars["SKILL_IMAGE_PATH"] = str(image_path)

            # stdin = arguments.get("stdin") if isinstance(arguments, dict) else None
            # if stdin is None and extra_field:
            #     stdin = extra_field.get("stdin")

            loop = asyncio.get_event_loop()

            def _run():
                start = time.time()
                try:
                    completed = subprocess.run(
                        cmd,
                        text=True,
                        capture_output=True,
                        timeout=RUN_SKILL_TIMEOUT,
                        cwd=str(skill_dir_resolved),
                        env=env_vars,
                    )
                    latency = time.time() - start
                    return completed, latency, None
                except subprocess.TimeoutExpired as exc:
                    latency = time.time() - start
                    return None, latency, exc

            completed, latency, error = await loop.run_in_executor(None, _run)
            if error:
                observation = {
                    "obs": f"run_skill timeout after {RUN_SKILL_TIMEOUT}s.",
                    "tool": self.run_skill_tool_name,
                    "skill_name": skill.get("name", skill_name),
                    "entrypoint": str(entrypoint),
                    "latency": latency,
                    "invalid_reason": "timeout",
                }
                return _finalize(observation, False, False)

            stdout = (completed.stdout or "").strip() if completed else ""
            stderr = (completed.stderr or "").strip() if completed else ""
            output_parts = []
            if stdout:
                output_parts.append(stdout)
            if stderr:
                output_parts.append(f"[stderr]\n{stderr}")
            obs_text = "\n".join(output_parts).strip() or "run_skill completed with no output."

            is_valid_run = completed.returncode == 0 if completed else False
            observation = {
                "obs": obs_text,
                "tool": self.run_skill_tool_name,
                "skill_name": skill.get("name", skill_name),
                "entrypoint": str(entrypoint),
                "latency": latency,
            }
            if not is_valid_run:
                observation["invalid_reason"] = "runtime_error"
            return _finalize(observation, False, is_valid_run)

        self._prepare_images(env, extra_field)

        if tool_name == self.opencv_tool_name:
            operation = arguments.get("operation")
            image_index = arguments.get("image_index")
            if operation is None or image_index is None:
                observation = {
                    "obs": "Missing parameters: operation and image_index are required.",
                    "invalid_reason": "missing_parameters",
                }
                return _finalize(observation, False, False)

            op = str(operation).lower()
            valid_ops = {
                "crop",
                "resize",
                "rotate",
                "flip",
                "grayscale",
                "blur",
                "threshold",
                "canny",
            }
            if op not in valid_ops:
                observation = {
                    "obs": "Invalid operation. Supported: crop, resize, rotate, flip, grayscale, blur, threshold, canny.",
                    "invalid_reason": "invalid_parameters",
                }
                return _finalize(observation, False, False)

            try:
                image_index = int(image_index)
            except Exception:
                observation = {
                    "obs": "Invalid image_index format. It should be an integer.",
                    "invalid_reason": "invalid_image_index",
                }
                return _finalize(observation, False, False)

            if image_index <= 0 or image_index > len(env["images"]):
                observation = {
                    "obs": f"Invalid image_index. It should be between 1 and {len(env['images'])}.",
                    "invalid_reason": "invalid_image_index",
                }
                return _finalize(observation, False, False)

            required_params = []
            threshold_type = None
            if op == "crop":
                required_params = ["x", "y", "w", "h"]
            elif op == "resize":
                required_params = ["width", "height"]
            elif op == "rotate":
                required_params = ["angle"]
            elif op == "flip":
                required_params = ["flip_code"]
            elif op == "blur":
                required_params = ["ksize"]
            elif op == "threshold":
                threshold_type = str(arguments.get("type", "BINARY")).upper()
                if threshold_type != "OTSU":
                    required_params = ["thresh"]
            elif op == "canny":
                required_params = ["threshold1", "threshold2"]

            missing = [key for key in required_params if arguments.get(key) is None]
            if missing:
                observation = {
                    "obs": f"Missing parameters: {', '.join(missing)} are required.",
                    "invalid_reason": "missing_parameters",
                }
                return _finalize(observation, False, False)

            image_obj = env["images"][image_index - 1]
            try:
                pil_image = _load_pil_image(image_obj)
            except Exception as exc:
                observation = {
                    "obs": f"Failed to load image for OpenCV: {exc}",
                    "invalid_reason": "missing_image",
                }
                return _finalize(observation, False, False)

            op_args = dict(arguments)
            op_args.pop("operation", None)
            op_args.pop("image_index", None)
            if threshold_type is not None:
                op_args["type"] = threshold_type

            try:
                result_image = OpenCVProcessor.process(pil_image, op, **op_args)
            except ValueError as exc:
                observation = {
                    "obs": f"Invalid parameters: {exc}",
                    "invalid_reason": "invalid_parameters",
                }
                return _finalize(observation, False, False)
            except Exception as exc:
                observation = {
                    "obs": f"multimodal_processor_tool failed: {exc}",
                    "invalid_reason": "request_failed",
                }
                return _finalize(observation, False, False)

            image_url = _image_to_data_url(result_image)
            observation = {
                "obs": f"[Tool: {self.opencv_tool_name}] Operation '{op}' completed.",
                "tool": self.opencv_tool_name,
                "model": self.opencv_tool_name,
                "image": image_url,
            }
            return _finalize(observation, False, True)

        if tool_name == self.sam3_tool_name:
            segment_type = arguments.get("segment_type")
            image_index = arguments.get("image_index")
            if segment_type is None or image_index is None:
                observation = {
                    "obs": "Missing parameters: segment_type and image_index are required.",
                    "invalid_reason": "missing_parameters",
                }
                return _finalize(observation, False, False)

            try:
                image_index = int(image_index)
            except Exception:
                observation = {
                    "obs": "Invalid image_index format. It should be an integer.",
                    "invalid_reason": "invalid_image_index",
                }
                return _finalize(observation, False, False)

            if image_index <= 0 or image_index > len(env["images"]):
                observation = {
                    "obs": f"Invalid image_index. It should be between 1 and {len(env['images'])}.",
                    "invalid_reason": "invalid_image_index",
                }
                return _finalize(observation, False, False)

            image_obj = env["images"][image_index - 1]
            image_base64 = _image_to_base64(image_obj)

            if segment_type == "text":
                text_prompt = arguments.get("text_prompt")
                if text_prompt is None:
                    observation = {
                        "obs": "Missing parameters: text_prompt is required for text segmentation.",
                        "invalid_reason": "missing_parameters",
                    }
                    return _finalize(observation, False, False)
                threshold = arguments.get("threshold", 0.5)
                try:
                    threshold = float(threshold)
                except Exception:
                    threshold = 0.5

                try:
                    result, meta = await self._call_sam3_text_async(
                        image_base64=image_base64,
                        text_prompt=str(text_prompt),
                        threshold=threshold,
                    )
                except ValueError as exc:
                    observation = {
                        "obs": f"multimodal_processor_tool failed: {exc}",
                        "invalid_reason": "unknown_model",
                    }
                    return _finalize(observation, False, False)
                except Exception as exc:
                    observation = {
                        "obs": f"multimodal_processor_tool failed: {exc}",
                        "invalid_reason": "request_failed",
                    }
                    return _finalize(observation, False, False)

                data = result.get("data", {}) if isinstance(result, dict) else {}
                masks_base64 = data.get("masks_base64", []) if isinstance(data, dict) else []
                if masks_base64:
                    labels = [str(text_prompt)] * len(masks_base64)
                    vis_image = _visualize_sam3(
                        image_obj,
                        masks_base64=masks_base64,
                        boxes=data.get("boxes"),
                        scores=data.get("scores"),
                        labels=labels,
                    )
                    vis_image_url = _image_to_data_url(vis_image)
                else:
                    vis_image_url = None

                cleaned = result
                if isinstance(result, dict) and isinstance(result.get("data"), dict):
                    cleaned = dict(result)
                    cleaned_data = dict(result["data"])
                    cleaned_data.pop("masks_base64", None)
                    cleaned["data"] = cleaned_data

                observation = {
                    "obs": f"[Tool: {self.sam3_tool_name} (text)] {json.dumps(cleaned, ensure_ascii=False)}",
                    "tool": self.sam3_tool_name,
                    "model": self.sam3_model_name,
                    "latency": meta.get("latency"),
                }
                if vis_image_url:
                    observation["image"] = vis_image_url
                return _finalize(observation, False, True)

            if segment_type == "box":
                boxes = arguments.get("boxes")
                labels = arguments.get("labels")
                if boxes is None or labels is None:
                    observation = {
                        "obs": "Missing parameters: boxes and labels are required for box segmentation.",
                        "invalid_reason": "missing_parameters",
                    }
                    return _finalize(observation, False, False)

                try:
                    result, meta = await self._call_sam3_box_async(
                        image_base64=image_base64,
                        boxes=boxes,
                        labels=labels,
                    )
                except ValueError as exc:
                    observation = {
                        "obs": f"multimodal_processor_tool failed: {exc}",
                        "invalid_reason": "unknown_model",
                    }
                    return _finalize(observation, False, False)
                except Exception as exc:
                    observation = {
                        "obs": f"multimodal_processor_tool failed: {exc}",
                        "invalid_reason": "request_failed",
                    }
                    return _finalize(observation, False, False)

                data = result.get("data", {}) if isinstance(result, dict) else {}
                masks_base64 = data.get("masks_base64", []) if isinstance(data, dict) else []
                if masks_base64:
                    label_texts = [str(label) for label in labels] if isinstance(labels, list) else None
                    vis_image = _visualize_sam3(
                        image_obj,
                        masks_base64=masks_base64,
                        boxes=data.get("boxes"),
                        scores=data.get("scores"),
                        labels=label_texts,
                    )
                    vis_image_url = _image_to_data_url(vis_image)
                else:
                    vis_image_url = None

                cleaned = result
                if isinstance(result, dict) and isinstance(result.get("data"), dict):
                    cleaned = dict(result)
                    cleaned_data = dict(result["data"])
                    cleaned_data.pop("masks_base64", None)
                    cleaned["data"] = cleaned_data

                observation = {
                    "obs": f"[Tool: {self.sam3_tool_name} (box)] {json.dumps(cleaned, ensure_ascii=False)}",
                    "tool": self.sam3_tool_name,
                    "model": self.sam3_model_name,
                    "latency": meta.get("latency"),
                }
                if vis_image_url:
                    observation["image"] = vis_image_url
                return _finalize(observation, False, True)

            observation = {
                "obs": "Invalid segment_type. It should be 'text' or 'box'.",
                "invalid_reason": "invalid_parameters",
            }
            return _finalize(observation, False, False)

        if tool_name == self.mineru_tool_name:
            extract_type = arguments.get("extract_type")
            image_index = arguments.get("image_index")
            if extract_type is None or image_index is None:
                observation = {
                    "obs": "Missing parameters: extract_type and image_index are required.",
                    "invalid_reason": "missing_parameters",
                }
                return _finalize(observation, False, False)

            if extract_type not in {"two_step", "layout", "content"}:
                observation = {
                    "obs": "Invalid extract_type. It should be 'two_step', 'layout', or 'content'.",
                    "invalid_reason": "invalid_parameters",
                }
                return _finalize(observation, False, False)

            try:
                image_index = int(image_index)
            except Exception:
                observation = {
                    "obs": "Invalid image_index format. It should be an integer.",
                    "invalid_reason": "invalid_image_index",
                }
                return _finalize(observation, False, False)

            if image_index <= 0 or image_index > len(env["images"]):
                observation = {
                    "obs": f"Invalid image_index. It should be between 1 and {len(env['images'])}.",
                    "invalid_reason": "invalid_image_index",
                }
                return _finalize(observation, False, False)

            content_type = arguments.get("content_type")
            if extract_type == "content":
                if content_type is None:
                    observation = {
                        "obs": "Missing parameters: content_type is required when extract_type is content.",
                        "invalid_reason": "missing_parameters",
                    }
                    return _finalize(observation, False, False)
                if content_type not in {"text", "table", "equation"}:
                    observation = {
                        "obs": "Invalid content_type. It should be 'text', 'table', or 'equation'.",
                        "invalid_reason": "invalid_parameters",
                    }
                    return _finalize(observation, False, False)

            image_obj = env["images"][image_index - 1]
            image_base64 = _image_to_base64(image_obj)

            try:
                result, meta = await self._call_mineru_async(
                    image_base64=image_base64,
                    extract_type=str(extract_type),
                    content_type=str(content_type) if content_type else None,
                )
            except ValueError as exc:
                observation = {
                    "obs": f"multimodal_processor_tool failed: {exc}",
                    "invalid_reason": "unknown_model",
                }
                return _finalize(observation, False, False)
            except Exception as exc:
                observation = {
                    "obs": f"multimodal_processor_tool failed: {exc}",
                    "invalid_reason": "request_failed",
                }
                return _finalize(observation, False, False)

            if extract_type == "content":
                content = ""
                if isinstance(result, dict):
                    content = result.get("data", {}).get("content", "")
                observation = {
                    "obs": f"[Tool: {self.mineru_tool_name} (content, {content_type})] {content}",
                    "tool": self.mineru_tool_name,
                    "model": self.mineru_model_name,
                    "latency": meta.get("latency"),
                }
                return _finalize(observation, False, True)

            observation = {
                "obs": f"[Tool: {self.mineru_tool_name} ({extract_type})] "
                f"{json.dumps(result, ensure_ascii=False)}",
                "tool": self.mineru_tool_name,
                "model": self.mineru_model_name,
                "latency": meta.get("latency"),
            }
            return _finalize(observation, False, True)

        if tool_name == self.paddle_tool_name:
            image_index = arguments.get("image_index")
            if image_index is None:
                observation = {
                    "obs": "Missing parameters: image_index is required.",
                    "invalid_reason": "missing_parameters",
                }
                return _finalize(observation, False, False)

            try:
                image_index = int(image_index)
            except Exception:
                observation = {
                    "obs": "Invalid image_index format. It should be an integer.",
                    "invalid_reason": "invalid_image_index",
                }
                return _finalize(observation, False, False)

            if image_index <= 0 or image_index > len(env["images"]):
                observation = {
                    "obs": f"Invalid image_index. It should be between 1 and {len(env['images'])}.",
                    "invalid_reason": "invalid_image_index",
                }
                return _finalize(observation, False, False)

            image_obj = env["images"][image_index - 1]
            image_base64 = _image_to_base64(image_obj)
            temp_file = None

            try:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                temp_file.write(base64.b64decode(image_base64))
                temp_file.close()
                result, meta = await self._call_paddle_async(Path(temp_file.name))
            except ValueError as exc:
                observation = {
                    "obs": f"multimodal_processor_tool failed: {exc}",
                    "invalid_reason": "unknown_model",
                }
                return _finalize(observation, False, False)
            except Exception as exc:
                observation = {
                    "obs": f"multimodal_processor_tool failed: {exc}",
                    "invalid_reason": "request_failed",
                }
                return _finalize(observation, False, False)
            finally:
                if temp_file is not None:
                    try:
                        Path(temp_file.name).unlink()
                    except FileNotFoundError:
                        pass

            text = ""
            if isinstance(result, dict):
                text = result.get("result", "")
            observation = {
                "obs": f"[Tool: {self.paddle_tool_name}] {text}",
                "tool": self.paddle_tool_name,
                "model": self.paddle_model_name,
                "latency": meta.get("latency"),
            }
            return _finalize(observation, False, True)

        if tool_name == self.easyocr_tool_name:
            image_index = arguments.get("image_index")
            if image_index is None:
                observation = {
                    "obs": "Missing parameters: image_index is required.",
                    "invalid_reason": "missing_parameters",
                }
                return _finalize(observation, False, False)

            try:
                image_index = int(image_index)
            except Exception:
                observation = {
                    "obs": "Invalid image_index format. It should be an integer.",
                    "invalid_reason": "invalid_image_index",
                }
                return _finalize(observation, False, False)

            if image_index <= 0 or image_index > len(env["images"]):
                observation = {
                    "obs": f"Invalid image_index. It should be between 1 and {len(env['images'])}.",
                    "invalid_reason": "invalid_image_index",
                }
                return _finalize(observation, False, False)

            image_obj = env["images"][image_index - 1]
            image_base64 = _image_to_base64(image_obj)

            try:
                result, meta = await self._call_easyocr_async(image_base64=image_base64)
            except ValueError as exc:
                observation = {
                    "obs": f"multimodal_processor_tool failed: {exc}",
                    "invalid_reason": "unknown_model",
                }
                return _finalize(observation, False, False)
            except Exception as exc:
                observation = {
                    "obs": f"multimodal_processor_tool failed: {exc}",
                    "invalid_reason": "request_failed",
                }
                return _finalize(observation, False, False)

            observation = {
                "obs": f"[Tool: {self.easyocr_tool_name}] {json.dumps(result, ensure_ascii=False)}",
                "tool": self.easyocr_tool_name,
                "model": self.easyocr_model_name,
                "latency": meta.get("latency"),
            }
            return _finalize(observation, False, True)

        if tool_name == self.groundingdino_tool_name:
            prompt = arguments.get("prompt")
            image_index = arguments.get("image_index")
            if prompt is None or image_index is None:
                observation = {
                    "obs": "Missing parameters: prompt and image_index are required.",
                    "invalid_reason": "missing_parameters",
                }
                return _finalize(observation, False, False)

            try:
                image_index = int(image_index)
            except Exception:
                observation = {
                    "obs": "Invalid image_index format. It should be an integer.",
                    "invalid_reason": "invalid_image_index",
                }
                return _finalize(observation, False, False)

            if image_index <= 0 or image_index > len(env["images"]):
                observation = {
                    "obs": f"Invalid image_index. It should be between 1 and {len(env['images'])}.",
                    "invalid_reason": "invalid_image_index",
                }
                return _finalize(observation, False, False)

            box_threshold = arguments.get("box_threshold", 0.25)
            text_threshold = arguments.get("text_threshold", 0.25)
            try:
                box_threshold = float(box_threshold)
            except Exception:
                box_threshold = 0.25
            try:
                text_threshold = float(text_threshold)
            except Exception:
                text_threshold = 0.25

            image_obj = env["images"][image_index - 1]
            image_base64 = _image_to_base64(image_obj)

            try:
                result, meta = await self._call_groundingdino_async(
                    image_base64=image_base64,
                    prompt=str(prompt),
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
            except ValueError as exc:
                observation = {
                    "obs": f"multimodal_processor_tool failed: {exc}",
                    "invalid_reason": "unknown_model",
                }
                return _finalize(observation, False, False)
            except Exception as exc:
                observation = {
                    "obs": f"multimodal_processor_tool failed: {exc}",
                    "invalid_reason": "request_failed",
                }
                return _finalize(observation, False, False)

            observation = {
                "obs": f"[Tool: {self.groundingdino_tool_name}] {json.dumps(result, ensure_ascii=False)}",
                "tool": self.groundingdino_tool_name,
                "model": self.groundingdino_model_name,
                "latency": meta.get("latency"),
            }
            return _finalize(observation, False, True)

        if tool_name == self.image_edit_tool_name:
            prompt = arguments.get("prompt")
            image_index = arguments.get("image_index")
            if prompt is None or image_index is None:
                observation = {
                    "obs": "Missing parameters: prompt and image_index are required.",
                    "invalid_reason": "missing_parameters",
                }
                return _finalize(observation, False, False)

            try:
                image_index = int(image_index)
            except Exception:
                observation = {
                    "obs": "Invalid image_index format. It should be an integer.",
                    "invalid_reason": "invalid_image_index",
                }
                return _finalize(observation, False, False)

            if image_index <= 0 or image_index > len(env["images"]):
                observation = {
                    "obs": f"Invalid image_index. It should be between 1 and {len(env['images'])}.",
                    "invalid_reason": "invalid_image_index",
                }
                return _finalize(observation, False, False)

            image_obj = env["images"][image_index - 1]
            image_base64 = _image_to_base64(image_obj)

            try:
                result, meta = await self._call_image_edit_async([image_base64], str(prompt))
            except ValueError as exc:
                observation = {
                    "obs": f"multimodal_processor_tool failed: {exc}",
                    "invalid_reason": "unknown_model",
                }
                return _finalize(observation, False, False)
            except Exception as exc:
                observation = {
                    "obs": f"multimodal_processor_tool failed: {exc}",
                    "invalid_reason": "request_failed",
                }
                return _finalize(observation, False, False)

            image_base64_out = ""
            if isinstance(result, dict):
                image_base64_out = result.get("image_base64", "")
            image_url = _image_to_data_url(image_base64_out) if image_base64_out else None

            observation = {
                "obs": f"[Tool: {self.image_edit_tool_name}] You have generated the following image.",
                # tool is the user-facing tool name; model is the full backend image-edit model name.
                "tool": self.image_edit_tool_name,
                "model": self.image_edit_model_name,
                "latency": meta.get("latency"),
            }
            if image_url:
                observation["image"] = image_url
            return _finalize(observation, False, True)

        # Handle skill calls (non-forever tools)
        if tool_name not in self.valid_mcp_func_names:
            skill = self._resolve_skill(trajectory_id, tool_name)
            if not skill:
                observation = {
                    "obs": f"Unknown skill: {tool_name}",
                    "invalid_reason": "unknown_skill",
                }
                return _finalize(observation, False, False)

            observation = {
                "obs": f"[Skill: {skill.get('name', tool_name)}] SKILL.md:\n{skill.get('skill_md', '')}",
                "tool": skill.get("name", tool_name),
                "skill_name": skill.get("name", tool_name),
                "skill_path": skill.get("path", ""),
            }
            return _finalize(observation, False, True)

        prompt = arguments.get("prompt")
        image_index = arguments.get("image_index")
        if prompt is None or image_index is None:
            observation = {
                "obs": "Missing parameters: prompt and image_index are required.",
                "invalid_reason": "missing_parameters",
            }
            return _finalize(observation, False, False)

        try:
            image_index = int(image_index)
        except Exception:
            observation = {
                "obs": "Invalid image_index format. It should be an integer.",
                "invalid_reason": "invalid_image_index",
            }
            return _finalize(observation, False, False)

        if image_index <= 0 or image_index > len(env["images"]):
            observation = {
                "obs": f"Invalid image_index. It should be between 1 and {len(env['images'])}.",
                "invalid_reason": "invalid_image_index",
            }
            return _finalize(observation, False, False)

        image_obj = env["images"][image_index - 1]
        image_data_url = self._resolve_image_to_data_url(image_obj)
        if not image_data_url:
            observation = {
                "obs": "Failed to resolve image data for the given image_index.",
                "invalid_reason": "missing_image",
            }
            return _finalize(observation, False, False)

        try:
            content, meta = await self._call_model_async(tool_name, image_data_url, str(prompt))
            observation = {
                "obs": f"[Tool: {tool_name}] {content}",
                "tool": tool_name,
                "model": tool_name,
                "latency": meta.get("latency"),
                "usage": meta.get("usage", {}),
            }
            return _finalize(observation, False, True)
        except ValueError as exc:
            observation = {
                "obs": f"multimodal_processor_tool failed: {exc}",
                "invalid_reason": "unknown_model",
            }
            return _finalize(observation, False, False)
        except Exception as exc:
            observation = {
                "obs": f"multimodal_processor_tool failed: {exc}",
                "invalid_reason": "request_failed",
            }
            return _finalize(observation, False, False)

    def conduct_action(
        self, trajectory_id: str, action: str, extra_field: Dict[str, Any]
    ) -> Tuple[Any, bool, bool]:
        """Synchronous entrypoint that runs async logic in an internal event loop."""
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._conduct_action_async(trajectory_id, action, extra_field))
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    async def aget_observations(
        self, trajectory_ids: List[str], actions: List[str], extra_fields: List[Dict[str, Any]]
    ) -> Tuple[List[Any], List[bool], List[bool]]:
        """Async batch interface for concurrent request processing."""
        tasks = []
        for trajectory_id, action, extra_field in zip(trajectory_ids, actions, extra_fields):
            tasks.append(self._conduct_action_async(trajectory_id, action, extra_field))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        observations, dones, valids = [], [], []
        for result in results:
            if isinstance(result, Exception):
                observations.append(f"Processing error: {result}")
                dones.append(False)
                valids.append(False)
            else:
                obs, done, valid = result
                observations.append(obs)
                dones.append(done)
                valids.append(valid)
        return observations, dones, valids
