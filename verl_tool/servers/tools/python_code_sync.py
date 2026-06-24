"""
Synchronous version of python_code tool for execution in Firejail sandbox.
This version can be used with Ray for parallel processing.

tool_type = "python_code" to maintain compatibility with the original tool.
"""
import time
import subprocess
import os
import uuid
import shutil
import resource
import json
import regex as re
from typing import Tuple, Dict, Any, Optional, Union, List

from .base import BaseTool, register_tool

# Timeout for code execution in seconds
TIMEOUT = 10
PRE_IMPORT_LIBS = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n\n"

firejail_command_exists = shutil.which("firejail") is not None


def check_forbidden_imports(code: str) -> bool:
    """Check if code contains forbidden imports."""
    forbidden_modules = [
        'subprocess', 'multiprocessing', 'threading',
        'socket', 'psutil', 'resource', 'ctypes'
    ]
    for module in forbidden_modules:
        if f"import {module}" in code or f"from {module}" in code:
            return True

    dangerous_patterns = [
        "os.system", "os.popen", "os.spawn", "os.fork",
        "os.exec", "sys.exit", "os._exit", "os.kill"
    ]
    for pattern in dangerous_patterns:
        if pattern in code:
            return True
    return False


def wrap_code_blocks(code: Union[str, List[str]]) -> str:
    """
    Wrap code blocks with error handling and auto-print for last expression.
    
    ✅ This version captures the last expression value even without explicit print().
    """
    wrapped_code = ""
    if isinstance(code, str):
        code = [code]

    # Setup: Import necessary modules and prepare expression capture
    wrapped_code += "import sys, os, io, ast\n\n"
    
    # Helper function for salvageable execution (existing logic)
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

    # ✅ NEW: Helper function to capture last expression
    wrapped_code += """
def execute_with_last_expr_capture(code_string):
    \"\"\"
    Execute code and capture the last expression value.
    Returns: (stdout, last_expr_value)
    \"\"\"
    import sys
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    # Capture stdout
    stdout_capture = io.StringIO()
    
    # Split code into lines
    lines = code_string.strip().split('\\n')
    if not lines:
        return "", None
    
    # Separate last line from the rest
    if len(lines) == 1:
        setup_code = ""
        last_line = lines[0]
    else:
        setup_code = '\\n'.join(lines[:-1])
        last_line = lines[-1].strip()
    
    # Execute setup code
    namespace = {}
    if setup_code:
        with redirect_stdout(stdout_capture):
            try:
                exec(setup_code, globals(), namespace)
            except Exception as e:
                print(f"Error in setup: {e}", file=sys.stderr)
                return stdout_capture.getvalue(), None
    
    # Update globals with namespace
    globals().update(namespace)
    
    # Try to evaluate last line as expression
    last_value = None
    try:
        # Try to compile as expression first
        compile(last_line, '<string>', 'eval')
        with redirect_stdout(stdout_capture):
            last_value = eval(last_line, globals(), namespace)
    except SyntaxError:
        # Not an expression, execute as statement
        with redirect_stdout(stdout_capture):
            try:
                exec(last_line, globals(), namespace)
            except Exception as e:
                print(f"Error in last line: {e}", file=sys.stderr)
    except Exception as e:
        # Evaluation error
        with redirect_stdout(stdout_capture):
            print(f"Error evaluating expression: {e}", file=sys.stderr)
    
    return stdout_capture.getvalue(), last_value

"""

    # Process each code block
    for i, block in enumerate(code):
        is_last_block = i == len(code) - 1
        
        if not is_last_block:
            # Previous blocks: suppress output but preserve variables
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
            # ✅ Last block: capture and auto-print last expression
            wrapped_block = f"""
# Code block {i+1} (current - with auto-print)
_stdout_output, _last_expr_value = execute_with_last_expr_capture('''{block}''')

# Print captured stdout
if _stdout_output:
    print(_stdout_output, end='')

# Print last expression value if exists and not already printed
if _last_expr_value is not None:
    # Check if it was already printed
    _value_str = str(_last_expr_value)
    if _value_str not in _stdout_output:
        print(_last_expr_value)
"""
        
        wrapped_code += wrapped_block

    return wrapped_code


def clean_traceback(text, base_path):
    pattern = re.compile(re.escape('File "' + base_path + "/"))
    return pattern.sub('File "', text)


def set_limits():
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, resource.RLIM_INFINITY))
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT, resource.RLIM_INFINITY))
    resource.setrlimit(resource.RLIMIT_FSIZE, (500*1024*1024, 500*1024*1024))


def execute_python(code: Union[str, List[str]], timeout: int = TIMEOUT, stdin: Optional[str] = None,
                   python_path: str = None, pre_import_lib: bool = False, use_firejail: bool = False) -> Tuple[str, str, bool]:
    """Execute Python code in a Firejail sandbox with a timeout."""
    if check_forbidden_imports(code if isinstance(code, str) else "\n".join(code)):
        return "", "Execution blocked: Code contains potentially dangerous operations or imports.", True

    original_env = os.environ.copy()
    cwd = os.path.join(os.getcwd(), "tmp/firejail", str(uuid.uuid4().hex))
    if not os.path.exists(cwd):
        os.makedirs(cwd, exist_ok=True)

    file_name = "main.py"
    file_path = os.path.join(cwd, file_name)
    code_wrapped = wrap_code_blocks(code)
    if pre_import_lib:
        code_wrapped = PRE_IMPORT_LIBS + code_wrapped
    with open(file_path, "w") as f:
        f.write(code_wrapped)

    if not python_path:
        python_path = "python3"
    else:
        assert os.path.exists(python_path), f"Python path {python_path} does not exist."

    if use_firejail and firejail_command_exists:
        env = {}
        essential_vars = [
            "PATH", "HOME", "USER", "SHELL",
            "LANG", "LC_ALL", "LC_CTYPE", "TERM",
            "PYTHONIOENCODING", "PYTHONUNBUFFERED", "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE",
            "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "TMPDIR", "TEMP", "TMP",
            "DISPLAY", "XAUTHORITY"
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
            "--rlimit-as=1096m"
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
            preexec_fn=set_limits,
            timeout=timeout,
            cwd=subprocess_cwd,
        )
        stdout = clean_traceback(result.stdout, cwd)
        stderr = clean_traceback(result.stderr, cwd)
        stderr = stderr if stderr else ""
        if stderr:
            has_error = True
    except subprocess.TimeoutExpired as e:
        has_error = True
        stdout = e.stdout if e.stdout else ""
        stderr = e.stderr if e.stderr else ""
        stdout = stdout.decode('utf-8') if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode('utf-8') if isinstance(stderr, bytes) else stderr
        stderr += f"Execution timed out after {timeout} seconds.\n"

    try:
        if os.path.exists(cwd):
            shutil.rmtree(cwd)
    except Exception:
        pass

    return stdout, stderr, has_error


MAX_OBS_LENGTH = 100000


@register_tool
class PythonCodeTool(BaseTool):
    """
    Synchronous Python code execution tool.
    Can be used with Ray for parallel processing.
    
    Expects action format:
        <tool_call>{"name": "python_code", "arguments": {"code": "print('hello')"}}</tool_call>
    
    ✅ Auto-captures last expression value even without explicit print()
    """
    tool_type = "python_code_sync"
    timeout = TIMEOUT
    stop_tokens = ["```output", "<output>", "<tool_call>"]
    enable_history_code_execution = False
    done_without_error = False
    python_path = None
    pre_import_lib = True
    use_firejail = True

    def get_usage_inst(self):
        return "python_code: You are able to write and execute Python code securely inside a Firejail sandbox. Format: <tool_call>{\"name\": \"python_code\", \"arguments\": {\"code\": \"...\"}}</tool_call>"

    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse action to extract Python code.
        Supports <tool_call> JSON format.
        Returns (code_string, is_valid).
        """
        print(f"[PythonCodeTool SYNC] parse_action called, action preview: {action[:200]}")
        
        try:
            # Extract JSON from <tool_call> tags
            if "<tool_call>" in action and "</tool_call>" in action:
                payload_str = action.split("<tool_call>")[1].split("</tool_call>")[0].strip()
                print(f"[PythonCodeTool SYNC] Extracted payload: {payload_str[:150]}")
                
                payload = json.loads(payload_str)
                print(f"[PythonCodeTool SYNC] Parsed JSON: {payload}")
                
                if payload.get("name") == "python_code":
                    arguments = payload.get("arguments", {})
                    if isinstance(arguments, dict) and "code" in arguments:
                        code = arguments["code"]
                        
                        # ✅ Clean markdown code blocks and escape sequences
                        # 1. Handle escaped newlines
                        code = code.replace('\\n', '\n')
                        code = code.replace('\\t', '\t')
                        code = code.replace('\\r', '\r')
                        
                        # 2. Remove markdown code block markers
                        code = re.sub(r'^\s*```\w*\s*\n?', '', code)
                        code = re.sub(r'\n?\s*```\s*$', '', code)
                        
                        # 3. Clean whitespace
                        code = code.strip()
                        
                        print(f"[PythonCodeTool SYNC] ✓ Successfully extracted code ({len(code)} chars)")
                        print(f"[PythonCodeTool SYNC] Code preview: {code[:100]}")
                        return code, True
                else:
                    print(f"[PythonCodeTool SYNC] Tool name mismatch: {payload.get('name')} != python_code")
            else:
                print(f"[PythonCodeTool SYNC] No <tool_call> tags found in action")
                
        except json.JSONDecodeError as e:
            print(f"[PythonCodeTool SYNC] JSON decode error: {e}")
        except Exception as e:
            print(f"[PythonCodeTool SYNC] Unexpected error: {e}")
        
        print(f"[PythonCodeTool SYNC] ✗ Returning invalid")
        return "", False


    def conduct_action(self, trajectory_id, action, extra_field):
        """Execute the parsed action in a Firejail sandbox."""
        print(f"[PythonCodeTool SYNC] conduct_action called for trajectory: {trajectory_id}")
        
        parsed_code, is_valid = self.parse_action(action)
        
        if not is_valid:
            print(f"[PythonCodeTool SYNC] Invalid action, returning error")
            observation = {
                "obs": "Invalid action format for python_code. Expected: <tool_call>{\"name\": \"python_code\", \"arguments\": {\"code\": \"...\"}}</tool_call>",
                "invalid_reason": "parse_failed",
            }
            return observation, False, False

        stdin = extra_field.get("stdin", "") if extra_field else None

        print(f"[PythonCodeTool SYNC] Executing code ({len(parsed_code)} chars)")
        start = time.time()
        stdout, stderr, has_error = execute_python(
            parsed_code, self.timeout, stdin, self.python_path,
            self.pre_import_lib, self.use_firejail
        )
        latency = time.time() - start

        execution_result = stdout + "\n" + stderr
        execution_result = execution_result.strip(' \n')
        obs_content = execution_result[:MAX_OBS_LENGTH]

        observation = {
            "obs": obs_content,
            "latency": latency,
            "cost": 0.0,
            "usage": {},
            "tool": "python_code",
            "has_error": has_error,
        }

        print(f"[PythonCodeTool SYNC] Execution complete, has_error: {has_error}")
        print(f"[PythonCodeTool SYNC] obs preview: {obs_content[:200]}")

        if self.done_without_error:
            done = not has_error
        else:
            done = False

        return observation, done, True
