#!/usr/bin/env python
"""
Test script for python_code tool (synchronous version).
"""
import json
import requests
import fire
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _send_request(url: str, action: str) -> dict:
    """Send a request to the python_code tool server."""
    payload = {
        "trajectory_ids": ["python-code-test-001"],
        "actions": [action],
        "extra_fields": [{}],
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    result = resp.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _parse_observation(obs):
    """Parse observation, supporting both string and dict formats."""
    if isinstance(obs, dict):
        return obs
    if isinstance(obs, str):
        return {"raw_response": obs}
    raise AssertionError(f"Unexpected observation type: {type(obs)}")


def _assert_success(result: dict):
    """Assert that the result indicates success."""
    obs = _parse_observation(result["observations"][0])
    print(obs)
    assert isinstance(obs, dict), f"expect dict obs, got {type(obs)}"
    
    # Check for latency and cost fields
    for key in ["latency", "cost"]:
        assert key in obs, f"missing {key}"
        assert isinstance(obs[key], (int, float)), f"{key} not numeric"
    
    return True


def _assert_contains(result: dict, expected_substring: str):
    """Assert that the observation contains the expected substring."""
    obs = _parse_observation(result["observations"][0])
    
    if "obs" in obs:
        output = obs["obs"]
    elif isinstance(obs, str):
        output = obs
    else:
        output = str(obs)
    
    assert expected_substring in output, f"Expected '{expected_substring}' in output, got: {output}"
    return True


def _assert_invalid(result: dict, expected_reasons: list = None):
    """Assert that the result indicates an invalid request."""
    if expected_reasons is None:
        expected_reasons = ["parse_failed", "No valid tool found for action"]
    
    obs = result["observations"][0]
    if isinstance(obs, str):
        try:
            obs = json.loads(obs) if obs else {}
        except json.JSONDecodeError:
            obs = {}
    assert isinstance(obs, dict)
    actual_reason = obs.get("invalid_reason")
    assert actual_reason in expected_reasons, f"expect one of {expected_reasons}, got {actual_reason}"


def test_basic_execution(
    url: str = "http://localhost:6661/get_observation",
):
    """Test basic code execution: print("Hello")"""
    payload = {"name": "python_code", "arguments": {"code": "print('Hello')"}}
    action = f'<tool_call>{json.dumps(payload)}</tool_call>'
    print(action)
    result = _send_request(url, action)
    _assert_success(result)
    _assert_contains(result, "Hello")
    return result


def test_math_calculation(
    url: str = "http://localhost:6661/get_observation",
):
    """Test mathematical calculation."""
    code = "x = 2\ny = 4\nprint(x + y)"
    payload = {"name": "python_code", "arguments": {"code": code}}
    action = f'<tool_call>{json.dumps(payload)}</tool_call>'
    result = _send_request(url, action)
    _assert_success(result)
    _assert_contains(result, "6")
    return result


def test_with_imports(
    url: str = "http://localhost:6661/get_observation",
):
    """Test code with imports (math module is pre-imported)."""
    payload = {"name": "python_code", "arguments": {"code": "print(math.sqrt(16))"}}
    action = f'<tool_call>{json.dumps(payload)}</tool_call>'
    result = _send_request(url, action)
    _assert_success(result)
    _assert_contains(result, "4.0")
    return result


def test_syntax_error(
    url: str = "http://localhost:6661/get_observation",
):
    """Test syntax error handling."""
    payload = {"name": "python_code", "arguments": {"code": "prnit('Hello')"}}
    action = f'<tool_call>{json.dumps(payload)}</tool_call>'
    result = _send_request(url, action)
    _assert_success(result)  # Tool should still return success with error in obs
    return result


def test_bad_function(
    url: str = "http://localhost:6661/get_observation",
):
    """Test invalid function name."""
    payload = {"name": "wrong_function", "arguments": {"code": "print(1)"}}
    action = f'<tool_call>{json.dumps(payload)}</tool_call>'
    result = _send_request(url, action)
    _assert_invalid(result, ["No valid tool found for action"])
    return result


def test_missing_code(
    url: str = "http://localhost:6661/get_observation",
):
    """Test missing code argument."""
    payload = {"name": "python_code", "arguments": {}}
    action = f'<tool_call>{json.dumps(payload)}</tool_call>'
    result = _send_request(url, action)
    _assert_invalid(result, ["parse_failed", "No valid tool found for action"])
    return result

def test_markdown_code_blocks(
    url: str = "http://localhost:6661/get_observation",
):
    """Test code wrapped in markdown code blocks (```py or ```python)."""
    # Test case 1: ```py
    code_with_py = """```py
x = 10
y = 20
print(x + y)
```"""
    payload = {"name": "python_code", "arguments": {"code": code_with_py}}
    action = f'<tool_call>{json.dumps(payload)}</tool_call>'
    result = _send_request(url, action)
    _assert_success(result)
    _assert_contains(result, "30")
    logger.info("✓ Test passed: ```py code blocks")
    
    # Test case 2: ```python
    code_with_python = """```python
result = 5 * 5
print(result)
```"""
    payload = {"name": "python_code", "arguments": {"code": code_with_python}}
    action = f'<tool_call>{json.dumps(payload)}</tool_call>'
    result = _send_request(url, action)
    _assert_success(result)
    _assert_contains(result, "25")
    logger.info("✓ Test passed: ```python code blocks")
    
    # Test case 3: ``` (no language specified)
    code_with_generic = """```
message = "Hello World"
print(message)
```"""
    payload = {"name": "python_code", "arguments": {"code": code_with_generic}}
    action = f'<tool_call>{json.dumps(payload)}</tool_call>'
    result = _send_request(url, action)
    _assert_success(result)
    _assert_contains(result, "Hello World")
    logger.info("✓ Test passed: ``` generic code blocks")
    
    # Test case 4: Real-world example from your issue
    code_real_world = """```py
from math import trunc

# Extract data for Dark Orange curve
data_dark_orange = [
    (0, 32.0),
    (20, 31.7),
    (40, 31.2),
    (60, 30.8),
    (80, 30.5),
    (100, 30.0)
]

# Calculate area under the curve using trapezoidal rule
area_dark_orange = 0
for i in range(len(data_dark_orange) - 1):
    x1, y1 = data_dark_orange[i]
    x2, y2 = data_dark_orange[i + 1]
    area_dark_orange += ((x2 - x1) * (y1 + y2)) / 2

print(area_dark_orange)
```"""
    payload = {"name": "python_code", "arguments": {"code": code_real_world}}
    action = f'<tool_call>{json.dumps(payload)}</tool_call>'
    result = _send_request(url, action)
    _assert_success(result)
    # The trapezoidal area calculation should produce a number
    logger.info("✓ Test passed: Real-world trapezoidal calculation")
    
    logger.info("✅ All markdown code block tests passed!")
    return result


def main():
    fire.Fire(
        {
            "basic_execution": test_basic_execution,
            "math_calculation": test_math_calculation,
            "with_imports": test_with_imports,
            "syntax_error": test_syntax_error,
            "bad_function": test_bad_function,
            "missing_code": test_missing_code,
            "markdown_code_blocks": test_markdown_code_blocks,  # 新增
        }
    )


if __name__ == "__main__":
    main()
