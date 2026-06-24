#!/usr/bin/env python
import base64
import json
from pathlib import Path

import fire
import requests

IMAGE_PATH = '/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_duo_final/ChartQA_2000/images/train/train_0.png'

def _encode_image_as_data_url(image_path: str) -> str:
    path_obj = Path(image_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    encoded = base64.b64encode(path_obj.read_bytes()).decode()
    return f"data:image/jpeg;base64,{encoded}"


def _load_models_and_pricing():
    url_cfg = json.loads(
        Path("/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/verl_tool/servers/tools/url.json").read_text()
    )
    services = list(url_cfg.keys())
    assert len(services) >= 2, "Need at least two services in url.json"
    import random
    random.shuffle(services)
    chosen = services[:2]

    pricing_cfg = json.loads(
        Path("/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/model_tool_pricing.json").read_text()
    )
    model_pricing = pricing_cfg["model_tool_pricing"]
    tool_pricing = {
        chosen[0]: {
            "input_tokens_per_million": model_pricing[chosen[0]]["input_tokens_per_million"],
            "output_tokens_per_million": model_pricing[chosen[0]]["output_tokens_per_million"],
        },
        chosen[1]: {
            "input_tokens_per_million": model_pricing[chosen[1]]["input_tokens_per_million"],
            "output_tokens_per_million": model_pricing[chosen[1]]["output_tokens_per_million"],
        },
    }
    model_mapping = {
        "calculate_statistics": {
            "calculate_statistics-1": chosen[0],
            "calculate_statistics-2": chosen[1],
        }
    }
    print(f"[test_async] chosen services: {chosen}, tool_pricing keys: {list(tool_pricing.keys())}")
    return model_mapping, tool_pricing


def _send_request(url: str, action: str, image_data_url: str, question: str) -> dict:
    model_mapping, tool_pricing = _load_models_and_pricing()
    payload = {
        "trajectory_ids": ["stats-async-001"],
        "actions": [action],
        "extra_fields": [
            {
                "images": [image_data_url],
                "question": question,
                "model_mapping": model_mapping,
                "tool_pricing": tool_pricing,
            }
        ],
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _parse_observation(obs):
    """解析 observation，支持字符串或字典格式"""
    if isinstance(obs, dict):
        return obs
    if isinstance(obs, str):
        if obs == "":
            raise AssertionError(
                "Server returned empty observation!\n"
                "Possible causes:\n"
                "  1. Tool handler did not return a result\n"
                "  2. Model service is not running\n"
                "  3. Check server logs for errors"
            )
        try:
            return json.loads(obs)
        except json.JSONDecodeError:
            return {"raw_response": obs}
    raise AssertionError(f"Unexpected observation type: {type(obs)}")


def _assert_success(result: dict):
    obs = _parse_observation(result["observations"][0])
    
    assert isinstance(obs, dict), f"expect dict obs, got {type(obs)}"
    
    if "error" in obs:
        raise AssertionError(f"Server returned error: {obs['error']}")
    
    if "raw_response" in obs:
        print(f"[WARNING] Server returned raw text: {obs['raw_response'][:200]}...")
        return
    
    for key in ["latency", "cost"]:
        assert key in obs, f"missing {key}"
        assert isinstance(obs[key], (int, float)), f"{key} not numeric"


def _assert_invalid(result: dict, reason: str):
    obs = result["observations"][0]
    if isinstance(obs, str):
        try:
            obs = json.loads(obs) if obs else {}
        except json.JSONDecodeError:
            obs = {}
    assert isinstance(obs, dict)
    assert obs.get("invalid_reason") == reason, f"expect {reason}, got {obs.get('invalid_reason')}"


def test_calculate_statistics_model1(
    url: str = "http://localhost:5601/get_observation",
    image_path: str = IMAGE_PATH,
    question: str = "Calculate the maximum value from the chart.",
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "calculate_statistics", "arguments": {"statistics_type": "max", "model": "calculate_statistics-1"}})}</tool_call>'
    result = _send_request(url, action, img, question)
    _assert_success(result)
    return result


def test_calculate_statistics_model2(
    url: str = "http://localhost:5601/get_observation",
    image_path: str = IMAGE_PATH,
    question: str = "Calculate the standard deviation from the chart.",
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "calculate_statistics", "arguments": {"statistics_type": "std", "model": "calculate_statistics-2"}})}</tool_call>'
    result = _send_request(url, action, img, question)
    _assert_success(result)
    return result


def test_bad_name(
    url: str = "http://localhost:5601/get_observation",
    image_path: str = IMAGE_PATH,
    question: str = "Should fail on model name.",
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "calculate_statistics", "arguments": {"statistics_type": "max", "model": "non-existent-model"}})}</tool_call>'
    result = _send_request(url, action, img, question)
    _assert_invalid(result, "unknown_model")
    return result


def test_bad_function(
    url: str = "http://localhost:5601/get_observation",
    image_path: str = IMAGE_PATH,
    question: str = "Should fail on function name.",
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "wrong_function", "arguments": {"statistics_type": "max", "model": "calculate_statistics-1"}})}</tool_call>'
    result = _send_request(url, action, img, question)
    _assert_invalid(result, "parse_failed")
    return result


def main():
    fire.Fire(
        {
            "model1": test_calculate_statistics_model1,
            "model2": test_calculate_statistics_model2,
            "bad_name": test_bad_name,
            "bad_function": test_bad_function,
        }
    )


if __name__ == "__main__":
    main()
