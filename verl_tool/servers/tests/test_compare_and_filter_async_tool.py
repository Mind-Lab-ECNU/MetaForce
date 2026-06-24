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
    tools_dir = Path(__file__).resolve().parents[2] / "tools"
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
            "input_per_million": model_pricing[chosen[0]]["input_tokens_per_million"],
            "output_per_million": model_pricing[chosen[0]]["output_tokens_per_million"],
        },
        chosen[1]: {
            "input_per_million": model_pricing[chosen[1]]["input_tokens_per_million"],
            "output_per_million": model_pricing[chosen[1]]["output_tokens_per_million"],
        },
    }
    model_mapping = {
        "compare_and_filter": {
            "compare_and_filter-1": chosen[0],
            "compare_and_filter-2": chosen[1],
        }
    }
    print(f"[test_async] chosen services: {chosen}, tool_pricing keys: {list(tool_pricing.keys())}")
    return model_mapping, tool_pricing


def _send_request(url: str, action: str, image_data_url: str, question: str) -> dict:
    model_mapping, tool_pricing = _load_models_and_pricing()
    payload = {
        "trajectory_ids": ["compare-async-001"],
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


def _assert_success(result: dict):
    obs = result["observations"][0]
    assert isinstance(obs, dict), f"expect dict obs, got {type(obs)}"
    for key in ["latency", "cost"]:
        assert key in obs, f"missing {key}"
        assert isinstance(obs[key], (int, float)), f"{key} not numeric"


def _assert_invalid(result: dict, expected_reasons: list = None):
    if expected_reasons is None:
        expected_reasons = ["parse_failed", "unknown_model"]
    obs = result["observations"][0]
    if isinstance(obs, str):
        try:
            obs = json.loads(obs) if obs else {}
        except json.JSONDecodeError:
            obs = {}
    assert isinstance(obs, dict)
    actual_reason = obs.get("invalid_reason")
    assert actual_reason in expected_reasons, f"expect one of {expected_reasons}, got {actual_reason}"


def test_compare_and_filter_model1(
    url: str = "http://localhost:5602/get_observation",
    image_path: str = IMAGE_PATH,
    question: str = "Compare values to find which ones are greater than the median.",
):
    img = _encode_image_as_data_url(image_path)
    action_payload = {
        "name": "compare_and_filter_async",
        "arguments": {
            "comparison_type": "value_to_statistic",
            "condition": "greater",
            "model": "compare_and_filter-1"
        }
    }
    action = f'<tool_call>{json.dumps(action_payload)}</tool_call>'
    result = _send_request(url, action, img, question)
    _assert_success(result)
    return result


def test_compare_and_filter_model2(
    url: str = "http://localhost:5602/get_observation",
    image_path: str = IMAGE_PATH,
    question: str = "Filter data to find values above the median.",
):
    img = _encode_image_as_data_url(image_path)
    action_payload = {
        "name": "compare_and_filter_async",
        "arguments": {
            "comparison_type": "filter_by_condition",
            "condition": "above_median",
            "model": "compare_and_filter-2"
        }
    }
    action = f'<tool_call>{json.dumps(action_payload)}</tool_call>'
    result = _send_request(url, action, img, question)
    _assert_success(result)
    return result


def test_bad_name(
    url: str = "http://localhost:5602/get_observation",
    image_path: str = IMAGE_PATH,
    question: str = "Should fail on model name.",
):
    img = _encode_image_as_data_url(image_path)
    action_payload = {
        "name": "compare_and_filter_async",
        "arguments": {
            "comparison_type": "value_to_statistic",
            "condition": "greater",
            "model": "non-existent-model"
        }
    }
    action = f'<tool_call>{json.dumps(action_payload)}</tool_call>'
    result = _send_request(url, action, img, question)
    _assert_invalid(result, ["unknown_model"])
    return result


def test_bad_function(
    url: str = "http://localhost:5602/get_observation",
    image_path: str = IMAGE_PATH,
    question: str = "Should fail on function name.",
):
    img = _encode_image_as_data_url(image_path)
    action_payload = {
        "name": "wrong_function",
        "arguments": {
            "comparison_type": "value_to_statistic",
            "condition": "greater",
            "model": "compare_and_filter-1"
        }
    }
    action = f'<tool_call>{json.dumps(action_payload)}</tool_call>'
    result = _send_request(url, action, img, question)
    _assert_invalid(result, ["parse_failed"])
    return result


def main():
    fire.Fire(
        {
            "model1": test_compare_and_filter_model1,
            "model2": test_compare_and_filter_model2,
            "bad_name": test_bad_name,
            "bad_function": test_bad_function,
        }
    )


if __name__ == "__main__":
    main()

