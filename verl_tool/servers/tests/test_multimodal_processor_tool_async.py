#!/usr/bin/env python
import base64
import json
import os
from pathlib import Path

import fire
import requests

IMAGE_PATH = "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/add/general/HatefulMemes_2000/images/train/train_1.png"


def _encode_image_as_data_url(image_path: str) -> str:
    path_obj = Path(image_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    encoded = base64.b64encode(path_obj.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"


def _encode_image_as_base64(image_path: str) -> str:
    path_obj = Path(image_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return base64.b64encode(path_obj.read_bytes()).decode()


def _save_data_url_image(data_url: str, output_path: str) -> None:
    if not data_url.startswith("data:image") or "base64," not in data_url:
        raise ValueError("Invalid data URL image format.")
    b64 = data_url.split("base64,", 1)[1]
    img_data = base64.b64decode(b64)
    with open(output_path, "wb") as f:
        f.write(img_data)


def _send_request(url: str, action: str, image_data_url: str) -> dict:
    payload = {
        "trajectory_ids": ["mm-async-001"],
        "actions": [action],
        "extra_fields": [
            {
                "images": [image_data_url],
            }
        ],
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    result = resp.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _send_request_no_images(url: str, action: str) -> dict:
    payload = {
        "trajectory_ids": ["mm-async-002"],
        "actions": [action],
        "extra_fields": [{}],
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    result = resp.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _send_request_with_images(url: str, action: str, images: list) -> dict:
    payload = {
        "trajectory_ids": ["mm-async-b64-001"],
        "actions": [action],
        "extra_fields": [
            {
                "images": images,
            }
        ],
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    result = resp.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _assert_success(result: dict):
    assert result.get("valids", [False])[0] is True, f"expected valid=True, got {result.get('valids')}"
    obs = result["observations"][0]
    assert isinstance(obs, dict), f"expect dict obs, got {type(obs)}"


def _assert_invalid(result: dict, reason: str):
    assert result.get("valids", [True])[0] is False, f"expected valid=False, got {result.get('valids')}"
    obs = result["observations"][0]
    assert isinstance(obs, dict)
    assert obs.get("invalid_reason") == reason, f"expect {reason}, got {obs.get('invalid_reason')}"


def test_vlm_call(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
    prompt: str = "请描述图片",
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "Qwen3-VL-8B-Instruct", "arguments": {"prompt": prompt, "image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "Qwen3-VL-8B-Instruct"
    return result


def test_parameter_validation(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    action = '<tool_call>{"name": "Qwen3-VL-8B-Instruct", "arguments": {"prompt": "hi"}}</tool_call>'
    result = _send_request(url, action, img)
    _assert_invalid(result, "missing_parameters")
    return result


def test_wolfram_call(
    url: str = "http://localhost:5000/get_observation",
    query: str = "10 densest elemental metals",
):
    action = f'<tool_call>{json.dumps({"name": "WolframAlpha", "arguments": {"query": query}})}</tool_call>'
    payload = {
        "trajectory_ids": ["mm-async-wolfram-001"],
        "actions": [action],
        "extra_fields": [{}],
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "WolframAlpha"
    return result


def test_mineru_content(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "MinerU2.5", "arguments": {"extract_type": "content", "content_type": "text", "image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "MinerU2.5"
    return result


def test_paddle_ocr(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "PaddleOCR", "arguments": {"image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "PaddleOCR"
    return result


def test_sam3_text(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
    text_prompt: str = "person",
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "SAM3", "arguments": {"segment_type": "text", "text_prompt": text_prompt, "image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_success(result)
    obs = result["observations"][0]
    if "image" in obs:
        output_path = os.path.join(os.getcwd(), "sam3_text_async_output.png")
        _save_data_url_image(obs["image"], output_path)
        print(f"saved image to: {output_path}")
    return result


def test_image_edit(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
    prompt: str = "make the background blue",
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "Qwen-Image-Edit", "arguments": {"prompt": prompt, "image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_success(result)
    obs = result["observations"][0]
    if "image" in obs:
        output_path = os.path.join(os.getcwd(), "image_edit_async_output.png")
        _save_data_url_image(obs["image"], output_path)
        print(f"saved image to: {output_path}")
    return result


def test_python_code_call(
    url: str = "http://localhost:5000/get_observation",
):
    action = f'<tool_call>{json.dumps({"name": "python_code", "arguments": {"code": "1 + 1"}})}</tool_call>'
    result = _send_request_no_images(url, action)
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "python_code"
    assert "2" in obs.get("obs", ""), f"Expected auto-printed result, got: {obs}"
    return result


def test_python_code_parameter_validation(
    url: str = "http://localhost:5000/get_observation",
):
    action = f'<tool_call>{json.dumps({"name": "python_code", "arguments": {}})}</tool_call>'
    result = _send_request_no_images(url, action)
    _assert_invalid(result, "missing_parameters")
    return result


def test_opencv_operations(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    operations = [
        ("crop", {"operation": "crop", "image_index": 1, "x": 10, "y": 10, "w": 64, "h": 64}),
        ("resize", {"operation": "resize", "image_index": 1, "width": 128, "height": 96}),
        ("rotate", {"operation": "rotate", "image_index": 1, "angle": 30}),
        ("flip", {"operation": "flip", "image_index": 1, "flip_code": 1}),
        ("grayscale", {"operation": "grayscale", "image_index": 1}),
        ("blur", {"operation": "blur", "image_index": 1, "ksize": 4}),
        ("threshold", {"operation": "threshold", "image_index": 1, "thresh": 127, "type": "BINARY"}),
        ("canny", {"operation": "canny", "image_index": 1, "threshold1": 100, "threshold2": 200}),
    ]

    results = {}
    for name, args in operations:
        action = f'<tool_call>{json.dumps({"name": "OpenCV", "arguments": args})}</tool_call>'
        result = _send_request(url, action, img)
        _assert_success(result)
        obs = result["observations"][0]
        assert obs.get("tool") == "OpenCV"
        if "image" in obs:
            assert obs.get("image", "").startswith("data:image"), f"OpenCV {name} should return image"
            output_path = os.path.join(os.getcwd(), f"opencv_async_{name}_output.png")
            _save_data_url_image(obs["image"], output_path)
            print(f"saved image to: {output_path}")
        results[name] = result
    return results


def test_opencv_parameter_validation(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)

    action = f'<tool_call>{json.dumps({"name": "OpenCV", "arguments": {"operation": "crop", "x": 0, "y": 0, "w": 10, "h": 10}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_invalid(result, "missing_parameters")

    action = f'<tool_call>{json.dumps({"name": "OpenCV", "arguments": {"operation": "unknown", "image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_invalid(result, "invalid_parameters")

    action = f'<tool_call>{json.dumps({"name": "OpenCV", "arguments": {"operation": "threshold", "image_index": 1, "type": "BINARY"}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_invalid(result, "missing_parameters")
    return result


def test_opencv_base64_image(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    image_base64 = _encode_image_as_base64(image_path)
    action = f'<tool_call>{json.dumps({"name": "OpenCV", "arguments": {"operation": "grayscale", "image_index": 1}})}</tool_call>'
    result = _send_request_with_images(url, action, [image_base64])
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "OpenCV"
    if "image" in obs:
        assert obs.get("image", "").startswith("data:image")
        output_path = os.path.join(os.getcwd(), "opencv_async_base64_output.png")
        _save_data_url_image(obs["image"], output_path)
        print(f"saved image to: {output_path}")
    return result


def test_bad_function(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "wrong_function", "arguments": {"prompt": "hi", "image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_invalid(result, "No valid tool found for action")
    return result


def test_env_image_accumulation(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    trajectory_id = "mm-async-env-001"

    action1 = f'<tool_call>{json.dumps({"name": "SAM3", "arguments": {"segment_type": "text", "text_prompt": "person", "image_index": 1}})}</tool_call>'
    payload1 = {
        "trajectory_ids": [trajectory_id],
        "actions": [action1],
        "extra_fields": [{"images": [img]}],
    }
    resp1 = requests.post(url, json=payload1, timeout=180)
    resp1.raise_for_status()
    result1 = resp1.json()
    print(json.dumps(result1, indent=2, ensure_ascii=False))
    _assert_success(result1)

    action2 = f'<tool_call>{json.dumps({"name": "Qwen3-VL-8B-Instruct", "arguments": {"prompt": "请描述图片", "image_index": 1}})}</tool_call>'
    payload2 = {
        "trajectory_ids": [trajectory_id],
        "actions": [action2],
        "extra_fields": [{}],
    }
    resp2 = requests.post(url, json=payload2, timeout=180)
    resp2.raise_for_status()
    result2 = resp2.json()
    print(json.dumps(result2, indent=2, ensure_ascii=False))
    _assert_success(result2)

    action3 = f'<tool_call>{json.dumps({"name": "Qwen-Image-Edit", "arguments": {"prompt": "make the background blue", "image_index": 2}})}</tool_call>'
    payload3 = {
        "trajectory_ids": [trajectory_id],
        "actions": [action3],
        "extra_fields": [{}],
    }
    resp3 = requests.post(url, json=payload3, timeout=180)
    resp3.raise_for_status()
    result3 = resp3.json()
    print(json.dumps(result3, indent=2, ensure_ascii=False))
    _assert_success(result3)
    obs3 = result3["observations"][0]
    if "image" in obs3:
        output_path = os.path.join(os.getcwd(), "env_accumulation_async_edit_1.png")
        _save_data_url_image(obs3["image"], output_path)
        print(f"saved image to: {output_path}")

    action4 = f'<tool_call>{json.dumps({"name": "PaddleOCR", "arguments": {"image_index": 1}})}</tool_call>'
    payload4 = {
        "trajectory_ids": [trajectory_id],
        "actions": [action4],
        "extra_fields": [{}],
    }
    resp4 = requests.post(url, json=payload4, timeout=180)
    resp4.raise_for_status()
    result4 = resp4.json()
    print(json.dumps(result4, indent=2, ensure_ascii=False))
    _assert_success(result4)

    action5 = f'<tool_call>{json.dumps({"name": "WolframAlpha", "arguments": {"query": "10 densest elemental metals"}})}</tool_call>'
    payload5 = {
        "trajectory_ids": [trajectory_id],
        "actions": [action5],
        "extra_fields": [{}],
    }
    resp5 = requests.post(url, json=payload5, timeout=180)
    resp5.raise_for_status()
    result5 = resp5.json()
    print(json.dumps(result5, indent=2, ensure_ascii=False))
    _assert_success(result5)

    action6 = f'<tool_call>{json.dumps({"name": "python_code", "arguments": {"code": "1 + 1"}})}</tool_call>'
    payload6 = {
        "trajectory_ids": [trajectory_id],
        "actions": [action6],
        "extra_fields": [{}],
    }
    resp6 = requests.post(url, json=payload6, timeout=180)
    resp6.raise_for_status()
    result6 = resp6.json()
    print(json.dumps(result6, indent=2, ensure_ascii=False))
    _assert_success(result6)

    action7 = f'<tool_call>{json.dumps({"name": "OpenCV", "arguments": {"operation": "grayscale", "image_index": 3}})}</tool_call>'
    payload7 = {
        "trajectory_ids": [trajectory_id],
        "actions": [action7],
        "extra_fields": [{}],
    }
    resp7 = requests.post(url, json=payload7, timeout=180)
    resp7.raise_for_status()
    result7 = resp7.json()
    print(json.dumps(result7, indent=2, ensure_ascii=False))
    _assert_success(result7)
    obs7 = result7["observations"][0]
    if "image" in obs7:
        output_path = os.path.join(os.getcwd(), "env_accumulation_async_opencv_1.png")
        _save_data_url_image(obs7["image"], output_path)
        print(f"saved image to: {output_path}")

    action8 = f'<tool_call>{json.dumps({"name": "Qwen-Image-Edit", "arguments": {"prompt": "add a red border", "image_index": 4}})}</tool_call>'
    payload8 = {
        "trajectory_ids": [trajectory_id],
        "actions": [action8],
        "extra_fields": [{}],
    }
    resp8 = requests.post(url, json=payload8, timeout=180)
    resp8.raise_for_status()
    result8 = resp8.json()
    print(json.dumps(result8, indent=2, ensure_ascii=False))
    _assert_success(result8)
    obs8 = result8["observations"][0]
    if "image" in obs8:
        output_path = os.path.join(os.getcwd(), "env_accumulation_async_edit_2.png")
        _save_data_url_image(obs8["image"], output_path)
        print(f"saved image to: {output_path}")

    return {
        "sam3": result1,
        "vlm": result2,
        "edit1": result3,
        "paddle": result4,
        "wolfram": result5,
        "python": result6,
        "opencv": result7,
        "edit2": result8,
    }


def main():
    fire.Fire(
        {
            "vlm": test_vlm_call,
            "params": test_parameter_validation,
            "wolfram_call": test_wolfram_call,
            "mineru_content": test_mineru_content,
            "paddle_call": test_paddle_ocr,
            "sam3_text": test_sam3_text,
            "image_edit": test_image_edit,
            "python_code_call": test_python_code_call,
            "python_code_params": test_python_code_parameter_validation,
            "opencv_ops": test_opencv_operations,
            "opencv_params": test_opencv_parameter_validation,
            "opencv_base64": test_opencv_base64_image,
            "env_image_accumulation": test_env_image_accumulation,
            "bad_function": test_bad_function,
        }
    )


if __name__ == "__main__":
    main()
