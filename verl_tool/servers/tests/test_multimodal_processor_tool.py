#!/usr/bin/env python
import base64
import json
import sys
import os

import fire

# 方便直接运行测试脚本时导入工具
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.multimodal_processor_tool import MultimodalProcessorTool

# 测试用图片路径，请替换为真实图片路径
IMAGE_PATH = "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/add/general/HatefulMemes_2000/images/train/train_1.png"


def _build_action(model_name: str, prompt: str, image_index: int) -> str:
    """构建 <tool_call> 格式的 action 字符串。"""
    payload = {
        "name": model_name,
        "arguments": {
            "prompt": prompt,
            "image_index": image_index,
        },
    }
    return f"<tool_call>{json.dumps(payload)}</tool_call>"


def _build_wolfram_action(query: str) -> str:
    payload = {
        "name": "WolframAlpha",
        "arguments": {
            "query": query,
        },
    }
    return f"<tool_call>{json.dumps(payload)}</tool_call>"


def _build_generic_action(name: str, arguments: dict) -> str:
    payload = {
        "name": name,
        "arguments": arguments,
    }
    return f"<tool_call>{json.dumps(payload)}</tool_call>"


def _save_data_url_image(data_url: str, output_path: str) -> None:
    """将 data URL 图片保存到本地文件。"""
    if not data_url.startswith("data:image") or "base64," not in data_url:
        raise ValueError("Invalid data URL image format.")
    b64 = data_url.split("base64,", 1)[1]
    img_data = base64.b64decode(b64)
    with open(output_path, "wb") as f:
        f.write(img_data)


def _encode_image_as_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def test_tool_initialization():
    """测试工具初始化与子工具注册。"""
    print(">>> [START] test_tool_initialization")
    print("  初始化 MultimodalProcessorTool...")
    tool = MultimodalProcessorTool()
    print(f"  tool_type: {tool.tool_type}")
    assert tool.tool_type == "multimodal_processor_tool"
    print("  验证子工具注册...")
    assert "Qwen3-VL-8B-Instruct" in tool.valid_mcp_func_names
    assert "Qwen3-VL-32B-Instruct" in tool.valid_mcp_func_names
    assert "InternVL3.5-38B-Instruct" in tool.valid_mcp_func_names
    assert "InternVL3.5-14B-Instruct" in tool.valid_mcp_func_names
    assert "WolframAlpha" in tool.valid_mcp_func_names
    assert "SAM3" in tool.valid_mcp_func_names
    assert "MinerU2.5" in tool.valid_mcp_func_names
    assert "PaddleOCR" in tool.valid_mcp_func_names
    assert "Qwen-Image-Edit" in tool.valid_mcp_func_names
    assert "OpenCV" in tool.valid_mcp_func_names
    assert "python_code" in tool.valid_mcp_func_names
    print(f"  已注册子工具数量: {len(tool.valid_mcp_func_names)}")
    print(">>> [PASSED] test_tool_initialization")


def test_parameter_validation():
    """测试参数校验（缺失 prompt / image_index）。"""
    print(">>> [START] test_parameter_validation")
    tool = MultimodalProcessorTool()
    print("  构建缺失 image_index 的 action...")
    action = "<tool_call>{\"name\": \"Qwen3-VL-8B-Instruct\", \"arguments\": {\"prompt\": \"hi\"}}</tool_call>"
    obs, done, valid = tool.conduct_action("t-001", action, {"images": [IMAGE_PATH]})
    print(f"  obs={obs}, valid={valid}, invalid_reason={obs.get('invalid_reason')}")
    assert valid is False
    assert obs.get("invalid_reason") == "missing_parameters"
    print(">>> [PASSED] test_parameter_validation")


def test_wolfram_parameter_validation():
    """测试 WolframAlpha 参数校验（缺失 query）。"""
    print(">>> [START] test_wolfram_parameter_validation")
    tool = MultimodalProcessorTool()
    print("  构建缺失 query 的 action...")
    action = "<tool_call>{\"name\": \"WolframAlpha\", \"arguments\": {}}</tool_call>"
    obs, done, valid = tool.conduct_action("t-004", action, {})
    print(f"  obs={obs}, valid={valid}, invalid_reason={obs.get('invalid_reason')}")
    assert valid is False
    assert obs.get("invalid_reason") == "missing_parameters"
    print(">>> [PASSED] test_wolfram_parameter_validation")


def test_model_call():
    """测试 VLM 模型调用流程（真实请求）。"""
    print(">>> [START] test_model_call")
    tool = MultimodalProcessorTool()

    model_names = [
        "Qwen3-VL-8B-Instruct",
        "Qwen3-VL-32B-Instruct",
        "InternVL3.5-38B-Instruct",
        "InternVL3.5-14B-Instruct",
    ]
    for idx, model_name in enumerate(model_names, start=2):
        print(f"  调用 {model_name}...")
        action = _build_action(model_name, "请描述图片", 1)
        obs, done, valid = tool.conduct_action(f"t-00{idx}", action, {"images": [IMAGE_PATH]})
        print(f"    obs={obs}, valid={valid}")
        assert valid is True, f"{model_name} call should be valid, got: {obs}"
        assert obs.get("tool") == model_name
        assert model_name in obs.get("obs", "")
        print(f"  - {model_name} passed")
    print(">>> [PASSED] test_model_call")


def test_wolfram_call():
    """测试 WolframAlpha 调用流程（真实请求）。"""
    print(">>> [START] test_wolfram_call")
    tool = MultimodalProcessorTool()

    print("  调用 WolframAlpha...")
    action = _build_wolfram_action("10 densest elemental metals")
    obs, done, valid = tool.conduct_action("t-005", action, {})
    print(f"  obs={obs}, valid={valid}")
    assert valid is True, f"WolframAlpha call should be valid, got: {obs}"
    assert obs.get("tool") == "WolframAlpha"
    assert "[Tool: WolframAlpha]" in obs.get("obs", "")
    print(">>> [PASSED] test_wolfram_call")


def test_sam3_text_call():
    """测试 SAM3 文本分割流程（真实请求）。"""
    print(">>> [START] test_sam3_text_call")
    tool = MultimodalProcessorTool()

    print("  调用 SAM3 text segmentation...")
    action = _build_generic_action(
        "SAM3",
        {"segment_type": "text", "text_prompt": "person", "image_index": 1},
    )
    obs, done, valid = tool.conduct_action("t-006", action, {"images": [IMAGE_PATH]})
    print(f"  obs={obs}, valid={valid}, has_image: {'image' in obs}")
    assert valid is True, f"SAM3 text call should be valid, got: {obs}"
    assert "[Tool: SAM3 (text)]" in obs.get("obs", "")
    # SAM3 成功时应返回可视化图片
    if "image" in obs:
        assert obs.get("image", "").startswith("data:image")
        output_path = os.path.join(os.getcwd(), "sam3_text_output.png")
        _save_data_url_image(obs["image"], output_path)
        print(f"  saved image to: {output_path}")
    print(">>> [PASSED] test_sam3_text_call")


def test_mineru_content_call():
    """测试 MinerU2.5 内容提取（真实请求）。"""
    print(">>> [START] test_mineru_content_call")
    tool = MultimodalProcessorTool()

    print("  调用 MinerU2.5 content extraction...")
    action = _build_generic_action(
        "MinerU2.5",
        {"extract_type": "content", "content_type": "text", "image_index": 1},
    )
    obs, done, valid = tool.conduct_action("t-007", action, {"images": [IMAGE_PATH]})
    print(f"  obs={obs}, valid={valid}")
    assert valid is True, f"MinerU2.5 call should be valid, got: {obs}"
    assert "[Tool: MinerU2.5 (content, text)]" in obs.get("obs", "")
    print(">>> [PASSED] test_mineru_content_call")


def test_paddle_call():
    """测试 PaddleOCR 调用流程（真实请求）。"""
    print(">>> [START] test_paddle_call")
    tool = MultimodalProcessorTool()

    print("  调用 PaddleOCR...")
    action = _build_generic_action("PaddleOCR", {"image_index": 1})
    obs, done, valid = tool.conduct_action("t-008", action, {"images": [IMAGE_PATH]})
    print(f"  obs={obs}, valid={valid}")
    assert valid is True, f"PaddleOCR call should be valid, got: {obs}"
    assert "[Tool: PaddleOCR]" in obs.get("obs", "")
    print(">>> [PASSED] test_paddle_call")


def test_image_edit_call():
    """测试 Qwen-Image-Edit 调用流程（真实请求）。"""
    print(">>> [START] test_image_edit_call")
    tool = MultimodalProcessorTool()

    print("  调用 Qwen-Image-Edit...")
    action = _build_generic_action(
        "Qwen-Image-Edit",
        {"prompt": "make the background blue", "image_index": 1},
    )
    obs, done, valid = tool.conduct_action("t-009", action, {"images": [IMAGE_PATH]})
    print(f"  obs={obs}, valid={valid}, has_image: {'image' in obs}")
    assert valid is True, f"Qwen-Image-Edit call should be valid, got: {obs}"
    assert "[Tool: Qwen-Image-Edit]" in obs.get("obs", "")
    # Image Edit 成功时应返回编辑后的图片
    if "image" in obs:
        assert obs.get("image", "").startswith("data:image")
        output_path = os.path.join(os.getcwd(), "image_edit_output.png")
        _save_data_url_image(obs["image"], output_path)
        print(f"  saved image to: {output_path}")
    print(">>> [PASSED] test_image_edit_call")


def test_python_code_call():
    """测试 python_code 调用流程（本地执行）。"""
    print(">>> [START] test_python_code_call")
    tool = MultimodalProcessorTool()

    print("  调用 python_code...")
    action = _build_generic_action("python_code", {"code": "1 + 1"})
    obs, done, valid = tool.conduct_action("t-010", action, {})
    print(f"  obs={obs}, valid={valid}")
    assert valid is True, f"python_code call should be valid, got: {obs}"
    assert obs.get("tool") == "python_code"
    assert "2" in obs.get("obs", ""), f"Expected auto-printed result, got: {obs}"
    print(">>> [PASSED] test_python_code_call")


def test_python_code_parameter_validation():
    """测试 python_code 参数校验（缺失 code）。"""
    print(">>> [START] test_python_code_parameter_validation")
    tool = MultimodalProcessorTool()

    print("  构建缺失 code 的 action...")
    action = _build_generic_action("python_code", {})
    obs, done, valid = tool.conduct_action("t-011", action, {})
    print(f"  obs={obs}, valid={valid}, invalid_reason={obs.get('invalid_reason')}")
    assert valid is False
    assert obs.get("invalid_reason") == "missing_parameters"
    print(">>> [PASSED] test_python_code_parameter_validation")


def test_opencv_operations():
    """测试 OpenCV 子操作流程（本地执行）。"""
    print(">>> [START] test_opencv_operations")
    tool = MultimodalProcessorTool()

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

    for name, args in operations:
        print(f"  调用 OpenCV operation={name}...")
        action = _build_generic_action("OpenCV", args)
        obs, done, valid = tool.conduct_action(f"t-opencv-{name}", action, {"images": [IMAGE_PATH]})
        print(f"    obs={obs}, valid={valid}")
        assert valid is True, f"OpenCV {name} call should be valid, got: {obs}"
        assert obs.get("tool") == "OpenCV"
        if "image" in obs:
            assert obs.get("image", "").startswith("data:image"), f"OpenCV {name} should return image"
            output_path = os.path.join(os.getcwd(), f"opencv_{name}_output.png")
            _save_data_url_image(obs["image"], output_path)
            print(f"  saved image to: {output_path}")
    print(">>> [PASSED] test_opencv_operations")


def test_opencv_parameter_validation():
    """测试 OpenCV 参数校验（缺失参数/非法操作）。"""
    print(">>> [START] test_opencv_parameter_validation")
    tool = MultimodalProcessorTool()

    print("  构建缺失 image_index 的 action...")
    action = _build_generic_action("OpenCV", {"operation": "crop", "x": 0, "y": 0, "w": 10, "h": 10})
    obs, done, valid = tool.conduct_action("t-opencv-params-001", action, {"images": [IMAGE_PATH]})
    print(f"    obs={obs}, valid={valid}")
    assert valid is False
    assert obs.get("invalid_reason") == "missing_parameters"

    print("  构建非法 operation 的 action...")
    action = _build_generic_action("OpenCV", {"operation": "unknown", "image_index": 1})
    obs, done, valid = tool.conduct_action("t-opencv-params-002", action, {"images": [IMAGE_PATH]})
    print(f"    obs={obs}, valid={valid}")
    assert valid is False
    assert obs.get("invalid_reason") == "invalid_parameters"

    print("  构建缺失 thresh 的 threshold action...")
    action = _build_generic_action("OpenCV", {"operation": "threshold", "image_index": 1, "type": "BINARY"})
    obs, done, valid = tool.conduct_action("t-opencv-params-003", action, {"images": [IMAGE_PATH]})
    print(f"    obs={obs}, valid={valid}")
    assert valid is False
    assert obs.get("invalid_reason") == "missing_parameters"

    print(">>> [PASSED] test_opencv_parameter_validation")


def test_opencv_base64_image():
    """测试 OpenCV 接收裸 base64 图片输入。"""
    print(">>> [START] test_opencv_base64_image")
    tool = MultimodalProcessorTool()

    image_base64 = _encode_image_as_base64(IMAGE_PATH)
    action = _build_generic_action(
        "OpenCV",
        {"operation": "grayscale", "image_index": 1},
    )
    obs, done, valid = tool.conduct_action("t-opencv-b64-001", action, {"images": [image_base64]})
    print(f"  obs={obs}, valid={valid}")
    assert valid is True, f"OpenCV base64 call should be valid, got: {obs}"
    assert obs.get("tool") == "OpenCV"
    if "image" in obs:
        assert obs.get("image", "").startswith("data:image")
        output_path = os.path.join(os.getcwd(), "opencv_base64_output.png")
        _save_data_url_image(obs["image"], output_path)
        print(f"  saved image to: {output_path}")
    print(">>> [PASSED] test_opencv_base64_image")


def test_env_image_accumulation():
    """测试环境中图片累积保存功能（真实请求）。

    验证同一个 trajectory 多次调用生成图片的工具后，
    所有生成的图片都会被累积保存到 env['images'] 中，
    并且后续调用可以使用之前生成的图片。
    """
    print(">>> [START] test_env_image_accumulation")
    tool = MultimodalProcessorTool()
    trajectory_id = "t-env-img-001"

    # 初始图片
    initial_images = [IMAGE_PATH]

    # 第一次调用：SAM3 分割，应生成第 2 张图片
    print("  第一次调用: SAM3 分割...")
    action1 = _build_generic_action(
        "SAM3",
        {"segment_type": "text", "text_prompt": "person", "image_index": 1},
    )
    obs1, done1, valid1 = tool.conduct_action(trajectory_id, action1, {"images": initial_images})
    print(f"    obs1={obs1}, valid1={valid1}")
    assert valid1 is True, f"SAM3 call should be valid, got: {obs1}"
    if "image" in obs1:
        output_path = os.path.join(os.getcwd(), "env_accumulation_sam3_1.png")
        _save_data_url_image(obs1["image"], output_path)
        print(f"  saved image to: {output_path}")
        env = tool.load_env(trajectory_id)
        print(f"  - SAM3 生成图后，图片数量: {len(env['images'])}")

    # 检查环境中的图片数量
    env = tool.load_env(trajectory_id)
    if "image" in obs1:
        assert len(env["images"]) == 2, f"Expected 2 images after SAM3, got {len(env['images'])}"
        assert env["images"][0] == IMAGE_PATH, "Original image should be preserved"
        print("  - SAM3 分割完成，图片数量: 2")
    else:
        print("  - SAM3 未返回图片，跳过图片累积检查")

    # 第二次调用：VLM（不生成图片）
    print("  第二次调用: VLM...")
    action2 = _build_action("Qwen3-VL-8B-Instruct", "请描述图片", 1)
    obs2, done2, valid2 = tool.conduct_action(trajectory_id, action2, {"images": initial_images})
    print(f"    obs2={obs2}, valid2={valid2}")
    assert valid2 is True, f"VLM call should be valid, got: {obs2}"

    # 第三次调用：Image Edit 对第 2 张图片进行编辑（SAM3 生成的图片）
    print("  第三次调用: Image Edit (第 2 张图片)...")
    action3 = _build_generic_action(
        "Qwen-Image-Edit",
        {"prompt": "make the background blue", "image_index": 2},
    )
    obs3, done3, valid3 = tool.conduct_action(trajectory_id, action3, {"images": initial_images})
    print(f"    obs3={obs3}, valid3={valid3}")
    assert valid3 is True, f"Image Edit call should be valid, got: {obs3}"
    if "image" in obs3:
        output_path = os.path.join(os.getcwd(), "env_accumulation_edit_1.png")
        _save_data_url_image(obs3["image"], output_path)
        print(f"  saved image to: {output_path}")
        env = tool.load_env(trajectory_id)
        print(f"  - Image Edit 生成图后，图片数量: {len(env['images'])}")

    # 第四次调用：PaddleOCR（不生成图片）
    print("  第四次调用: PaddleOCR...")
    action4 = _build_generic_action("PaddleOCR", {"image_index": 1})
    obs4, done4, valid4 = tool.conduct_action(trajectory_id, action4, {"images": initial_images})
    print(f"    obs4={obs4}, valid4={valid4}")
    assert valid4 is True, f"PaddleOCR call should be valid, got: {obs4}"

    # 第五次调用：WolframAlpha（不生成图片）
    print("  第五次调用: WolframAlpha...")
    action5 = _build_wolfram_action("10 densest elemental metals")
    obs5, done5, valid5 = tool.conduct_action(trajectory_id, action5, {})
    print(f"    obs5={obs5}, valid5={valid5}")
    assert valid5 is True, f"WolframAlpha call should be valid, got: {obs5}"

    # 检查环境中的图片数量
    env = tool.load_env(trajectory_id)
    current_image_count = len(env["images"])
    print(f"  - 截至 WolframAlpha，图片数量: {current_image_count}")

    # 第六次调用：对第 3 张图片（如果存在）再次进行编辑
    if current_image_count >= 2:
        # 第六次调用：python_code（不生成图片）
        print("  第六次调用: python_code...")
        action6 = _build_generic_action("python_code", {"code": "1 + 1"})
        obs6, done6, valid6 = tool.conduct_action(trajectory_id, action6, {})
        print(f"    obs6={obs6}, valid6={valid6}")
        assert valid6 is True, f"python_code call should be valid, got: {obs6}"

        # 第七次调用：OpenCV（对第 3 张图片进行处理）
        print("  第七次调用: OpenCV (第 3 张图片)...")
        action7 = _build_generic_action(
            "OpenCV",
            {"operation": "grayscale", "image_index": 3},
        )
        obs7, done7, valid7 = tool.conduct_action(trajectory_id, action7, {"images": initial_images})
        print(f"    obs7={obs7}, valid7={valid7}")
        assert valid7 is True, f"OpenCV call should be valid, got: {obs7}"
        if "image" in obs7:
            output_path = os.path.join(os.getcwd(), "env_accumulation_opencv_1.png")
            _save_data_url_image(obs7["image"], output_path)
            print(f"  saved image to: {output_path}")
            env = tool.load_env(trajectory_id)
            print(f"  - OpenCV 生成图后，图片数量: {len(env['images'])}")

        # 第八次调用：Image Edit（第 4 张图片）
        print("  第八次调用: Image Edit (第 4 张图片)...")
        action8 = _build_generic_action(
            "Qwen-Image-Edit",
            {"prompt": "add a red border", "image_index": 4},
        )
        obs8, done8, valid8 = tool.conduct_action(trajectory_id, action8, {"images": initial_images})
        print(f"    obs8={obs8}, valid8={valid8}")
        assert valid8 is True, f"Second Image Edit call should be valid, got: {obs8}"
        if "image" in obs8:
            output_path = os.path.join(os.getcwd(), "env_accumulation_edit_2.png")
            _save_data_url_image(obs8["image"], output_path)
            print(f"  saved image to: {output_path}")
            env = tool.load_env(trajectory_id)
            print(f"  - 第二次 Image Edit 生成图后，图片数量: {len(env['images'])}")

        # 检查环境中的图片数量
        env = tool.load_env(trajectory_id)
        final_image_count = len(env["images"])
        print(f"  - 第二次 Image Edit 完成，图片数量: {final_image_count}")

        # 验证图片确实在累积
        assert final_image_count > current_image_count, "Images should accumulate"

    # 验证 previous_obs 记录了所有操作
    env = tool.load_env(trajectory_id)
    print('env', env)
    assert len(env["previous_obs"]) >= 2, f"Expected at least 2 observations, got {len(env['previous_obs'])}"
    assert env["metadata"]["turns"] >= 2, f"Expected at least 2 turns, got {env['metadata']['turns']}"
    print(f"  - 操作记录数: {len(env['previous_obs'])}, turns: {env['metadata']['turns']}")

    # 清理环境
    tool.delete_env(trajectory_id)
    assert not tool.has_env(trajectory_id), "Environment should be deleted"
    print("  - 环境清理完成")

    print(">>> [PASSED] test_env_image_accumulation")


def main():
    """主入口，支持命令行运行单个测试。"""
    fire.Fire(
        {
            "init": test_tool_initialization,
            "params": test_parameter_validation,
            "wolfram_params": test_wolfram_parameter_validation,
            "call": test_model_call,
            "wolfram_call": test_wolfram_call,
            "sam3_text": test_sam3_text_call,
            "mineru_content": test_mineru_content_call,
            "paddle_call": test_paddle_call,
            "image_edit_call": test_image_edit_call,
            "python_code_call": test_python_code_call,
            "python_code_params": test_python_code_parameter_validation,
            "opencv_ops": test_opencv_operations,
            "opencv_params": test_opencv_parameter_validation,
            "opencv_base64": test_opencv_base64_image,
            "env_image_accumulation": test_env_image_accumulation,
        }
    )


if __name__ == "__main__":
    main()
