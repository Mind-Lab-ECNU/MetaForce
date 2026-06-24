#!/usr/bin/env python3
"""
Qwen Image Edit 请求工具
"""

import requests
import base64
import json
from typing import List


def image_edit_health_check(server_url: str, timeout: int = 10) -> dict:
    """
    检查图像编辑服务健康状态

    Args:
        server_url: 服务端地址，如 "http://localhost:30020"
        timeout: 请求超时时间（秒）

    Returns:
        dict: 健康检查响应
    """
    url = f"{server_url.rstrip('/')}/health"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def image_edit(
    server_url: str,
    file_paths: List[str],
    prompt: str,
    negative_prompt: str = " ",
    num_inference_steps: int = 40,
    true_cfg_scale: float = 4.0,
    guidance_scale: float = 1.0,
    seed: int = 0,
    output_path: str = "output.png",
    timeout: int = 300,
) -> str:
    """
    上传图片进行编辑

    Args:
        server_url: 服务端地址
        file_paths: 图片路径列表
        prompt: 编辑提示词
        negative_prompt: 负面提示词
        num_inference_steps: 推理步数
        true_cfg_scale: CFG 缩放
        guidance_scale: 引导缩放
        seed: 随机种子
        output_path: 输出图片保存路径
        timeout: 请求超时时间（秒）

    Returns:
        str: 输出图片保存路径
    """
    url = f"{server_url.rstrip('/')}/edit"

    files = []
    try:
        for path in file_paths:
            files.append(("files", open(path, "rb")))

        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": true_cfg_scale,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }

        resp = requests.post(url, files=files, data=data, timeout=timeout)
        print('resp', resp.content)

        if resp.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return output_path
        else:
            raise Exception(f"Error: {resp.text}")
    finally:
        for _, f in files:
            f.close()


def image_edit_base64(
    server_url: str,
    images_base64: List[str],
    prompt: str,
    negative_prompt: str = " ",
    num_inference_steps: int = 40,
    true_cfg_scale: float = 4.0,
    guidance_scale: float = 1.0,
    seed: int = 0,
    output_path: str = "output.png",
    timeout: int = 300,
) -> str:
    """
    使用 base64 方式上传图片进行编辑

    Args:
        server_url: 服务端地址
        images_base64: base64 编码的图片列表
        prompt: 编辑提示词
        negative_prompt: 负面提示词
        num_inference_steps: 推理步数
        true_cfg_scale: CFG 缩放
        guidance_scale: 引导缩放
        seed: 随机种子
        output_path: 输出图片保存路径
        timeout: 请求超时时间（秒）

    Returns:
        str: 输出图片保存路径
    """
    url = f"{server_url.rstrip('/')}/edit_base64"

    data = {
        "prompt": prompt,
        "images_base64": json.dumps(images_base64),
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "guidance_scale": guidance_scale,
        "seed": seed,
    }

    resp = requests.post(url, data=data, timeout=timeout)
    resp.raise_for_status()

    result = resp.json()
    if result.get("success"):
        img_data = base64.b64decode(result["image_base64"])
        with open(output_path, "wb") as f:
            f.write(img_data)
        return output_path
    else:
        raise Exception(f"Error: {result.get('error')}")


if __name__ == "__main__":
    # 配置
    SERVER_URL = "https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-c6264b38-46a1-4bb6-a3a7-1db39453f6f8/vscode/5e251c6d-8e0c-4e8c-be04-4f64e5d8e5eb/cbf73fa3-6fdd-47fa-a3d9-14de85230f07/proxy/30020"
    IMAGE_PATH = "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/add/general/HatefulMemes_2000/images/train/train_1.png"

    print(f"连接服务: {SERVER_URL[:80]}...")
    print("正在检查服务状态...")
    health = image_edit_health_check(SERVER_URL)
    print(f"服务状态: {health}")

    if health.get("status") != "ok":
        print("服务未就绪!")
        exit(1)

    # 图像编辑
    print("\n" + "=" * 50)
    print("开始图像编辑（可能需要几分钟）...")
    print("提示词: 'turn the girl into Trump'")

    output_path = image_edit(
        SERVER_URL,
        [IMAGE_PATH],
        prompt="turn the girl into Trump",
        output_path="edited_output_turn.png"
    )

    print(f"\n编辑完成，保存到: {output_path}")
    
    # 图像剪裁
    print("\n" + "=" * 50)
    print("开始图像剪裁（可能需要几分钟）...")
    print("提示词:剪裁目标区域")

    output_path = image_edit(
        SERVER_URL,
        [IMAGE_PATH],
        prompt="剪裁女生",
        output_path="edited_output_crop.png"
    )

    print(f"\n编辑完成，保存到: {output_path}")

    # 图像旋转
    print("\n" + "=" * 50)
    print("开始图像旋转（可能需要几分钟）...")
    print("提示词:旋转目标区域")

    output_path = image_edit(
        SERVER_URL,
        [IMAGE_PATH],
        prompt="旋转图片中的三个女生",
        output_path="edited_output_s.png"
    )

    print(f"\n编辑完成，保存到: {output_path}")
