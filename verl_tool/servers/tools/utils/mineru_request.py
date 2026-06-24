#!/usr/bin/env python3
"""
MinerU 文档提取请求工具
"""

import base64
import requests
from pathlib import Path
from typing import Union, List
from PIL import Image
from io import BytesIO


def image_to_base64(image: Union[str, Path]) -> str:
    """将图片路径转换为 base64"""
    img = Image.open(image)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def mineru_health_check(server_url: str, timeout: int = 10) -> dict:
    """
    检查 MinerU 服务健康状态

    Args:
        server_url: 服务端地址，如 "http://localhost:30009"
        timeout: 请求超时时间（秒）

    Returns:
        dict: 健康检查响应
    """
    url = f"{server_url.rstrip('/')}/health"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    
    # 处理空响应
    if not resp.text:
        return {"status": "ok", "model_loaded": True}
    
    return resp.json()


def mineru_extract(
    server_url: str,
    image_base64: str,
    extract_type: str = "two_step",
    content_type: str = None,
    timeout: int = 300,
) -> dict:
    """
    文档提取

    Args:
        server_url: 服务端地址
        image_base64: base64 编码的图片
        extract_type: 提取类型
            - "two_step": 两步提取（布局检测 + 内容提取）
            - "layout": 仅布局检测
            - "content": 仅内容提取
        content_type: 内容类型，当 extract_type="content" 时使用
            - "text": 文本
            - "table": 表格
            - "equation": 公式
        timeout: 请求超时时间（秒）

    Returns:
        dict: 提取结果响应
    """
    url = f"{server_url.rstrip('/')}/extract"

    payload = {
        "image_base64": image_base64,
        "extract_type": extract_type,
    }

    if content_type:
        payload["content_type"] = content_type

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def mineru_content_extract(
    server_url: str,
    image_base64: str,
    content_type: str = "text",
    timeout: int = 300,
) -> str:
    """
    内容提取

    Args:
        server_url: 服务端地址
        image_base64: base64 编码的图片
        content_type: 内容类型
            - "text": 文本
            - "table": 表格
            - "equation": 公式
        timeout: 请求超时时间（秒）

    Returns:
        str: 提取的内容
    """
    url = f"{server_url.rstrip('/')}/extract"

    payload = {
        "image_base64": image_base64,
        "extract_type": "content",
        "content_type": content_type
    }

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    return data["data"]["content"]


def mineru_batch_extract(
    server_url: str,
    requests_list: List[dict],
    timeout: int = 300,
) -> dict:
    """
    批量两步提取

    Args:
        server_url: 服务端地址
        requests_list: 请求列表，每项包含:
            {
                "image_base64": str,
                "extract_type": str
            }
        timeout: 请求超时时间（秒）

    Returns:
        dict: 批量提取结果响应
    """
    url = f"{server_url.rstrip('/')}/batch_extract"
    resp = requests.post(url, json=requests_list, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def format_blocks(blocks: list) -> str:
    """格式化输出块信息"""
    lines = []
    for i, block in enumerate(blocks, 1):
        lines.append(f"【Block {i}】")
        lines.append(f"  类型: {block.get('type')}")
        lines.append(f"  位置: {block.get('bbox')}")
        if block.get('angle') is not None:
            lines.append(f"  角度: {block.get('angle')}")
        if block.get('content'):
            content = block['content']
            if len(str(content)) > 200:
                content = str(content)[:200] + "..."
            lines.append(f"  内容: {content}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    # 配置
    SERVER_URL = "https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-c6264b38-46a1-4bb6-a3a7-1db39453f6f8/vscode/5e251c6d-8e0c-4e8c-be04-4f64e5d8e5eb/cbf73fa3-6fdd-47fa-a3d9-14de85230f07/proxy/30009"
    IMAGE_PATH = "/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/demo.png"

    print(f"连接服务: {SERVER_URL[:80]}...")
    print("正在检查服务状态...")

    try:
        health = mineru_health_check(SERVER_URL)
        print(f"服务状态: {health}")

        if not health.get("model_loaded"):
            print("⚠️ 警告: 模型尚未加载完成")
            exit(1)
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        exit(1)

    # 读取图片
    print(f"\n读取图片: {IMAGE_PATH}")
    img_b64 = image_to_base64(IMAGE_PATH)

    # 测试1：两步提取
    print("\n" + "=" * 50)
    print("测试1：两步提取（布局检测 + 内容提取）")
    print("-" * 50)
    try:
        result = mineru_extract(SERVER_URL, img_b64, extract_type="two_step")
        print("两步提取：", result)

        if result.get("success"):
            print("✅ 提取成功!")
            blocks = result.get("data", {}).get("blocks", [])
            print(f"提取到 {len(blocks)} 个内容块")
            print(format_blocks(blocks[:5]))
            if len(blocks) > 5:
                print(f"... 还有 {len(blocks) - 5} 个块")
        else:
            print("❌ 提取失败!")
            print("错误:", result.get("error", "未知错误"))
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP错误: {e}")
        print(f"响应状态码: {e.response.status_code}")
        print(f"响应内容: {e.response.text[:200]}")
    except Exception as e:
        print(f"❌ 失败: {e}")

    # 测试2：仅布局检测
    print("\n" + "=" * 50)
    print("测试2：布局检测")
    print("-" * 50)
    try:
        result = mineru_extract(SERVER_URL, img_b64, extract_type="layout")
        print("布局检测:", result)
        if result.get("success"):
            print("✅ 布局检测成功!")
            blocks = result.get("data", {}).get("blocks", [])
            print(f"检测到 {len(blocks)} 个布局块")
            print(format_blocks(blocks[:3]))
        else:
            print("❌ 布局检测失败!")
            print("错误:", result.get("error", "未知错误"))
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP错误: {e}")
        print(f"响应状态码: {e.response.status_code}")
        print(f"响应内容: {e.response.text[:200]}")
    except Exception as e:
        print(f"❌ 失败: {e}")

    # 测试3：内容提取
    print("\n" + "=" * 50)
    print("测试3：内容提取（文本）")
    print("-" * 50)
    try:
        content = mineru_content_extract(SERVER_URL, img_b64, content_type="text")
        print("✅ 内容提取成功!")
        print(f"提取内容:\n{content}")
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP错误: {e}")
        print(f"响应状态码: {e.response.status_code}")
        print(f"响应内容: {e.response.text[:200]}")
    except Exception as e:
        print(f"❌ 失败: {e}")

# ==================================================
# 测试1：两步提取（布局检测 + 内容提取）
# --------------------------------------------------
# 两步提取： {'success': True, 'data': {'blocks': [{'type': 'header', 'bbox': [0.003, 0.016, 0.067, 0.032], 'angle': 0, 'content': '更改后自动刷新数据'}, {'type': 'header', 'bbox': [0.071, 0.016, 0.088, 0.032], 'angle': 0, 'content': '0'}, {'type': 'header', 'bbox': [0.742, 0.016, 0.772, 0.032], 'angle': 0, 'content': '数据解释'}, {'type': 'header', 'bbox': [0.787, 0.016, 0.804, 0.032], 'angle': 0, 'content': '</>'}, {'type': 'header', 'bbox': [0.811, 0.016, 0.832, 0.032], 'angle': 0, 'content': '面'}, {'type': 'header', 'bbox': [0.839, 0.016, 0.854, 0.032], 'angle': 0, 'content': '#'}, {'type': 'header', 'bbox': [0.869, 0.016, 0.889, 0.032], 'angle': 0, 'content': '可视化'}, {'type': 'image_caption', 'bbox': [0.009, 0.06, 0.085, 0.079], 'angle': 0, 'content': '各区域销售情况'}, {'type': 'image_caption', 'bbox': [0.008, 0.086, 0.054, 0.103], 'angle': 0, 'content': '\\(\\oplus\\) 汇总指标'}, {'type': 'image', 'bbox': [0.005, 0.124, 0.855, 0.932], 'angle': 0}, {'type': 'image_caption', 'bbox': [0.869, 0.042, 0.973, 0.058], 'angle': 0, 'content': '常用类型'}, {'type': 'image_caption', 'bbox': [0.869, 0.06, 0.973, 0.076], 'angle': 0, 'content': '- 带有天王'}, {'type': 'image', 'bbox': [0.871, 0.082, 0.991, 0.209], 'angle': 0}, {'type': 'image', 'bbox': [0.871, 0.216, 0.961, 0.254], 'angle': 0}, {'type': 'title', 'bbox': [0.866, 0.274, 0.899, 0.292], 'angle': 0, 'content': '图表属性'}, {'type': 'text', 'bbox': [0.873, 0.304, 0.916, 0.321], 'angle': 0, 'content': '参数默认值'}, {'type': 'text', 'bbox': [0.873, 0.341, 0.911, 0.357], 'angle': 0, 'content': '主题/颜色'}, {'type': 'text', 'bbox': [0.873, 0.378, 0.902, 0.394], 'angle': 0, 'content': '坐标轴'}, {'type': 'text', 'bbox': [0.873, 0.415, 0.91, 0.431], 'angle': 0, 'content': '数据标签'}, {'type': 'text', 'bbox': [0.873, 0.452, 0.895, 0.467], 'angle': 0, 'content': '图例'}, {'type': 'text', 'bbox': [0.873, 0.488, 0.902, 0.504], 'angle': 0, 'content': '辅助线'}, {'type': 'text', 'bbox': [0.873, 0.525, 0.909, 0.541], 'angle': 0, 'content': '工具提示'}, {'type': 'text', 'bbox': [0.873, 0.561, 0.915, 0.577], 'angle': 0, 'content': '分组累计线'}, {'type': 'text', 'bbox': [0.873, 0.599, 0.896, 0.615], 'angle': 0, 'content': '标题'}, {'type': 'text', 'bbox': [0.873, 0.635, 0.909, 0.651], 'angle': 0, 'content': '卡片设置'}, {'type': 'list', 'bbox': [0.873, 0.304, 0.916, 0.651], 'angle': 0}, {'type': 'page_number', 'bbox': [0.003, 0.962, 0.034, 0.979], 'angle': 0, 'content': '\\(\\oplus\\) 描述'}]}, 'error': None}
# ✅ 提取成功!
# 提取到 27 个内容块
# 【Block 1】
#   类型: header
#   位置: [0.003, 0.016, 0.067, 0.032]
#   角度: 0
#   内容: 更改后自动刷新数据

# 【Block 2】
#   类型: header
#   位置: [0.071, 0.016, 0.088, 0.032]
#   角度: 0
#   内容: 0

# 【Block 3】
#   类型: header
#   位置: [0.742, 0.016, 0.772, 0.032]
#   角度: 0
#   内容: 数据解释

# 【Block 4】
#   类型: header
#   位置: [0.787, 0.016, 0.804, 0.032]
#   角度: 0
#   内容: </>

# 【Block 5】
#   类型: header
#   位置: [0.811, 0.016, 0.832, 0.032]
#   角度: 0
#   内容: 面

# ... 还有 22 个块

# ==================================================
# 测试2：布局检测
# --------------------------------------------------
# 布局检测: {'success': True, 'data': {'blocks': [{'type': 'header', 'bbox': [0.003, 0.016, 0.067, 0.032], 'angle': 0}, {'type': 'header', 'bbox': [0.071, 0.016, 0.088, 0.032], 'angle': 0}, {'type': 'header', 'bbox': [0.742, 0.016, 0.772, 0.032], 'angle': 0}, {'type': 'header', 'bbox': [0.787, 0.016, 0.804, 0.032], 'angle': 0}, {'type': 'header', 'bbox': [0.811, 0.016, 0.832, 0.032], 'angle': 0}, {'type': 'header', 'bbox': [0.839, 0.016, 0.854, 0.032], 'angle': 0}, {'type': 'header', 'bbox': [0.869, 0.016, 0.889, 0.032], 'angle': 0}, {'type': 'image_caption', 'bbox': [0.009, 0.06, 0.085, 0.079], 'angle': 0}, {'type': 'image_caption', 'bbox': [0.008, 0.086, 0.054, 0.103], 'angle': 0}, {'type': 'image', 'bbox': [0.005, 0.124, 0.855, 0.932], 'angle': 0}, {'type': 'image_caption', 'bbox': [0.869, 0.042, 0.973, 0.058], 'angle': 0}, {'type': 'image_caption', 'bbox': [0.869, 0.06, 0.973, 0.076], 'angle': 0}, {'type': 'image', 'bbox': [0.871, 0.082, 0.991, 0.209], 'angle': 0}, {'type': 'image', 'bbox': [0.871, 0.216, 0.961, 0.254], 'angle': 0}, {'type': 'title', 'bbox': [0.866, 0.274, 0.899, 0.292], 'angle': 0}, {'type': 'text', 'bbox': [0.873, 0.304, 0.916, 0.321], 'angle': 0}, {'type': 'text', 'bbox': [0.873, 0.341, 0.911, 0.357], 'angle': 0}, {'type': 'text', 'bbox': [0.873, 0.378, 0.902, 0.394], 'angle': 0}, {'type': 'text', 'bbox': [0.873, 0.415, 0.91, 0.431], 'angle': 0}, {'type': 'text', 'bbox': [0.873, 0.452, 0.895, 0.467], 'angle': 0}, {'type': 'text', 'bbox': [0.873, 0.488, 0.902, 0.504], 'angle': 0}, {'type': 'text', 'bbox': [0.873, 0.525, 0.909, 0.541], 'angle': 0}, {'type': 'text', 'bbox': [0.873, 0.561, 0.915, 0.577], 'angle': 0}, {'type': 'text', 'bbox': [0.873, 0.599, 0.896, 0.615], 'angle': 0}, {'type': 'text', 'bbox': [0.873, 0.635, 0.909, 0.651], 'angle': 0}, {'type': 'list', 'bbox': [0.873, 0.304, 0.916, 0.651], 'angle': 0}, {'type': 'page_number', 'bbox': [0.003, 0.962, 0.034, 0.979], 'angle': 0}]}, 'error': None}
# ✅ 布局检测成功!
# 检测到 27 个布局块
# 【Block 1】
#   类型: header
#   位置: [0.003, 0.016, 0.067, 0.032]
#   角度: 0

# 【Block 2】
#   类型: header
#   位置: [0.071, 0.016, 0.088, 0.032]
#   角度: 0

# 【Block 3】
#   类型: header
#   位置: [0.742, 0.016, 0.772, 0.032]
#   角度: 0


# ==================================================
# 测试3：内容提取（文本）
# --------------------------------------------------
# ✅ 内容提取成功!
# 提取内容:
# 更改后自动刷新数据 数据解释√ m图可视化 可视化常用类型 Q各区域销售情况汇总指标图表属性参数默认值主题/颜色坐标轴数据标签 不显示图例 显示辅助线工具提示 显示分组累计线标题 显示卡片设置华东 中南 东北 华北 西南 西北消费者 公司 小型企业