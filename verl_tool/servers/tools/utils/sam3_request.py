#!/usr/bin/env python3
"""
SAM3 分割请求工具
支持: 文本提示、框提示
"""

import base64
import requests
from pathlib import Path
from typing import Union, List, Optional
from PIL import Image
from io import BytesIO
import numpy as np


def image_to_base64(image: Union[str, Path]) -> str:
    """将图片路径转换为 base64"""
    img = Image.open(image)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def sam3_health_check(server_url: str, timeout: int = 10) -> dict:
    """
    检查 SAM3 服务健康状态

    Args:
        server_url: 服务端地址，如 "http://localhost:30018"
        timeout: 请求超时时间（秒）

    Returns:
        dict: 健康检查响应
    """
    url = f"{server_url.rstrip('/')}/health"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def sam3_segment_text(
    server_url: str,
    image_base64: str,
    text_prompt: str,
    threshold: float = 0.5,
    timeout: int = 300,
) -> dict:
    """
    使用文本提示进行分割

    Args:
        server_url: 服务端地址
        image_base64: base64 编码的图片
        text_prompt: 文本提示，如 "person", "car", "shoe"
        threshold: 置信度阈值 (0-1)
        timeout: 请求超时时间（秒）

    Returns:
        dict: 分割结果响应
    """
    url = f"{server_url.rstrip('/')}/segment/text"
    payload = {
        "image_base64": image_base64,
        "text_prompt": text_prompt,
        "threshold": threshold,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def sam3_segment_box(
    server_url: str,
    image_base64: str,
    boxes: List[List[float]],
    labels: List[bool],
    timeout: int = 300,
) -> dict:
    """
    使用边界框进行分割

    Args:
        server_url: 服务端地址
        image_base64: base64 编码的图片
        boxes: 边界框列表 [[x1, y1, x2, y2], ...]，像素坐标
        labels: 标签列表 [True, False, ...]
                True = 前景（要分割的对象）
                False = 背景（排除的区域）
        timeout: 请求超时时间（秒）

    Returns:
        dict: 分割结果响应
    """
    url = f"{server_url.rstrip('/')}/segment/box"
    payload = {
        "image_base64": image_base64,
        "boxes": boxes,
        "labels": labels,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def base64_to_mask(mask_b64: str) -> np.ndarray:
    """将 base64 掩码转换为 numpy 数组"""
    mask_data = base64.b64decode(mask_b64)
    mask_image = Image.open(BytesIO(mask_data))
    return np.array(mask_image)


def visualize(
    image: Union[str, Path, Image.Image],
    masks: List[np.ndarray],
    boxes: Optional[List[List[float]]] = None,
    scores: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
    alpha: float = 0.5,
    show_boxes: bool = True,
) -> Image.Image:
    """
    可视化分割结果

    Args:
        image: 原始图片
        masks: 掩码列表
        boxes: 边界框列表
        scores: 分数列表
        labels: 标签列表
        alpha: 掩码透明度
        show_boxes: 是否显示边界框

    Returns:
        PIL Image: 可视化结果
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    image = image.convert("RGBA")

    n_masks = len(masks)
    if n_masks == 0:
        return image

    # 生成颜色
    import colorsys
    colors = []
    for i in range(n_masks):
        hue = i / max(n_masks, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(tuple(int(c * 255) for c in rgb))

    # 叠加掩码
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

    # 绘制边界框和标签
    if show_boxes and boxes and len(boxes) > 0:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

        for i, (box, color) in enumerate(zip(boxes, colors)):
            if len(box) >= 4:
                x1, y1, x2, y2 = box[:4]
                draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)

                # 标签文本
                label_parts = []
                if labels and i < len(labels):
                    label_parts.append(labels[i])
                if scores and i < len(scores):
                    label_parts.append(f"{scores[i]:.2f}")

                if label_parts:
                    label_text = " ".join(label_parts)
                    draw.text((x1 + 5, y1 + 5), label_text, fill=(255, 255, 255, 255), font=font)

    return image


if __name__ == "__main__":
    # 配置
    SERVER_URL = "https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-c6264b38-46a1-4bb6-a3a7-1db39453f6f8/vscode/5e251c6d-8e0c-4e8c-be04-4f64e5d8e5eb/cbf73fa3-6fdd-47fa-a3d9-14de85230f07/proxy/30018"
    IMAGE_PATH = "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/add/general/HatefulMemes_2000/images/train/train_1.png"

    print("检查服务状态...")
    health = sam3_health_check(SERVER_URL)
    print(f"健康状态: {health}")

    # 文本分割测试
    print("\n" + "=" * 50)
    print("测试文本分割...")
    text_prompts = ["person", "face", "shoe"]
    for prompt in text_prompts:
        print(f"\n--- 测试: '{prompt}' ---")
        img_b64 = image_to_base64(IMAGE_PATH)
        result = sam3_segment_text(SERVER_URL, img_b64, prompt, threshold=0.1)
        print('文本分割', {prompt}, result)
        if result.get("success"):
            data = result['data']
            num_objects = data.get('num_objects', 0)
            print(f"✅ 检测到 {num_objects} 个对象")
            if data.get('scores'):
                print(f"   分数: {[f'{s:.3f}' for s in data['scores']]}")

            # 可视化并保存
            if num_objects > 0 and data.get('masks_base64'):
                masks = [base64_to_mask(m) for m in data['masks_base64']]
                vis_image = visualize(
                    IMAGE_PATH,
                    masks,
                    data.get('boxes'),
                    data.get('scores'),
                    labels=[prompt] * num_objects,
                )
                output_path = f"output_text_{prompt}.png"
                vis_image.save(output_path)
                print(f"   💾 保存到 {output_path}")
        else:
            print(f"❌ 失败: {result.get('error')}")

    # 框分割测试
    print("\n" + "=" * 50)
    print("测试框分割...")
    test_img = Image.open(IMAGE_PATH)
    img_width, img_height = test_img.size
    center_x, center_y = img_width // 2, img_height // 2
    box_size = min(img_width, img_height) // 3
    test_box = [
        center_x - box_size // 2,
        center_y - box_size // 2,
        center_x + box_size // 2,
        center_y + box_size // 2,
    ]

    img_b64 = image_to_base64(IMAGE_PATH)
    result = sam3_segment_box(SERVER_URL, img_b64, [test_box], [True])
    print('框分割', result)
    if result.get("success"):
        data = result['data']
        num_objects = data.get('num_objects', 0)
        print(f"✅ 分割了 {num_objects} 个区域")

        # 可视化并保存
        if num_objects > 0 and data.get('masks_base64'):
            masks = [base64_to_mask(m) for m in data['masks_base64']]
            vis_image = visualize(
                IMAGE_PATH,
                masks,
                [test_box],
                data.get('scores'),
            )
            output_path = "output_box.png"
            vis_image.save(output_path)
            print(f"   💾 保存到 {output_path}")
    else:
        print(f"❌ 失败: {result.get('error')}")

    print("\n" + "=" * 50)
    print("测试完成!")


# 健康状态: {'status': 'ok', 'model_loaded': True, 'device': 'cuda:0', 'physical_gpu': 1, 'gpu_name': 'NVIDIA H200', 'supported_modes': ['text', 'box']}

# ==================================================
# 测试文本分割...

# --- 测试: 'person' ---
# 文本分割 {'person'} {'success': True, 'data': {'num_objects': 5, 'masks_base64': ['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'], 'boxes': [[20.948482513427734, 340.0607604980469, 166.49765014648438, 465.8026123046875], [19.841875076293945, 339.87158203125, 167.21749877929688, 465.1597900390625], [6.486913204193115, 475.950439453125, 177.2208709716797, 590.4954833984375], [6.981389999389648, 529.0288696289062, 43.600982666015625, 590.8674926757812], [65.73770141601562, 249.13787841796875, 151.26025390625, 334.9702453613281]], 'scores': [0.10888671875, 0.94921875, 0.94921875, 0.1630859375, 0.953125], 'image_size': [177, 800], 'prompt': 'person'}, 'error': None}
# ✅ 检测到 5 个对象
#    分数: ['0.109', '0.949', '0.949', '0.163', '0.953']
#    💾 保存到 output_text_person.png

# --- 测试: 'face' ---
# 文本分割 {'face'} {'success': True, 'data': {'num_objects': 3, 'masks_base64': ['iVBORw0KGgoAAAANSUhEUgAAALEAAAMgCAAAAABJkNxiAAACA0lEQVR4nO3bQU7DQBAEQIz4/5fDAQVZQcE20zMmctUBKRw2vc2sF6Hw9gYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPx3t1vn6kt6wXXa+OIdiz7U25D5I7lY6zTcxUp4kjZfcmrF5/WmM2fW+30cspnfE4tszG92vBOJNxNFIwcS78iTjFxPvCtNMHJkjnfIRS4nHrk11qY6zu2smni84rmOY3srJp6veLDj1O4GE4fUEh+rLVPyaMeRyFebiqMSJet4Q6DkUuIT7o/5qahvspS45W8+W5y8fuOJy4Os434Sb6sOso77SdzvhMTFo3e1js/45e2MjmtjcbWpOEMx8QmDfLmO/6Z09KqJ58fCVPSTuF858fjRu2DH4yUHOh6OnJiK45Erl15kjpfJmkMn71jk0gav+Kz4MjcXsY4PRK7tLjcVR3JUMgfneG+M4gAlT96+KNWJjz4r9oQpH9HmT0w3vN91n8d3y8+X6zs88CPNP/lXc7E8fj/xbtHP0T9YHl9FPivQcLveczVd3C0nb/n+8iJuzf91k/dicQEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAS/gEyActi9SbwRQAAAAASUVORK5CYII=', 'iVBORw0KGgoAAAANSUhEUgAAALEAAAMgCAAAAABJkNxiAAABKElEQVR4nO3YQQ7CMAwEwJT//zmcEKoqoE4aWxUzp9y8XfVitwYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwL/bSqb2iekVifv7OTD+cV2Qs/qH90n5iQdC7qQnng1c8VfsxD8gO/F0xeUdxyUnPlYcLl3H383/xTrOIPF6Eq8n8XrlicNbSHnisNzEx0Lji56Og+6xS0+qTTxyLdHxeqWJh05o9+s4/VLY2/ba98Zm11xjW+t1o0ddsVYDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPz0Bn1YPOUYpfkUAAAAASUVORK5CYII=', 'iVBORw0KGgoAAAANSUhEUgAAALEAAAMgCAAAAABJkNxiAAABrklEQVR4nO3Zy27CMBAFUE/V//9ldwGIUh5F8lyTxTmLVhFSfLlMHNKOAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMCRVe7UM7PMV9+p/phPD9akOr6P2LVSqOMHnXbVnJuKlEzih302lbyz457IkcTPorVE3jvHHZE3X3kNkROJG28XD+ze3dbfzfb9eDmyO8hIj/EHOl59Q6Yib0/iOvYzSJ2+vJ9SXn5UW+jgc97ZvC4yO5b7Xj/F+1rqyc9x96dor/hl3vxqk+z4LmtL+NheMes23/mwlveLQOI56p8ylxbd+mT67usv2SvyPpN4ZSy23qUvjnflRUk84l9gdZwn8RjpQdZxnsRjjPAg6/gkWbKOz4IlpzrORY5NRSyyOb56XvJa/TrOCyauUaOquq/B/H8V7v86cfw5rpeHi2eLufS8az0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB43w8kUSN6B8nz6gAAAABJRU5ErkJggg=='], 'boxes': [[60.49418640136719, 485.92108154296875, 118.17552185058594, 562.61083984375], [108.49445343017578, 257.79425048828125, 129.32904052734375, 290.24444580078125], [80.79415130615234, 355.4683532714844, 119.26688385009766, 413.49395751953125]], 'scores': [0.8515625, 0.875, 0.83984375], 'image_size': [177, 800], 'prompt': 'face'}, 'error': None}
# ✅ 检测到 3 个对象
#    分数: ['0.852', '0.875', '0.840']
#    💾 保存到 output_text_face.png

# --- 测试: 'shoe' ---
# 文本分割 {'shoe'} {'success': True, 'data': {'num_objects': 0, 'masks_base64': [], 'boxes': [], 'scores': [], 'image_size': [177, 800], 'prompt': 'shoe'}, 'error': None}
# ✅ 检测到 0 个对象

# ==================================================
# 测试框分割...
# 框分割 {'success': True, 'data': {'num_objects': 1, 'masks_base64': ['iVBORw0KGgoAAAANSUhEUgAAALEAAAMgCAAAAABJkNxiAAABr0lEQVR4nO3a0WoCQRAEwN2Q///lyYOenomHJDutCakC9VQ4m2Z2VXQMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4C+q5TO8NaT46nOuOn7q2xKJq0Zdo9UYtcs5V8++fII7tnhzjJrne32vE+i4bo7WB/eTzByf1K7aGlU94fsT18HxGKPqdFnSn3geHG8PrU50ZCrmzc05ZdfaS87x6Ax6EV552yjP2ZY9kXi3QexTN0UOvIPc2wxmX+j+xA93r8WXjK68+xY35Ox+nPCCjhdJnCdxnsQjvr29ouO1txBTkRf9tBmh47zkN9MMHZ88KHlpQ9bxWXCSdbzJlRzrOBbZVORJnCfxVWqz0HHeSxIvDcx7V4oDc/fJco7bXyV/5okdNy3F5BecU5+Xkn/xb02bubvuPmtWb8vplXfxjGr+rfa/hAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAT/MBUJ8tZa80BVwAAAAASUVORK5CYII='], 'boxes': [[59.0, 371.0, 117.0, 429.0]], 'scores': [0.8984375], 'image_size': [177, 800], 'input_boxes': [[59.0, 371.0, 117.0, 429.0]], 'input_labels': [True]}, 'error': None}
# ✅ 分割了 1 个区域
#    💾 保存到 output_box.png

# ==================================================