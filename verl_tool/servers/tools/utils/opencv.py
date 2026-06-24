"""
OpenCV Processor - 提供基于 OpenCV 的图像处理功能

参考文档: opencv.md
"""

import cv2
import numpy as np
from PIL import Image
from typing import Any, Dict, Optional
import base64
from io import BytesIO


class OpenCVProcessor:
    """
    OpenCV 图像处理器

    提供 8 种基础图像操作：
    - 几何变换: crop, resize, rotate, flip
    - 图像增强: grayscale, blur, threshold, canny
    """

    # 插值方式映射
    INTERPOLATION_MAP = {
        "NEAREST": cv2.INTER_NEAREST,
        "LINEAR": cv2.INTER_LINEAR,
        "AREA": cv2.INTER_AREA,
        "CUBIC": cv2.INTER_CUBIC,
    }

    # 阈值类型映射
    THRESHOLD_TYPE_MAP = {
        "BINARY": cv2.THRESH_BINARY,
        "BINARY_INV": cv2.THRESH_BINARY_INV,
        "OTSU": cv2.THRESH_OTSU,
    }

    @staticmethod
    def process(pil_image: Image.Image, operation: str, **args) -> Image.Image:
        """
        执行指定的图像处理操作

        Args:
            pil_image: PIL Image 对象 (RGB 格式)
            operation: 操作名称
            **args: 操作所需的参数

        Returns:
            处理后的 PIL Image 对象

        Raises:
            ValueError: 不支持的操作或参数无效
        """
        # 转换 PIL Image 为 numpy array (RGB)
        img = np.array(pil_image)

        # 获取图像尺寸用于边界检查
        img_height, img_width = img.shape[:2]

        # 根据操作类型执行相应处理
        if operation == "crop":
            out = OpenCVProcessor._crop(img, img_width, img_height, **args)

        elif operation == "resize":
            out = OpenCVProcessor._resize(img, **args)

        elif operation == "rotate":
            out = OpenCVProcessor._rotate(img, **args)

        elif operation == "flip":
            out = OpenCVProcessor._flip(img, **args)

        elif operation == "grayscale":
            out = OpenCVProcessor._grayscale(img)

        elif operation == "blur":
            out = OpenCVProcessor._blur(img, **args)

        elif operation == "threshold":
            out = OpenCVProcessor._threshold(img, **args)

        elif operation == "canny":
            out = OpenCVProcessor._canny(img, **args)

        else:
            raise ValueError(f"Unsupported operation: {operation}")

        # 转换回 PIL Image
        if out.ndim == 2:
            return Image.fromarray(out, mode="L")
        if out.shape[2] == 4:
            return Image.fromarray(out)
        return Image.fromarray(out)

    @staticmethod
    def _crop(img: np.ndarray, img_width: int, img_height: int,
              x: int, y: int, w: int, h: int, **kwargs) -> np.ndarray:
        """裁剪图像"""
        x, y, w, h = int(x), int(y), int(w), int(h)

        # 参数验证
        if x < 0 or y < 0:
            raise ValueError(f" crop coordinates (x={x}, y={y}) cannot be negative")
        if w <= 0 or h <= 0:
            raise ValueError(f"crop dimensions (w={w}, h={h}) must be positive")
        if x + w > img_width:
            raise ValueError(f"crop region x+w={x+w} exceeds image width={img_width}")
        if y + h > img_height:
            raise ValueError(f"crop region y+h={y+h} exceeds image height={img_height}")

        return img[y:y + h, x:x + w]

    @staticmethod
    def _resize(img: np.ndarray, width: int, height: int,
                interpolation: Optional[str] = None, **kwargs) -> np.ndarray:
        """缩放图像"""
        width, height = int(width), int(height)

        if width <= 0 or height <= 0:
            raise ValueError(f"resize dimensions (width={width}, height={height}) must be positive")

        # 获取插值方式
        interp_key = (interpolation or "LINEAR").upper()
        interp_mode = OpenCVProcessor.INTERPOLATION_MAP.get(interp_key, cv2.INTER_LINEAR)

        return cv2.resize(img, (width, height), interpolation=interp_mode)

    @staticmethod
    def _rotate(img: np.ndarray, angle: float, center: tuple, scale: float = 1.0, **kwargs) -> np.ndarray:
        """旋转图像"""
        angle = float(angle)
        scale = float(scale)

        # center 格式: (x, y)
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            raise ValueError(f"center must be a tuple of (x, y), got {center}")

        center = (int(center[0]), int(center[1]))

        h, w = img.shape[:2]

        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, scale)

        # 执行旋转
        return cv2.warpAffine(img, M, (w, h))

    @staticmethod
    def _flip(img: np.ndarray, flip_code: int = 1, **kwargs) -> np.ndarray:
        """翻转图像"""
        flip_code = int(flip_code)

        if flip_code not in {-1, 0, 1}:
            raise ValueError(f"flip_code must be -1, 0, or 1, got {flip_code}")

        return cv2.flip(img, flip_code)

    @staticmethod
    def _grayscale(img: np.ndarray) -> np.ndarray:
        """转换为灰度图"""
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img  # 已经是灰度图

    @staticmethod
    def _blur(img: np.ndarray, ksize: int = 5, **kwargs) -> np.ndarray:
        """高斯模糊"""
        ksize = int(ksize)

        if ksize <= 0:
            raise ValueError(f"ksize must be positive, got {ksize}")

        # 确保是奇数
        if ksize % 2 == 0:
            ksize = ksize + 1

        return cv2.GaussianBlur(img, (ksize, ksize), 0)

    @staticmethod
    def _threshold(img: np.ndarray, thresh: int = 127, maxval: int = 255,
                   type_: Optional[str] = None, **kwargs) -> np.ndarray:
        """二值化"""
        thresh = int(thresh)
        maxval = int(maxval)

        if not (0 <= thresh <= 255):
            raise ValueError(f"thresh must be in [0, 255], got {thresh}")
        if not (0 <= maxval <= 255):
            raise ValueError(f"maxval must be in [0, 255], got {maxval}")

        # 先转为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

        # 获取阈值类型
        type_key = (type_ or "BINARY").upper()

        if type_key == "OTSU":
            _, out = cv2.threshold(gray, 0, maxval, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        elif type_key == "BINARY_INV":
            _, out = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY_INV)
        else:  # BINARY (默认)
            _, out = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY)

        return out

    @staticmethod
    def _canny(img: np.ndarray, threshold1: int, threshold2: int, **kwargs) -> np.ndarray:
        """Canny 边缘检测"""
        threshold1 = int(threshold1)
        threshold2 = int(threshold2)

        # 先转为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

        return cv2.Canny(gray, threshold1, threshold2)


# ============================================================================
# 测试函数
# ============================================================================

# 测试图像路径
TEST_IMAGE_PATH = "/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/demo.png"


def load_test_image() -> Image.Image:
    """加载测试图像"""
    return Image.open(TEST_IMAGE_PATH)


def test_crop():
    """测试裁剪操作"""
    print("Testing crop...")
    img = load_test_image()
    w, h = img.size
    # 裁剪中心区域
    crop_w, crop_h = min(200, w), min(200, h)
    x, y = (w - crop_w) // 2, (h - crop_h) // 2
    result = OpenCVProcessor.process(img, "crop", x=x, y=y, w=crop_w, h=crop_h)
    print(f"  Original size: {img.size}, Crop size: {result.size}")
    assert result.size == (crop_w, crop_h), "Crop size mismatch"
    print("  Crop test PASSED")
    return result


def test_resize():
    """测试缩放操作"""
    print("Testing resize...")
    img = load_test_image()
    result = OpenCVProcessor.process(img, "resize", width=200, height=150)
    print(f"  Original size: {img.size}, Resized to: {result.size}")
    assert result.size == (200, 150), "Resize size mismatch"
    print("  Resize test PASSED")
    return result


def test_rotate():
    """测试旋转操作"""
    print("Testing rotate...")
    img = load_test_image()
    w, h = img.size
    # 使用图像中心作为旋转中心
    center = (w // 3, h // 4)
    result = OpenCVProcessor.process(img, "rotate", angle=45, center=center)
    print(f"  Rotated by 45 degrees around center {center}, size: {result.size}")
    assert result.size == img.size, "Rotate should maintain size"
    print("  Rotate test PASSED")
    return result


def test_flip():
    """测试翻转操作"""
    print("Testing flip...")
    img = load_test_image()

    # 水平翻转
    result_h = OpenCVProcessor.process(img, "flip", flip_code=1)
    print(f"  Horizontal flip, size: {result_h.size}")

    # 垂直翻转
    result_v = OpenCVProcessor.process(img, "flip", flip_code=0)
    print(f"  Vertical flip, size: {result_v.size}")

    # 两者都翻转
    result_b = OpenCVProcessor.process(img, "flip", flip_code=-1)
    print(f"  Both flip, size: {result_b.size}")

    assert result_h.size == img.size, "Flip should maintain size"
    print("  Flip test PASSED")
    return result_h


def test_grayscale():
    """测试灰度化操作"""
    print("Testing grayscale...")
    img = load_test_image()
    result = OpenCVProcessor.process(img, "grayscale")
    print(f"  Original mode: RGB, Result mode: {result.mode}")
    assert result.mode == "L", "Grayscale should be mode 'L'"
    print("  Grayscale test PASSED")
    return result


def test_blur():
    """测试模糊操作"""
    print("Testing blur...")
    img = load_test_image()
    result = OpenCVProcessor.process(img, "blur", ksize=15)
    print(f"  Blurred with ksize=15, size: {result.size}")
    assert result.size == img.size, "Blur should maintain size"
    print("  Blur test PASSED")
    return result


def test_threshold():
    """测试二值化操作"""
    print("Testing threshold...")
    img = load_test_image()

    # 普通二值化
    result1 = OpenCVProcessor.process(img, "threshold", thresh=127, maxval=255, type_="BINARY")
    print(f"  Binary threshold, mode: {result1.mode}")
    assert result1.mode == "L", "Threshold output should be grayscale"

    # OTSU 自动阈值
    result2 = OpenCVProcessor.process(img, "threshold", type_="OTSU")
    print(f"  OTSU threshold, mode: {result2.mode}")

    # 反向二值化
    result3 = OpenCVProcessor.process(img, "threshold", thresh=127, type_="BINARY_INV")
    print(f"  Binary_INV threshold, mode: {result3.mode}")

    print("  Threshold test PASSED")
    return result1


def test_canny():
    """测试 Canny 边缘检测"""
    print("Testing canny...")
    img = load_test_image()
    result = OpenCVProcessor.process(img, "canny", threshold1=100, threshold2=200)
    print(f"  Canny edge detection, mode: {result.mode}, size: {result.size}")
    assert result.mode == "L", "Canny output should be grayscale"
    assert result.size == img.size, "Canny should maintain size"
    print("  Canny test PASSED")
    return result


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("OpenCVProcessor Test Suite")
    print("=" * 60)

    results = {}

    try:
        results["crop"] = test_crop()
        print()
    except Exception as e:
        print(f"  Crop test FAILED: {e}")
        print()

    try:
        results["resize"] = test_resize()
        print()
    except Exception as e:
        print(f"  Resize test FAILED: {e}")
        print()

    try:
        results["rotate"] = test_rotate()
        print()
    except Exception as e:
        print(f"  Rotate test FAILED: {e}")
        print()

    try:
        results["flip"] = test_flip()
        print()
    except Exception as e:
        print(f"  Flip test FAILED: {e}")
        print()

    try:
        results["grayscale"] = test_grayscale()
        print()
    except Exception as e:
        print(f"  Grayscale test FAILED: {e}")
        print()

    try:
        results["blur"] = test_blur()
        print()
    except Exception as e:
        print(f"  Blur test FAILED: {e}")
        print()

    try:
        results["threshold"] = test_threshold()
        print()
    except Exception as e:
        print(f"  Threshold test FAILED: {e}")
        print()

    try:
        results["canny"] = test_canny()
        print()
    except Exception as e:
        print(f"  Canny test FAILED: {e}")
        print()

    print("=" * 60)
    print(f"Tests completed. Passed: {len(results)}/8")
    print("=" * 60)

    return results


def save_test_results(results: Dict[str, Image.Image], output_dir: str = None):
    """保存测试结果图像"""
    import os

    # 默认保存到当前脚本所在目录的 test_output 文件夹
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "test_output")

    os.makedirs(output_dir, exist_ok=True)

    for name, img in results.items():
        path = os.path.join(output_dir, f"{name}.png")
        img.save(path)
        print(f"Saved: {path}")


def image_to_data_url(pil_image: Image.Image) -> str:
    """将 PIL Image 转换为 data URL 格式"""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


if __name__ == "__main__":
    # 运行所有测试
    results = run_all_tests()

    # 保存结果
    if results:
        print("\nSaving test results...")
        save_test_results(results)

        # 测试 data URL 转换
        print("\nTesting data URL conversion...")
        for name, img in results.items():
            url = image_to_data_url(img)
            print(f"  {name}: data URL length = {len(url)} chars")
