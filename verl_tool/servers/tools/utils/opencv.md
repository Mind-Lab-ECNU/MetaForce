# OpenCV 工具集成指南

本文档详细描述了如何将 OpenCV 功能集成到 `multimodal_processor_tool.py` 中，作为多模态工具集的一个新组件。

## 1. 工具概述

**工具名称**: `OpenCV`  
**功能描述**: 提供一系列基于 OpenCV 的图像处理原子操作，用于辅助多模态模型进行图像预处理、增强或特征提取。  
**调用方式**: 通过 `MultimodalProcessorTool` 的标准 `tool_call` 接口调用，指定 `operation` 参数来选择具体的子操作。

## 2. OpenCV Python 依赖与安装选择

OpenCV 在 Python 中的入口模块名是 `cv2`（代码里写 `import cv2`），常见的安装方式是直接使用 PyPI 上的预编译 wheel 包，不需要你们自行编译 OpenCV。

### 2.1 推荐安装方式（四选一，且同环境只能选一个）

这 4 个包**共享同一个命名空间 `cv2`**，不存在插件式叠加机制；同一个 Python 环境里不要同时安装多个，否则容易出现导入/符号冲突。参考：PyPI 项目说明与 OpenCV 官方 pip 安装文档（[PyPI opencv-python](https://pypi.org/project/opencv-python/)；[OpenCV pip 安装](https://docs.opencv.org/4.x/db/dd1/tutorial_py_pip_install.html)）。

- `opencv-python`：主模块（桌面环境常用）
- `opencv-contrib-python`：主模块 + contrib/extra 模块
- `opencv-python-headless`：无 GUI/窗口后端依赖（服务器/容器/CI 推荐）
- `opencv-contrib-python-headless`：contrib + headless

### 2.2 对你们工具端的建议

你们的多模态工具更像服务端组件，不需要 `cv2.imshow` 等 GUI 能力，建议默认优先使用 `opencv-python-headless`（如果确实需要 contrib 模块再切换到 `opencv-contrib-python-headless`）。

## 3. 工具协议（tool_call）与返回约定

### 3.1 tool_call 结构

OpenCV 工具作为 `MultimodalProcessorTool` 的一个子工具，调用时的基本形状如下：

```json
{
  "name": "OpenCV",
  "arguments": {
    "operation": "crop",
    "image_index": 1
  }
}
```

其中：

- `name`: 固定为 `"OpenCV"`
- `arguments.operation`: 子操作名称（字符串枚举）
- `arguments.image_index`: 输入图片索引（从 1 开始）
- `arguments` 里其余字段由不同 `operation` 决定（例如裁剪需要 x/y/w/h）

如果你们的调用方需要 `<tool_call>...</tool_call>` 包裹，可以沿用现有工具的格式：

```text
<tool_call>{"name":"OpenCV","arguments":{"operation":"crop","image_index":1,"x":10,"y":10,"w":100,"h":80}}</tool_call>
```

### 3.2 返回（observation）约定

建议返回结构与现有工具保持一致：

- `obs`：人类可读文本，包含 operation 执行状态或错误原因
- `tool` / `model`：均可填 `"OpenCV"`（与现有工具风格保持一致）
- `image`：可选。若 operation 产生图像输出，建议返回 `data:image/png;base64,...` 形式，便于后续工具链继续使用

## 4. 可执行子操作 (Sub-operations) 详解

OpenCV 工具支持以下子操作。每个操作都需要通过 `arguments` 字典传递相应的参数。所有操作都必须包含 `image_index` 参数。

### 4.1 基础几何变换 (Geometric Transformations)

#### `crop` (裁剪)
裁剪出图像的特定矩形区域 (Region of Interest)。
*   **参数**:
    *   `image_index` (int): 目标图片在环境中的索引 (从1开始)。
    *   `operation` (str): 固定为 `"crop"`。
    *   `x` (int): 裁剪区域左上角的 x 坐标。
    *   `y` (int): 裁剪区域左上角的 y 坐标。
    *   `w` (int): 裁剪区域的宽度。
    *   `h` (int): 裁剪区域的高度。
*   **行为约定**:
    *   坐标系原点在左上角，`x` 向右、`y` 向下。
    *   `w/h` 必须大于 0。
    *   建议采用严格模式：当裁剪区域越界时，直接返回 `invalid_parameters` 错误，而不是自动截断。
*   **常见错误**:
    *   `x/y/w/h` 不是整数或为负数/零。
    *   `x + w > image_width` 或 `y + h > image_height` 导致越界。
*   **示例**:
    ```json
    {
      "name": "OpenCV",
      "arguments": {
        "operation": "crop",
        "image_index": 1,
        "x": 100,
        "y": 100,
        "w": 200,
        "h": 150
      }
    }
    ```

#### `resize` (缩放)
改变图像的分辨率。
*   **参数**:
    *   `image_index` (int): 目标图片索引。
    *   `operation` (str): 固定为 `"resize"`。
    *   `width` (int): 目标宽度。
    *   `height` (int): 目标高度。
    *   `interpolation` (str, 可选): 插值方式，默认 `"LINEAR"`。
        *   `"NEAREST"`：最近邻（最快、锯齿明显）
        *   `"LINEAR"`：双线性（默认，通用）
        *   `"AREA"`：区域插值（缩小时通常更好）
        *   `"CUBIC"`：双三次（放大时更平滑但更慢）
*   **行为约定**:
    *   `width/height` 必须大于 0。
    *   输出尺寸固定为 `(width, height)`，不保持原始宽高比；若要保持比例，应由上层先计算好目标尺寸。
*   **常见错误**:
    *   `width/height` 非法（非整数、<=0）。
*   **示例**:
    ```json
    {
      "name": "OpenCV",
      "arguments": {
        "operation": "resize",
        "image_index": 1,
        "width": 800,
        "height": 600
      }
    }
    ```

#### `rotate` (旋转)
围绕指定中心点旋转指定角度。
*   **参数**:
    *   `image_index` (int): 目标图片索引。
    *   `operation` (str): 固定为 `"rotate"`。
    *   `angle` (float): 旋转角度（正值为逆时针，负值为顺时针）。
    *   `center` (tuple): 旋转中心点坐标，格式为 `(x, y)`。
    *   `scale` (float, 可选): 缩放比例，默认为 1.0。
*   **行为约定**:
    *   必须指定 `center` 参数，不再使用默认的图像中心。
    *   输出尺寸保持与输入一致（即旋转后仍输出原图大小），超出画布部分会被裁掉。
    *   画布外填充值使用黑色（OpenCV 默认行为）。
*   **常见错误**:
    *   `angle/scale` 无法转换为数值。
*   **示例**:
    ```json
    {
      "name": "OpenCV",
      "arguments": {
        "operation": "rotate",
        "image_index": 1,
        "angle": 45
      }
    }
    ```

#### `flip` (翻转)
沿水平或垂直轴翻转图像。
*   **参数**:
    *   `image_index` (int): 目标图片索引。
    *   `operation` (str): 固定为 `"flip"`。
    *   `flip_code` (int): 翻转模式。
        *   `1`: 水平翻转 (左右镜像)。
        *   `0`: 垂直翻转 (上下镜像)。
        *   `-1`: 同时水平和垂直翻转。
*   **行为约定**:
    *   输出尺寸与输入一致。
*   **常见错误**:
    *   `flip_code` 不在 `{-1, 0, 1}`，建议返回 `invalid_parameters`。
*   **示例**:
    ```json
    {
      "name": "OpenCV",
      "arguments": {
        "operation": "flip",
        "image_index": 1,
        "flip_code": 1
      }
    }
    ```

### 4.2 图像增强与处理 (Enhancement & Processing)

#### `grayscale` (灰度化)
将彩色图像转换为灰度图像。
*   **参数**:
    *   `image_index` (int): 目标图片索引。
    *   `operation` (str): 固定为 `"grayscale"`。
*   **行为约定**:
    *   若输入为 RGB，则执行 `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`。
    *   若输入本身已是单通道灰度图，直接返回（幂等）。
    *   输出为单通道（H×W），在返回给工具框架时仍可编码为 PNG。
*   **示例**:
    ```json
    {
      "name": "OpenCV",
      "arguments": {
        "operation": "grayscale",
        "image_index": 1
      }
    }
    ```

#### `blur` (模糊/去噪)
使用高斯模糊对图像进行平滑处理，常用于去除噪声。
*   **参数**:
    *   `image_index` (int): 目标图片索引。
    *   `operation` (str): 固定为 `"blur"`。
    *   `ksize` (int): 高斯核大小，必须是奇数 (如 3, 5, 7, 9)。值越大越模糊。
*   **行为约定**:
    *   建议严格要求 `ksize` 为正奇数；若传入偶数，可以选择：
        *   严格模式：直接报错；或
        *   容错模式：自动调整到最近的下一个奇数（例如 4→5）。
*   **常见错误**:
    *   `ksize` 非法（<=0、无法转为整数）。
*   **示例**:
    ```json
    {
      "name": "OpenCV",
      "arguments": {
        "operation": "blur",
        "image_index": 1,
        "ksize": 5
      }
    }
    ```

#### `threshold` (二值化)
将图像转换为黑白两色。如果输入是彩色图，工具内部应先自动转为灰度图。
*   **参数**:
    *   `image_index` (int): 目标图片索引。
    *   `operation` (str): 固定为 `"threshold"`。
    *   `thresh` (int): 阈值 (0-255)。像素值大于此值设为 maxval，否则设为0。
    *   `maxval` (int, 可选): 最大值，默认为 255。
    *   `type` (str, 可选): 阈值类型，默认为 `"BINARY"`。
        *   `"BINARY"`: 二值化。
        *   `"BINARY_INV"`: 反向二值化。
        *   `"OTSU"`: 使用 Otsu 算法自动计算阈值 (此时 thresh 参数被忽略)。
*   **行为约定**:
    *   内部先确保输入为灰度图（RGB → Gray）。
    *   输出为单通道二值图，像素值通常为 `0` 或 `maxval`。
    *   `"OTSU"` 适用于前景/背景分布较明显的场景，`thresh` 将被忽略。
*   **常见错误**:
    *   `thresh/maxval` 不在 0-255 范围内或无法转为整数。
    *   `type` 非法时建议回退到 `"BINARY"` 或直接报错（两者择一并保持一致）。
*   **示例**:
    ```json
    {
      "name": "OpenCV",
      "arguments": {
        "operation": "threshold",
        "image_index": 1,
        "thresh": 127,
        "type": "BINARY"
      }
    }
    ```

#### `canny` (边缘检测)
使用 Canny 算法检测图像边缘。
*   **参数**:
    *   `image_index` (int): 目标图片索引。
    *   `operation` (str): 固定为 `"canny"`。
    *   `threshold1` (int): 第一个滞后性阈值 (低阈值)。
    *   `threshold2` (int): 第二个滞后性阈值 (高阈值)。
*   **行为约定**:
    *   Canny 常用于单通道输入；若输入为 RGB，建议先转灰度再执行 Canny，以获得稳定结果。
    *   输出为单通道边缘图（边缘像素为 255，非边缘为 0）。
*   **常见错误**:
    *   `threshold1/threshold2` 无法转为整数或阈值关系不合理（通常 `threshold2 > threshold1`）。
*   **示例**:
    ```json
    {
      "name": "OpenCV",
      "arguments": {
        "operation": "canny",
        "image_index": 1,
        "threshold1": 100,
        "threshold2": 200
      }
    }
    ```

## 5. 实现建议：用类封装并直接调用 cv2

本工具的“算法能力”来自 OpenCV（`cv2`）本身，因此实现侧最重要的是把逻辑组织清晰：建议将 OpenCV 子操作封装到独立类中，让 `MultimodalProcessorTool` 只做协议解析与调度。

### 5.1 设计目标

- 单一职责：`MultimodalProcessorTool` 负责解析 `tool_call`、取图/存图；`OpenCVProcessor` 负责图像处理。
- 直接调库：`OpenCVProcessor` 的每个方法直接调用 `cv2` 的对应函数（`resize/flip/cvtColor/...`）。
- 统一数据格式：内部统一用 `numpy.ndarray`（RGB）做处理，只在输入/输出边界做 PIL↔NumPy 转换。

### 5.2 OpenCVProcessor 伪代码（可直接映射到实现）

```python
class OpenCVProcessor:
    @staticmethod
    def process(pil_image, operation: str, **args):
        import numpy as np
        import cv2
        from PIL import Image

        img = np.array(pil_image)  # RGB

        if operation == "crop":
            x, y, w, h = int(args["x"]), int(args["y"]), int(args["w"]), int(args["h"])
            out = img[y:y + h, x:x + w]

        elif operation == "resize":
            width, height = int(args["width"]), int(args["height"])
            interpolation = (args.get("interpolation") or "LINEAR").upper()
            interp_map = {
                "NEAREST": cv2.INTER_NEAREST,
                "LINEAR": cv2.INTER_LINEAR,
                "AREA": cv2.INTER_AREA,
                "CUBIC": cv2.INTER_CUBIC,
            }
            out = cv2.resize(img, (width, height), interpolation=interp_map.get(interpolation, cv2.INTER_LINEAR))

        elif operation == "rotate":
            angle = float(args["angle"])
            scale = float(args.get("scale", 1.0))
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
            out = cv2.warpAffine(img, M, (w, h))

        elif operation == "flip":
            flip_code = int(args.get("flip_code", 1))
            out = cv2.flip(img, flip_code)

        elif operation == "grayscale":
            out = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

        elif operation == "blur":
            ksize = int(args.get("ksize", 5))
            out = cv2.GaussianBlur(img, (ksize, ksize), 0)

        elif operation == "threshold":
            thresh = int(args.get("thresh", 127))
            maxval = int(args.get("maxval", 255))
            t = (args.get("type") or "BINARY").upper()
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
            if t == "OTSU":
                _, out = cv2.threshold(gray, 0, maxval, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            elif t == "BINARY_INV":
                _, out = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY_INV)
            else:
                _, out = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY)

        elif operation == "canny":
            t1, t2 = int(args["threshold1"]), int(args["threshold2"])
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
            out = cv2.Canny(gray, t1, t2)

        else:
            raise ValueError(f"Unsupported operation: {operation}")

        if out.ndim == 2:
            return Image.fromarray(out, mode="L")
        return Image.fromarray(out, mode="RGB")
```

### 5.3 和你们工具框架的衔接点

在 `MultimodalProcessorTool` 中，OpenCV 分支建议只做三件事：

1. 校验 `operation/image_index` 和该 operation 的必填参数
2. 从 env 取图并确保转换成 PIL Image
3. 调用 `OpenCVProcessor.process(...)` 得到 PIL Image，复用现有 `_image_to_data_url` 把结果回传给 agent

相关的图片编码/回传工具函数可复用现有实现：[_image_to_data_url](verl_tool/servers/tools/multimodal_processor_tool.py#L76-L115)。

## 6. 集成步骤

请按照以下步骤将 OpenCV 工具集成到 `multimodal_processor_tool.py` 文件中。

### 步骤 1: 导入依赖
在文件头部的导入区域添加：
```python
import cv2
import numpy as np
from PIL import Image
```
*(注意：请确保运行环境已安装 `opencv-python` 和 `numpy`)*

### 步骤 2: 定义工具名称常量
在 `MultimodalProcessorTool` 类定义中，添加 OpenCV 相关的常量：

```python
class MultimodalProcessorTool(BaseTool):
    # ... 其他代码 ...
    
    # 在 valid_mcp_func_names 列表中添加 "OpenCV"
    valid_mcp_func_names = [
        # ... 现有列表 ...
        "OpenCV",
    ]
    
    # 定义常量
    opencv_tool_name = "OpenCV"
    
    # ... 其他代码 ...
```

### 步骤 3: 更新使用说明
在 `get_usage_inst` 方法中，添加 OpenCV 的 usage 字符串：

```python
    def get_usage_inst(self) -> str:
        return (
            "multimodal_processor_tool:\n"
            # ... 现有说明 ...
            "OpenCV 示例：<tool_call>{\"name\": \"OpenCV\", "
            "\"arguments\": {\"operation\": \"crop\", \"x\": 100, \"y\": 100, \"w\": 200, \"h\": 200, \"image_index\": 1}}</tool_call>\n"
        )
```

### 步骤 4: 实现核心逻辑
在 `_conduct_action_async` 方法中，添加对 `OpenCV` 工具调用的处理逻辑。

**代码逻辑伪代码**:

```python
        if model_name == self.opencv_tool_name:
            try:
                # 1) 解析与校验公共参数
                operation = arguments.get("operation")
                image_index = arguments.get("image_index")
                if operation is None or image_index is None:
                    # 返回 missing_parameters
                    pass

                # 2) 从 env 取图，转换为 PIL Image
                # image_obj 可能是 Path / data_url / base64 / PIL 等
                image_obj = env["images"][int(image_index) - 1]
                pil_image = ...  # 统一转换成 PIL.Image.Image

                # 3) 委托给 OpenCVProcessor（内部直接调 cv2）
                result_pil = OpenCVProcessor.process(pil_image, **arguments)

                # 4) 结果回传：PIL -> data_url
                image_url = _image_to_data_url(result_pil)
                observation = {
                    "obs": f"[Tool: OpenCV] Operation '{operation}' completed.",
                    "tool": self.opencv_tool_name,
                    "model": self.opencv_tool_name,
                    "image": image_url,
                }
                return _finalize(observation, False, True)

            except ValueError as exc:
                # 返回 invalid_parameters
                pass
            except Exception as exc:
                # 返回 internal_error
                pass
```

## 7. 注意事项

1.  **坐标系统**: OpenCV 中的图像坐标系原点 (0,0) 在左上角。
2.  **通道顺序**: OpenCV 默认读取图片为 BGR 格式，而 PIL 默认为 RGB。在 `PIL -> Numpy` 转换时，`np.array(pil_image)` 会得到 RGB 顺序的数组。调用 `cv2` 函数时（如 `cvtColor`）需注意保持颜色空间一致性。本指南建议统一在 RGB 空间操作，只在特定需要灰度转换时使用 `cv2.COLOR_RGB2GRAY`。
3.  **异常处理**: 图像处理操作容易因参数越界（如裁剪超出范围）抛出异常，务必包裹在 `try-except` 块中，并返回清晰的错误信息给模型。
4.  **环境兼容性**: 确保服务器环境中安装了 `libgl1` 等 OpenCV 系统依赖库。
