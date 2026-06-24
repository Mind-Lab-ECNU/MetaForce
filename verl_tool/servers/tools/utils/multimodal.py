import requests
import base64
import json

def analyze_image_with_requests(image_path: str, question: str):
    """使用 requests 调用 Qwen3-VL"""

    # API 配置 - 修正 URL
    base_url = "https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-c6264b38-46a1-4bb6-a3a7-1db39453f6f8/vscode/002d3d72-c1da-4d7e-93ef-ff71f371ee67/a692fc76-719c-43ae-974b-6c64db6e6adc/proxy/30005"
    url = f"{base_url}/v1/chat/completions"  # 正确的端点

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer token-abc123"
    }

    # 读取并编码图片
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()

    # 构建请求数据
    data = {
        "model": "Qwen3-VL-235B-A22B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    # 发送请求
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)

        # 解析响应
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"请求失败: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"网络请求错误: {str(e)}")

# 使用示例
if __name__ == "__main__":
    try:
        result = analyze_image_with_requests(
            image_path="/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/demo.png",
            question="提取柱状图中的每一个柱子的值和对应的柱子类别名称,整理成表格形式"
        )
        print("=" * 50)
        print("分析结果:")
        print("=" * 50)
        print(result)
    except Exception as e:
        print(f"错误: {e}")