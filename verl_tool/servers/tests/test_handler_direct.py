#!/usr/bin/env python
"""直接测试 handler，绕过服务端"""
import base64
import json
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m")

from verl_tool.servers.tools.extract_chart_data import ExtractChartDataTool

IMAGE_PATH = '/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_duo_final/ChartQA_2000/images/train/train_0.png'

def encode_image(path):
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

def main():
    # 创建 handler 实例
    tool = ExtractChartDataTool()
    print(f"[DEBUG] tool_type: {tool.tool_type}")
    print(f"[DEBUG] available_models: {tool.available_models}")
    
    # 准备测试数据
    image_data_url = encode_image(IMAGE_PATH)
    
    action = json.dumps({
        "name": "extract_chart_data",
        "arguments": {
            "chart_type": "bar",
            "model": "extract_chart_data-1"
        }
    })
    action = f"<tool_call>{action}</tool_call>"
    
    # 加载配置
    url_cfg = json.loads(Path("/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/verl_tool/servers/tools/url.json").read_text())
    pricing_cfg = json.loads(Path("/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/model_tool_pricing.json").read_text())
    
    services = list(url_cfg.keys())[:2]
    model_mapping = {
        "extract_chart_data": {
            "extract_chart_data-1": services[0],
            "extract_chart_data-2": services[1] if len(services) > 1 else services[0],
        }
    }
    tool_pricing = {
        s: {
            "input_per_million": pricing_cfg["model_tool_pricing"][s]["input_tokens_per_million"],
            "output_per_million": pricing_cfg["model_tool_pricing"][s]["output_tokens_per_million"],
        }
        for s in services
    }
    
    extra_field = {
        "images": [image_data_url],
        "question": "Extract all bars with their labels and values.",
        "model_mapping": model_mapping,
        "tool_pricing": tool_pricing,
    }
    
    print(f"[DEBUG] model_mapping: {model_mapping}")
    print(f"[DEBUG] tool_pricing: {tool_pricing}")
    print(f"[DEBUG] action: {action}")
    
    # 直接调用 handler
    print("\n[DEBUG] Calling conduct_action...")
    observation, done, valid = tool.conduct_action("test-001", action, extra_field)
    
    print(f"\n[RESULT]")
    print(f"  observation: {json.dumps(observation, indent=2, ensure_ascii=False)}")
    print(f"  done: {done}")
    print(f"  valid: {valid}")

if __name__ == "__main__":
    main()
