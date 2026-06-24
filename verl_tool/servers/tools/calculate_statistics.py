import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from .base import BaseTool, register_tool


def _load_url_mapping() -> Dict[str, str]:
    url_path = Path(__file__).with_name("url.json")
    if not url_path.exists():
        return {}
    try:
        return json.loads(url_path.read_text())
    except Exception:
        return {}


def _to_data_url(image_path: Path) -> str:
    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"


@register_tool
class CalculateStatisticsTool(BaseTool):
    """
    单步同步版本（可由 ray_utils 包装），负责把 calculate_statistics 请求转发到多模态模型。
    """

    tool_type = "calculate_statistics"

    def __init__(self, num_workers: int = 8, request_timeout: int = 60):
        super().__init__(num_workers)
        self.request_timeout = request_timeout
        self.url_mapping = _load_url_mapping()
        self.available_models = list(self.url_mapping.keys())
        self.tool_pricing: Dict[str, Any] = {}

    def get_usage_inst(self) -> str:
        return (
            "calculate_statistics: 计算数据的统计指标，如最大值、最小值、中位数、均值、方差、标准差等。\n"
            "action 示例：<tool_call>"
            '{"name": "calculate_statistics", '
            '"arguments": {"statistics_type": "max", "model": "calculate_statistics-1"}}</tool_call> '
            "对于具体数据序列，可指定 target_label 或 target_color；return_info 可选 'value'(默认)/'index'/'label'/'color'。"
        )

    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        try:
            payload_str = action
            if "<tool_call>" in action and "</tool_call>" in action:
                payload_str = action.split("<tool_call>")[1].split("</tool_call>")[0]
            payload = json.loads(payload_str)
            if payload.get("name") != "calculate_statistics":
                return {}, False
            arguments = payload.get("arguments", {})
            if not isinstance(arguments, dict):
                return {}, False
            if "statistics_type" not in arguments or "model" not in arguments:
                return {}, False
            payload["arguments"] = arguments
            return payload, True
        except Exception:
            return {}, False

    def _build_prompt(self, question: Optional[str], arguments: Dict[str, Any]) -> str:
        """
        Build detailed, context-aware prompt for statistical analysis.
        Dynamically generates instructions based on user-provided parameters.
        """
        
        # Extract parameters
        statistics_type = arguments.get('statistics_type', 'unknown')
        target_label = arguments.get('target_label')
        target_color = arguments.get('target_color')
        target_index = arguments.get('target_index')
        return_info = arguments.get('return_info', 'value')
        
        # Base system instruction
        base_instruction = (
            "You are a precise statistical calculation assistant specializing in quantitative analysis "
            "of data visualizations. Your task is to accurately calculate statistical metrics from charts "
            "and return the requested information."
        )
        
        # Detailed instructions and formulas for each statistics type
        statistics_instructions = {
            'max': (
                "MAXIMUM (MAX): Find the largest value in the dataset.\n"
                "- Calculation: max(dataset) = the data point with the highest y-value\n"
                "- If multiple points share the maximum value, return information based on return_info parameter"
            ),
            'min': (
                "MINIMUM (MIN): Find the smallest value in the dataset.\n"
                "- Calculation: min(dataset) = the data point with the lowest y-value\n"
                "- If multiple points share the minimum value, return information based on return_info parameter"
            ),
            'median': (
                "MEDIAN: The middle value when data is sorted.\n"
                "- Calculation:\n"
                "  · Odd number of points: median = middle value after sorting\n"
                "  · Even number of points: median = average of the two middle values\n"
                "- Median is resistant to outliers"
            ),
            'low_median': (
                "LOW MEDIAN: When dataset has an even number of points, take the lower of the two middle values.\n"
                "- Calculation:\n"
                "  · Odd number of points: same as regular median\n"
                "  · Even number of points: take the value at position n/2 after sorting"
            ),
            'high_median': (
                "HIGH MEDIAN: When dataset has an even number of points, take the higher of the two middle values.\n"
                "- Calculation:\n"
                "  · Odd number of points: same as regular median\n"
                "  · Even number of points: take the value at position (n/2 + 1) after sorting"
            ),
            'mean': (
                "MEAN (Average): The arithmetic average of all data points.\n"
                "- Formula: mean = Σ(all y-values) / total number of points\n"
                "- Also known as expected value, reflects central tendency\n"
                "- Note: Mean is sensitive to outliers"
            ),
            'variance': (
                "VARIANCE: Measures the spread of data points relative to the mean.\n"
                "- Calculation steps:\n"
                "  1. Calculate mean: mean_y = mean(all y-values)\n"
                "  2. Calculate variance: Var(Y) = mean((y[i] - mean_y)²)\n"
                "- Higher variance indicates more volatile data\n"
                "- Units are squared units of the original data"
            ),
            'std': (
                "STANDARD DEVIATION (STD): Square root of variance, measures typical deviation from mean.\n"
                "- Calculation steps:\n"
                "  1. Calculate variance: Var(Y) = mean((y[i] - mean_y)²)\n"
                "  2. Take square root: σ = sqrt(Var(Y))\n"
                "- Standard deviation has the same units as original data\n"
                "- Optional: Also report coefficient of variation CV = σ/mean (if mean ≠ 0)"
            ),
            'percentage': (
                "PERCENTAGE: Calculate the proportion of data points meeting specific conditions.\n"
                "- Based on filter conditions (label/color/index), calculate the percentage of qualifying points\n"
                "- Formula: percentage = (number of qualifying points / total points) × 100%\n"
                "- If no filter conditions, report distribution across all categories"
            )
        }
        
        # Get detailed description for the statistics type
        analysis_detail = statistics_instructions.get(
            statistics_type,
            f"Calculate the {statistics_type} statistical metric. Use appropriate quantitative methods "
            f"and provide clear numerical results."
        )
        
        # Build dynamic context based on user filter parameters
        context_parts = []
        
        # Target data series filtering
        filter_conditions = []
        if target_label:
            filter_conditions.append(f"legend label is '{target_label}'")
        if target_color:
            filter_conditions.append(f"color is '{target_color}'")
        if target_index is not None:
            filter_conditions.append(f"index position is {target_index}")
        
        if filter_conditions:
            context_parts.append(
                f"DATA FILTER: Analyze only data series or points where {' AND '.join(filter_conditions)}.\n"
                f"If the chart contains multiple data series, first identify the series matching the criteria, "
                f"then perform statistical calculations on that series only."
            )
        else:
            context_parts.append(
                "DATA SCOPE: Analyze ALL visible data series in the chart.\n"
                "If the chart contains multiple data series (e.g., multiple curves, bar groups), "
                "calculate statistics for EACH series separately."
            )
        
        # Return information type instructions
        return_info_instructions = {
            'value': (
                "RETURN TYPE: Numerical Value (VALUE)\n"
                "- Return the calculated statistical value itself\n"
                "- Include units if axis labels provide them\n"
                "- Maintain at least 2 decimal places"
            ),
            'index': (
                "RETURN TYPE: Index Position (INDEX)\n"
                "- Return the index position(s) of data point(s) achieving this statistical value\n"
                "- If multiple points share the value, list all indices\n"
                "- Index starts from 0 or 1 (based on chart display convention)"
            ),
            'label': (
                "RETURN TYPE: Label Name (LABEL)\n"
                "- Return the label name of the data point or series achieving this statistical value\n"
                "- Label source: legend, x-axis labels, or data point annotations\n"
                "- If label is ambiguous, provide color or other identifying information"
            ),
            'color': (
                "RETURN TYPE: Color (COLOR)\n"
                "- Return the color of the data series or point achieving this statistical value\n"
                "- Use common color names (e.g., red, blue) or RGB/HEX values\n"
                "- If multiple series share the same color, additionally specify distinguishing features"
            )
        }
        
        context_parts.append(
            return_info_instructions.get(
                return_info,
                f"RETURN TYPE: {return_info}\nReturn information according to this type."
            )
        )
        
        context_section = "\n".join(context_parts)
        
        # Output format requirements
        output_format = (
            "OUTPUT FORMAT REQUIREMENTS:\n"
            "1. DATA SERIES IDENTIFICATION:\n"
            "   - List all relevant data series in the chart\n"
            "   - Provide clear identifiers (legend names, colors, line styles, etc.)\n"
            "2. STATISTICAL RESULTS:\n"
            "   - Clearly state which statistical metric was calculated\n"
            "   - Return based on return_info type:\n"
            "     · value: Numerical result (with units, at least 2 decimals)\n"
            "     · index: Index position (e.g., 'index=5' or 'indices=[3, 7, 9]')\n"
            "     · label: Label name (e.g., 'Series A' or 'point at x=10')\n"
            "     · color: Color description (e.g., 'red' or '#FF0000')\n"
            "3. MULTI-SERIES HANDLING:\n"
            "   - If no filter conditions and multiple series exist, report results for EACH series\n"
            "   - Use clear structure (sections, tables, or lists) to organize multi-series results\n"
            "4. CALCULATION NOTES (optional):\n"
            "   - If helpful, briefly explain the calculation method used\n"
            "   - For complex cases (e.g., multiple maximum points), explain handling approach\n"
            "5. DATA COMPLETENESS:\n"
            "   - If chart data cannot be fully read, state best-effort analysis with brief note\n"
            "   - If filter conditions match no data, explicitly state this"
        )
        
        # User question section
        question_section = ""
        if question:
            question_section = f"\nUSER'S SPECIFIC QUESTION:\n{question}\n"
        
        # Assemble complete prompt
        full_prompt = (
            f"{base_instruction}\n\n"
            f"STATISTICS TYPE: {statistics_type.upper()}\n\n"
            f"{analysis_detail}\n\n"
            f"CONTEXT & SCOPE:\n{context_section}\n\n"
            f"{output_format}"
            f"{question_section}"
        )
        
        return full_prompt.strip()

    def _prepare_image(self, arguments: Dict[str, Any], extra_field: Dict[str, Any]) -> Optional[str]:
        images = extra_field.get("images") or []
        if isinstance(images, list) and images:
            return images[0]
        image_path = arguments.get("image_path")
        if image_path and Path(image_path).exists():
            return _to_data_url(Path(image_path))
        return None

    def _estimate_cost(
        self,
        model_variant: str,
        usage: Dict[str, Any],
        extra_field: Dict[str, Any],
        service_name: Optional[str] = None,
    ) -> float:
        pricing_map = extra_field.get("tool_pricing") or {}
        pricing = None
        if service_name and service_name in pricing_map:
            pricing = pricing_map.get(service_name)
        if pricing is None:
            pricing = pricing_map.get(model_variant) or self.tool_pricing.get(model_variant)
        if not pricing or not usage:
            return 0.0
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        input_price = pricing.get("input_per_million", pricing.get("input_tokens_per_million", 0.0))
        output_price = pricing.get("output_per_million", pricing.get("output_tokens_per_million", 0.0))
        return (
            prompt_tokens * input_price
            + completion_tokens * output_price
        ) 

    def _resolve_service_name(self, model_variant: str, extra_field: Dict[str, Any]) -> str:
        mapping = extra_field.get("model_mapping") or {}
        if isinstance(mapping, dict) and "calculate_statistics" in mapping:
            mapping = mapping["calculate_statistics"]
        if isinstance(mapping, dict) and model_variant in mapping:
            return mapping[model_variant]
        if isinstance(mapping, list) and mapping:
            try:
                idx = int(str(model_variant).split("-")[-1]) - 1
                if 0 <= idx < len(mapping):
                    return mapping[idx]
            except Exception:
                pass
        # Only allow numeric-suffix fallback to url order when suffix exists
        parts = str(model_variant).rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            idx = int(parts[1]) - 1
            if 0 <= idx < len(self.available_models):
                return self.available_models[idx]
        raise ValueError(f"Unknown model variant: {model_variant}")

    def _call_model(
        self,
        model_variant: str,
        image_data_url: str,
        prompt: str,
        extra_field: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        service_name = self._resolve_service_name(model_variant, extra_field)
        base_url = self.url_mapping.get(service_name)
        if not base_url:
            raise ValueError(f"Model URL for {service_name} not found.")

        url = f"{base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer token-abc123"}
        payload = {
            "model": service_name,
            "messages": [
                {"role": "system", "content": "You are a precise statistical calculation assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 4096,
        }

        start = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=self.request_timeout)
        latency = time.time() - start
        response.raise_for_status()
        data = response.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {}) or {}
        estimated_cost = self._estimate_cost(model_variant, usage, extra_field, service_name=service_name)
        metadata = {
            "latency": latency,
            "cost": estimated_cost,
            "usage": usage,
            "model": model_variant,
            "service_name": service_name,
        }
        return content, metadata

    def conduct_action(
        self, trajectory_id: str, action: str, extra_field: Dict[str, Any]
    ) -> Tuple[Any, bool, bool]:
        parsed, is_valid = self.parse_action(action)
        if not is_valid:
            observation = {
                "obs": "Invalid tool_call format for calculate_statistics.",
                "invalid_reason": "parse_failed",
            }
            return observation, False, False

        arguments = parsed["arguments"]
        model_variant = arguments.get("model", "calculate_statistics-1")
        image_data_url = self._prepare_image(arguments, extra_field)
        if not image_data_url:
            observation = {
                "obs": "No image provided for calculate_statistics.",
                "invalid_reason": "missing_image",
            }
            return observation, False, False

        question = extra_field.get("question") or extra_field.get("prompt")
        prompt = self._build_prompt(question, arguments)

        try:
            content, meta = self._call_model(model_variant, image_data_url, prompt, extra_field)
            print(
                f"[calculate_statistics] model_variant={model_variant}, service_name={meta.get('service_name')}, "
                f"pricing={(extra_field.get('tool_pricing') or {}).get(model_variant) or self.tool_pricing.get(model_variant)}, "
                f"arguments={arguments}, latency={meta.get('latency')}, cost={meta.get('cost')}"
            )
            observation = {
                "obs": content,
                "latency": meta["latency"],
                "cost": meta["cost"],
                "usage": meta.get("usage", {}),
                "tool": "calculate_statistics",
                "model": meta["model"],
                "model_variant": model_variant
            }
            return observation, False, True
        except ValueError as exc:
            observation = {
                "obs": f"calculate_statistics failed: {exc}",
                "invalid_reason": "unknown_model",
            }
            return observation, False, False
        except Exception as exc:
            observation = {
                "obs": f"calculate_statistics failed: {exc}",
                "invalid_reason": "request_failed",
            }
            return observation, False, False
