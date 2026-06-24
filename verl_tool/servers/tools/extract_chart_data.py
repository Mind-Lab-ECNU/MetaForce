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
class ExtractChartDataTool(BaseTool):
    """
    单步同步版本（可由 ray_utils 包装），负责把 <tool_call> 转发到多模态模型。
    """

    tool_type = "extract_chart_data"

    def __init__(self, num_workers: int = 8, request_timeout: int = 60):
        super().__init__(num_workers)
        self.request_timeout = request_timeout
        self.url_mapping = _load_url_mapping()
        self.available_models = list(self.url_mapping.keys())
        self.tool_pricing: Dict[str, Any] = {}

    def get_usage_inst(self) -> str:
        return (
            "extract_chart_data: 提取图表数据。\n"
            "action 示例：<tool_call>{\"name\": \"extract_chart_data\", "
            "\"arguments\": {\"chart_type\": \"bar\", \"model\": \"extract_chart_data-1\"}}</tool_call>"
        )

    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        try:
            payload_str = action
            if "<tool_call>" in action and "</tool_call>" in action:
                payload_str = action.split("<tool_call>")[1].split("</tool_call>")[0]
            payload = json.loads(payload_str)
            if payload.get("name") != "extract_chart_data":
                return {}, False
            arguments = payload.get("arguments", {})
            if not isinstance(arguments, dict):
                return {}, False
            if "chart_type" not in arguments or "model" not in arguments:
                return {}, False
            payload["arguments"] = arguments
            return payload, True
        except Exception:
            return {}, False

    def _build_prompt(self, question: Optional[str], arguments: Dict[str, Any]) -> str:
        """
        Build detailed, context-aware prompt for chart data extraction.
        Dynamically generates instructions based on user-provided parameters.
        """
        
        # Extract parameters
        chart_type = arguments.get('chart_type', 'unknown')
        target_label = arguments.get('target_label')
        target_color = arguments.get('target_color')
        target_index = arguments.get('target_index')
        
        # Base system instruction
        base_instruction = (
            "You are a precise chart data extraction assistant specializing in quantitative analysis "
            "of data visualizations. Your task is to accurately extract numerical data, labels, colors, "
            "and structural information from charts and return them in a structured JSON format."
        )
        
        # Detailed instructions for each chart type
        chart_type_instructions = {
            'bar': (
                "BAR CHART EXTRACTION:\n\n"
                "STRUCTURE ANALYSIS:\n"
                "- Identify axis orientation: horizontal bars or vertical bars\n"
                "- Determine if chart contains grouped bars, stacked bars, or single series\n"
                "- Identify all data series (by legend labels, colors, or patterns)\n\n"
                "DATA EXTRACTION:\n"
                "1. X-axis (category axis): Extract all category labels\n"
                "2. Y-axis (value axis): Extract numerical values for each bar\n"
                "3. For each bar:\n"
                "   - Category label (x-value)\n"
                "   - Numerical value (bar height/length)\n"
                "   - Series identifier (legend label or color)\n"
                "   - Bar color (if distinguishing feature)\n\n"
                "SPECIAL CASES:\n"
                "- Grouped bars: Return values for each sub-bar with series identifier\n"
                "- Stacked bars: Return both individual segment values and cumulative total\n"
                "- Negative values: Preserve sign (bars extending below zero)\n\n"
                "OUTPUT FORMAT:\n"
                "```json\n"
                "[\n"
                "  {\"category\": \"A\", \"series\": \"Series1\", \"value\": 25.5, \"color\": \"blue\"},\n"
                "  {\"category\": \"A\", \"series\": \"Series2\", \"value\": 18.3, \"color\": \"red\"},\n"
                "  {\"category\": \"B\", \"series\": \"Series1\", \"value\": 30.2, \"color\": \"blue\"}\n"
                "]\n"
                "```"
            ),
            'pie': (
                "PIE CHART EXTRACTION:\n\n"
                "STRUCTURE ANALYSIS:\n"
                "- Identify all pie slices/wedges\n"
                "- Check for labels (inside slices, outside with leaders, or in legend)\n"
                "- Determine if percentages or absolute values are displayed\n\n"
                "DATA EXTRACTION:\n"
                "1. For each slice:\n"
                "   - Label/category name\n"
                "   - Percentage of total (calculate if not shown: angle/360° × 100%)\n"
                "   - Absolute value (if displayed)\n"
                "   - Color of the slice\n"
                "   - Angular position (optional: starting angle to ending angle)\n"
                "2. Total sum: Calculate sum of all values (should equal 100% or stated total)\n\n"
                "SPECIAL CASES:\n"
                "- Exploded slices: Note which slices are separated from the main pie\n"
                "- Donut charts: Extract same data as pie chart (ignore center hole)\n"
                "- Small slices: If slices are too small to label clearly, mark as 'others' and sum\n\n"
                "OUTPUT FORMAT:\n"
                "```json\n"
                "[\n"
                "  {\"label\": \"Category A\", \"value\": 45.5, \"percentage\": 35.2, \"color\": \"blue\"},\n"
                "  {\"label\": \"Category B\", \"value\": 32.3, \"percentage\": 25.0, \"color\": \"red\"},\n"
                "  {\"label\": \"Others\", \"value\": 51.2, \"percentage\": 39.8, \"color\": \"gray\"}\n"
                "]\n"
                "```\n"
                "Verification: Sum of percentages should equal 100% (±0.5% tolerance for rounding)"
            ),
            'line': (
                "LINE CHART EXTRACTION:\n\n"
                "STRUCTURE ANALYSIS:\n"
                "- Identify all line series (by legend, color, or line style)\n"
                "- Determine X-axis type: continuous (numerical) or categorical (discrete)\n"
                "- Check for markers/data points on lines\n\n"
                "DATA EXTRACTION:\n"
                "1. For each visible data point on each line:\n"
                "   - X-coordinate (read from X-axis)\n"
                "   - Y-coordinate (read from Y-axis)\n"
                "   - Series identifier (legend label or line color)\n"
                "   - Point marker style (if applicable)\n"
                "2. If lines have no visible markers:\n"
                "   - Sample key points: peaks, valleys, inflection points, endpoints\n"
                "   - Sample regularly spaced points to capture line shape\n\n"
                "SPECIAL CASES:\n"
                "- Multiple lines: Extract data for ALL series separately\n"
                "- Area charts: Extract the line boundary values (ignore filled area)\n"
                "- Missing data: Mark gaps with null or skip index positions\n\n"
                "OUTPUT FORMAT:\n"
                "```json\n"
                "[\n"
                "  {\"series\": \"Revenue\", \"x\": 0, \"y\": 25.5, \"color\": \"blue\"},\n"
                "  {\"series\": \"Revenue\", \"x\": 1, \"y\": 28.3, \"color\": \"blue\"},\n"
                "  {\"series\": \"Cost\", \"x\": 0, \"y\": 18.2, \"color\": \"red\"},\n"
                "  {\"series\": \"Cost\", \"x\": 1, \"y\": 22.1, \"color\": \"red\"}\n"
                "]\n"
                "```"
            ),
            'scatter': (
                "SCATTER PLOT EXTRACTION:\n\n"
                "STRUCTURE ANALYSIS:\n"
                "- Identify all point clusters or series (by color, shape, or legend)\n"
                "- Check for point labels or annotations\n"
                "- Determine if point sizes carry meaning (bubble chart)\n\n"
                "DATA EXTRACTION:\n"
                "1. For each visible point:\n"
                "   - X-coordinate (read from X-axis)\n"
                "   - Y-coordinate (read from Y-axis)\n"
                "   - Series/cluster identifier (if multiple groups exist)\n"
                "   - Point color\n"
                "   - Point size (if bubble chart)\n"
                "   - Point label (if present)\n"
                "2. Extract ALL visible points (do not skip or sample)\n\n"
                "SPECIAL CASES:\n"
                "- Bubble charts: Include 'size' field representing bubble diameter or area\n"
                "- Overlapping points: Report all points even if they overlap visually\n"
                "- Outliers: Include all points regardless of distance from cluster\n\n"
                "OUTPUT FORMAT:\n"
                "```json\n"
                "[\n"
                "  {\"series\": \"Group A\", \"x\": 12.5, \"y\": 45.3, \"color\": \"blue\", \"size\": 10},\n"
                "  {\"series\": \"Group A\", \"x\": 15.2, \"y\": 48.7, \"color\": \"blue\", \"size\": 15},\n"
                "  {\"series\": \"Group B\", \"x\": 20.1, \"y\": 35.2, \"color\": \"red\", \"size\": 8}\n"
                "]\n"
                "```"
            ),
            'heatmap': (
                "HEATMAP EXTRACTION:\n\n"
                "STRUCTURE ANALYSIS:\n"
                "- Identify row labels (Y-axis categories)\n"
                "- Identify column labels (X-axis categories)\n"
                "- Locate color scale/legend mapping colors to values\n\n"
                "DATA EXTRACTION:\n"
                "1. For each cell in the grid:\n"
                "   - Row label (y-category)\n"
                "   - Column label (x-category)\n"
                "   - Cell value (read from color scale or cell annotation)\n"
                "   - Cell color (optional)\n"
                "2. Extract color scale metadata:\n"
                "   - Minimum value and corresponding color\n"
                "   - Maximum value and corresponding color\n"
                "   - Scale type (linear, logarithmic, diverging, etc.)\n\n"
                "OUTPUT FORMAT:\n"
                "```json\n"
                "{\n"
                "  \"data\": [\n"
                "    {\"row\": \"Product A\", \"column\": \"Q1\", \"value\": 125.5, \"color\": \"#FF5733\"},\n"
                "    {\"row\": \"Product A\", \"column\": \"Q2\", \"value\": 98.3, \"color\": \"#FFA500\"}\n"
                "  ],\n"
                "  \"color_scale\": {\"min\": 0, \"max\": 200, \"type\": \"linear\"}\n"
                "}\n"
                "```"
            )
        }
        
        # Get detailed description for the chart type
        chart_detail = chart_type_instructions.get(
            chart_type,
            f"CHART TYPE: {chart_type.upper()}\n"
            f"Extract all visible data points from this chart type. Return data in JSON format with "
            f"appropriate fields for this visualization type (e.g., labels, values, coordinates, colors)."
        )
        
        # Build filter context based on target parameters
        filter_context = ""
        filter_conditions = []
        
        if target_label:
            filter_conditions.append(f"label/category equals '{target_label}'")
        if target_color:
            filter_conditions.append(f"color equals '{target_color}'")
        if target_index is not None:
            filter_conditions.append(f"index position equals {target_index}")
        
        if filter_conditions:
            filter_context = (
                f"\nDATA FILTERING:\n"
                f"- Extract ONLY data points/series where {' AND '.join(filter_conditions)}\n"
                f"- If no data matches the filter criteria, return empty array [] with a note\n"
                f"- If filter is ambiguous (e.g., color name matches multiple series), extract all matches\n"
            )
        else:
            filter_context = (
                f"\nDATA SCOPE:\n"
                f"- Extract ALL visible data from the chart\n"
                f"- If multiple series exist, extract data for EACH series separately\n"
                f"- Maintain series identifiers (labels, colors) to distinguish data points\n"
            )
        
        # Output format and quality requirements
        output_requirements = (
            "\nOUTPUT REQUIREMENTS:\n\n"
            "1. JSON FORMAT:\n"
            "   - Return valid JSON (parseable by standard JSON parsers)\n"
            "   - Use array of objects: [{\"field1\": value1, \"field2\": value2}, ...]\n"
            "   - Do NOT include markdown code blocks, explanatory text, or comments\n\n"
            "2. NUMERICAL PRECISION:\n"
            "   - Read values as accurately as possible from axis scales\n"
            "   - Maintain at least 1-2 decimal places for continuous values\n"
            "   - For integer data (e.g., counts), return whole numbers\n\n"
            "3. REQUIRED FIELDS (adapt based on chart type):\n"
            "   - Labels/categories: Use exact text from chart (preserve capitalization, spacing)\n"
            "   - Values: Numerical data read from axes or displayed on chart\n"
            "   - Colors: Use common color names (red, blue) or hex codes (#FF0000)\n"
            "   - Series identifiers: Legend labels or distinguishing features\n\n"
            "4. COMPLETENESS:\n"
            "   - Extract ALL visible data points (do not sample or skip unless chart is too dense)\n"
            "   - For dense charts (>100 points): sample evenly or focus on key features\n"
            "   - If data cannot be fully read: extract best effort and add note field\n\n"
            "5. DATA VERIFICATION:\n"
            "   - For pie charts: verify percentages sum to ~100%\n"
            "   - For bar charts: verify all categories are represented\n"
            "   - For line/scatter: verify coordinate ranges match axis limits\n\n"
            "6. ERROR HANDLING:\n"
            "   - If chart is unclear/unreadable: return {\"error\": \"reason\", \"partial_data\": [...]}\n"
            "   - If filter finds no matches: return [] with {\"note\": \"no data matches filter\"}\n"
            "   - If chart type mismatch: return {\"error\": \"expected {type1}, found {type2}\"}\n"
        )
        
        # User question section
        question_section = ""
        if question:
            question_section = f"\nUSER'S SPECIFIC QUESTION:\n{question}\n"
        
        # Assemble complete prompt
        full_prompt = (
            f"{base_instruction}\n\n"
            f"{chart_detail}"
            f"{filter_context}"
            f"{output_requirements}"
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
        """
        Calculate cost based on token usage and pricing information.
        Supports multiple pricing key formats for compatibility.
        """
        pricing_map = extra_field.get("tool_pricing") or {}
        pricing = None
        
        # Try to get pricing from service_name first, then model_variant
        if service_name and service_name in pricing_map:
            pricing = pricing_map.get(service_name)
        if pricing is None:
            pricing = pricing_map.get(model_variant) or self.tool_pricing.get(model_variant)
        
        # If no pricing info or no usage, return 0
        if not pricing or not usage:
            return 0.0
        
        # Ensure pricing is a dict
        if not isinstance(pricing, dict):
            print(f"[WARNING] Invalid pricing format for {service_name or model_variant}: {type(pricing)}")
            return 0.0
        
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        # Support multiple key name formats with fallback
        input_price = (
            pricing.get("input_per_million") or 
            pricing.get("input_tokens_per_million") or 
            0.0
        )
        output_price = (
            pricing.get("output_per_million") or 
            pricing.get("output_tokens_per_million") or 
            0.0
        )
        
        cost = (prompt_tokens * input_price + completion_tokens * output_price) 
        return cost


    def _resolve_service_name(self, model_variant: str, extra_field: Dict[str, Any]) -> str:
        mapping = extra_field.get("model_mapping") or {}
        if isinstance(mapping, dict) and "extract_chart_data" in mapping:
            mapping = mapping["extract_chart_data"]
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
                {"role": "system", "content": "You are a precise chart data extraction assistant."},
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
                "obs": "Invalid tool_call format for extract_chart_data.",
                "invalid_reason": "parse_failed",
            }
            return observation, False, False

        arguments = parsed["arguments"]
        model_variant = arguments.get("model", "extract_chart_data-1")
        image_data_url = self._prepare_image(arguments, extra_field)
        if not image_data_url:
            observation = {
                "obs": "No image provided for extract_chart_data.",
                "invalid_reason": "missing_image",
            }
            return observation, False, False

        question = extra_field.get("question") or extra_field.get("prompt")
        prompt = self._build_prompt(question, arguments)

        try:
            content, meta = self._call_model(model_variant, image_data_url, prompt, extra_field)
            print(
                f"[extract_chart_data] model_variant={model_variant}, service_name={meta.get('service_name')}, "
                f"pricing={(extra_field.get('tool_pricing') or {}).get(model_variant) or self.tool_pricing.get(model_variant)}, "
                f"arguments={arguments}, latency={meta.get('latency')}, cost={meta.get('cost')}"
            )
            observation = {
                "obs": content,
                "latency": meta["latency"],
                "cost": meta["cost"],
                "usage": meta.get("usage", {}),
                "tool": "extract_chart_data",
                "model": meta["model"],
                "model_variant": model_variant
            }
            return observation, False, True
        except ValueError as exc:
            observation = {
                "obs": f"extract_chart_data failed: {exc}",
                "invalid_reason": "unknown_model",
            }
            return observation, False, False
        except Exception as exc:
            observation = {
                "obs": f"extract_chart_data failed: {exc}",
                "invalid_reason": "request_failed",
            }
            return observation, False, False
