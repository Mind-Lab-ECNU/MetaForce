import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from .base import BaseTool, register_tool


def _load_url_mapping() -> Dict[str, str]:
    """Load model deployment URLs from the colocated url.json file."""
    url_path = Path(__file__).with_name("url.json")
    if not url_path.exists():
        return {}
    try:
        return json.loads(url_path.read_text())
    except Exception:
        return {}


def _to_data_url(image_path: Path) -> str:
    """Encode an image file into a data URL."""
    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"


@register_tool
class CompareAndFilterAsyncTool(BaseTool):
    """
    Asynchronous compare and filter tool.
    
    Expected action format:
        <tool_call>{"name": "compare_and_filter_async", "arguments": {"comparison_type": "value_to_statistic", "condition": "greater", "model": "compare_and_filter-1"}}</tool_call>
    
    DO NOT wrap this tool with Ray - it manages its own event loop internally.
    """

    tool_type = "compare_and_filter_async"

    def __init__(self, num_workers: int = 8, request_timeout: int = 60):
        super().__init__(num_workers)
        self.request_timeout = request_timeout
        self.url_mapping = _load_url_mapping()
        self.available_models = list(self.url_mapping.keys())
        self.tool_pricing: Dict[str, Any] = {}

    def get_usage_inst(self) -> str:
        return (
            "compare_and_filter_async: Compare and filter data (async). "
            "Format: <tool_call>{\"name\": \"compare_and_filter_async\", "
            "\"arguments\": {\"comparison_type\": \"value_to_statistic\", \"condition\": \"greater\", \"model\": \"compare_and_filter-1\"}}</tool_call>"
        )

    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        """
        Parse action to extract comparison parameters.
        Only supports <tool_call>JSON</tool_call> format.
        Returns (payload_dict, is_valid).
        """
        print(f"[CompareAndFilterAsync] parse_action called, action preview: {action[:200]}")
        
        try:
            # Extract JSON from <tool_call> tags
            if "<tool_call>" not in action or "</tool_call>" not in action:
                print(f"[CompareAndFilterAsync] No <tool_call> tags found")
                return {}, False
                
            start = action.find("<tool_call>") + len("<tool_call>")
            end = action.find("</tool_call>")
            payload_str = action[start:end].strip()
            
            print(f"[CompareAndFilterAsync] Extracted payload: {payload_str[:150]}")
            
            payload = json.loads(payload_str)
            print(f"[CompareAndFilterAsync] Parsed JSON: {payload}")
            
            if payload.get("name") != "compare_and_filter_async":
                print(f"[CompareAndFilterAsync] Tool name mismatch: {payload.get('name')}")
                return {}, False
                
            arguments = payload.get("arguments", {})
            if not isinstance(arguments, dict):
                print(f"[CompareAndFilterAsync] Arguments not a dict")
                return {}, False
                
            if "comparison_type" not in arguments or "condition" not in arguments or "model" not in arguments:
                print(f"[CompareAndFilterAsync] Missing required fields")
                return {}, False
                
            payload["arguments"] = arguments
            print(f"[CompareAndFilterAsync] ✓ Successfully parsed action")
            return payload, True
            
        except json.JSONDecodeError as e:
            print(f"[CompareAndFilterAsync] JSON decode error: {e}")
            return {}, False
        except Exception as e:
            print(f"[CompareAndFilterAsync] Unexpected error: {e}")
            return {}, False

    def _build_prompt(self, question: Optional[str], arguments: Dict[str, Any]) -> str:
        """
        Build detailed, context-aware prompt for comparison and filtering operations.
        Dynamically generates instructions based on user-provided parameters.
        """
        
        # Extract parameters
        comparison_type = arguments.get('comparison_type', 'unknown')
        condition = arguments.get('condition', 'unknown')
        target_value = arguments.get('target_value', '')
        reference_value = arguments.get('reference_value', '')
        
        # Base system instruction
        base_instruction = (
            "You are a precise data comparison and filtering assistant specializing in quantitative analysis "
            "of data visualizations. Your task is to accurately perform comparison or filtering operations "
            "on chart data and return structured results."
        )
        
        # Detailed instructions for each comparison type
        comparison_type_instructions = {
            'value_to_value': (
                "VALUE-TO-VALUE COMPARISON: Compare two specific data values or data series.\n"
                "- Identify the two target values/series based on provided identifiers (labels, colors, indices)\n"
                "- Perform direct numerical comparison between them\n"
                "- Return: which value is greater/less/equal, the numerical difference, and percentage difference if applicable"
            ),
            'value_to_statistic': (
                "VALUE-TO-STATISTIC COMPARISON: Compare a specific data value against a statistical metric.\n"
                "- First, identify the target value/series using the provided identifier\n"
                "- Calculate the specified statistic (max, min, median, mean, etc.) for the dataset or series\n"
                "- Compare the target value against this statistic\n"
                "- Return: comparison result (greater/less/equal), the statistic value, and numerical difference"
            ),
            'filter_by_condition': (
                "FILTER BY CONDITION: Filter and return data points or series meeting specific criteria.\n"
                "- Apply the specified condition to all data points or series\n"
                "- Conditions may reference statistics (above_median, below_median, is_max, is_min) or thresholds\n"
                "- Return: list of all data points/series satisfying the condition with their identifiers and values\n"
                "- Include count and percentage of filtered results relative to total dataset"
            ),
            'label_match': (
                "LABEL MATCH CHECK: Verify if specific labels or identifiers exist in the dataset.\n"
                "- Search for the target label/identifier in legend labels, axis labels, or data annotations\n"
                "- Return: match status (found/not found), matching elements, and their associated data values\n"
                "- If multiple matches exist, list all occurrences"
            )
        }
        
        # Get detailed description for the comparison type
        type_detail = comparison_type_instructions.get(
            comparison_type,
            f"Perform {comparison_type} operation. Analyze the data and return structured results based on the specified condition."
        )
        
        # Detailed instructions for each condition
        condition_instructions = {
            'greater': (
                "CONDITION: Greater Than (>)\n"
                "- Compare if target value > reference value (for value_to_value or value_to_statistic)\n"
                "- Return: boolean result, both values, and numerical difference"
            ),
            'less': (
                "CONDITION: Less Than (<)\n"
                "- Compare if target value < reference value\n"
                "- Return: boolean result, both values, and numerical difference"
            ),
            'equal': (
                "CONDITION: Equal To (=)\n"
                "- Compare if target value = reference value (within reasonable precision tolerance)\n"
                "- Return: boolean result, both values, and absolute difference"
            ),
            'above_median': (
                "CONDITION: Above Median\n"
                "- Filter/check for values greater than the median of the dataset\n"
                "- Calculation: First compute median, then filter values > median\n"
                "- Return: filtered values, median value, count and percentage of qualifying points"
            ),
            'below_median': (
                "CONDITION: Below Median\n"
                "- Filter/check for values less than the median of the dataset\n"
                "- Calculation: First compute median, then filter values < median\n"
                "- Return: filtered values, median value, count and percentage of qualifying points"
            ),
            'above_high_median': (
                "CONDITION: Above High Median\n"
                "- Filter for values greater than the high median (upper middle value when dataset has even count)\n"
                "- Calculation: Compute high_median = value at position (n/2 + 1) after sorting\n"
                "- Return: filtered values, high median value, count and percentage"
            ),
            'below_low_median': (
                "CONDITION: Below Low Median\n"
                "- Filter for values less than the low median (lower middle value when dataset has even count)\n"
                "- Calculation: Compute low_median = value at position n/2 after sorting\n"
                "- Return: filtered values, low median value, count and percentage"
            ),
            'is_max': (
                "CONDITION: Is Maximum\n"
                "- Check if target value equals the maximum value in the dataset\n"
                "- Return: boolean result, maximum value, and identifiers of all points achieving this maximum"
            ),
            'is_min': (
                "CONDITION: Is Minimum\n"
                "- Check if target value equals the minimum value in the dataset\n"
                "- Return: boolean result, minimum value, and identifiers of all points achieving this minimum"
            )
        }
        
        # Get detailed description for the condition
        condition_detail = condition_instructions.get(
            condition,
            f"Apply condition: {condition}. Return boolean result and supporting data."
        )
        
        # Build context section based on provided parameters
        context_parts = []
        
        # Target value specification
        if target_value:
            context_parts.append(
                f"TARGET SPECIFICATION: Identify and analyze data identified as '{target_value}'.\n"
                f"- Match by: legend label, axis label, color name, or data annotation\n"
                f"- If multiple series/points match, process all matching elements"
            )
        else:
            context_parts.append(
                "TARGET SPECIFICATION: Apply operation to all data in the chart.\n"
                "- Process all visible data series or points\n"
                "- Return results organized by series if multiple series exist"
            )
        
        # Reference value specification (for comparison operations)
        if reference_value:
            context_parts.append(
                f"REFERENCE SPECIFICATION: Use '{reference_value}' as the comparison baseline.\n"
                f"- Identify reference by: legend label, axis label, color name, or numeric value\n"
                f"- Perform comparison: target vs reference"
            )
        
        context_section = "\n".join(context_parts)
        
        # Output format requirements
        output_format = (
            "OUTPUT FORMAT REQUIREMENTS:\n"
            "1. OPERATION SUMMARY:\n"
            "   - State the comparison/filter type performed\n"
            "   - List the condition applied\n"
            "   - Identify target and reference (if applicable)\n"
            "2. NUMERICAL RESULTS:\n"
            "   - For comparisons: boolean result (true/false), both values, numerical difference, percentage difference\n"
            "   - For filters: list of qualifying data points with identifiers and values\n"
            "   - For statistics: include calculated statistic value with label\n"
            "3. DATA IDENTIFICATION:\n"
            "   - Clearly identify all data elements (by label, color, index, or other distinguishing features)\n"
            "   - For multiple series: organize results by series\n"
            "4. COUNTS AND PERCENTAGES (for filtering):\n"
            "   - Total count of data points/series analyzed\n"
            "   - Count of points meeting the condition\n"
            "   - Percentage: (matching count / total count) × 100%\n"
            "5. STRUCTURED FORMAT:\n"
            "   - Use JSON format when returning multiple results or filtered datasets\n"
            "   - Example: {\"result\": true, \"target_value\": 45.2, \"reference_value\": 38.7, \"difference\": 6.5}\n"
            "   - For filters: {\"matching_points\": [{\"label\": \"A\", \"value\": 50}, ...], \"count\": 5, \"percentage\": 41.7}\n"
            "6. PRECISION AND UNITS:\n"
            "   - Maintain at least 2 decimal places for numerical results\n"
            "   - Include units from axis labels if available\n"
            "   - For percentage differences: use formula ((target - reference) / reference) × 100%\n"
            "7. EDGE CASES:\n"
            "   - If target/reference not found: explicitly state \"not found\" with available identifiers listed\n"
            "   - If no data meets filter condition: return empty list with count=0, percentage=0%\n"
            "   - If data is ambiguous: list all possible matches and ask for clarification if needed"
        )
        
        # User question section
        question_section = ""
        if question:
            question_section = f"\nUSER'S SPECIFIC QUESTION:\n{question}\n"
        
        # Assemble complete prompt
        full_prompt = (
            f"{base_instruction}\n\n"
            f"COMPARISON TYPE: {comparison_type.upper()}\n\n"
            f"{type_detail}\n\n"
            f"CONDITION: {condition.upper()}\n\n"
            f"{condition_detail}\n\n"
            f"CONTEXT & PARAMETERS:\n{context_section}\n\n"
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
        if isinstance(mapping, dict) and "compare_and_filter" in mapping:
            mapping = mapping["compare_and_filter"]
        if isinstance(mapping, dict) and model_variant in mapping:
            return mapping[model_variant]
        if isinstance(mapping, list) and mapping:
            try:
                idx = int(str(model_variant).split("-")[-1]) - 1
                if 0 <= idx < len(mapping):
                    return mapping[idx]
            except Exception:
                pass
        parts = str(model_variant).rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            idx = int(parts[1]) - 1
            if 0 <= idx < len(self.available_models):
                return self.available_models[idx]
        raise ValueError(f"Unknown model variant: {model_variant}")

    async def _call_model_async(
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
                {"role": "system", "content": "You are a precise data comparison and filtering assistant."},
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

        loop = asyncio.get_event_loop()

        def _post():
            start = time.time()
            resp = requests.post(url, headers=headers, json=payload, timeout=self.request_timeout)
            latency = time.time() - start
            resp.raise_for_status()
            return resp.json(), latency

        data, latency = await loop.run_in_executor(None, _post)
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

    async def _conduct_action_async(
        self, trajectory_id: str, action: str, extra_field: Dict[str, Any]
    ) -> Tuple[Any, bool, bool]:
        print(f"[CompareAndFilterAsync] _conduct_action_async called for trajectory: {trajectory_id}")
        
        parsed, is_valid = self.parse_action(action)
        if not is_valid:
            print(f"[CompareAndFilterAsync] Invalid action, returning error")
            observation = {
                "obs": "Invalid action format for compare_and_filter_async. Expected: <tool_call>{\"name\": \"compare_and_filter_async\", \"arguments\": {...}}</tool_call>",
                "invalid_reason": "parse_failed",
            }
            return observation, False, False

        arguments = parsed["arguments"]
        model_variant = arguments.get("model", "compare_and_filter-1")
        image_data_url = self._prepare_image(arguments, extra_field)
        if not image_data_url:
            observation = {
                "obs": "No image provided for compare_and_filter_async.",
                "invalid_reason": "missing_image",
            }
            return observation, False, False

        question = extra_field.get("question") or extra_field.get("prompt")
        prompt = self._build_prompt(question, arguments)

        try:
            content, meta = await self._call_model_async(model_variant, image_data_url, prompt, extra_field)
            print(
                f"[compare_and_filter_async] model_variant={model_variant}, service_name={meta.get('service_name')}, "
                f"latency={meta.get('latency'):.3f}s, cost={meta.get('cost'):.6f}"
            )
            observation = {
                "obs": content,
                "latency": meta["latency"],
                "cost": meta["cost"],
                "usage": meta.get("usage", {}),
                "tool": "compare_and_filter",
                "model": meta["model"],
                "model_variant": model_variant
            }
            return observation, False, True
        except ValueError as exc:
            observation = {
                "obs": f"compare_and_filter_async failed: {exc}",
                "invalid_reason": "unknown_model",
            }
            return observation, False, False
        except Exception as exc:
            observation = {
                "obs": f"compare_and_filter_async failed: {exc}",
                "invalid_reason": "request_failed",
            }
            return observation, False, False

    def conduct_action(
        self, trajectory_id: str, action: str, extra_field: Dict[str, Any]
    ) -> Tuple[Any, bool, bool]:
        """Synchronous wrapper that runs the async implementation."""
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._conduct_action_async(trajectory_id, action, extra_field))
        finally:
            asyncio.set_event_loop(None)
            loop.close()

