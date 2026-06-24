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
class DetectIntersectionAsyncTool(BaseTool):
    """
    Asynchronous intersection detection tool.
    
    Expected action format:
        <tool_call>{"name": "detect_intersection_async", "arguments": {"series1": "line A", "series2": "line B", "detection_method": "y_value_comparison", "model": "detect_intersection-1"}}</tool_call>
    
    DO NOT wrap this tool with Ray - it manages its own event loop internally.
    """

    tool_type = "detect_intersection_async"

    def __init__(self, num_workers: int = 8, request_timeout: int = 60):
        super().__init__(num_workers)
        self.request_timeout = request_timeout
        self.url_mapping = _load_url_mapping()
        self.available_models = list(self.url_mapping.keys())
        self.tool_pricing: Dict[str, Any] = {}

    def get_usage_inst(self) -> str:
        return (
            "detect_intersection_async: Detect if two data series/lines intersect (async). "
            "Format: <tool_call>{\"name\": \"detect_intersection_async\", "
            "\"arguments\": {\"series1\": \"line A\", \"series2\": \"line B\", "
            "\"detection_method\": \"y_value_comparison\", \"model\": \"detect_intersection-1\"}}</tool_call>"
        )

    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        """
        Parse action to extract intersection detection parameters.
        Only supports <tool_call>JSON</tool_call> format.
        Returns (payload_dict, is_valid).
        """
        print(f"[DetectIntersectionAsync] parse_action called, action preview: {action[:200]}")
        
        try:
            # Extract JSON from <tool_call> tags
            if "<tool_call>" not in action or "</tool_call>" not in action:
                print(f"[DetectIntersectionAsync] No <tool_call> tags found")
                return {}, False
                
            start = action.find("<tool_call>") + len("<tool_call>")
            end = action.find("</tool_call>")
            payload_str = action[start:end].strip()
            
            print(f"[DetectIntersectionAsync] Extracted payload: {payload_str[:150]}")
            
            payload = json.loads(payload_str)
            print(f"[DetectIntersectionAsync] Parsed JSON: {payload}")
            
            if payload.get("name") != "detect_intersection_async":
                print(f"[DetectIntersectionAsync] Tool name mismatch: {payload.get('name')}")
                return {}, False
                
            arguments = payload.get("arguments", {})
            if not isinstance(arguments, dict):
                print(f"[DetectIntersectionAsync] Arguments not a dict")
                return {}, False
                
            # Check required fields
            required_fields = ["series1", "series2", "detection_method", "model"]
            for field in required_fields:
                if field not in arguments:
                    print(f"[DetectIntersectionAsync] Missing required field: {field}")
                    return {}, False
                    
            payload["arguments"] = arguments
            print(f"[DetectIntersectionAsync] ✓ Successfully parsed action")
            return payload, True
            
        except json.JSONDecodeError as e:
            print(f"[DetectIntersectionAsync] JSON decode error: {e}")
            return {}, False
        except Exception as e:
            print(f"[DetectIntersectionAsync] Unexpected error: {e}")
            return {}, False

    def _build_prompt(self, question: Optional[str], arguments: Dict[str, Any]) -> str:
        """
        Build detailed, context-aware prompt for intersection detection.
        Dynamically generates instructions based on user-provided parameters.
        """
        
        # Extract parameters
        series1 = arguments.get('series1', '')
        series2 = arguments.get('series2', '')
        detection_method = arguments.get('detection_method', 'unknown')
        x_range = arguments.get('x_range', 'full')
        
        # Base system instruction
        base_instruction = (
            "You are a precise intersection detection assistant specializing in geometric and quantitative "
            "analysis of data visualizations. Your task is to accurately determine whether two data series "
            "intersect and provide detailed information about intersection points."
        )
        
        # Detailed instructions for each detection method
        detection_method_instructions = {
            'y_value_comparison': (
                "Y-VALUE COMPARISON METHOD: Detect intersections by comparing Y-values at shared X-coordinates.\n\n"
                "ALGORITHM:\n"
                "1. Identify all X-coordinates where both series have data points\n"
                "2. For each shared X-coordinate:\n"
                "   - Extract Y-values from both series: y1[x] and y2[x]\n"
                "   - Check if y1[x] ≈ y2[x] (within reasonable tolerance for visual precision)\n"
                "3. Also check between consecutive points:\n"
                "   - If (y1[x_i] - y2[x_i]) and (y1[x_{i+1}] - y2[x_{i+1}]) have opposite signs\n"
                "   - Then an intersection exists between x_i and x_{i+1}\n\n"
                "RETURN INFORMATION:\n"
                "- Boolean: Do the series intersect? (true/false)\n"
                "- If true: Approximate X-coordinate(s) of intersection point(s)\n"
                "- Y-value(s) at intersection point(s)\n"
                "- Total count of intersection points detected"
            ),
            'range_overlap': (
                "RANGE OVERLAP METHOD: Detect potential intersections by checking if Y-value ranges overlap.\n\n"
                "ALGORITHM:\n"
                "1. For each series, determine Y-value range over the specified X-range:\n"
                "   - Series 1: min_y1 to max_y1\n"
                "   - Series 2: min_y2 to max_y2\n"
                "2. Check for range overlap:\n"
                "   - Overlap exists if: max(min_y1, min_y2) ≤ min(max_y1, max_y2)\n"
                "3. If ranges overlap, intersection is POSSIBLE (not guaranteed)\n"
                "4. For definitive answer, additionally check for crossing patterns\n\n"
                "RETURN INFORMATION:\n"
                "- Boolean: Do Y-ranges overlap? (true/false)\n"
                "- Overlapping Y-range: [overlap_min, overlap_max]\n"
                "- Confidence level: 'possible' if ranges overlap, 'confirmed' if actual crossing detected\n"
                "- If confirmed: approximate intersection location(s)"
            ),
            'sign_change': (
                "SIGN CHANGE METHOD: Detect intersections by monitoring sign changes in the difference (y1 - y2).\n\n"
                "ALGORITHM:\n"
                "1. Calculate difference function: diff[x] = y1[x] - y2[x] for all X-coordinates\n"
                "2. Track sign of diff[x] across the X-range:\n"
                "   - Positive: series1 is above series2\n"
                "   - Negative: series1 is below series2\n"
                "   - Zero: series intersect at this point\n"
                "3. Detect sign changes:\n"
                "   - If sign(diff[x_i]) ≠ sign(diff[x_{i+1}]) and neither is zero\n"
                "   - Then intersection occurs between x_i and x_{i+1}\n"
                "4. Count total number of sign changes = number of intersections\n\n"
                "RETURN INFORMATION:\n"
                "- Boolean: Are there any sign changes? (true/false)\n"
                "- Count: Total number of intersection points\n"
                "- Locations: Approximate X-coordinates where sign changes occur\n"
                "- For each intersection: which series transitions from above to below (or vice versa)"
            ),
            'point_overlap': (
                "POINT OVERLAP METHOD: Detect exact point overlaps between scatter datasets or discrete points.\n\n"
                "ALGORITHM:\n"
                "1. Extract all (x, y) coordinate pairs from both series:\n"
                "   - Series 1: {(x1_i, y1_i)} for all i\n"
                "   - Series 2: {(x2_j, y2_j)} for all j\n"
                "2. For each point in series1, check if it matches any point in series2:\n"
                "   - Match criteria: |x1_i - x2_j| < ε_x AND |y1_i - y2_j| < ε_y\n"
                "   - Use visual precision tolerance (typically 1-2% of axis range)\n"
                "3. Record all matching point pairs\n\n"
                "RETURN INFORMATION:\n"
                "- Boolean: Are there any overlapping points? (true/false)\n"
                "- Count: Number of overlapping points\n"
                "- Coordinates: List of all (x, y) coordinates where overlap occurs\n"
                "- If no exact overlaps: report closest approach distance and location"
            )
        }
        
        # Get detailed description for the detection method
        method_detail = detection_method_instructions.get(
            detection_method,
            f"DETECTION METHOD: {detection_method}\n"
            f"Apply appropriate geometric or algebraic analysis to determine if the two series intersect. "
            f"Return boolean result and intersection location(s) if found."
        )
        
        # Build series identification context
        series_context = f"SERIES IDENTIFICATION:\n"
        series_context += f"- SERIES 1: '{series1}'\n"
        series_context += f"  Identify by: legend label, line color, line style, or marker type\n"
        series_context += f"- SERIES 2: '{series2}'\n"
        series_context += f"  Identify by: legend label, line color, line style, or marker type\n"
        series_context += f"- If series names are ambiguous, use visual features (color, style) to distinguish\n"
        series_context += f"- If a series is not found, explicitly report which series is missing"
        
        # X-range specification
        x_range_instructions = {
            'full': (
                "X-RANGE: Full range (analyze entire visible X-axis range)\n"
                "- Check for intersections across all available data points\n"
                "- Report all intersection points found"
            ),
            '0-10': (
                f"X-RANGE: Restricted to X ∈ [{x_range}]\n"
                f"- Only analyze data points within this X-range\n"
                f"- Ignore intersection points outside this range\n"
                f"- If series don't have data in this range, report 'no data in specified range'"
            ),
            'specific_point': (
                "X-RANGE: Specific point analysis\n"
                "- Check if series intersect at or near a specific X-coordinate\n"
                "- Use tolerance window around the specified point\n"
                "- Report whether intersection occurs at this location"
            )
        }
        
        # Determine appropriate x_range instruction
        if x_range == 'full':
            x_range_detail = x_range_instructions['full']
        elif '-' in str(x_range) or 'to' in str(x_range).lower():
            x_range_detail = (
                f"X-RANGE: Restricted to X ∈ {x_range}\n"
                f"- Only analyze data points within this X-range\n"
                f"- Ignore intersection points outside this range\n"
                f"- If series don't have data in this range, report 'no data in specified range'"
            )
        elif x_range == 'specific_point':
            x_range_detail = x_range_instructions['specific_point']
        else:
            x_range_detail = f"X-RANGE: {x_range}\n- Analyze the specified range"
        
        # Output format requirements
        output_format = (
            "OUTPUT FORMAT REQUIREMENTS:\n\n"
            "1. SERIES CONFIRMATION:\n"
            "   - Confirm identification of both series with their visual characteristics\n"
            "   - Example: 'Series 1 (blue solid line) and Series 2 (red dashed line) identified'\n\n"
            "2. INTERSECTION RESULT (Primary Answer):\n"
            "   - Boolean: intersect = true OR intersect = false\n"
            "   - Clear statement: 'The two series DO intersect' or 'The two series DO NOT intersect'\n\n"
            "3. INTERSECTION DETAILS (if intersect = true):\n"
            "   A. Count: Total number of intersection points\n"
            "   B. Locations: For each intersection point:\n"
            "      - Approximate X-coordinate (with precision: ±0.5 units or better)\n"
            "      - Approximate Y-coordinate (with precision: ±0.5 units or better)\n"
            "      - Format: '(x ≈ 5.2, y ≈ 12.7)'\n"
            "   C. Crossing behavior:\n"
            "      - Which series crosses from above to below (or vice versa)\n"
            "      - Example: 'At x≈5.2, Series 1 crosses from below to above Series 2'\n\n"
            "4. STRUCTURED JSON FORMAT (preferred for programmatic use):\n"
            "   ```json\n"
            "   {\n"
            "     \"intersect\": true,\n"
            "     \"intersection_count\": 2,\n"
            "     \"intersections\": [\n"
            "       {\"x\": 5.2, \"y\": 12.7, \"crossing\": \"series1_above_to_below\"},\n"
            "       {\"x\": 8.9, \"y\": 15.3, \"crossing\": \"series1_below_to_above\"}\n"
            "     ],\n"
            "     \"method\": \"y_value_comparison\",\n"
            "     \"x_range_analyzed\": \"full\"\n"
            "   }\n"
            "   ```\n\n"
            "5. NO INTERSECTION CASE (if intersect = false):\n"
            "   - State clearly: 'No intersection points detected'\n"
            "   - Provide context:\n"
            "     · Are the series parallel?\n"
            "     · Which series is consistently above/below the other?\n"
            "     · Minimum distance between the series (if calculable)\n"
            "   - Example: 'Series 1 remains consistently above Series 2 throughout the range (min vertical distance ≈ 3.5 units)'\n\n"
            "6. EDGE CASES:\n"
            "   - If one or both series not found: 'Series [name] not found in chart. Available series: [list]'\n"
            "   - If no data in specified X-range: 'No data available in X-range [range] for one or both series'\n"
            "   - If series are identical: 'Series 1 and Series 2 are identical (complete overlap)'\n"
            "   - If series tangent (touch but don't cross): 'Series touch at x≈[value] but do not cross'\n\n"
            "7. CONFIDENCE AND PRECISION:\n"
            "   - State confidence level: 'high' (clear intersection), 'medium' (approximate), 'low' (ambiguous)\n"
            "   - Note any limitations due to chart resolution, overlapping elements, or visual clarity"
        )
        
        # User question section
        question_section = ""
        if question:
            question_section = f"\nUSER'S SPECIFIC QUESTION:\n{question}\n"
        
        # Assemble complete prompt
        full_prompt = (
            f"{base_instruction}\n\n"
            f"{series_context}\n\n"
            f"{x_range_detail}\n\n"
            f"DETECTION METHOD: {detection_method.upper()}\n\n"
            f"{method_detail}\n\n"
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
        if isinstance(mapping, dict) and "detect_intersection" in mapping:
            mapping = mapping["detect_intersection"]
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
                {"role": "system", "content": "You are a precise intersection detection assistant."},
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
        print(f"[DetectIntersectionAsync] _conduct_action_async called for trajectory: {trajectory_id}")
        
        parsed, is_valid = self.parse_action(action)
        if not is_valid:
            print(f"[DetectIntersectionAsync] Invalid action, returning error")
            observation = {
                "obs": "Invalid action format for detect_intersection_async. Expected: <tool_call>{\"name\": \"detect_intersection_async\", \"arguments\": {...}}</tool_call>",
                "invalid_reason": "parse_failed",
            }
            return observation, False, False

        arguments = parsed["arguments"]
        model_variant = arguments.get("model", "detect_intersection-1")
        image_data_url = self._prepare_image(arguments, extra_field)
        if not image_data_url:
            observation = {
                "obs": "No image provided for detect_intersection_async.",
                "invalid_reason": "missing_image",
            }
            return observation, False, False

        question = extra_field.get("question") or extra_field.get("prompt")
        prompt = self._build_prompt(question, arguments)

        try:
            content, meta = await self._call_model_async(model_variant, image_data_url, prompt, extra_field)
            print(
                f"[detect_intersection_async] model_variant={model_variant}, service_name={meta.get('service_name')}, "
                f"latency={meta.get('latency'):.3f}s, cost={meta.get('cost'):.6f}"
            )
            observation = {
                "obs": content,
                "latency": meta["latency"],
                "cost": meta["cost"],
                "usage": meta.get("usage", {}),
                "tool": "detect_intersection",
                "model": meta["model"],
                "model_variant": model_variant
            }
            return observation, False, True
        except ValueError as exc:
            observation = {
                "obs": f"detect_intersection_async failed: {exc}",
                "invalid_reason": "unknown_model",
            }
            return observation, False, False
        except Exception as exc:
            observation = {
                "obs": f"detect_intersection_async failed: {exc}",
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
