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
class AnalyzeCurvePropertiesTool(BaseTool):
    """
    Synchronous curve properties analysis tool (can be wrapped by Ray).
    
    Expected action format:
        <tool_call>{"name": "analyze_curve_properties", "arguments": {"analysis_type": "smoothness", "model": "analyze_curve_properties-1"}}</tool_call>
    """

    tool_type = "analyze_curve_properties"

    def __init__(self, num_workers: int = 8, request_timeout: int = 60):
        super().__init__(num_workers)
        self.request_timeout = request_timeout
        self.url_mapping = _load_url_mapping()
        self.available_models = list(self.url_mapping.keys())
        self.tool_pricing: Dict[str, Any] = {}

    def get_usage_inst(self) -> str:
        return (
            "analyze_curve_properties: Analyze geometric and statistical properties of curves. "
            "Format: <tool_call>{\"name\": \"analyze_curve_properties\", "
            "\"arguments\": {\"analysis_type\": \"smoothness\", \"model\": \"analyze_curve_properties-1\"}}</tool_call>"
        )

    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        """
        Parse action to extract curve analysis parameters.
        Only supports <tool_call>JSON</tool_call> format.
        Returns (payload_dict, is_valid).
        """
        print(f"[AnalyzeCurveProperties] parse_action called, action preview: {action[:200]}")
        
        try:
            # Extract JSON from <tool_call> tags
            if "<tool_call>" not in action or "</tool_call>" not in action:
                print(f"[AnalyzeCurveProperties] No <tool_call> tags found")
                return {}, False
                
            start = action.find("<tool_call>") + len("<tool_call>")
            end = action.find("</tool_call>")
            payload_str = action[start:end].strip()
            
            print(f"[AnalyzeCurveProperties] Extracted payload: {payload_str[:150]}")
            
            payload = json.loads(payload_str)
            print(f"[AnalyzeCurveProperties] Parsed JSON: {payload}")
            
            if payload.get("name") != "analyze_curve_properties":
                print(f"[AnalyzeCurveProperties] Tool name mismatch: {payload.get('name')}")
                return {}, False
                
            arguments = payload.get("arguments", {})
            if not isinstance(arguments, dict):
                print(f"[AnalyzeCurveProperties] Arguments not a dict")
                return {}, False
                
            if "analysis_type" not in arguments or "model" not in arguments:
                print(f"[AnalyzeCurveProperties] Missing required fields")
                return {}, False
                
            payload["arguments"] = arguments
            print(f"[AnalyzeCurveProperties] ✓ Successfully parsed action")
            return payload, True
            
        except json.JSONDecodeError as e:
            print(f"[AnalyzeCurveProperties] JSON decode error: {e}")
            return {}, False
        except Exception as e:
            print(f"[AnalyzeCurveProperties] Unexpected error: {e}")
            return {}, False

    def _build_prompt(self, question: Optional[str], arguments: Dict[str, Any]) -> str:
        """Build detailed, context-aware prompt for curve properties analysis."""
        
        # Extract parameters
        analysis_type = arguments.get('analysis_type', 'unknown')
        target_series = arguments.get('target_series')
        comparison_mode = arguments.get('comparison_mode')
        reference_series = arguments.get('reference_series')
        
        # Base system instruction
        base_instruction = (
            "You are an expert curve properties analyst specializing in quantitative geometric "
            "and statistical analysis of data visualizations. Your task is to precisely analyze "
            "curves in charts and provide accurate numerical measurements."
        )
        
        # Analysis type specific instructions with formulas
        analysis_instructions = {
            'smoothness': (
                "SMOOTHNESS measures how gradually a curve changes direction. "
                "It is quantified by calculating the variance of slope changes between consecutive data points:\n"
                "- Calculate slopes: slope[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])\n"
                "- Calculate slope changes: Δslope[i] = slope[i+1] - slope[i]\n"
                "- Smoothness metric: Var(Δslope) = mean((Δslope[i] - mean(Δslope))²)\n"
                "A LOWER variance indicates a SMOOTHER curve (more gradual changes)."
            ),
            'roughness': (
                "ROUGHNESS measures the magnitude of fluctuations in a curve. "
                "It is calculated as the average absolute difference between consecutive y-values:\n"
                "- Roughness = mean(|y[i+1] - y[i]|) for all consecutive points\n"
                "A HIGHER value indicates a ROUGHER curve (more jagged or fluctuating)."
            ),
            'area_under_curve': (
                "AREA UNDER THE CURVE is calculated using the trapezoidal rule:\n"
                "- Area = Σ[(y[i] + y[i+1]) / 2 * (x[i+1] - x[i])] for all consecutive points\n"
                "Important considerations:\n"
                "- If the curve goes below the x-axis, report both signed area (net area) and absolute area\n"
                "- Clearly specify the integration bounds (x-axis range)\n"
                "- Include units if axis labels provide them"
            ),
            'slope': (
                "SLOPE analysis examines the rate of change of the curve:\n"
                "- For linear portions: slope = Δy/Δx = (y₂ - y₁)/(x₂ - x₁)\n"
                "- For non-linear curves, provide:\n"
                "  1. Average slope across the entire x-range\n"
                "  2. Maximum slope (steepest ascent)\n"
                "  3. Minimum slope (steepest descent)\n"
                "  4. Identify x-intervals where slope changes significantly"
            ),
            'variance': (
                "VARIANCE measures the spread of y-values around their mean:\n"
                "- First calculate: mean_y = mean(all y-values)\n"
                "- Then: Var(Y) = mean((y[i] - mean_y)²)\n"
                "Interpretation:\n"
                "- HIGHER variance = data points spread far from the mean\n"
                "- LOWER variance = data points clustered near the mean\n"
                "Also report the mean value for context."
            ),
            'std_deviation': (
                "STANDARD DEVIATION (σ) measures typical distance from the mean:\n"
                "- First calculate variance: Var(Y) = mean((y[i] - mean_y)²)\n"
                "- Then: σ = sqrt(Var(Y))\n"
                "Additionally calculate:\n"
                "- Coefficient of Variation: CV = σ/mean_y (if mean_y ≠ 0)\n"
                "- CV expresses variability relative to the mean (useful for comparing different scales)"
            )
        }
        
        # Get analysis-specific instruction
        analysis_detail = analysis_instructions.get(
            analysis_type,
            f"Perform {analysis_type} analysis on the curve(s) using appropriate quantitative methods. "
            f"Provide clear numerical results with explanations."
        )
        
        # Build dynamic context based on user parameters
        context_parts = []
        
        # Target series filter
        if target_series:
            context_parts.append(
                f"FOCUS: Analyze only the curve identified as '{target_series}' "
                f"(match by legend label, color description, or line style)."
            )
        
        # Comparison mode
        if comparison_mode:
            mode_instructions = {
                'find_max': (
                    f"COMPARISON TASK: Among all curves in the chart, identify which curve has the "
                    f"MAXIMUM (highest) value for {analysis_type}. Report that curve's identifier and value."
                ),
                'find_min': (
                    f"COMPARISON TASK: Among all curves in the chart, identify which curve has the "
                    f"MINIMUM (lowest) value for {analysis_type}. Report that curve's identifier and value."
                ),
                'compare_all': (
                    f"COMPARISON TASK: Calculate {analysis_type} for ALL curves in the chart. "
                    f"Rank them from highest to lowest value with their identifiers."
                )
            }
            context_parts.append(
                mode_instructions.get(
                    comparison_mode,
                    f"COMPARISON: Use mode '{comparison_mode}' to compare curves."
                )
            )
        
        # Reference series for pairwise comparison
        if reference_series:
            context_parts.append(
                f"PAIRWISE COMPARISON: Compare the target curve against the reference curve '{reference_series}'. "
                f"Report the {analysis_type} value for both curves and calculate the difference/ratio."
            )
        
        # Default context if no filters specified
        if not context_parts:
            context_parts.append(
                "SCOPE: Analyze ALL visible curves in the chart. "
                "Identify each curve clearly and report metrics for each."
            )
        
        context_section = "\n".join(context_parts)
        
        # Output format requirements
        output_format = (
            "OUTPUT FORMAT REQUIREMENTS:\n"
            "1. CURVE IDENTIFICATION: List all curves in the chart with clear identifiers "
            "(e.g., 'red solid line', 'Series A', 'blue dashed curve')\n"
            "2. NUMERICAL RESULTS: For each analyzed curve, provide:\n"
            "   - Curve identifier (name/color/style)\n"
            "   - Calculated metric value with at least 2 decimal places\n"
            "   - Units (if applicable from axis labels)\n"
            "3. CALCULATION NOTES: If helpful, briefly explain the calculation method used\n"
            "4. COMPARISON SUMMARY: If comparing curves, clearly state:\n"
            "   - Which curve has max/min value\n"
            "   - Ranking if requested\n"
            "   - Numerical differences between curves\n"
            "5. STRUCTURE: Use clear headers or bullet points to organize information by curve"
        )
        
        # User question section
        question_section = ""
        if question:
            question_section = f"\nUSER'S SPECIFIC QUESTION:\n{question}\n"
        
        # Assemble complete prompt
        full_prompt = (
            f"{base_instruction}\n\n"
            f"ANALYSIS TYPE: {analysis_type.upper().replace('_', ' ')}\n\n"
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
        if isinstance(mapping, dict) and "analyze_curve_properties" in mapping:
            mapping = mapping["analyze_curve_properties"]
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
                {
                    "role": "system", 
                    "content": "You are a precise quantitative analyst specializing in curve properties and data visualization analysis. Provide accurate numerical measurements with clear explanations."
                },
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
        print(f"[AnalyzeCurveProperties] conduct_action called for trajectory: {trajectory_id}")
        
        parsed, is_valid = self.parse_action(action)
        if not is_valid:
            print(f"[AnalyzeCurveProperties] Invalid action, returning error")
            observation = {
                "obs": "Invalid action format for analyze_curve_properties. Expected: <tool_call>{\"name\": \"analyze_curve_properties\", \"arguments\": {...}}</tool_call>",
                "invalid_reason": "parse_failed",
            }
            return observation, False, False

        arguments = parsed["arguments"]
        model_variant = arguments.get("model", "analyze_curve_properties-1")
        image_data_url = self._prepare_image(arguments, extra_field)
        if not image_data_url:
            observation = {
                "obs": "No image provided for analyze_curve_properties.",
                "invalid_reason": "missing_image",
            }
            return observation, False, False

        question = extra_field.get("question") or extra_field.get("prompt")
        prompt = self._build_prompt(question, arguments)

        try:
            content, meta = self._call_model(model_variant, image_data_url, prompt, extra_field)
            print(
                f"[analyze_curve_properties] model_variant={model_variant}, service_name={meta.get('service_name')}, "
                f"latency={meta.get('latency'):.3f}s, cost={meta.get('cost'):.6f}"
            )
            observation = {
                "obs": content,
                "latency": meta["latency"],
                "cost": meta["cost"],
                "usage": meta.get("usage", {}),
                "tool": "analyze_curve_properties",
                "model": meta["model"],
                "model_variant": model_variant
            }
            return observation, False, True
        except ValueError as exc:
            observation = {
                "obs": f"analyze_curve_properties failed: {exc}",
                "invalid_reason": "unknown_model",
            }
            return observation, False, False
        except Exception as exc:
            observation = {
                "obs": f"analyze_curve_properties failed: {exc}",
                "invalid_reason": "request_failed",
            }
            return observation, False, False
