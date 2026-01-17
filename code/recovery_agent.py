"""
VLM-based image recovery agent.

This module provides a VLM-powered agent that can analyze images for quality issues
and apply recovery tools to improve them. The agent supports multiple VLM backends
(Anthropic, OpenAI, Gemini, local Transformers models).

The agent loop:
1. Analyze image for quality issues (blur, noise, color problems, etc.)
2. Select appropriate recovery tool based on detected issues
3. Apply the tool to recover/enhance the image
4. Optionally verify the improvement
5. Repeat until image is acceptable or max steps reached
"""

import os
import re
import json
import copy
import base64
import hashlib
from io import BytesIO
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image

try:
    from .recovery_tools import (
        RecoveryTool, ToolRegistry, TOOL_REGISTRY,
        get_all_metrics, format_metrics_for_vlm
    )
except ImportError:
    from recovery_tools import (
        RecoveryTool, ToolRegistry, TOOL_REGISTRY,
        get_all_metrics, format_metrics_for_vlm
    )


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

ANALYSIS_PROMPT_TEMPLATE = """You are an image quality analyst. Analyze this image for visual quality issues that might affect a Vision Language Model's ability to understand its content.

Look for these types of issues:
1. **Blur**: Is the image blurry, out of focus, or has motion blur?
2. **Noise**: Does the image have visible noise (grainy, speckled, salt-and-pepper)?
3. **Color issues**: Is contrast too low/high? Colors washed out or oversaturated? Unusual color cast?
4. **Geometric distortion**: Is the image rotated, skewed, flipped, or has perspective issues?
5. **Weather effects**: Is there fog, haze, rain streaks, or frost patterns?
6. **Compression artifacts**: Are there JPEG blocking artifacts or pixelation?
7. **Resolution**: Is the image too small or low resolution?
8. **Occlusion**: Are there overlays, watermarks, text overlays, or borders obscuring content?

Respond in this exact JSON format:
```json
{
    "quality_assessment": {
        "overall_quality": "good|moderate|poor",
        "issues_detected": [
            {
                "type": "issue_type",
                "severity": "mild|moderate|severe",
                "description": "brief description"
            }
        ]
    },
    "recommended_action": "none|recover",
    "reasoning": "brief explanation"
}
```

If the image quality is acceptable for VLM analysis, set recommended_action to "none".
If issues are detected that could be improved, set recommended_action to "recover".
"""

TOOL_SELECTION_PROMPT_TEMPLATE = """You are an image recovery specialist. Based on the quality issues identified, select the best recovery tool to apply.

**Detected Issues:**
{issues_json}

**Available Tools:**
{tool_descriptions}

Select ONE tool to apply. Consider:
1. Match the tool to the specific issue type
2. Start with the most impactful issue first
3. Use conservative parameters to avoid over-correction

Respond in this exact JSON format:
```json
{{
    "selected_tool": "tool_name",
    "parameters": {{
        "param1": value1,
        "param2": value2
    }},
    "reasoning": "why this tool for this issue"
}}
```

If no recovery is needed or no suitable tool exists, respond:
```json
{{
    "selected_tool": "none",
    "parameters": {{}},
    "reasoning": "explanation"
}}
```
"""

VERIFICATION_PROMPT_TEMPLATE = """You are evaluating whether an image recovery operation improved the image quality.

The recovery tool "{tool_name}" was applied to address detected issues.

Compare the recovered image to what you would expect from a high-quality image. Assess if:
1. The targeted issue was reduced or eliminated
2. No significant new artifacts were introduced
3. The image is now more suitable for VLM analysis

Respond in JSON format:
```json
{{
    "improvement": "yes|no|partial",
    "continue_recovery": true|false,
    "remaining_issues": ["issue1", "issue2"],
    "reasoning": "brief explanation"
}}
```

Set continue_recovery to true if there are still significant issues that could be addressed.
"""


# =============================================================================
# METRIC-BASED PROMPT TEMPLATES (New workflow)
# =============================================================================

METRIC_DECISION_PROMPT_TEMPLATE = """You are an image recovery expert. Your job is to analyze image quality metrics and decide if and how to improve the image.

## Current Image Metrics
{metrics_formatted}

## Available Recovery Tools
{tool_descriptions}

## Your Task
Based on the metrics above:
1. Identify which metrics indicate problems (compare to reference values)
2. Decide if recovery is needed
3. If yes, select ONE tool and specify its parameters

### Decision Guidelines
- **Sharpness < 500**: Image is blurry, consider sharpen/unsharp_mask/high_pass_sharpen
- **Contrast pixel_range < 100**: Low contrast, consider auto_contrast/contrast_adjust
- **Brightness mean < 50 or > 200**: Exposure issues, consider brightness_adjust/gamma_correction
- **Noise > 20**: Noisy image, consider median_filter/gaussian_smooth/bilateral_filter
- **Saturation mean < 0.15**: Desaturated, consider saturation_adjust
- **JPEG artifacts > 15**: Compression artifacts, consider deblock

Respond in JSON format:
```json
{{
    "needs_recovery": true|false,
    "primary_issue": "blur|contrast|brightness|noise|saturation|jpeg_artifacts|none",
    "selected_tool": "tool_name or none",
    "parameters": {{
        "param1": value1
    }},
    "reasoning": "why this tool with these parameters based on the metrics"
}}
```

IMPORTANT: Choose parameters based on the severity shown in metrics. For example:
- Sharpness of 300 (blurry) → use stronger sharpening (higher percent/strength)
- Sharpness of 450 (slightly blurry) → use milder sharpening
- Contrast pixel_range of 50 → needs strong contrast boost
- Contrast pixel_range of 90 → needs mild contrast boost
"""

METRIC_DECISION_WITH_QUESTION_PROMPT_TEMPLATE = """You are an image recovery expert. Your job is to analyze image quality metrics and decide if and how to improve the image FOR A SPECIFIC TASK.

## Task Context
The image will be used to answer this question:
{question}

## Current Image Metrics
{metrics_formatted}

## Available Recovery Tools
{tool_descriptions}

## Your Task
Based on the metrics AND the question:
1. Identify which metrics indicate problems (compare to reference values)
2. Consider what parts of the image are most important for answering the question
3. Decide if recovery is needed and prioritize accordingly
4. If yes, select ONE tool and specify its parameters

### Decision Guidelines
- **Sharpness < 500**: Image is blurry, consider sharpen/unsharp_mask/high_pass_sharpen
- **Contrast pixel_range < 100**: Low contrast, consider auto_contrast/contrast_adjust
- **Brightness mean < 50 or > 200**: Exposure issues, consider brightness_adjust/gamma_correction
- **Noise > 20**: Noisy image, consider median_filter/gaussian_smooth/bilateral_filter
- **Saturation mean < 0.15**: Desaturated, consider saturation_adjust
- **JPEG artifacts > 15**: Compression artifacts, consider deblock

### Task-Aware Prioritization
- If question asks about TEXT (signs, labels, numbers): prioritize sharpening
- If question asks about COLORS: prioritize contrast/saturation recovery
- If question asks about DETAILS (small objects): prioritize sharpening and noise reduction
- If question asks about OVERALL SCENE: balance all metrics

Respond in JSON format:
```json
{{
    "needs_recovery": true|false,
    "primary_issue": "blur|contrast|brightness|noise|saturation|jpeg_artifacts|none",
    "selected_tool": "tool_name or none",
    "parameters": {{
        "param1": value1
    }},
    "reasoning": "why this tool with these parameters based on the metrics AND the question"
}}
```
"""

METRIC_VERIFICATION_PROMPT_TEMPLATE = """You are evaluating if the image recovery was successful by comparing metrics.

## BEFORE Recovery
{metrics_before}

## AFTER Recovery (tool: {tool_name})
{metrics_after}

## Your Task
Compare the before/after metrics and decide:
1. Did the targeted metric improve?
2. Did any other metrics get worse?
3. Should we continue with more recovery?

Respond in JSON format:
```json
{{
    "improvement": "yes|no|partial",
    "metric_changes": {{
        "improved": ["list of improved metrics"],
        "degraded": ["list of degraded metrics"],
        "unchanged": ["list of unchanged metrics"]
    }},
    "continue_recovery": true|false,
    "reasoning": "analysis of the metric changes"
}}
```

Guidelines:
- "yes" = target metric significantly improved, no major degradation
- "partial" = some improvement but target not fully fixed
- "no" = no improvement or made things worse
- continue_recovery = true if other metrics still need fixing
"""


# =============================================================================
# VLM CLIENT ABSTRACTION
# =============================================================================

class VLMClient(ABC):
    """Abstract base class for VLM clients."""

    @abstractmethod
    def analyze_image(self, image: Image.Image, prompt: str) -> str:
        """Send image and prompt to VLM, return response text."""
        pass

    @staticmethod
    def encode_image_base64(image: Image.Image, format: str = "PNG") -> str:
        """Encode PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


class AnthropicVLMClient(VLMClient):
    """Anthropic Claude client for agent VLM."""

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment or arguments")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def analyze_image(self, image: Image.Image, prompt: str) -> str:
        base64_image = self.encode_image_base64(image.convert('RGB'))
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        return response.content[0].text


class OpenAIVLMClient(VLMClient):
    """OpenAI GPT-4V client for agent VLM."""

    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or arguments")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def analyze_image(self, image: Image.Image, prompt: str) -> str:
        base64_image = self.encode_image_base64(image.convert('RGB'))
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        return response.choices[0].message.content


class GeminiVLMClient(VLMClient):
    """Google Gemini client for agent VLM."""

    def __init__(self, api_key: str = None, model: str = "gemini-1.5-pro"):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")

        api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment or arguments")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def analyze_image(self, image: Image.Image, prompt: str) -> str:
        response = self.model.generate_content([prompt, image.convert('RGB')])
        return response.text


class LocalTransformersVLMClient(VLMClient):
    """Local HuggingFace Transformers client for agent VLM."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "auto"):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError:
            raise ImportError("transformers and torch required. Install with: pip install transformers torch")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        self.model_name = model_name

    def analyze_image(self, image: Image.Image, prompt: str) -> str:
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert('RGB')},
                {"type": "text", "text": prompt}
            ]
        }]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output_ids = self.model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        generated_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
        return self.processor.decode(generated_ids, skip_special_tokens=True)


class Qwen3VLClient(VLMClient):
    """Qwen3-VL client for agent VLM (local model).

    Supports: Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct, Qwen3-VL-4B-Thinking, etc.
    """

    # Models that need AutoModelForImageTextToText (MoE)
    MOE_MODELS = ["Qwen/Qwen3-VL-30B-A3B-Instruct"]

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "auto",
        use_flash_attn: bool = False,
        enable_thinking: bool = False
    ):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError:
            raise ImportError("transformers and torch required. Install with: pip install transformers torch")

        self.model_name = model
        self.enable_thinking = enable_thinking or "Thinking" in model

        # Model loading kwargs
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": device,
        }
        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Load model - use appropriate class based on model type
        if model in self.MOE_MODELS:
            self.model = AutoModelForImageTextToText.from_pretrained(model, **model_kwargs)
        else:
            try:
                from transformers import Qwen3VLForConditionalGeneration
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(model, **model_kwargs)
            except ImportError:
                # Fallback to AutoModel if Qwen3VL class not available
                self.model = AutoModelForImageTextToText.from_pretrained(model, **model_kwargs)

        self.processor = AutoProcessor.from_pretrained(model)
        print(f"Loaded Qwen3-VL model: {model} (thinking={self.enable_thinking})")

    def analyze_image(self, image: Image.Image, prompt: str) -> str:
        import torch

        # Prepare messages in Qwen3-VL format
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert('RGB')},
                {"type": "text", "text": prompt}
            ]
        }]

        # Process inputs using chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.enable_thinking
        )
        inputs = inputs.to(self.model.device)

        # Generate with appropriate params based on thinking mode
        if self.enable_thinking:
            # Thinking mode: use sampling
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=8192,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                )
        else:
            # Non-thinking mode: greedy decoding
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                )

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        generated_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return generated_text


def create_vlm_client(provider: str, **kwargs) -> VLMClient:
    """Factory function to create VLM client.

    Args:
        provider: One of "anthropic", "openai", "gemini", "transformers", "qwen3vl"
        **kwargs: Provider-specific arguments (api_key, model, device, etc.)

    Returns:
        VLMClient instance

    Examples:
        # API-based providers
        client = create_vlm_client("anthropic", model="claude-sonnet-4-20250514")
        client = create_vlm_client("openai", model="gpt-4o")

        # Local Qwen3-VL
        client = create_vlm_client("qwen3vl", model="Qwen/Qwen3-VL-8B-Instruct")
        client = create_vlm_client("qwen3vl", model="Qwen/Qwen3-VL-8B-Thinking", enable_thinking=True)
    """
    providers = {
        "anthropic": AnthropicVLMClient,
        "openai": OpenAIVLMClient,
        "gemini": GeminiVLMClient,
        "transformers": LocalTransformersVLMClient,
        "qwen3vl": Qwen3VLClient,
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")

    return providers[provider](**kwargs)


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from VLM response, handling markdown code blocks."""
    # Try to find JSON in code blocks first
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: return empty dict
    return {}


def parse_analysis_response(response: str) -> Tuple[str, List[Dict], str]:
    """Parse image analysis response.

    Returns:
        Tuple of (overall_quality, issues_detected, recommended_action)
    """
    data = extract_json_from_response(response)

    quality = data.get("quality_assessment", {})
    overall = quality.get("overall_quality", "unknown")
    issues = quality.get("issues_detected", [])
    action = data.get("recommended_action", "none")

    return overall, issues, action


def parse_tool_selection_response(response: str) -> Tuple[str, Dict[str, Any], str]:
    """Parse tool selection response.

    Returns:
        Tuple of (tool_name, parameters, reasoning)
    """
    data = extract_json_from_response(response)

    tool_name = data.get("selected_tool", "none")
    params = data.get("parameters", {})
    reasoning = data.get("reasoning", "")

    return tool_name, params, reasoning


def parse_verification_response(response: str) -> Tuple[str, bool, List[str], str]:
    """Parse verification response.

    Returns:
        Tuple of (improvement, continue_recovery, remaining_issues, reasoning)
    """
    data = extract_json_from_response(response)

    improvement = data.get("improvement", "no")
    continue_recovery = data.get("continue_recovery", False)
    remaining_issues = data.get("remaining_issues", [])
    reasoning = data.get("reasoning", "")

    return improvement, continue_recovery, remaining_issues, reasoning


def parse_metric_decision_response(response: str) -> Tuple[bool, str, str, Dict[str, Any], str]:
    """Parse metric-based decision response.

    Returns:
        Tuple of (needs_recovery, primary_issue, tool_name, parameters, reasoning)
    """
    data = extract_json_from_response(response)

    needs_recovery = data.get("needs_recovery", False)
    primary_issue = data.get("primary_issue", "none")
    tool_name = data.get("selected_tool", "none")
    params = data.get("parameters", {})
    reasoning = data.get("reasoning", "")

    return needs_recovery, primary_issue, tool_name, params, reasoning


def parse_metric_verification_response(response: str) -> Tuple[str, Dict, bool, str]:
    """Parse metric-based verification response.

    Returns:
        Tuple of (improvement, metric_changes, continue_recovery, reasoning)
    """
    data = extract_json_from_response(response)

    improvement = data.get("improvement", "no")
    metric_changes = data.get("metric_changes", {})
    continue_recovery = data.get("continue_recovery", False)
    reasoning = data.get("reasoning", "")

    return improvement, metric_changes, continue_recovery, reasoning


# =============================================================================
# RECOVERY AGENT
# =============================================================================

@dataclass
class RecoveryStep:
    """Record of a single recovery step."""
    step_number: int
    tool_name: str
    parameters: Dict[str, Any]
    reasoning: str
    improvement: str
    image_before_hash: str
    image_after_hash: str
    metrics_before: Optional[Dict] = None
    metrics_after: Optional[Dict] = None
    # For full trace
    vlm_decision_prompt: Optional[str] = None
    vlm_decision_response: Optional[str] = None
    vlm_verify_prompt: Optional[str] = None
    vlm_verify_response: Optional[str] = None

    def to_trace_dict(self) -> Dict[str, Any]:
        """Convert to dict for trace saving (excludes image hashes)."""
        return {
            "step_number": self.step_number,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "reasoning": self.reasoning,
            "improvement": self.improvement,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "vlm_decision_prompt": self.vlm_decision_prompt,
            "vlm_decision_response": self.vlm_decision_response,
            "vlm_verify_prompt": self.vlm_verify_prompt,
            "vlm_verify_response": self.vlm_verify_response,
        }


@dataclass
class RecoveryResult:
    """Complete result of recovery process."""
    original_image: Image.Image
    recovered_image: Image.Image
    steps_taken: List[RecoveryStep]
    initial_issues: List[Dict]
    final_quality: str
    total_steps: int
    early_stopped: bool
    stop_reason: str
    initial_metrics: Optional[Dict] = None
    final_metrics: Optional[Dict] = None

    def to_trace_dict(self) -> Dict[str, Any]:
        """Export full trace as dict (for saving to JSON).

        Does NOT include images - only the tool calls and metrics.
        Images can be reconstructed by replaying the tool calls.
        """
        return {
            "initial_metrics": self.initial_metrics,
            "final_metrics": self.final_metrics,
            "final_quality": self.final_quality,
            "total_steps": self.total_steps,
            "early_stopped": self.early_stopped,
            "stop_reason": self.stop_reason,
            "steps": [step.to_trace_dict() for step in self.steps_taken],
            # Tool call sequence for easy replay
            "tool_sequence": [
                {"tool": step.tool_name, "params": step.parameters}
                for step in self.steps_taken
            ],
        }


class RecoveryAgent:
    """VLM-based image recovery agent.

    The agent analyzes images for quality issues and applies recovery tools
    to improve them for better VLM understanding.
    """

    def __init__(
        self,
        vlm_client: VLMClient,
        tool_registry: ToolRegistry = None,
        max_steps: int = 5,
        verify_improvements: bool = True,
        verbose: bool = False,
    ):
        """Initialize the recovery agent.

        Args:
            vlm_client: VLM client for image analysis and tool selection
            tool_registry: Registry of available recovery tools (default: global registry)
            max_steps: Maximum number of recovery steps per image
            verify_improvements: Whether to verify improvements after each step
            verbose: Print progress information
        """
        self.vlm_client = vlm_client
        self.tool_registry = tool_registry or TOOL_REGISTRY
        self.max_steps = max_steps
        self.verify_improvements = verify_improvements
        self.verbose = verbose

    def _log(self, msg: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[RecoveryAgent] {msg}")

    def _hash_image(self, image: Image.Image) -> str:
        """Generate a short hash for tracking image changes."""
        return hashlib.md5(image.tobytes()).hexdigest()[:8]

    def analyze_image(self, image: Image.Image) -> Tuple[str, List[Dict], str]:
        """Analyze image for quality issues.

        Args:
            image: PIL Image to analyze

        Returns:
            Tuple of (overall_quality, issues_detected, recommended_action)
        """
        self._log("Analyzing image quality...")
        response = self.vlm_client.analyze_image(image, ANALYSIS_PROMPT_TEMPLATE)
        self._log(f"Analysis response: {response[:200]}...")
        return parse_analysis_response(response)

    def select_tool(self, image: Image.Image, issues: List[Dict]) -> Tuple[str, Dict[str, Any], str]:
        """Select recovery tool based on detected issues.

        Args:
            image: The image being processed (for context)
            issues: List of detected issues

        Returns:
            Tuple of (tool_name, parameters, reasoning)
        """
        self._log("Selecting recovery tool...")

        prompt = TOOL_SELECTION_PROMPT_TEMPLATE.format(
            issues_json=json.dumps(issues, indent=2),
            tool_descriptions=self.tool_registry.get_tool_descriptions_for_vlm()
        )

        response = self.vlm_client.analyze_image(image, prompt)
        self._log(f"Tool selection response: {response[:200]}...")
        return parse_tool_selection_response(response)

    def apply_tool(self, image: Image.Image, tool_name: str, params: Dict[str, Any]) -> Image.Image:
        """Apply recovery tool to image.

        Args:
            image: PIL Image to process
            tool_name: Name of the tool to apply
            params: Tool parameters

        Returns:
            Processed image
        """
        tool = self.tool_registry.get(tool_name)
        if tool is None:
            self._log(f"Tool '{tool_name}' not found, returning original image")
            return image

        self._log(f"Applying tool '{tool_name}' with params {params}")
        try:
            return tool(image, params)
        except Exception as e:
            self._log(f"Error applying tool: {e}")
            return image

    def verify_improvement(
        self,
        original: Image.Image,
        recovered: Image.Image,
        tool_name: str
    ) -> Tuple[str, bool, List[str]]:
        """Verify if recovery improved the image.

        Args:
            original: Image before recovery
            recovered: Image after recovery
            tool_name: Name of the applied tool

        Returns:
            Tuple of (improvement, continue_recovery, remaining_issues)
        """
        if not self.verify_improvements:
            return "assumed", True, []

        self._log("Verifying improvement...")

        prompt = VERIFICATION_PROMPT_TEMPLATE.format(tool_name=tool_name)
        response = self.vlm_client.analyze_image(recovered, prompt)
        self._log(f"Verification response: {response[:200]}...")

        improvement, continue_recovery, remaining_issues, _ = parse_verification_response(response)
        return improvement, continue_recovery, remaining_issues

    def recover(
        self,
        image: Image.Image,
        known_corruption: str = None
    ) -> RecoveryResult:
        """Run the full recovery loop on an image.

        Args:
            image: Input image to recover
            known_corruption: If known, the corruption type (for deterministic mode)

        Returns:
            RecoveryResult with recovered image and metadata
        """
        original_image = image.copy()
        current_image = image.copy()
        steps = []

        # Initial analysis
        overall_quality, issues, action = self.analyze_image(current_image)
        initial_issues = copy.deepcopy(issues)

        self._log(f"Initial quality: {overall_quality}, issues: {len(issues)}, action: {action}")

        # If using known corruption, add it as a known issue for deterministic recovery
        if known_corruption:
            tools_for_corruption = self.tool_registry.get_tools_for_corruption(known_corruption)
            if tools_for_corruption:
                self._log(f"Using deterministic mode for known corruption: {known_corruption}")
                issues = [{"type": known_corruption, "severity": "moderate", "description": f"known corruption: {known_corruption}"}]
                action = "recover"

        # Check if recovery is needed
        if action == "none" and not known_corruption:
            self._log("No recovery needed, image quality is acceptable")
            return RecoveryResult(
                original_image=original_image,
                recovered_image=current_image,
                steps_taken=[],
                initial_issues=initial_issues,
                final_quality=overall_quality,
                total_steps=0,
                early_stopped=True,
                stop_reason="no_issues_detected"
            )

        # Recovery loop
        for step_num in range(self.max_steps):
            self._log(f"=== Step {step_num + 1}/{self.max_steps} ===")

            # Select tool
            tool_name, params, reasoning = self.select_tool(current_image, issues)

            if tool_name == "none" or tool_name is None:
                self._log("No tool selected, stopping recovery")
                break

            # Apply tool
            before_hash = self._hash_image(current_image)
            recovered_image = self.apply_tool(current_image, tool_name, params)
            after_hash = self._hash_image(recovered_image)

            # Verify improvement
            improvement, continue_recovery, remaining_issues = self.verify_improvement(
                current_image, recovered_image, tool_name
            )

            # Record step
            steps.append(RecoveryStep(
                step_number=step_num + 1,
                tool_name=tool_name,
                parameters=params,
                reasoning=reasoning,
                improvement=improvement,
                image_before_hash=before_hash,
                image_after_hash=after_hash
            ))

            self._log(f"Step {step_num + 1} complete: tool={tool_name}, improvement={improvement}")

            # Update current image if improvement detected
            if improvement in ("yes", "partial"):
                current_image = recovered_image

            # Check if we should continue
            if not continue_recovery:
                self._log("Verification says no more recovery needed")
                break

            # Update issues for next iteration
            if remaining_issues:
                issues = [{"type": issue, "severity": "moderate", "description": issue} for issue in remaining_issues]
            else:
                # Re-analyze if no remaining issues specified
                overall_quality, issues, action = self.analyze_image(current_image)
                if action == "none" or not issues:
                    self._log("No more issues to address")
                    break

        # Final assessment
        final_quality, _, _ = self.analyze_image(current_image)

        return RecoveryResult(
            original_image=original_image,
            recovered_image=current_image,
            steps_taken=steps,
            initial_issues=initial_issues,
            final_quality=final_quality,
            total_steps=len(steps),
            early_stopped=len(steps) < self.max_steps,
            stop_reason="completed" if len(steps) == self.max_steps else "converged"
        )

    def recover_deterministic(
        self,
        image: Image.Image,
        corruption_name: str,
        max_tools: int = 1
    ) -> RecoveryResult:
        """Apply deterministic recovery based on known corruption type.

        This skips VLM analysis and directly applies tools that counter
        the specified corruption.

        Args:
            image: Input image to recover
            corruption_name: Name of the corruption to counter
            max_tools: Maximum number of tools to apply

        Returns:
            RecoveryResult with recovered image and metadata
        """
        original_image = image.copy()
        current_image = image.copy()
        steps = []

        # Get tools for this corruption
        tools = self.tool_registry.get_tools_for_corruption(corruption_name)

        if not tools:
            self._log(f"No tools found for corruption: {corruption_name}")
            return RecoveryResult(
                original_image=original_image,
                recovered_image=current_image,
                steps_taken=[],
                initial_issues=[{"type": corruption_name, "severity": "unknown"}],
                final_quality="unknown",
                total_steps=0,
                early_stopped=True,
                stop_reason="no_tools_available"
            )

        self._log(f"Found {len(tools)} tools for corruption '{corruption_name}'")

        # Apply tools
        for i, tool in enumerate(tools[:max_tools]):
            self._log(f"Applying tool: {tool.name}")

            before_hash = self._hash_image(current_image)
            recovered_image = tool(current_image, {})  # Use default params
            after_hash = self._hash_image(recovered_image)

            steps.append(RecoveryStep(
                step_number=i + 1,
                tool_name=tool.name,
                parameters={},
                reasoning=f"Deterministic counter for {corruption_name}",
                improvement="assumed",
                image_before_hash=before_hash,
                image_after_hash=after_hash
            ))

            current_image = recovered_image

        return RecoveryResult(
            original_image=original_image,
            recovered_image=current_image,
            steps_taken=steps,
            initial_issues=[{"type": corruption_name, "severity": "known"}],
            final_quality="improved",
            total_steps=len(steps),
            early_stopped=False,
            stop_reason="deterministic_complete"
        )

    def recover_with_metrics(
        self,
        image: Image.Image,
        max_iterations: int = 5,
        question: str = None
    ) -> RecoveryResult:
        """Run metric-based recovery loop.

        New workflow:
        1. Compute all image quality metrics
        2. Send metrics + image to VLM (optionally with question for task-aware recovery)
        3. VLM decides tool + parameters based on metrics
        4. Apply tool
        5. Re-compute metrics
        6. VLM verifies improvement by comparing metrics
        7. Repeat until satisfied or max iterations

        Args:
            image: Input image to recover
            max_iterations: Maximum recovery iterations (default: 5)
            question: Optional question/task context for task-aware recovery

        Returns:
            RecoveryResult with recovered image, metrics, and metadata
        """
        original_image = image.copy()
        current_image = image.copy()
        steps = []

        # Compute initial metrics
        initial_metrics = get_all_metrics(current_image)
        current_metrics = initial_metrics.copy()

        self._log("=== Metric-Based Recovery ===")
        self._log(f"Initial metrics:")
        self._log(f"  Sharpness: {initial_metrics['sharpness']:.1f}")
        self._log(f"  Contrast (pixel_range): {initial_metrics['contrast']['pixel_range']}")
        self._log(f"  Brightness (mean): {initial_metrics['brightness']['mean']}")
        self._log(f"  Noise: {initial_metrics['noise']}")
        self._log(f"  Saturation (mean): {initial_metrics['saturation']['mean']}")

        for iteration in range(max_iterations):
            self._log(f"\n=== Iteration {iteration + 1}/{max_iterations} ===")

            # Format metrics for VLM
            metrics_formatted = format_metrics_for_vlm(current_metrics)

            # Ask VLM for decision (use question-aware prompt if question provided)
            if question:
                prompt = METRIC_DECISION_WITH_QUESTION_PROMPT_TEMPLATE.format(
                    question=question,
                    metrics_formatted=metrics_formatted,
                    tool_descriptions=self.tool_registry.get_tool_descriptions_for_vlm()
                )
            else:
                prompt = METRIC_DECISION_PROMPT_TEMPLATE.format(
                    metrics_formatted=metrics_formatted,
                    tool_descriptions=self.tool_registry.get_tool_descriptions_for_vlm()
                )

            self._log("Asking VLM for recovery decision...")
            response = self.vlm_client.analyze_image(current_image, prompt)
            self._log(f"VLM response: {response[:300]}...")

            # Parse decision
            needs_recovery, primary_issue, tool_name, params, reasoning = parse_metric_decision_response(response)

            self._log(f"Decision: needs_recovery={needs_recovery}, issue={primary_issue}, tool={tool_name}")
            self._log(f"Reasoning: {reasoning[:200]}...")

            # Check if VLM says no recovery needed
            if not needs_recovery or tool_name == "none" or tool_name is None:
                self._log("VLM decided no more recovery needed")
                break

            # Apply the selected tool
            before_hash = self._hash_image(current_image)
            metrics_before = current_metrics.copy()

            tool = self.tool_registry.get(tool_name)
            if tool is None:
                self._log(f"Tool '{tool_name}' not found, skipping")
                continue

            self._log(f"Applying tool '{tool_name}' with params: {params}")
            try:
                recovered_image = tool(current_image, params)
            except Exception as e:
                self._log(f"Error applying tool: {e}")
                continue

            after_hash = self._hash_image(recovered_image)

            # Compute new metrics
            metrics_after = get_all_metrics(recovered_image)

            self._log(f"Metrics after recovery:")
            self._log(f"  Sharpness: {metrics_before['sharpness']:.1f} → {metrics_after['sharpness']:.1f}")
            self._log(f"  Contrast: {metrics_before['contrast']['pixel_range']} → {metrics_after['contrast']['pixel_range']}")
            self._log(f"  Brightness: {metrics_before['brightness']['mean']} → {metrics_after['brightness']['mean']}")
            self._log(f"  Noise: {metrics_before['noise']} → {metrics_after['noise']}")

            # Ask VLM to verify improvement
            verify_prompt = None
            verify_response = None
            if self.verify_improvements:
                verify_prompt = METRIC_VERIFICATION_PROMPT_TEMPLATE.format(
                    metrics_before=format_metrics_for_vlm(metrics_before),
                    tool_name=tool_name,
                    metrics_after=format_metrics_for_vlm(metrics_after)
                )

                self._log("Asking VLM to verify improvement...")
                verify_response = self.vlm_client.analyze_image(recovered_image, verify_prompt)
                self._log(f"Verification response: {verify_response[:200]}...")

                improvement, metric_changes, continue_recovery, verify_reasoning = parse_metric_verification_response(verify_response)
                self._log(f"Verification: improvement={improvement}, continue={continue_recovery}")
            else:
                # Skip verification, assume improvement
                improvement = "assumed"
                continue_recovery = True

            # Record step (with full trace info)
            steps.append(RecoveryStep(
                step_number=iteration + 1,
                tool_name=tool_name,
                parameters=params,
                reasoning=reasoning,
                improvement=improvement,
                image_before_hash=before_hash,
                image_after_hash=after_hash,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                vlm_decision_prompt=prompt,
                vlm_decision_response=response,
                vlm_verify_prompt=verify_prompt,
                vlm_verify_response=verify_response,
            ))

            # Update current image and metrics if improvement detected
            if improvement in ("yes", "partial", "assumed"):
                current_image = recovered_image
                current_metrics = metrics_after
            else:
                self._log("No improvement detected, keeping previous image")

            # Check if we should continue
            if not continue_recovery:
                self._log("VLM says no more recovery needed")
                break

        # Final metrics
        final_metrics = get_all_metrics(current_image)

        # Determine final quality based on metrics
        quality_score = 0
        if final_metrics['sharpness'] > 500:
            quality_score += 1
        if final_metrics['contrast']['pixel_range'] > 100:
            quality_score += 1
        if 50 < final_metrics['brightness']['mean'] < 200:
            quality_score += 1
        if final_metrics['noise'] < 20:
            quality_score += 1

        final_quality = "good" if quality_score >= 3 else "moderate" if quality_score >= 2 else "poor"

        self._log(f"\n=== Recovery Complete ===")
        self._log(f"Total steps: {len(steps)}")
        self._log(f"Final quality: {final_quality}")
        self._log(f"Final sharpness: {final_metrics['sharpness']:.1f}")
        self._log(f"Final contrast: {final_metrics['contrast']['pixel_range']}")

        return RecoveryResult(
            original_image=original_image,
            recovered_image=current_image,
            steps_taken=steps,
            initial_issues=[{"type": "metric_based", "metrics": initial_metrics}],
            final_quality=final_quality,
            total_steps=len(steps),
            early_stopped=len(steps) < max_iterations,
            stop_reason="converged" if len(steps) < max_iterations else "max_iterations",
            initial_metrics=initial_metrics,
            final_metrics=final_metrics
        )


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def batch_recover(
    images: List[Image.Image],
    agent: RecoveryAgent,
    known_corruptions: List[str] = None,
    use_deterministic: bool = False,
    progress_callback=None
) -> List[RecoveryResult]:
    """Apply recovery agent to a batch of images.

    Args:
        images: List of PIL images
        agent: RecoveryAgent instance
        known_corruptions: Optional list of known corruption types (one per image)
        use_deterministic: If True and known_corruptions provided, use deterministic mode
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        List of RecoveryResult objects
    """
    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(images), total=len(images), desc="Recovering images")
    except ImportError:
        iterator = enumerate(images)

    results = []

    for idx, image in iterator:
        corruption = known_corruptions[idx] if known_corruptions else None

        if use_deterministic and corruption:
            result = agent.recover_deterministic(image, corruption)
        else:
            result = agent.recover(image, known_corruption=corruption)

        results.append(result)

        if progress_callback:
            progress_callback(idx + 1, len(images))

    return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_recovery_agent(
    provider: str = "anthropic",
    model: str = None,
    api_key: str = None,
    max_steps: int = 3,
    verify: bool = True,
    verbose: bool = False,
    **kwargs
) -> RecoveryAgent:
    """Create a recovery agent with specified configuration.

    Args:
        provider: VLM provider ("anthropic", "openai", "gemini", "transformers", "qwen3vl")
        model: Model name (provider-specific, uses default if None)
        api_key: API key (uses environment variable if None)
        max_steps: Maximum recovery steps per image
        verify: Whether to verify improvements
        verbose: Print progress information
        **kwargs: Additional provider-specific arguments (e.g., enable_thinking for qwen3vl)

    Returns:
        Configured RecoveryAgent instance
    """
    # Default models per provider
    default_models = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "gemini": "gemini-1.5-pro",
        "transformers": "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen3vl": "Qwen/Qwen3-VL-8B-Instruct",
    }

    model = model or default_models.get(provider)

    # Create VLM client
    client_kwargs = {"model": model}
    if api_key:
        client_kwargs["api_key"] = api_key

    # Add any extra kwargs (e.g., enable_thinking for qwen3vl)
    client_kwargs.update(kwargs)

    vlm_client = create_vlm_client(provider, **client_kwargs)

    # Create agent
    return RecoveryAgent(
        vlm_client=vlm_client,
        max_steps=max_steps,
        verify_improvements=verify,
        verbose=verbose
    )


def recover_image(
    image: Image.Image,
    provider: str = "anthropic",
    model: str = None,
    known_corruption: str = None,
    max_steps: int = 3,
    verbose: bool = False
) -> RecoveryResult:
    """Convenience function to recover a single image.

    Args:
        image: PIL Image to recover
        provider: VLM provider
        model: Model name
        known_corruption: Optional known corruption type
        max_steps: Maximum recovery steps
        verbose: Print progress

    Returns:
        RecoveryResult
    """
    agent = create_recovery_agent(
        provider=provider,
        model=model,
        max_steps=max_steps,
        verbose=verbose
    )
    return agent.recover(image, known_corruption=known_corruption)
