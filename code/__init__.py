"""
VL Agent - VLM-based Image Recovery Agent

This package provides tools for recovering corrupted images using
Vision Language Models (VLMs) to detect and fix image quality issues.

Main components:
- recovery_tools: Image processing tools for recovery (blur, noise, color, etc.)
- recovery_agent: VLM-based agent that analyzes images and applies recovery tools

Example usage:
    from code.recovery_agent import create_recovery_agent, recover_image
    from PIL import Image

    # Quick recovery
    image = Image.open("corrupted_image.png")
    result = recover_image(image, provider="anthropic", verbose=True)
    result.recovered_image.save("recovered.png")

    # Using the agent directly
    agent = create_recovery_agent(provider="openai", model="gpt-4o", max_steps=3)
    result = agent.recover(image)
"""

from .recovery_tools import (
    RecoveryTool,
    ToolRegistry,
    TOOL_REGISTRY,
    get_tool_registry,
    # Metrics
    measure_sharpness,
    measure_contrast,
    measure_brightness,
    measure_noise,
    measure_saturation,
    measure_colorfulness,
    measure_jpeg_artifacts,
    get_all_metrics,
    format_metrics_for_vlm,
)

from .recovery_agent import (
    VLMClient,
    AnthropicVLMClient,
    OpenAIVLMClient,
    GeminiVLMClient,
    LocalTransformersVLMClient,
    Qwen3VLClient,
    create_vlm_client,
    RecoveryAgent,
    RecoveryResult,
    RecoveryStep,
    create_recovery_agent,
    recover_image,
    batch_recover,
)

__all__ = [
    # Tools
    "RecoveryTool",
    "ToolRegistry",
    "TOOL_REGISTRY",
    "get_tool_registry",
    # Metrics
    "measure_sharpness",
    "measure_contrast",
    "measure_brightness",
    "measure_noise",
    "measure_saturation",
    "measure_colorfulness",
    "measure_jpeg_artifacts",
    "get_all_metrics",
    "format_metrics_for_vlm",
    # VLM Clients
    "VLMClient",
    "AnthropicVLMClient",
    "OpenAIVLMClient",
    "GeminiVLMClient",
    "LocalTransformersVLMClient",
    "Qwen3VLClient",
    "create_vlm_client",
    # Agent
    "RecoveryAgent",
    "RecoveryResult",
    "RecoveryStep",
    "create_recovery_agent",
    "recover_image",
    "batch_recover",
]

__version__ = "0.1.0"
