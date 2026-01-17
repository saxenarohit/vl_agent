# VL Agent - VLM-based Image Recovery Agent

A Vision Language Model (VLM) powered agent that analyzes corrupted images and applies recovery tools to improve them for better VLM understanding.

## Overview

VL Agent is designed to work with the RobustBench framework to improve VLM accuracy on corrupted images. The agent:

1. **Analyzes** images for quality issues (blur, noise, color problems, etc.)
2. **Selects** appropriate recovery tools based on detected issues
3. **Applies** image processing operations to recover/enhance the image
4. **Verifies** improvements and iterates if needed

## Architecture

```
┌────────────────┐    ┌──────────────────────┐    ┌─────────────┐
│ Corrupted      │───►│ Recovery Agent       │───►│ Recovered   │
│ Image          │    │ (VLM + Tools)        │    │ Image       │
└────────────────┘    └──────────────────────┘    └─────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Recovery Tools    │
                    │ (PIL/scipy)       │
                    └───────────────────┘
```

## Installation

```bash
# Core dependencies
pip install pillow numpy scipy

# For VLM providers (install as needed)
pip install anthropic    # For Claude
pip install openai       # For GPT-4o
pip install google-generativeai  # For Gemini
pip install transformers torch   # For local models
```

## Quick Start

### Single Image Recovery

```python
from code.recovery_agent import recover_image
from PIL import Image

# Load a corrupted image
image = Image.open("blurry_image.png")

# Recover with default settings (Anthropic Claude)
result = recover_image(image, provider="anthropic", verbose=True)

# Save recovered image
result.recovered_image.save("recovered.png")

# Check what was done
print(f"Steps taken: {result.total_steps}")
for step in result.steps_taken:
    print(f"  - {step.tool_name}: {step.reasoning}")
```

### Using the Agent Directly

```python
from code.recovery_agent import create_recovery_agent

# Create agent with custom settings
agent = create_recovery_agent(
    provider="openai",      # or "anthropic", "gemini", "transformers"
    model="gpt-4o-mini",    # Use cheaper model for faster recovery
    max_steps=3,
    verify=True,
    verbose=True
)

# Recover image
result = agent.recover(image)

# Or use deterministic mode if you know the corruption
result = agent.recover_deterministic(image, "gaussian_blur")
```

### Applying Tools Directly

```python
from code.recovery_tools import get_tool_registry

registry = get_tool_registry()

# Get a specific tool
sharpen = registry.get("unsharp_mask")
recovered = sharpen(image, {"radius": 2, "percent": 150})

# Find tools for a specific corruption
blur_tools = registry.get_tools_for_corruption("gaussian_blur")
for tool in blur_tools:
    print(f"{tool.name}: {tool.description}")
```

## Available Recovery Tools

### Blur Recovery
| Tool | Description |
|------|-------------|
| `sharpen` | Basic sharpening for mild blur |
| `unsharp_mask` | Edge enhancement for moderate blur |
| `high_pass_sharpen` | Aggressive sharpening for severe blur |

### Noise Recovery
| Tool | Description |
|------|-------------|
| `median_filter` | Salt-and-pepper noise removal |
| `gaussian_smooth` | General noise reduction |
| `bilateral_filter` | Edge-preserving denoising |

### Color Recovery
| Tool | Description |
|------|-------------|
| `auto_contrast` | Automatic contrast stretching |
| `gamma_correction` | Brightness adjustment via gamma |
| `brightness_adjust` | Direct brightness control |
| `contrast_adjust` | Direct contrast control |
| `saturation_adjust` | Color intensity adjustment |
| `histogram_equalize` | Full histogram equalization |
| `white_balance` | Color cast correction |
| `invert_colors` | Recover from inverted images |

### Geometric Recovery
| Tool | Description |
|------|-------------|
| `rotate_correction` | Fix rotated images |
| `flip_horizontal` | Mirror horizontally |
| `flip_vertical` | Flip vertically |
| `crop_borders` | Remove fixed borders |
| `auto_crop_borders` | Auto-detect and remove borders |

### Weather Recovery
| Tool | Description |
|------|-------------|
| `dehaze` | Remove fog/haze |
| `remove_rain` | Reduce rain streaks |
| `defrost` | Reduce frost patterns |

### Compression Recovery
| Tool | Description |
|------|-------------|
| `deblock` | Reduce JPEG artifacts |
| `depixelate` | Smooth pixelated images |

### Resolution Enhancement
| Tool | Description |
|------|-------------|
| `upscale` | Increase resolution |
| `zoom_crop` | Zoom into center |
| `smart_crop` | Focus on specific region |

## Supported VLM Providers

| Provider | Models | API Key Env Var |
|----------|--------|-----------------|
| `anthropic` | claude-sonnet-4-20250514, claude-3-haiku, etc. | `ANTHROPIC_API_KEY` |
| `openai` | gpt-4o, gpt-4o-mini, etc. | `OPENAI_API_KEY` |
| `gemini` | gemini-1.5-pro, gemini-1.5-flash | `GOOGLE_API_KEY` |
| `qwen3vl` | Qwen3-VL-8B-Instruct, Qwen3-VL-4B-Instruct, etc. | N/A (local) |
| `transformers` | Qwen2.5-VL, InternVL, etc. | N/A (local) |

### Using Qwen3-VL (Recommended for local inference)

```python
from code.recovery_agent import create_recovery_agent

# Standard Qwen3-VL
agent = create_recovery_agent(
    provider="qwen3vl",
    model="Qwen/Qwen3-VL-8B-Instruct",
    max_steps=3,
    verbose=True
)

# With thinking mode (for complex analysis)
agent = create_recovery_agent(
    provider="qwen3vl",
    model="Qwen/Qwen3-VL-8B-Thinking",
    enable_thinking=True,
    max_steps=3,
    verbose=True
)
```

## Integration with RobustBench

Add to inference scripts after augmentation:

```python
from code.recovery_agent import create_recovery_agent

# In main()
if args.enable_recovery:
    recovery_agent = create_recovery_agent(
        provider=args.recovery_provider,
        model=args.recovery_model,
        max_steps=args.recovery_max_steps,
        verbose=True
    )

# In inference loop, after augmentation
images = augmenter(images, idx)

if recovery_agent:
    recovered_images = []
    for img in images:
        result = recovery_agent.recover(img)
        recovered_images.append(result.recovered_image)
    images = recovered_images
```

## Recovery Modes

### Adaptive Mode (Default)
The VLM analyzes each image to detect issues and selects appropriate tools:
```python
result = agent.recover(image)
```

### Deterministic Mode
Skip VLM analysis and directly apply tools that counter the known corruption:
```python
result = agent.recover_deterministic(image, "gaussian_blur")
```

## Example Commands

```bash
# Show available tools
python example_usage.py --demo registry

# Apply single tool
python example_usage.py --demo single_tool --image test.png --output recovered.png

# Run full agent
python example_usage.py --demo agent --image test.png --provider anthropic

# Deterministic recovery
python example_usage.py --demo deterministic --image test.png --corruption fog

# Integration guide
python example_usage.py --demo integration
```

## Project Structure

```
vl_agent/
├── code/
│   ├── __init__.py           # Package exports
│   ├── recovery_tools.py     # Tool definitions (25+ tools)
│   └── recovery_agent.py     # VLM clients and agent logic
├── example_usage.py          # Demo script
└── README.md
```

## Adding Custom Tools

```python
from code.recovery_tools import RecoveryTool, TOOL_REGISTRY

def my_custom_tool(image, params):
    # Your processing logic
    return processed_image

custom_tool = RecoveryTool(
    name="my_tool",
    display_name="My Custom Tool",
    description="Description for VLM to understand when to use",
    category="custom",
    function=my_custom_tool,
    parameters={
        "strength": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0}
    },
    counters=["some_corruption"]
)

TOOL_REGISTRY.register(custom_tool)
```

## Future Work

- [ ] Add deep learning-based recovery tools (Real-ESRGAN, NAFNet, etc.)
- [ ] Add inpainting for occlusion recovery
- [ ] Add text/watermark removal
- [ ] Optimize for batch processing
- [ ] Add caching for repeated corruptions
