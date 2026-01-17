#!/usr/bin/env python3
"""
Example usage of the VL Agent for image recovery.

This script demonstrates how to use the recovery agent to:
1. Recover a single corrupted image
2. Use deterministic recovery for known corruptions
3. Integrate with the robustbench framework
"""

import os
import sys
import argparse
from PIL import Image

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from recovery_tools import TOOL_REGISTRY, get_tool_registry
from recovery_agent import (
    create_recovery_agent,
    recover_image,
    RecoveryAgent,
    create_vlm_client
)


def demo_tool_registry():
    """Demonstrate the tool registry."""
    print("=" * 60)
    print("TOOL REGISTRY DEMO")
    print("=" * 60)

    registry = get_tool_registry()

    print(f"\nRegistered tools: {len(registry.get_all_tools())}")
    print(f"Categories: {list(registry.categories.keys())}")

    print("\nTools by category:")
    for category, tools in registry.categories.items():
        print(f"  {category}: {', '.join(tools)}")

    print("\nTools for 'gaussian_blur' corruption:")
    blur_tools = registry.get_tools_for_corruption("gaussian_blur")
    for tool in blur_tools:
        print(f"  - {tool.name}: {tool.description[:60]}...")

    print()


def demo_single_tool(image_path: str, output_path: str = None):
    """Apply a single recovery tool to an image."""
    print("=" * 60)
    print("SINGLE TOOL DEMO")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    image = Image.open(image_path)
    print(f"Loaded image: {image.size}")

    # Apply sharpening
    registry = get_tool_registry()
    sharpen_tool = registry.get("unsharp_mask")

    if sharpen_tool:
        print(f"\nApplying tool: {sharpen_tool.name}")
        recovered = sharpen_tool(image, {"radius": 2, "percent": 150})

        if output_path:
            recovered.save(output_path)
            print(f"Saved to: {output_path}")
        else:
            print("Recovered image ready (not saved)")

    print()


def demo_recovery_agent(image_path: str, provider: str = "anthropic", output_path: str = None):
    """Run the full recovery agent on an image."""
    print("=" * 60)
    print("RECOVERY AGENT DEMO")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    image = Image.open(image_path)
    print(f"Loaded image: {image.size}")

    try:
        # Create agent
        print(f"\nCreating recovery agent with provider: {provider}")
        agent = create_recovery_agent(
            provider=provider,
            max_steps=3,
            verify=True,
            verbose=True
        )

        # Run recovery
        print("\nStarting recovery...")
        result = agent.recover(image)

        # Print results
        print("\n" + "=" * 40)
        print("RECOVERY RESULTS")
        print("=" * 40)
        print(f"Initial issues: {len(result.initial_issues)}")
        for issue in result.initial_issues:
            print(f"  - {issue.get('type', 'unknown')}: {issue.get('description', '')}")

        print(f"\nSteps taken: {result.total_steps}")
        for step in result.steps_taken:
            print(f"  Step {step.step_number}: {step.tool_name} (improvement: {step.improvement})")
            print(f"    Params: {step.parameters}")
            print(f"    Reasoning: {step.reasoning[:80]}...")

        print(f"\nFinal quality: {result.final_quality}")
        print(f"Stop reason: {result.stop_reason}")

        if output_path:
            result.recovered_image.save(output_path)
            print(f"\nSaved recovered image to: {output_path}")

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure the API key is set in the environment.")

    print()


def demo_deterministic_recovery(image_path: str, corruption: str, output_path: str = None):
    """Run deterministic recovery for a known corruption."""
    print("=" * 60)
    print("DETERMINISTIC RECOVERY DEMO")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    image = Image.open(image_path)
    print(f"Loaded image: {image.size}")
    print(f"Known corruption: {corruption}")

    # Get tools for this corruption
    registry = get_tool_registry()
    tools = registry.get_tools_for_corruption(corruption)
    print(f"Available tools: {[t.name for t in tools]}")

    if not tools:
        print(f"No tools available for corruption: {corruption}")
        return

    # Apply first matching tool with default params
    tool = tools[0]
    print(f"\nApplying tool: {tool.name}")
    print(f"Description: {tool.description}")

    recovered = tool(image, {})  # Use default params

    if output_path:
        recovered.save(output_path)
        print(f"Saved to: {output_path}")

    print()


def demo_robustbench_integration():
    """Show how to integrate with robustbench."""
    print("=" * 60)
    print("ROBUSTBENCH INTEGRATION EXAMPLE")
    print("=" * 60)

    print("""
To integrate with robustbench, add the following to the inference scripts:

1. Import the recovery agent:

    from recovery_agent import create_recovery_agent

2. Create the agent (in main):

    if args.enable_recovery:
        recovery_agent = create_recovery_agent(
            provider=args.recovery_provider,
            model=args.recovery_model,
            max_steps=args.recovery_max_steps,
            verbose=True
        )

3. Apply recovery after augmentation (in the inference loop):

    # After: images = augmenter(images, idx)
    if recovery_agent:
        recovered_images = []
        recovery_info = []

        for img in images:
            if args.recovery_mode == "deterministic" and args.aug != "none":
                result = recovery_agent.recover_deterministic(img, args.aug)
            else:
                result = recovery_agent.recover(img)

            recovered_images.append(result.recovered_image)
            recovery_info.append({
                "steps": [{"tool": s.tool_name, "params": s.parameters}
                          for s in result.steps_taken],
                "initial_issues": result.initial_issues,
                "final_quality": result.final_quality
            })

        images = recovered_images

4. Save recovery info in results:

    result_sample["recovery"] = recovery_info

Example command:

    python infer_anthropic.py \\
        --model claude-sonnet-4-20250514 \\
        --dataset_repo_id MMMU/MMMU_Pro \\
        --aug gaussian_blur --severity 3 \\
        --enable_recovery \\
        --recovery_provider anthropic \\
        --recovery_model claude-3-haiku-20240307 \\
        --recovery_mode adaptive
""")


def main():
    parser = argparse.ArgumentParser(description="VL Agent Demo")
    parser.add_argument("--demo", type=str, default="registry",
                        choices=["registry", "single_tool", "agent", "deterministic", "integration"],
                        help="Demo to run")
    parser.add_argument("--image", type=str, help="Input image path")
    parser.add_argument("--output", type=str, help="Output image path")
    parser.add_argument("--provider", type=str, default="anthropic",
                        choices=["anthropic", "openai", "gemini"],
                        help="VLM provider for agent demo")
    parser.add_argument("--corruption", type=str, default="gaussian_blur",
                        help="Corruption type for deterministic demo")

    args = parser.parse_args()

    if args.demo == "registry":
        demo_tool_registry()

    elif args.demo == "single_tool":
        if not args.image:
            print("--image required for single_tool demo")
            return
        demo_single_tool(args.image, args.output)

    elif args.demo == "agent":
        if not args.image:
            print("--image required for agent demo")
            return
        demo_recovery_agent(args.image, args.provider, args.output)

    elif args.demo == "deterministic":
        if not args.image:
            print("--image required for deterministic demo")
            return
        demo_deterministic_recovery(args.image, args.corruption, args.output)

    elif args.demo == "integration":
        demo_robustbench_integration()


if __name__ == "__main__":
    main()
