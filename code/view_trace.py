"""
View and replay recovery traces.

Usage:
    # View a single trace
    python code/view_trace.py --trace traces/sample_0000.json

    # View and replay (regenerate images)
    python code/view_trace.py --trace traces/sample_0000.json --replay --dataset MMMU/MMMU_Pro

    # List all traces in a directory
    python code/view_trace.py --trace_dir traces/ --list

    # Generate HTML report
    python code/view_trace.py --trace traces/sample_0000.json --html output/report.html
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional
from PIL import Image

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recovery_tools import get_tool_registry, get_all_metrics


def load_trace(trace_path: str) -> Dict:
    """Load a trace file."""
    with open(trace_path, 'r') as f:
        return json.load(f)


def print_trace(trace: Dict, show_prompts: bool = False):
    """Pretty print a trace."""
    print("=" * 70)
    print(f"TRACE: Sample {trace.get('sample_idx', '?')}")
    print("=" * 70)
    print(f"ID: {trace.get('id', 'N/A')}")
    print(f"Subject: {trace.get('subject', 'N/A')}")
    print(f"Question: {trace.get('question', 'N/A')[:200]}...")
    print()

    for img_data in trace.get("images", []):
        img_idx = img_data.get("image_idx", 0)
        img_trace = img_data.get("trace", {})

        print(f"\n--- Image {img_idx} ---")
        print(f"Total steps: {img_trace.get('total_steps', 0)}")
        print(f"Final quality: {img_trace.get('final_quality', 'N/A')}")
        print(f"Stop reason: {img_trace.get('stop_reason', 'N/A')}")

        # Initial metrics
        init_metrics = img_trace.get("initial_metrics", {})
        if init_metrics:
            print(f"\nInitial metrics:")
            print(f"  Sharpness: {init_metrics.get('sharpness', 'N/A')}")
            if isinstance(init_metrics.get('contrast'), dict):
                print(f"  Contrast: {init_metrics['contrast'].get('pixel_range', 'N/A')}")
            if isinstance(init_metrics.get('brightness'), dict):
                print(f"  Brightness: {init_metrics['brightness'].get('mean', 'N/A')}")
            print(f"  Noise: {init_metrics.get('noise', 'N/A')}")

        # Final metrics
        final_metrics = img_trace.get("final_metrics", {})
        if final_metrics:
            print(f"\nFinal metrics:")
            print(f"  Sharpness: {final_metrics.get('sharpness', 'N/A')}")
            if isinstance(final_metrics.get('contrast'), dict):
                print(f"  Contrast: {final_metrics['contrast'].get('pixel_range', 'N/A')}")
            if isinstance(final_metrics.get('brightness'), dict):
                print(f"  Brightness: {final_metrics['brightness'].get('mean', 'N/A')}")
            print(f"  Noise: {final_metrics.get('noise', 'N/A')}")

        # Tool sequence (compact)
        tool_seq = img_trace.get("tool_sequence", [])
        if tool_seq:
            print(f"\nTool sequence:")
            for i, step in enumerate(tool_seq, 1):
                print(f"  {i}. {step.get('tool', 'N/A')} {step.get('params', {})}")

        # Detailed steps
        steps = img_trace.get("steps", [])
        if steps:
            print(f"\nDetailed steps:")
            for step in steps:
                print(f"\n  Step {step.get('step_number', '?')}:")
                print(f"    Tool: {step.get('tool_name', 'N/A')}")
                print(f"    Params: {step.get('parameters', {})}")
                print(f"    Reasoning: {step.get('reasoning', 'N/A')[:150]}...")
                print(f"    Improvement: {step.get('improvement', 'N/A')}")

                if step.get('metrics_before') and step.get('metrics_after'):
                    mb = step['metrics_before']
                    ma = step['metrics_after']
                    print(f"    Sharpness: {mb.get('sharpness', 0):.1f} -> {ma.get('sharpness', 0):.1f}")
                    if isinstance(mb.get('contrast'), dict) and isinstance(ma.get('contrast'), dict):
                        print(f"    Contrast: {mb['contrast'].get('pixel_range', 0)} -> {ma['contrast'].get('pixel_range', 0)}")

                if show_prompts:
                    if step.get('vlm_decision_prompt'):
                        print(f"\n    [Decision Prompt]")
                        print("    " + step['vlm_decision_prompt'][:500].replace('\n', '\n    ') + "...")
                    if step.get('vlm_decision_response'):
                        print(f"\n    [Decision Response]")
                        print("    " + step['vlm_decision_response'][:500].replace('\n', '\n    ') + "...")

    print("\n" + "=" * 70)


def replay_trace(trace: Dict, original_image: Image.Image, output_dir: str = None) -> List[Image.Image]:
    """Replay a trace to regenerate intermediate images.

    Args:
        trace: Trace dict containing tool_sequence
        original_image: Original image to start from
        output_dir: If provided, save intermediate images

    Returns:
        List of images at each step (including original)
    """
    registry = get_tool_registry()
    images = [original_image]
    current = original_image.copy()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        original_image.save(os.path.join(output_dir, "step_0_original.png"))

    for img_data in trace.get("images", []):
        img_idx = img_data.get("image_idx", 0)
        img_trace = img_data.get("trace", {})
        tool_seq = img_trace.get("tool_sequence", [])

        print(f"\nReplaying image {img_idx} with {len(tool_seq)} steps...")

        for i, step in enumerate(tool_seq, 1):
            tool_name = step.get("tool", "")
            params = step.get("params", {})

            tool = registry.get(tool_name)
            if tool is None:
                print(f"  Step {i}: Tool '{tool_name}' not found, skipping")
                continue

            try:
                current = tool(current, params)
                images.append(current)
                print(f"  Step {i}: Applied {tool_name} {params}")

                if output_dir:
                    current.save(os.path.join(output_dir, f"step_{i}_{tool_name}.png"))
            except Exception as e:
                print(f"  Step {i}: Error applying {tool_name}: {e}")

    return images


def load_image_from_dataset(sample_idx: int, dataset_repo_id: str, variant: str, img_idx: int = 0) -> Image.Image:
    """Load original image from dataset."""
    from datasets import load_dataset
    import re

    print(f"Loading image from {dataset_repo_id}...")
    dataset = load_dataset(dataset_repo_id, name=variant, split="test")
    sample = dataset[sample_idx]

    # Extract image based on variant
    if "standard" in variant:
        question = sample.get("question", "")
        image_order = [int(num) for num in re.findall(r"<image\s+(\d+)>", question)]
        if img_idx < len(image_order):
            img_key = f"image_{image_order[img_idx]}"
        else:
            img_key = f"image_{img_idx + 1}"
        img = sample.get(img_key)
    else:
        img = sample.get("image")

    if img is None:
        raise ValueError(f"Image not found for sample {sample_idx}")

    if isinstance(img, Image.Image):
        return img.convert('RGB')
    elif hasattr(img, 'convert'):
        return img.convert('RGB')
    else:
        raise ValueError(f"Unknown image type: {type(img)}")


def generate_html_report(trace: Dict, output_path: str, images: List[Image.Image] = None):
    """Generate an HTML report from a trace."""
    import base64
    from io import BytesIO

    def img_to_base64(img: Image.Image, max_size: int = 400) -> str:
        """Convert PIL image to base64 string."""
        # Resize for display
        img = img.copy()
        img.thumbnail((max_size, max_size))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Recovery Trace Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: #333; color: white; padding: 20px; margin-bottom: 20px; }
        .section { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .step { border-left: 4px solid #4CAF50; padding-left: 15px; margin: 15px 0; }
        .metrics { display: flex; gap: 20px; flex-wrap: wrap; }
        .metric { background: #e8f5e9; padding: 10px; border-radius: 4px; min-width: 120px; }
        .images { display: flex; gap: 10px; flex-wrap: wrap; }
        .images img { max-width: 300px; border: 1px solid #ddd; border-radius: 4px; }
        pre { background: #f5f5f5; padding: 10px; overflow-x: auto; border-radius: 4px; }
        .tool-name { font-weight: bold; color: #1976D2; }
        .params { color: #666; font-size: 0.9em; }
        h2 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
    </style>
</head>
<body>
"""

    html += f"""
    <div class="header">
        <h1>Recovery Trace Report</h1>
        <p>Sample: {trace.get('id', 'N/A')} | Subject: {trace.get('subject', 'N/A')}</p>
    </div>
"""

    # Question
    html += f"""
    <div class="section">
        <h2>Question</h2>
        <p>{trace.get('question', 'N/A')}</p>
    </div>
"""

    # Each image trace
    for img_data in trace.get("images", []):
        img_idx = img_data.get("image_idx", 0)
        img_trace = img_data.get("trace", {})

        html += f"""
    <div class="section">
        <h2>Image {img_idx}</h2>
        <p><strong>Total steps:</strong> {img_trace.get('total_steps', 0)} |
           <strong>Final quality:</strong> {img_trace.get('final_quality', 'N/A')} |
           <strong>Stop reason:</strong> {img_trace.get('stop_reason', 'N/A')}</p>
"""

        # Metrics comparison
        init_m = img_trace.get("initial_metrics", {})
        final_m = img_trace.get("final_metrics", {})
        if init_m and final_m:
            html += """
        <h3>Metrics Comparison</h3>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
            <tr><th>Metric</th><th>Before</th><th>After</th><th>Change</th></tr>
"""
            # Sharpness
            s_before = init_m.get('sharpness', 0)
            s_after = final_m.get('sharpness', 0)
            s_change = s_after - s_before
            html += f"<tr><td>Sharpness</td><td>{s_before:.1f}</td><td>{s_after:.1f}</td><td>{s_change:+.1f}</td></tr>"

            # Contrast
            c_before = init_m.get('contrast', {}).get('pixel_range', 0) if isinstance(init_m.get('contrast'), dict) else 0
            c_after = final_m.get('contrast', {}).get('pixel_range', 0) if isinstance(final_m.get('contrast'), dict) else 0
            c_change = c_after - c_before
            html += f"<tr><td>Contrast</td><td>{c_before}</td><td>{c_after}</td><td>{c_change:+d}</td></tr>"

            # Brightness
            b_before = init_m.get('brightness', {}).get('mean', 0) if isinstance(init_m.get('brightness'), dict) else 0
            b_after = final_m.get('brightness', {}).get('mean', 0) if isinstance(final_m.get('brightness'), dict) else 0
            b_change = b_after - b_before
            html += f"<tr><td>Brightness</td><td>{b_before}</td><td>{b_after}</td><td>{b_change:+d}</td></tr>"

            html += "</table>"

        # Tool sequence
        tool_seq = img_trace.get("tool_sequence", [])
        if tool_seq:
            html += "<h3>Tool Sequence</h3><ol>"
            for step in tool_seq:
                html += f"<li><span class='tool-name'>{step.get('tool', 'N/A')}</span> <span class='params'>{step.get('params', {})}</span></li>"
            html += "</ol>"

        # Detailed steps
        steps = img_trace.get("steps", [])
        if steps:
            html += "<h3>Detailed Steps</h3>"
            for step in steps:
                html += f"""
        <div class="step">
            <h4>Step {step.get('step_number', '?')}: <span class='tool-name'>{step.get('tool_name', 'N/A')}</span></h4>
            <p><strong>Parameters:</strong> {step.get('parameters', {})}</p>
            <p><strong>Reasoning:</strong> {step.get('reasoning', 'N/A')[:300]}...</p>
            <p><strong>Improvement:</strong> {step.get('improvement', 'N/A')}</p>
        </div>
"""

        html += "</div>"  # End section

    # Images if provided
    if images:
        html += """
    <div class="section">
        <h2>Image Progression</h2>
        <div class="images">
"""
        for i, img in enumerate(images):
            label = "Original" if i == 0 else f"Step {i}"
            b64 = img_to_base64(img)
            html += f'<div><img src="data:image/png;base64,{b64}" alt="{label}"><p>{label}</p></div>'
        html += "</div></div>"

    html += """
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"HTML report saved to: {output_path}")


def list_traces(trace_dir: str):
    """List all traces in a directory."""
    traces = []
    for f in sorted(os.listdir(trace_dir)):
        if f.endswith('.json'):
            path = os.path.join(trace_dir, f)
            try:
                with open(path) as file:
                    data = json.load(file)
                    total_steps = sum(
                        img.get('trace', {}).get('total_steps', 0)
                        for img in data.get('images', [])
                    )
                    traces.append({
                        'file': f,
                        'id': data.get('id', 'N/A'),
                        'subject': data.get('subject', 'N/A'),
                        'steps': total_steps,
                    })
            except:
                pass

    print(f"\nFound {len(traces)} traces in {trace_dir}:\n")
    print(f"{'File':<25} {'ID':<25} {'Subject':<20} {'Steps'}")
    print("-" * 80)
    for t in traces:
        print(f"{t['file']:<25} {t['id']:<25} {t['subject']:<20} {t['steps']}")


def main():
    parser = argparse.ArgumentParser(description="View and replay recovery traces")

    parser.add_argument("--trace", type=str, help="Path to trace JSON file")
    parser.add_argument("--trace_dir", type=str, help="Directory containing traces")
    parser.add_argument("--list", action="store_true", help="List traces in directory")

    parser.add_argument("--show_prompts", action="store_true", help="Show VLM prompts and responses")

    parser.add_argument("--replay", action="store_true", help="Replay trace to regenerate images")
    parser.add_argument("--dataset", type=str, default="MMMU/MMMU_Pro", help="Dataset repo ID for replay")
    parser.add_argument("--variant", type=str, default="standard (10 options)", help="Dataset variant")
    parser.add_argument("--output_images", type=str, help="Directory to save replayed images")

    parser.add_argument("--html", type=str, help="Generate HTML report to this path")

    args = parser.parse_args()

    if args.list and args.trace_dir:
        list_traces(args.trace_dir)
        return

    if not args.trace:
        parser.print_help()
        return

    # Load trace
    trace = load_trace(args.trace)

    # Print trace
    print_trace(trace, show_prompts=args.show_prompts)

    images = None

    # Replay if requested
    if args.replay:
        sample_idx = trace.get('sample_idx', 0)
        try:
            original = load_image_from_dataset(sample_idx, args.dataset, args.variant)
            images = replay_trace(trace, original, args.output_images)
            print(f"\nReplayed {len(images)} images")
        except Exception as e:
            print(f"Error replaying: {e}")

    # Generate HTML report if requested
    if args.html:
        generate_html_report(trace, args.html, images)


if __name__ == "__main__":
    main()
