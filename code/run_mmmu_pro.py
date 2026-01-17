"""
Run recovery agent on MMMU_Pro dataset.

Uses the same sampling approach as robustbench:
- Dataset: MMMU/MMMU_Pro
- Variant: standard (10 options)
- Split: test
- Sampling: 20% stratified by subject with seed=42

Usage:
    python code/run_mmmu_pro.py --provider qwen3vl --model Qwen/Qwen3-VL-8B-Instruct
"""

import os
import sys
import json
import argparse
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recovery_agent import create_recovery_agent, RecoveryResult
from recovery_tools import get_all_metrics


def load_mmmu_pro(
    variant: str = "standard (10 options)",
    split: str = "test"
) -> Any:
    """Load MMMU_Pro dataset from HuggingFace."""
    from datasets import load_dataset

    print(f"Loading MMMU/MMMU_Pro dataset...")
    print(f"  Variant: {variant}")
    print(f"  Split: {split}")

    dataset = load_dataset(
        "MMMU/MMMU_Pro",
        name=variant,
        split=split
    )

    print(f"  Total samples: {len(dataset)}")
    return dataset


def apply_stratified_sampling(
    dataset,
    sample_ratio: float = 0.2,
    stratify_field: str = "subject",
    seed: int = 42
) -> Any:
    """Apply stratified sampling to dataset.

    Matches robustbench approach:
    - 20% sampling
    - Stratified by subject
    - Seed 42
    """
    rng = np.random.default_rng(seed)
    target_n = int(len(dataset) * sample_ratio)

    print(f"\nApplying stratified sampling...")
    print(f"  Sample ratio: {sample_ratio} ({sample_ratio*100:.0f}%)")
    print(f"  Stratify field: {stratify_field}")
    print(f"  Seed: {seed}")
    print(f"  Target samples: {target_n}")

    # Group indices by category
    cat_indices = defaultdict(list)
    for idx, sample in enumerate(dataset):
        cat = sample.get(stratify_field, 'unknown')
        cat_indices[cat].append(idx)

    print(f"  Categories found: {len(cat_indices)}")

    # Sample proportionally from each category
    selected_indices = []
    for cat, indices in sorted(cat_indices.items()):
        cat_n = max(1, int(len(indices) / len(dataset) * target_n))
        if len(indices) < cat_n:
            cat_n = len(indices)
        sampled = rng.choice(indices, size=cat_n, replace=False).tolist()
        selected_indices.extend(sampled)

    # Shuffle selected indices
    rng.shuffle(selected_indices)

    # Select samples
    sampled_dataset = dataset.select(selected_indices)
    print(f"  Final sample count: {len(sampled_dataset)}")

    return sampled_dataset


def extract_images_from_sample(sample: Dict) -> List[Image.Image]:
    """Extract all images from a dataset sample."""
    images = []

    # MMMU_Pro can have multiple images (image_1, image_2, etc.)
    for key in sorted(sample.keys()):
        if key.startswith('image') and sample[key] is not None:
            img = sample[key]
            if isinstance(img, Image.Image):
                images.append(img.convert('RGB'))
            elif hasattr(img, 'convert'):  # PIL-like
                images.append(img.convert('RGB'))

    return images


def run_recovery_on_dataset(
    dataset,
    agent,
    output_dir: str,
    max_samples: int = None,
    save_images: bool = True,
    verbose: bool = True
) -> List[Dict]:
    """Run recovery agent on dataset samples."""

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    results = []
    total = min(len(dataset), max_samples) if max_samples else len(dataset)

    print(f"\nRunning recovery on {total} samples...")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    for idx in range(total):
        sample = dataset[idx]
        images = extract_images_from_sample(sample)

        if not images:
            print(f"[{idx+1}/{total}] No images found, skipping")
            continue

        sample_results = {
            "sample_idx": idx,
            "subject": sample.get("subject", "unknown"),
            "question": sample.get("question", "")[:100],
            "num_images": len(images),
            "image_results": []
        }

        for img_idx, image in enumerate(images):
            if verbose:
                print(f"\n[{idx+1}/{total}] Image {img_idx+1}/{len(images)}")
                print(f"  Subject: {sample.get('subject', 'unknown')}")
                print(f"  Image size: {image.size}")

            # Get initial metrics
            initial_metrics = get_all_metrics(image)

            if verbose:
                print(f"  Initial sharpness: {initial_metrics['sharpness']:.1f}")
                print(f"  Initial contrast: {initial_metrics['contrast']['pixel_range']}")

            # Run recovery
            try:
                result = agent.recover_with_metrics(image, max_iterations=5)

                img_result = {
                    "image_idx": img_idx,
                    "original_size": list(image.size),
                    "initial_metrics": {
                        "sharpness": initial_metrics['sharpness'],
                        "contrast_pixel_range": initial_metrics['contrast']['pixel_range'],
                        "brightness_mean": initial_metrics['brightness']['mean'],
                        "noise": initial_metrics['noise'],
                        "saturation_mean": initial_metrics['saturation']['mean']
                    },
                    "final_metrics": {
                        "sharpness": result.final_metrics['sharpness'],
                        "contrast_pixel_range": result.final_metrics['contrast']['pixel_range'],
                        "brightness_mean": result.final_metrics['brightness']['mean'],
                        "noise": result.final_metrics['noise'],
                        "saturation_mean": result.final_metrics['saturation']['mean']
                    },
                    "steps_taken": result.total_steps,
                    "tools_applied": [s.tool_name for s in result.steps_taken],
                    "final_quality": result.final_quality,
                    "stop_reason": result.stop_reason
                }

                if verbose:
                    print(f"  Steps taken: {result.total_steps}")
                    print(f"  Tools: {img_result['tools_applied']}")
                    print(f"  Final sharpness: {result.final_metrics['sharpness']:.1f}")
                    print(f"  Final contrast: {result.final_metrics['contrast']['pixel_range']}")

                # Save images if requested
                if save_images:
                    orig_path = os.path.join(output_dir, "images", f"sample_{idx:04d}_img_{img_idx}_original.png")
                    recv_path = os.path.join(output_dir, "images", f"sample_{idx:04d}_img_{img_idx}_recovered.png")

                    image.save(orig_path)
                    result.recovered_image.save(recv_path)

                    img_result["original_path"] = orig_path
                    img_result["recovered_path"] = recv_path

                sample_results["image_results"].append(img_result)

            except Exception as e:
                print(f"  ERROR: {e}")
                sample_results["image_results"].append({
                    "image_idx": img_idx,
                    "error": str(e)
                })

        results.append(sample_results)

        # Save intermediate results every 10 samples
        if (idx + 1) % 10 == 0:
            with open(os.path.join(output_dir, "results_partial.json"), 'w') as f:
                json.dump(results, f, indent=2, default=str)

    return results


def compute_summary_stats(results: List[Dict]) -> Dict:
    """Compute summary statistics from results."""

    total_images = 0
    total_recovered = 0
    total_steps = 0

    sharpness_improvements = []
    contrast_improvements = []

    tools_used = defaultdict(int)
    quality_counts = defaultdict(int)

    for sample in results:
        for img_result in sample.get("image_results", []):
            if "error" in img_result:
                continue

            total_images += 1
            steps = img_result.get("steps_taken", 0)
            total_steps += steps

            if steps > 0:
                total_recovered += 1

            # Track improvements
            init_sharp = img_result.get("initial_metrics", {}).get("sharpness", 0)
            final_sharp = img_result.get("final_metrics", {}).get("sharpness", 0)
            if init_sharp > 0:
                sharpness_improvements.append((final_sharp - init_sharp) / init_sharp * 100)

            init_contrast = img_result.get("initial_metrics", {}).get("contrast_pixel_range", 0)
            final_contrast = img_result.get("final_metrics", {}).get("contrast_pixel_range", 0)
            if init_contrast > 0:
                contrast_improvements.append((final_contrast - init_contrast) / init_contrast * 100)

            # Track tools
            for tool in img_result.get("tools_applied", []):
                tools_used[tool] += 1

            # Track quality
            quality = img_result.get("final_quality", "unknown")
            quality_counts[quality] += 1

    return {
        "total_samples": len(results),
        "total_images": total_images,
        "images_recovered": total_recovered,
        "recovery_rate": total_recovered / total_images * 100 if total_images > 0 else 0,
        "avg_steps_per_image": total_steps / total_images if total_images > 0 else 0,
        "avg_sharpness_improvement_pct": np.mean(sharpness_improvements) if sharpness_improvements else 0,
        "avg_contrast_improvement_pct": np.mean(contrast_improvements) if contrast_improvements else 0,
        "tools_used": dict(tools_used),
        "final_quality_distribution": dict(quality_counts)
    }


def main():
    parser = argparse.ArgumentParser(description="Run recovery agent on MMMU_Pro")

    # Model arguments
    parser.add_argument("--provider", type=str, default="qwen3vl",
                        choices=["anthropic", "openai", "gemini", "qwen3vl", "transformers"],
                        help="VLM provider")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Model name")

    # Dataset arguments
    parser.add_argument("--variant", type=str, default="standard (10 options)",
                        help="MMMU_Pro variant")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split")

    # Sampling arguments (matching robustbench)
    parser.add_argument("--sample_ratio", type=float, default=0.2,
                        help="Sampling ratio (default: 0.2 for 20%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--no_stratified", action="store_true",
                        help="Disable stratified sampling")

    # Recovery arguments
    parser.add_argument("--max_iterations", type=int, default=5,
                        help="Max recovery iterations per image")
    parser.add_argument("--no_verify", action="store_true",
                        help="Disable verification step")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: output/mmmu_pro_<timestamp>)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process (for testing)")
    parser.add_argument("--no_save_images", action="store_true",
                        help="Don't save original/recovered images")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output/mmmu_pro_{timestamp}"

    print("=" * 60)
    print("VL Agent - MMMU_Pro Recovery Test")
    print("=" * 60)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Sample ratio: {args.sample_ratio}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Load dataset
    dataset = load_mmmu_pro(variant=args.variant, split=args.split)

    # Apply sampling
    if args.sample_ratio and args.sample_ratio < 1.0:
        stratify_field = None if args.no_stratified else "subject"
        dataset = apply_stratified_sampling(
            dataset,
            sample_ratio=args.sample_ratio,
            stratify_field=stratify_field,
            seed=args.seed
        )

    # Create agent
    print(f"\nCreating recovery agent...")
    agent = create_recovery_agent(
        provider=args.provider,
        model=args.model,
        max_steps=args.max_iterations,
        verify=not args.no_verify,
        verbose=args.verbose
    )

    # Run recovery
    results = run_recovery_on_dataset(
        dataset=dataset,
        agent=agent,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        save_images=not args.no_save_images,
        verbose=args.verbose
    )

    # Compute summary
    summary = compute_summary_stats(results)

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total samples: {summary['total_samples']}")
    print(f"Total images: {summary['total_images']}")
    print(f"Images recovered: {summary['images_recovered']} ({summary['recovery_rate']:.1f}%)")
    print(f"Avg steps per image: {summary['avg_steps_per_image']:.2f}")
    print(f"Avg sharpness improvement: {summary['avg_sharpness_improvement_pct']:.1f}%")
    print(f"Avg contrast improvement: {summary['avg_contrast_improvement_pct']:.1f}%")
    print(f"Tools used: {summary['tools_used']}")
    print(f"Quality distribution: {summary['final_quality_distribution']}")

    # Save results
    final_results = {
        "config": {
            "provider": args.provider,
            "model": args.model,
            "sample_ratio": args.sample_ratio,
            "seed": args.seed,
            "max_iterations": args.max_iterations,
            "timestamp": datetime.now().isoformat()
        },
        "summary": summary,
        "results": results
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
