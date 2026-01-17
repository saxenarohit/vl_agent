"""
Evaluate VLM on MMMU_Pro with optional image recovery.

Matches robustbench result format:
- JSONL output (one JSON per line)
- Preserves all original dataset fields
- Adds response and recovery info

Usage:
    # Direct (no recovery)
    python code/eval_mmmu_pro.py --provider qwen3vl --mode direct

    # With recovery
    python code/eval_mmmu_pro.py --provider qwen3vl --mode direct --enable_recovery

    # With task-aware recovery
    python code/eval_mmmu_pro.py --provider qwen3vl --mode direct --enable_recovery --recovery_with_question
"""

import os
import sys
import re
import json
import argparse
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recovery_agent import create_recovery_agent, create_vlm_client
from recovery_tools import get_all_metrics


# ============================================================================
# PROMPT TEMPLATES (matching robustbench)
# ============================================================================

PROMPT_TEMPLATES = {
    "direct": {
        "vision": "Please select the correct answer from the options above. Respond with only the letter of the correct option. Do not explain. Answer:",
        "standard": "Please select the correct answer from the options above. Respond with only the letter of the correct option. Do not explain. Answer:"
    },
    "cot": {
        "vision": "Write out the multiple-choice question in the image and then solve it. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.",
        "standard": "Answer the preceding multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering."
    }
}


# ============================================================================
# DATASET LOADING
# ============================================================================

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
    """Apply stratified sampling (matching robustbench)."""
    rng = np.random.default_rng(seed)
    target_n = int(len(dataset) * sample_ratio)

    print(f"\nApplying stratified sampling...")
    print(f"  Sample ratio: {sample_ratio} ({sample_ratio*100:.0f}%)")
    print(f"  Seed: {seed}")
    print(f"  Target samples: ~{target_n}")

    # Group indices by category
    cat_indices = defaultdict(list)
    for idx, sample in enumerate(dataset):
        cat = sample.get(stratify_field, 'unknown')
        cat_indices[cat].append(idx)

    # Sample proportionally from each category
    selected_indices = []
    for cat, indices in sorted(cat_indices.items()):
        cat_n = max(1, int(len(indices) / len(dataset) * target_n))
        if len(indices) < cat_n:
            cat_n = len(indices)
        sampled = rng.choice(indices, size=cat_n, replace=False).tolist()
        selected_indices.extend(sampled)

    rng.shuffle(selected_indices)
    sampled_dataset = dataset.select(selected_indices)
    print(f"  Final sample count: {len(sampled_dataset)}")

    return sampled_dataset


# ============================================================================
# PROMPT CONSTRUCTION (matching robustbench)
# ============================================================================

def parse_options(options):
    """Parse multiple choice options into formatted string."""
    import ast
    if isinstance(options, str):
        try:
            options = ast.literal_eval(options)
        except (ValueError, SyntaxError):
            options = [options]
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{opt_letter}. {opt}" for opt_letter, opt in zip(option_letters, options)])
    return choices_str


def replace_images_tokens(input_string):
    """Replace image tokens with [image] placeholder and return order."""
    image_order = [int(num) for num in re.findall(r"<image\s+(\d+)>", input_string)]
    input_string = re.sub(r"<image\s+\d+>", "[image]", input_string)
    return input_string, image_order


def extract_images_from_sample(sample: Dict, variant: str) -> List[Image.Image]:
    """Extract all images from a dataset sample."""
    images = []

    if "standard" in variant:
        # Standard variant: look for image_1, image_2, etc. based on question tokens
        question = sample.get("question", "")
        image_order = [int(num) for num in re.findall(r"<image\s+(\d+)>", question)]
        for idx in image_order:
            img_key = f"image_{idx}"
            if img_key in sample and sample[img_key] is not None:
                img = sample[img_key]
                if isinstance(img, Image.Image):
                    images.append(img.convert('RGB'))
                elif hasattr(img, 'convert'):
                    images.append(img.convert('RGB'))
    else:
        # Vision variant: single image field
        if "image" in sample and sample["image"] is not None:
            img = sample["image"]
            if isinstance(img, Image.Image):
                images.append(img.convert('RGB'))
            elif hasattr(img, 'convert'):
                images.append(img.convert('RGB'))

    return images


def construct_prompt(sample: Dict, variant: str, mode: str = "direct") -> Tuple[str, str]:
    """Construct prompt for MMMU_Pro sample (matching robustbench)."""
    question = sample.get("question", "")
    options = sample.get("options", [])

    if "standard" in variant:
        # Standard variant: question with image tokens + options
        parsed_options = parse_options(options)
        prompt_suffix = PROMPT_TEMPLATES[mode]['standard']
        full_prompt = f"{question}\n{parsed_options}\n{prompt_suffix}"
        # Replace image tokens
        full_prompt, _ = replace_images_tokens(full_prompt)
    else:
        # Vision variant: question is in image, just provide options + suffix
        if options:
            parsed_options = parse_options(options)
            prompt_suffix = PROMPT_TEMPLATES[mode]['vision']
            full_prompt = f"{parsed_options}\n{prompt_suffix}"
        else:
            full_prompt = PROMPT_TEMPLATES[mode]['vision']

    return full_prompt


def extract_answer(response: str) -> str:
    """Extract answer letter from VLM response."""
    # Try to find "Answer: X" pattern
    match = re.search(r'Answer:\s*([A-J])', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Try to find standalone letter at end
    match = re.search(r'\b([A-J])\s*$', response.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Try to find first letter mention
    match = re.search(r'\b([A-J])\b', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return ""


# ============================================================================
# RESULT SAVING (matching robustbench format)
# ============================================================================

def save_results_jsonl(results: List[Dict], output_path: str):
    """Save results to JSONL file (one JSON per line)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            str_result = json.dumps(result, ensure_ascii=False, default=str)
            f.write(str_result + "\n")


def setup_output_path(args) -> str:
    """Setup output path matching robustbench structure.

    Structure: output_dir/{model_name}/MMMU_Pro/{variant}_{mode}_{recovery}.jsonl
    """
    model_name = args.model.split("/")[-1] if args.model else args.provider

    if "standard" in args.variant:
        variant = "standard"
    else:
        variant = "vision"

    # Build recovery suffix
    if args.enable_recovery:
        if args.recovery_with_question:
            recovery_str = "recovery_with_q"
        else:
            recovery_str = "recovery"
    else:
        recovery_str = "none"

    filename = f"{variant}_{args.mode}_{recovery_str}.jsonl"
    output_path = os.path.join(args.output_dir, model_name, "MMMU_Pro", filename)

    return output_path


# ============================================================================
# EVALUATION
# ============================================================================

def run_evaluation(
    dataset,
    vlm_client,
    recovery_agent,
    variant: str,
    mode: str = "direct",
    enable_recovery: bool = False,
    recovery_with_question: bool = False,
    recovery_max_iter: int = 5,
    save_traces: bool = False,
    trace_dir: str = None,
    verbose: bool = True
) -> List[Dict]:
    """Run evaluation on dataset, returning results in robustbench format."""

    results = []
    traces = []  # For storing trace data
    total = len(dataset)

    # Setup trace directory
    if save_traces and trace_dir:
        os.makedirs(trace_dir, exist_ok=True)

    correct = 0
    total_processed = 0

    print(f"\nRunning evaluation on {total} samples...")
    print(f"Mode: {mode}")
    print(f"Recovery: {'enabled' if enable_recovery else 'disabled'}")
    if enable_recovery:
        print(f"  With question: {'yes' if recovery_with_question else 'no'}")
    print("-" * 60)

    for idx in range(total):
        sample = dataset[idx]
        images = extract_images_from_sample(sample, variant)

        if not images:
            print(f"[{idx+1}/{total}] No images found, skipping")
            continue

        ground_truth = sample.get("answer", "").strip().upper()

        if verbose:
            print(f"\n[{idx+1}/{total}] Subject: {sample.get('subject', 'unknown')}")

        # Recovery info (similar to augmentation in robustbench)
        recovery_info = {
            "enabled": enable_recovery,
            "with_question": recovery_with_question,
            "max_iter": recovery_max_iter,
            "sample_idx": idx,
        }

        # Process images
        processed_images = []
        sample_trace = None  # For this sample's trace

        if enable_recovery and recovery_agent:
            question_context = sample.get("question", "") if recovery_with_question else None
            sample_trace = {
                "sample_idx": idx,
                "id": sample.get("id", f"sample_{idx}"),
                "subject": sample.get("subject", "unknown"),
                "question": sample.get("question", "")[:500],
                "images": [],
            }

            for img_idx, image in enumerate(images):
                result = recovery_agent.recover_with_metrics(
                    image,
                    max_iterations=recovery_max_iter,
                    question=question_context
                )
                processed_images.append(result.recovered_image)

                # Add recovery details
                recovery_info[f"image_{img_idx}"] = {
                    "steps": result.total_steps,
                    "tools": [s.tool_name for s in result.steps_taken],
                    "quality": result.final_quality,
                    "initial_sharpness": result.initial_metrics.get('sharpness', 0) if result.initial_metrics else 0,
                    "final_sharpness": result.final_metrics.get('sharpness', 0) if result.final_metrics else 0,
                }

                # Add full trace for this image
                if save_traces:
                    sample_trace["images"].append({
                        "image_idx": img_idx,
                        "trace": result.to_trace_dict(),
                    })

                if verbose and result.total_steps > 0:
                    print(f"  Image {img_idx+1}: {result.total_steps} recovery steps, tools: {[s.tool_name for s in result.steps_taken]}")

            # Save trace file for this sample
            if save_traces and trace_dir and sample_trace:
                trace_path = os.path.join(trace_dir, f"sample_{idx:04d}.json")
                with open(trace_path, 'w') as f:
                    json.dump(sample_trace, f, indent=2, default=str)
        else:
            processed_images = images

        # Construct prompt and get answer
        prompt = construct_prompt(sample, variant, mode)

        try:
            # Call VLM with first image and prompt
            response = vlm_client.analyze_image(processed_images[0], prompt)
            predicted = extract_answer(response)
            is_correct = (predicted == ground_truth)

            if is_correct:
                correct += 1
            total_processed += 1

            if verbose:
                status = "+" if is_correct else "-"
                print(f"  {status} Pred: {predicted}, GT: {ground_truth}")

        except Exception as e:
            print(f"  ERROR: {e}")
            response = str(e)
            predicted = ""
            is_correct = False
            total_processed += 1

        # Build result in robustbench format
        # Include all original dataset fields (except images)
        result_entry = {"response": response}

        # Add all original fields from sample (except image fields)
        for key, value in sample.items():
            if not key.startswith("image"):
                result_entry[key] = value

        # Add recovery info (similar to augmentation in robustbench)
        result_entry["recovery"] = recovery_info

        results.append(result_entry)

        # Print running accuracy every 10 samples
        if (idx + 1) % 10 == 0:
            acc = correct / total_processed * 100 if total_processed > 0 else 0
            print(f"\n--- Running accuracy: {correct}/{total_processed} = {acc:.1f}% ---\n")

    return results


def compute_accuracy(results: List[Dict]) -> Dict:
    """Compute accuracy statistics from results."""
    import ast

    correct = 0
    total = 0

    # By subject
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        response = r.get("response", "")
        answer = r.get("answer", "").strip().upper()

        # Parse prediction
        predicted = extract_answer(response)
        is_correct = (predicted == answer)

        if is_correct:
            correct += 1
        total += 1

        # Track by subject
        subj = r.get("subject", "unknown")
        subject_stats[subj]["total"] += 1
        if is_correct:
            subject_stats[subj]["correct"] += 1

    overall_acc = correct / total * 100 if total > 0 else 0

    return {
        "accuracy": overall_acc,
        "correct": correct,
        "total": total,
        "subject_accuracy": {
            subj: stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            for subj, stats in subject_stats.items()
        }
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM on MMMU_Pro with optional recovery")

    # Model arguments
    parser.add_argument("--provider", type=str, default="qwen3vl",
                        choices=["anthropic", "openai", "gemini", "qwen3vl", "transformers"],
                        help="VLM provider")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (uses default for provider if not specified)")

    # Prompt mode
    parser.add_argument("--mode", type=str, default="direct",
                        choices=["direct", "cot"],
                        help="Prompting mode: direct or chain-of-thought")

    # Recovery arguments
    parser.add_argument("--enable_recovery", action="store_true",
                        help="Enable image recovery before QA")
    parser.add_argument("--recovery_max_iter", type=int, default=5,
                        help="Max recovery iterations")
    parser.add_argument("--recovery_with_question", action="store_true",
                        help="Pass question to recovery agent for task-aware recovery")

    # Dataset arguments
    parser.add_argument("--variant", type=str, default="standard (10 options)",
                        help="MMMU_Pro variant")
    parser.add_argument("--sample_ratio", type=float, default=0.2,
                        help="Sampling ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--save_traces", action="store_true",
                        help="Save detailed recovery traces for each sample")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    print("=" * 60)
    print("MMMU_Pro Evaluation")
    print("=" * 60)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or 'default'}")
    print(f"Mode: {args.mode}")
    print(f"Recovery: {'enabled' if args.enable_recovery else 'disabled'}")
    if args.enable_recovery:
        print(f"  - With question: {'yes' if args.recovery_with_question else 'no'}")
        print(f"  - Max iterations: {args.recovery_max_iter}")
    print(f"Sample ratio: {args.sample_ratio}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Load dataset
    dataset = load_mmmu_pro(variant=args.variant)

    # Apply sampling
    if args.sample_ratio and args.sample_ratio < 1.0:
        dataset = apply_stratified_sampling(
            dataset,
            sample_ratio=args.sample_ratio,
            seed=args.seed
        )

    # Create VLM client
    print(f"\nCreating VLM client...")
    vlm_client = create_vlm_client(
        args.provider,
        model=args.model
    )

    # Create recovery agent if needed
    recovery_agent = None
    if args.enable_recovery:
        print(f"Creating recovery agent...")
        recovery_agent = create_recovery_agent(
            provider=args.provider,
            model=args.model,
            max_steps=args.recovery_max_iter,
            verify=True,
            verbose=False
        )

    # Setup trace directory
    trace_dir = None
    if args.save_traces and args.enable_recovery:
        output_path = setup_output_path(args)
        trace_dir = output_path.replace(".jsonl", "_traces")
        os.makedirs(trace_dir, exist_ok=True)
        print(f"Saving traces to: {trace_dir}")

    # Run evaluation
    results = run_evaluation(
        dataset=dataset,
        vlm_client=vlm_client,
        recovery_agent=recovery_agent,
        variant=args.variant,
        mode=args.mode,
        enable_recovery=args.enable_recovery,
        recovery_with_question=args.recovery_with_question,
        recovery_max_iter=args.recovery_max_iter,
        save_traces=args.save_traces,
        trace_dir=trace_dir,
        verbose=args.verbose
    )

    # Compute stats
    stats = compute_accuracy(results)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total samples: {stats['total']}")
    print(f"Correct: {stats['correct']}")
    print(f"Accuracy: {stats['accuracy']:.2f}%")

    print("\nAccuracy by subject:")
    for subj, acc in sorted(stats['subject_accuracy'].items(), key=lambda x: -x[1]):
        print(f"  {subj}: {acc:.1f}%")

    # Save results in JSONL format
    output_path = setup_output_path(args)
    save_results_jsonl(results, output_path)
    print(f"\nResults saved to: {output_path}")

    # Also save summary stats
    summary_path = output_path.replace(".jsonl", "_summary.json")
    summary = {
        "config": {
            "provider": args.provider,
            "model": args.model,
            "mode": args.mode,
            "enable_recovery": args.enable_recovery,
            "recovery_with_question": args.recovery_with_question,
            "sample_ratio": args.sample_ratio,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat()
        },
        "stats": stats
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
