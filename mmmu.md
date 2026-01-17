# MMMU_Pro Evaluation Workflow

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     MMMU_Pro Evaluation                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Load Dataset (20% stratified by subject, seed=42)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. For each sample: extract images + question + options        │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌──────────────────────┐        ┌──────────────────────────────┐
│ --enable_recovery    │        │ No recovery flag             │
│ (WITH recovery)      │        │ (baseline)                   │
└──────────────────────┘        └──────────────────────────────┘
              │                               │
              ▼                               │
┌──────────────────────┐                      │
│ Recovery Loop:       │                      │
│ ┌──────────────────┐ │                      │
│ │Compute metrics   │ │                      │
│ └────────┬─────────┘ │                      │
│          ▼           │                      │
│ ┌──────────────────┐ │                      │
│ │VLM sees metrics, │ │                      │
│ │decides tool+params│ │                      │
│ └────────┬─────────┘ │                      │
│          ▼           │                      │
│ ┌──────────────────┐ │                      │
│ │Apply tool        │ │                      │
│ └────────┬─────────┘ │                      │
│          ▼           │                      │
│ ┌──────────────────┐ │                      │
│ │Repeat (max 5x)   │ │                      │
│ └──────────────────┘ │                      │
└──────────────────────┘                      │
              │                               │
              └───────────┬───────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Construct prompt with question + options                    │
│     (direct mode: answer immediately, cot mode: think first)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. VLM answers question (A-J)                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. Compare predicted vs ground truth, compute accuracy         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Command Line Usage

### Baseline (no recovery)

```bash
python code/eval_mmmu_pro.py \
    --provider qwen3vl \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --mode direct \
    --sample_ratio 0.2 \
    --seed 42 \
    --output_dir output/eval_direct_no_recovery
```

### With recovery (generic)

```bash
python code/eval_mmmu_pro.py \
    --provider qwen3vl \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --mode direct \
    --enable_recovery \
    --recovery_max_iter 5 \
    --sample_ratio 0.2 \
    --seed 42 \
    --output_dir output/eval_direct_with_recovery
```

### With recovery (task-aware, VLM sees question)

```bash
python code/eval_mmmu_pro.py \
    --provider qwen3vl \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --mode direct \
    --enable_recovery \
    --recovery_with_question \
    --recovery_max_iter 5 \
    --sample_ratio 0.2 \
    --seed 42 \
    --output_dir output/eval_direct_with_recovery_question
```

---

## Key Difference

| Without Recovery | With Recovery |
|-----------------|---------------|
| Image → VLM → Answer | Image → Metrics → VLM decides tool → Apply → (repeat) → VLM → Answer |
| Fast (1 VLM call) | Slower (N+1 VLM calls where N = recovery iterations) |
| Baseline accuracy | Potentially improved if images have quality issues |

The `--enable_recovery` flag is the toggle.

---

## Recovery Step Details

### Step 1: Load MMMU_Pro sample

```
Image + Question + Options (A-J) + Ground Truth
```

### Step 2: Optional Recovery (if `--enable_recovery`)

```
┌─────────────────────────────────────────────────┐
│  Compute metrics (sharpness, contrast, etc.)    │
│                     ↓                           │
│  VLM receives:                                  │
│    - Image (visual input)                       │
│    - Metrics (numeric values)                   │
│    - All available tools (descriptions)         │
│    - Question (if --recovery_with_question)     │
│                     ↓                           │
│  VLM decides:                                   │
│    - Does image need recovery? (yes/no)         │
│    - If yes: which tool? what parameters?       │
│                     ↓                           │
│  If yes: apply tool → re-measure → repeat       │
│  If no: stop (image is good enough)             │
│                     ↓                           │
│  Max 5 iterations                               │
└─────────────────────────────────────────────────┘
```

#### What VLM Sees (without `--recovery_with_question`)

```
## Current Image Metrics
- Sharpness: 145.2 (reference: >500 is sharp, <200 is blurry)
- Contrast pixel_range: 77 (reference: >150 is good, <100 is low)
- Brightness mean: 128 (reference: 80-180 is normal)
- Noise: 8.5 (reference: <10 is clean, >20 is noisy)
- Saturation mean: 0.45 (reference: 0.3-0.7 is normal)

## Available Recovery Tools
1. unsharp_mask - Sharpen blurry images using unsharp masking
2. auto_contrast - Automatically adjust contrast to use full dynamic range
... (27 tools total)
```

#### What VLM Sees (with `--recovery_with_question`)

```
## Task Context
The image will be used to answer this question:
What does the warning sign in the image say?

## Current Image Metrics
- Sharpness: 145.2 (reference: >500 is sharp, <200 is blurry)
- Contrast pixel_range: 77 (reference: >150 is good, <100 is low)
...

## Available Recovery Tools
1. unsharp_mask - Sharpen blurry images using unsharp masking
...

## Task-Aware Prioritization
- If question asks about TEXT: prioritize sharpening
- If question asks about COLORS: prioritize contrast/saturation
- If question asks about DETAILS: prioritize sharpening and noise reduction
```

### Step 3: Answer the question

```
VLM sees (recovered or original) image + question → outputs answer (A-J)
```

### Step 4: Score

```
Compare predicted answer vs ground truth
```

---

## What Recovery Does

| Metric | Threshold | Tool Applied |
|--------|-----------|--------------|
| Sharpness < 500 | Blurry | `unsharp_mask` |
| Contrast < 100 | Low contrast | `auto_contrast` |
| Brightness < 50 | Too dark | `gamma_correction` |
| Brightness > 200 | Too bright | `gamma_correction` |
| Noise > 15 | Noisy | `bilateral_filter` |

VLM reads the actual metric values and decides which tool to apply with what parameters.

---

## SLURM Script

The `slurm/eval_mmmu_pro.sh` script runs both experiments:

1. **Test 1**: Direct prompting WITHOUT recovery (baseline)
2. **Test 2**: Direct prompting WITH recovery

Then compares the accuracy between both.

```bash
# Submit to SLURM
sbatch slurm/eval_mmmu_pro.sh

# Or run directly
bash slurm/eval_mmmu_pro.sh
```

---

## Output (matching robustbench format)

**Directory structure:**
```
output/
└── {model_name}/
    └── MMMU_Pro/
        ├── standard_direct_none.jsonl          # Baseline (no recovery)
        ├── standard_direct_none_summary.json
        ├── standard_direct_recovery.jsonl      # With generic recovery
        ├── standard_direct_recovery_summary.json
        ├── standard_direct_recovery_with_q.jsonl  # With task-aware recovery
        └── standard_direct_recovery_with_q_summary.json
```

**JSONL format** (one JSON per line, matching robustbench):
```json
{
    "response": "B",
    "id": "test_History_1",
    "question": "Which of the following best explains...",
    "options": "['Option A', 'Option B', ...]",
    "answer": "B",
    "subject": "History",
    "recovery": {
        "enabled": true,
        "with_question": false,
        "max_iter": 5,
        "sample_idx": 0,
        "image_0": {
            "steps": 2,
            "tools": ["auto_contrast", "unsharp_mask"],
            "quality": "good",
            "initial_sharpness": 145.2,
            "final_sharpness": 612.8
        }
    }
}
```

**Summary JSON:**
```json
{
    "config": {
        "provider": "qwen3vl",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "mode": "direct",
        "enable_recovery": true,
        "recovery_with_question": false,
        "sample_ratio": 0.2,
        "seed": 42
    },
    "stats": {
        "accuracy": 45.2,
        "correct": 90,
        "total": 199,
        "subject_accuracy": {...}
    }
}
```

---

## Trace Saving

Save detailed traces with `--save_traces`:

```bash
python code/eval_mmmu_pro.py --enable_recovery --save_traces
```

**Trace directory structure:**
```
output/{model}/MMMU_Pro/
├── standard_direct_recovery.jsonl
├── standard_direct_recovery_traces/
│   ├── sample_0000.json
│   ├── sample_0001.json
│   └── ...
```

**Trace file format** (no images, just tool calls for replay):
```json
{
    "sample_idx": 0,
    "id": "test_History_1",
    "subject": "History",
    "question": "Which of the following...",
    "images": [{
        "image_idx": 0,
        "trace": {
            "initial_metrics": {"sharpness": 145.2, ...},
            "final_metrics": {"sharpness": 612.8, ...},
            "total_steps": 2,
            "tool_sequence": [
                {"tool": "auto_contrast", "params": {}},
                {"tool": "unsharp_mask", "params": {"percent": 150}}
            ],
            "steps": [{
                "step_number": 1,
                "tool_name": "auto_contrast",
                "reasoning": "Low contrast detected...",
                "vlm_decision_prompt": "...",
                "vlm_decision_response": "..."
            }]
        }
    }]
}
```

---

## Viewing Traces

Use `view_trace.py` to view and replay traces:

```bash
# View a trace
python code/view_trace.py --trace output/.../traces/sample_0000.json

# View with VLM prompts/responses
python code/view_trace.py --trace traces/sample_0000.json --show_prompts

# List all traces
python code/view_trace.py --trace_dir output/.../traces/ --list

# Replay trace to regenerate images
python code/view_trace.py --trace traces/sample_0000.json --replay --output_images replay/

# Generate HTML report
python code/view_trace.py --trace traces/sample_0000.json --html report.html
```
