"""
Step 1: Dataset Loading and Inspection for Math Reasoning with A*-PO
===================================================================

This script loads the MATH competition dataset and prepares it for A*-PO reinforcement learning.
A*-PO (Advantage-weighted Policy Optimization) is a technique that uses offline data to estimate
optimal value functions and then optimizes policy using estimated advantages.
"""

from datasets import load_dataset
import json
import os
from collections import Counter
import re


def inspect_dataset(ds):
    """
    Inspect the dataset structure and validate data integrity.

    A*-PO requires understanding the data distribution to properly estimate
    value functions and advantages during training.
    """
    print("=== Dataset Inspection ===")
    print(f"Dataset keys: {list(ds.keys())}")
    print(f"Train samples: {len(ds['train'])}")

    # Inspect sample structure
    sample = ds['train'][0]
    print(f"Sample keys: {list(sample.keys())}")

    # Analyze problem types and levels
    types = [sample['type'] for sample in ds['train']]
    levels = [sample['level'] for sample in ds['train']]

    print(f"\nProblem types distribution:")
    type_counts = Counter(types)
    for ptype, count in type_counts.most_common():
        print(f"  {ptype}: {count} problems")

    print(f"\nDifficulty levels distribution:")
    level_counts = Counter(levels)
    for level, count in sorted(level_counts.items()):
        print(f"  {level}: {count} problems")

    return type_counts, level_counts


def clean_latex_formatting(text):
    """
    Clean LaTeX formatting for better LLM processing.

    A*-PO training benefits from consistent formatting as it helps
    the value function estimation be more accurate.
    """
    # Basic LaTeX cleaning - can be expanded based on needs
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)  # Remove \text{}
    text = re.sub(r'\$([^$]+)\$', r'\1', text)  # Remove single $ delimiters
    text = text.replace('\\\\', '\n')  # Convert LaTeX line breaks
    return text.strip()


def filter_by_difficulty(ds, target_levels=None):
    """
    Filter dataset by difficulty for focused training.

    A*-PO can benefit from curriculum learning - starting with easier problems
    helps with better value function estimation before tackling harder ones.
    """
    if target_levels is None:
        # Start with easier problems
        target_levels = ['Level 1', 'Level 2', 'Level 3']

    filtered_samples = []
    for sample in ds['train']:
        if sample['level'] in target_levels:
            filtered_samples.append(sample)

    print(
        f"Filtered to {len(filtered_samples)} samples from levels: {target_levels}")
    return filtered_samples


# Load the dataset
print("Loading MATH competition dataset...")
ds = load_dataset("qwedsacf/competition_math")

# Inspect the dataset
type_counts, level_counts = inspect_dataset(ds)

# For A*-PO training, we'll start with a subset for better value function estimation
print("\n=== Data Preprocessing for A*-PO Training ===")

# Option 1: Use all data
prompt_response_list = []

for sample in ds["train"]:
    # Extract problem as prompt and solution as response
    prompt = sample["problem"]
    response = sample["solution"]

    # Optional: Clean LaTeX formatting
    # prompt = clean_latex_formatting(prompt)
    # response = clean_latex_formatting(response)

    # Format for A*-PO training - each sample needs prompt-response pair
    # A*-PO will use these to estimate optimal value functions
    prompt_response_list.append({
        "prompt": prompt,
        "response": response,
        "level": sample["level"],
        "type": sample["type"]
    })

print(
    f"Prepared {len(prompt_response_list)} prompt-response pairs for A*-PO training")

# Print samples to understand the data structure
print("\n=== Sample Problem-Solution Pairs ===")
for i, pr in enumerate(prompt_response_list[:3]):
    print(f"Sample {i+1}:")
    print(f"Type: {pr['type']}, Level: {pr['level']}")
    print(f"Prompt: {pr['prompt'][:200]}...")
    print(f"Response: {pr['response'][:200]}...")
    print("---")

# Step 2: Save processed data for efficient loading during A*-PO training
print("\n=== Saving Processed Data ===")

# Save full dataset in JSONL format for A*-PO training
output_file = 'math_prompts_responses.jsonl'
with open(output_file, 'w', encoding='utf-8') as f:
    for sample in prompt_response_list:
        # Each line is a JSON object - efficient for streaming during training
        record = json.dumps(sample, ensure_ascii=False)
        f.write(record + '\n')

print(f"Saved {len(prompt_response_list)} samples to {output_file}")

# Create a smaller subset for initial A*-PO experimentation
subset_size = 1000
subset_file = 'math_subset.jsonl'
with open(subset_file, 'w', encoding='utf-8') as f:
    for sample in prompt_response_list[:subset_size]:
        record = json.dumps(sample, ensure_ascii=False)
        f.write(record + '\n')

print(f"Saved {subset_size} samples to {subset_file} for initial experiments")

# Create evaluation set (MATH 500)
# Note: We'll use a portion of the data as evaluation set
eval_size = 500
eval_file = 'math_eval_500.jsonl'
with open(eval_file, 'w', encoding='utf-8') as f:
    # Use last 500 samples as evaluation set
    for sample in prompt_response_list[-eval_size:]:
        record = json.dumps(sample, ensure_ascii=False)
        f.write(record + '\n')

print(f"Saved {eval_size} samples to {eval_file} for evaluation")

print("\n=== Data Preparation Complete ===")
print("Files created:")
print(f"  - {output_file}: Full training dataset")
print(f"  - {subset_file}: Subset for experimentation")
print(f"  - {eval_file}: Evaluation set")
print("\nReady for A*-PO implementation!")
