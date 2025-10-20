"""
Step 5: Evaluation Script for A*-PO Math Reasoning Model
=======================================================

This script evaluates the A*-PO trained model on mathematical reasoning tasks,
comparing first-attempt vs final-attempt accuracy to measure self-correction benefits.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

# Import our math equivalence checker
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'modeling'))
try:
    from math_equivalence import is_equiv
    MATH_EQUIV_AVAILABLE = True
except ImportError:
    print("Warning: math_equivalence module not found. Using simplified equivalence checking.")
    MATH_EQUIV_AVAILABLE = False

    def is_equiv(a, b):
        """Simplified equivalence check"""
        return str(a).strip() == str(b).strip()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MathEvaluator:
    """
    Comprehensive evaluator for mathematical reasoning models
    """

    def __init__(self, model_path: str = "./apo_trained_model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """Load the trained A*-PO model"""
        logger.info(f"Loading model from {self.model_path}")

        try:
            # Check if trained model exists
            if Path(self.model_path).exists():
                logger.info("Found trained A*-PO model, loading...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                ).to(self.device)
                logger.info("Loaded trained A*-PO model successfully")
            else:
                # Fallback to base model for demonstration
                logger.warning(f"Trained model not found at {self.model_path}")
                logger.info("Loading base model for demonstration...")

                # Try multiple model options in order of preference
                model_options = [
                    # Most reliable fallback
                    ("microsoft/DialoGPT-small", False),
                    ("distilgpt2", False),  # Another reliable option
                    ("gpt2", False)  # Final fallback
                ]

                model_loaded = False
                for model_name, trust_remote in model_options:
                    try:
                        logger.info(f"Attempting to load {model_name}...")
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_name,
                            trust_remote_code=trust_remote
                        )
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float32,  # Use float32 for compatibility
                            trust_remote_code=trust_remote
                        ).to(self.device)
                        logger.info(f"Successfully loaded {model_name}")
                        model_loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {model_name}: {e}")
                        continue

                if not model_loaded:
                    logger.error("Failed to load any fallback model")
                    return False

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def extract_answer(self, solution: str) -> str:
        """
        Extract the final answer from a mathematical solution.

        This function looks for various answer formats:
        - \\boxed{answer}
        - Final answer: answer
        - Answer: answer
        - The answer is answer
        """
        if not solution:
            return ""

        # First, try to find boxed answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution)
        if boxed_match:
            return boxed_match.group(1).strip()

        # Try other common answer patterns (using raw strings)
        patterns = [
            r'[Ff]inal answer:?\s*([^\n\.]+)',
            r'[Aa]nswer:?\s*([^\n\.]+)',
            r'[Tt]he answer is\s*([^\n\.]+)',
            r'[Tt]herefore,?\s*([^\n\.]+)',
            r'[Ss]o,?\s*([^\n\.]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, solution)
            if match:
                return match.group(1).strip()

        # If no explicit answer found, try to extract the last mathematical expression
        math_expressions = re.findall(r'[=]\s*([^\n\.]+)', solution)
        if math_expressions:
            return math_expressions[-1].strip()

        # Return empty string if no answer found
        return ""

    def generate_solution(self, problem: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate a single solution attempt"""
        try:
            prompt = f"Problem: {problem}\nSolution:"

            inputs = self.tokenizer.encode(
                prompt, return_tensors='pt', truncation=True, max_length=256)
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=256  # Limit new tokens to prevent memory issues
                )

            solution = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            return solution.strip()

        except Exception as e:
            logger.warning(f"Error generating solution: {e}")
            return "Unable to generate solution due to error."

    def multi_turn_generate(self, problem: str, max_turns: int = 3) -> List[str]:
        """
        Generate multiple solution attempts with self-correction.

        This implements the multi-turn self-correction mechanism from A*-PO training.
        """
        solutions = []
        current_prompt = problem

        for turn in range(max_turns):
            if turn == 0:
                prompt_text = f"Problem: {current_prompt}\nSolution:"
            else:
                prompt_text = f"Problem: {problem}\nPrevious attempt: {solutions[-1]}\nLet me reconsider and provide a better solution:"

            solution = self.generate_solution(prompt_text)
            solutions.append(solution)

            # Simple verification check
            if self.verify_solution(problem, solution):
                logger.debug(f"Solution verified after {turn + 1} turns")
                break

        return solutions

    def verify_solution(self, problem: str, solution: str) -> bool:
        """
        Basic solution verification.

        In practice, this would use a trained verification model.
        For now, we use simple heuristics.
        """
        # Check if solution has reasonable length
        if len(solution) < 20:
            return False

        # Check if solution contains mathematical reasoning indicators
        reasoning_indicators = ['because', 'since',
                                'therefore', 'thus', 'so', '=', 'solve']
        has_reasoning = any(indicator in solution.lower()
                            for indicator in reasoning_indicators)

        # Check if solution has a final answer
        has_answer = bool(self.extract_answer(solution))

        return has_reasoning and has_answer

    def evaluate_dataset(self, eval_file: str, max_samples: int = 500) -> Dict:
        """
        Evaluate the model on a dataset of math problems.

        Returns detailed metrics including:
        - First attempt accuracy
        - Final attempt accuracy
        - Self-correction improvement
        - Performance by problem type and difficulty
        """
        logger.info(f"Evaluating on {eval_file}")

        # Load evaluation data
        eval_data = []
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                eval_data.append(json.loads(line))

        # Limit samples if specified
        if max_samples and len(eval_data) > max_samples:
            eval_data = eval_data[:max_samples]

        logger.info(f"Evaluating on {len(eval_data)} problems")

        # Initialize metrics
        results = {
            'total_problems': len(eval_data),
            'first_attempt_correct': 0,
            'final_attempt_correct': 0,
            'by_type': defaultdict(lambda: {'total': 0, 'first_correct': 0, 'final_correct': 0}),
            'by_level': defaultdict(lambda: {'total': 0, 'first_correct': 0, 'final_correct': 0}),
            'detailed_results': []
        }

        # Evaluate each problem
        for i, sample in enumerate(eval_data):
            if i % 50 == 0:
                logger.info(
                    f"Progress: {i}/{len(eval_data)} problems evaluated")

            problem = sample['prompt']
            expected_answer = self.extract_answer(sample['response'])
            problem_type = sample.get('type', 'Unknown')
            problem_level = sample.get('level', 'Unknown')

            # Generate multiple solution attempts
            solutions = self.multi_turn_generate(problem, max_turns=3)

            # Extract answers from solutions
            generated_answers = [self.extract_answer(sol) for sol in solutions]

            # Check correctness
            first_correct = self.check_equivalence(
                generated_answers[0], expected_answer)
            final_correct = self.check_equivalence(
                generated_answers[-1], expected_answer)

            # Update metrics
            if first_correct:
                results['first_attempt_correct'] += 1
            if final_correct:
                results['final_attempt_correct'] += 1

            # Update by type
            results['by_type'][problem_type]['total'] += 1
            if first_correct:
                results['by_type'][problem_type]['first_correct'] += 1
            if final_correct:
                results['by_type'][problem_type]['final_correct'] += 1

            # Update by level
            results['by_level'][problem_level]['total'] += 1
            if first_correct:
                results['by_level'][problem_level]['first_correct'] += 1
            if final_correct:
                results['by_level'][problem_level]['final_correct'] += 1

            # Store detailed result
            results['detailed_results'].append({
                'problem': problem[:100] + "..." if len(problem) > 100 else problem,
                'expected_answer': expected_answer,
                'generated_answers': generated_answers,
                'first_correct': first_correct,
                'final_correct': final_correct,
                'type': problem_type,
                'level': problem_level
            })

        # Calculate final metrics
        results['first_attempt_accuracy'] = results['first_attempt_correct'] / \
            results['total_problems']
        results['final_attempt_accuracy'] = results['final_attempt_correct'] / \
            results['total_problems']
        results['self_correction_improvement'] = results['final_attempt_accuracy'] - \
            results['first_attempt_accuracy']

        return results

    def check_equivalence(self, answer1: str, answer2: str) -> bool:
        """
        Check if two mathematical answers are equivalent.

        Uses the math_equivalence module if available, otherwise falls back to string comparison.
        """
        if not answer1 or not answer2:
            return False

        try:
            return is_equiv(answer1, answer2)
        except:
            # Fallback to simple string comparison
            return answer1.strip().lower() == answer2.strip().lower()

    def print_results(self, results: Dict):
        """Print evaluation results in a formatted way"""
        print("\n" + "="*60)
        print("MATH REASONING EVALUATION RESULTS")
        print("="*60)

        print(f"\nOverall Performance:")
        print(f"  Total Problems: {results['total_problems']}")
        print(
            f"  First Attempt Accuracy: {results['first_attempt_accuracy']:.3f} ({results['first_attempt_correct']}/{results['total_problems']})")
        print(
            f"  Final Attempt Accuracy: {results['final_attempt_accuracy']:.3f} ({results['final_attempt_correct']}/{results['total_problems']})")
        print(
            f"  Self-Correction Improvement: {results['self_correction_improvement']:.3f}")

        # Performance by problem type
        print(f"\nPerformance by Problem Type:")
        for ptype, stats in results['by_type'].items():
            if stats['total'] > 0:
                first_acc = stats['first_correct'] / stats['total']
                final_acc = stats['final_correct'] / stats['total']
                improvement = final_acc - first_acc
                print(
                    f"  {ptype}: First {first_acc:.3f}, Final {final_acc:.3f}, Improvement {improvement:.3f} ({stats['total']} problems)")

        # Performance by difficulty level
        print(f"\nPerformance by Difficulty Level:")
        for level, stats in results['by_level'].items():
            if stats['total'] > 0:
                first_acc = stats['first_correct'] / stats['total']
                final_acc = stats['final_correct'] / stats['total']
                improvement = final_acc - first_acc
                print(
                    f"  {level}: First {first_acc:.3f}, Final {final_acc:.3f}, Improvement {improvement:.3f} ({stats['total']} problems)")

        # Show some example results
        print(f"\nExample Results:")
        for i, result in enumerate(results['detailed_results'][:5]):
            print(f"\nExample {i+1}:")
            print(f"  Problem: {result['problem']}")
            print(f"  Expected: {result['expected_answer']}")
            print(f"  Generated (First): {result['generated_answers'][0]}")
            print(f"  Generated (Final): {result['generated_answers'][-1]}")
            print(
                f"  First Correct: {result['first_correct']}, Final Correct: {result['final_correct']}")

    def save_results(self, results: Dict, output_file: str = "evaluation_results.json"):
        """Save evaluation results to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")


def main():
    """Main evaluation pipeline"""
    logger.info("=== A*-PO Math Reasoning Model Evaluation ===")

    # Initialize evaluator
    evaluator = MathEvaluator()

    # Load model
    if not evaluator.load_model():
        logger.error("Failed to load any model for evaluation.")
        return

    # Check for evaluation dataset
    eval_file = "math_eval_500.jsonl"
    if not Path(eval_file).exists():
        logger.warning(f"Evaluation file {eval_file} not found.")
        logger.info("Generating evaluation data from prepared dataset...")

        # Try to create eval data from existing data
        if Path("math_prompts_responses.jsonl").exists():
            logger.info(
                "Using math_prompts_responses.jsonl to create evaluation set")
            eval_file = "math_prompts_responses.jsonl"
        elif Path("math_subset.jsonl").exists():
            logger.info("Using math_subset.jsonl for evaluation")
            eval_file = "math_subset.jsonl"
        else:
            logger.error(
                "No evaluation data available. Please run prepare_data.py first.")
            return

    # Run evaluation
    logger.info("Starting evaluation...")
    try:
        results = evaluator.evaluate_dataset(
            eval_file, max_samples=10)  # Small sample for demo

        # Print and save results
        evaluator.print_results(results)
        evaluator.save_results(results)

        logger.info("Evaluation complete!")

        # Additional demo output
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(
            f"Model Type: {'A*-PO Trained' if Path('./apo_trained_model').exists() else 'Base Model (Demo)'}")
        print(f"Problems Evaluated: {results['total_problems']}")
        print(
            f"First Attempt Success Rate: {results['first_attempt_accuracy']:.1%}")
        print(
            f"Final Attempt Success Rate: {results['final_attempt_accuracy']:.1%}")
        improvement = results['self_correction_improvement']
        print(f"Self-Correction Improvement: {improvement:+.1%}")

        if improvement > 0:
            print("✅ Multi-turn self-correction is working!")
        else:
            print("ℹ️  No improvement from self-correction (expected for base model)")

        print("="*60)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print("⚠️  Evaluation failed, but model loading was successful.")
        print("This is normal if no training data is available yet.")


if __name__ == "__main__":
    main()
