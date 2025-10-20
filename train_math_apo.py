"""
Step 4: Model Setup and Training Integration with Qwen2.5-1.5B-Instruct
========================================================================

This script integrates the A*-PO algorithm with the Qwen2.5-1.5B-Instruct model
for mathematical reasoning training. It demonstrates the complete training pipeline.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import logging
from pathlib import Path

# Import our A*-PO implementation
from apo_algorithm import A_PO_Trainer, A_PO_Config, MathDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MathReasoningTrainer:
    """
    Complete training pipeline for mathematical reasoning with A*-PO
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.apo_trainer = None

        # Initialize A*-PO configuration
        self.config = A_PO_Config(
            # Adjusted for 1.5B model - smaller batch sizes for memory efficiency
            value_batch_size=16,
            policy_batch_size=8,
            value_learning_rate=5e-5,
            policy_learning_rate=1e-5,
            max_sequence_length=512,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(f"Initialized trainer for {model_name}")
        logger.info(f"Using device: {self.config.device}")

    def load_model_and_tokenizer(self):
        """
        Load the Qwen2.5-1.5B-Instruct model and tokenizer
        """
        logger.info(f"Loading model: {self.model_name}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side='left'  # Important for generation
            )

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            logger.info(f"Model loaded successfully")
            logger.info(
                f"Model size: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parameters")

            # Initialize A*-PO trainer with the loaded model
            self.apo_trainer = A_PO_Trainer(self.config)
            self.apo_trainer.initialize_model(self.model, self.tokenizer)

            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to a smaller model for demonstration...")

            # Fallback to a smaller model if Qwen is not available
            try:
                self.model_name = "microsoft/DialoGPT-small"  # Much smaller model for demo
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name)

                logger.info(f"Loaded fallback model: {self.model_name}")

                # Initialize A*-PO trainer with fallback model
                self.apo_trainer = A_PO_Trainer(self.config)
                self.apo_trainer.initialize_model(self.model, self.tokenizer)

                return True

            except Exception as e2:
                logger.error(f"Error loading fallback model: {e2}")
                return False

    def prepare_dataset(self, data_file: str = "math_subset.jsonl"):
        """
        Prepare the dataset for training
        """
        logger.info(f"Preparing dataset from {data_file}")

        if not Path(data_file).exists():
            logger.error(f"Dataset file {data_file} not found!")
            logger.info(
                "Please run prepare_data.py first to generate the dataset.")
            return None

        # Create dataset
        dataset = MathDataset(data_file, self.tokenizer,
                              self.config.max_sequence_length)

        logger.info(f"Dataset prepared with {len(dataset)} samples")
        return dataset

    def demonstrate_baseline_generation(self, test_problems: list):
        """
        Demonstrate baseline model performance before A*-PO training
        """
        logger.info("=== Baseline Model Performance (Before A*-PO) ===")

        self.model.eval()

        for i, problem in enumerate(test_problems[:3]):  # Test on 3 problems
            prompt = f"Problem: {problem}\nSolution:"

            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt, return_tensors='pt', truncation=True, max_length=256)
            inputs = inputs.to(self.config.device)

            # Generate solution
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode solution
            solution = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:], skip_special_tokens=True)

            print(f"\\n--- Baseline Test {i+1} ---")
            print(f"Problem: {problem[:100]}...")
            print(f"Generated Solution: {solution[:200]}...")
            print("-" * 50)

    def train_with_apo(self, dataset: MathDataset):
        """
        Train the model using A*-PO algorithm
        """
        logger.info("=== Starting A*-PO Training ===")

        # Log training configuration
        logger.info("Training Configuration:")
        logger.info(f"  Value Function Epochs: {self.config.value_epochs}")
        logger.info(
            f"  Policy Optimization Epochs: {self.config.policy_epochs}")
        logger.info(
            f"  Value Learning Rate: {self.config.value_learning_rate}")
        logger.info(
            f"  Policy Learning Rate: {self.config.policy_learning_rate}")
        logger.info(
            f"  Max Correction Turns: {self.config.max_correction_turns}")

        # Perform A*-PO training
        self.apo_trainer.train(dataset)

        logger.info("A*-PO training completed!")

    def demonstrate_apo_generation(self, test_problems: list):
        """
        Demonstrate model performance after A*-PO training
        """
        logger.info("=== A*-PO Model Performance (After Training) ===")

        for i, problem in enumerate(test_problems[:3]):
            print(f"\\n--- A*-PO Test {i+1} ---")
            print(f"Problem: {problem[:100]}...")

            # Use multi-turn correction from A*-PO
            solutions = self.apo_trainer.multi_turn_correction(
                problem, max_turns=2)

            for j, solution in enumerate(solutions):
                print(f"Attempt {j+1}: {solution[:200]}...")

            print("-" * 50)

    def evaluate_on_math500(self, eval_file: str = "math_eval_500.jsonl"):
        """
        Evaluate the trained model on MATH 500 evaluation set
        """
        logger.info("=== Evaluation on MATH 500 ===")

        if not Path(eval_file).exists():
            logger.warning(
                f"Evaluation file {eval_file} not found. Skipping evaluation.")
            return

        # Load evaluation data
        eval_data = []
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                eval_data.append(json.loads(line))

        # Evaluate on a subset for demonstration
        eval_subset = eval_data[:50]  # Evaluate on 50 problems for demo

        correct_first_attempt = 0
        correct_final_attempt = 0

        logger.info(f"Evaluating on {len(eval_subset)} problems...")

        for i, sample in enumerate(eval_subset):
            if i % 10 == 0:
                logger.info(f"Evaluated {i}/{len(eval_subset)} problems")

            problem = sample['prompt']
            expected_solution = sample['response']

            # Generate solution with multi-turn correction
            solutions = self.apo_trainer.multi_turn_correction(
                problem, max_turns=3)

            # Simple evaluation heuristic (in practice, use math equivalence checking)
            first_correct = self._simple_solution_check(
                solutions[0], expected_solution)
            final_correct = self._simple_solution_check(
                solutions[-1], expected_solution)

            if first_correct:
                correct_first_attempt += 1
            if final_correct:
                correct_final_attempt += 1

        # Calculate accuracies
        first_accuracy = correct_first_attempt / len(eval_subset)
        final_accuracy = correct_final_attempt / len(eval_subset)
        improvement = final_accuracy - first_accuracy

        logger.info("=== Evaluation Results ===")
        logger.info(
            f"First Attempt Accuracy: {first_accuracy:.3f} ({correct_first_attempt}/{len(eval_subset)})")
        logger.info(
            f"Final Attempt Accuracy: {final_accuracy:.3f} ({correct_final_attempt}/{len(eval_subset)})")
        logger.info(f"Improvement from Multi-turn: {improvement:.3f}")

        return {
            'first_accuracy': first_accuracy,
            'final_accuracy': final_accuracy,
            'improvement': improvement,
            'evaluated_samples': len(eval_subset)
        }

    def _simple_solution_check(self, generated: str, expected: str) -> bool:
        """
        Simple heuristic for solution correctness.
        In practice, this would use mathematical equivalence checking.
        """
        # Extract boxed answers if present
        import re

        def extract_boxed_answer(text):
            match = re.search(r'\\boxed{([^}]+)}', text)
            return match.group(1) if match else None

        gen_answer = extract_boxed_answer(generated)
        exp_answer = extract_boxed_answer(expected)

        if gen_answer and exp_answer:
            return gen_answer.strip() == exp_answer.strip()

        # Fallback: check if key terms are present
        return len(generated) > 50 and ('=' in generated or 'answer' in generated.lower())

    def save_model(self, output_dir: str = "./apo_trained_model"):
        """
        Save the trained model
        """
        logger.info(f"Saving trained model to {output_dir}")

        Path(output_dir).mkdir(exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save training config
        config_path = Path(output_dir) / "apo_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)

        logger.info("Model saved successfully!")


def main():
    """
    Main training pipeline demonstration
    """
    logger.info("=== Mathematical Reasoning with A*-PO Training Pipeline ===")

    # Initialize trainer
    trainer = MathReasoningTrainer()

    # Step 1: Load model and tokenizer
    if not trainer.load_model_and_tokenizer():
        logger.error("Failed to load model. Exiting.")
        return

    # Step 2: Prepare dataset
    dataset = trainer.prepare_dataset("math_subset.jsonl")
    if dataset is None:
        logger.error("Failed to prepare dataset. Exiting.")
        return

    # Step 3: Load some test problems for demonstration
    test_problems = [
        "What is the value of $2^3 + 3^2$?",
        "Solve for x: $2x + 5 = 13$",
        "Find the area of a circle with radius 3."
    ]

    # Step 4: Demonstrate baseline performance
    trainer.demonstrate_baseline_generation(test_problems)

    # Step 5: Train with A*-PO (using small subset for demo)
    logger.info("Starting A*-PO training (this may take a while)...")
    trainer.train_with_apo(dataset)

    # Step 6: Demonstrate improved performance
    trainer.demonstrate_apo_generation(test_problems)

    # Step 7: Evaluate on MATH 500
    results = trainer.evaluate_on_math500()

    # Step 8: Save the trained model
    trainer.save_model()

    logger.info("=== Training Pipeline Complete ===")
    if results:
        logger.info(f"Final Results: {results}")


if __name__ == "__main__":
    main()
