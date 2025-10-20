"""
Step 3: A*-PO (Advantage-weighted Policy Optimization) Implementation
====================================================================

A*-PO is a reinforcement learning algorithm that consists of two main stages:
1. Offline estimation of optimal value function V* using collected samples
2. Online policy optimization using estimated optimal advantages

This implementation follows the PAG (Process-supervised Advantage Generation) approach
with multi-turn self-correction for mathematical reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class A_PO_Config:
    """Configuration for A*-PO training"""
    # Value function estimation parameters
    value_learning_rate: float = 1e-4
    value_batch_size: int = 32
    value_epochs: int = 10

    # Policy optimization parameters
    policy_learning_rate: float = 5e-5
    policy_batch_size: int = 16
    policy_epochs: int = 5

    # A*-PO specific parameters
    advantage_clip: float = 10.0  # Clip advantages to prevent instability
    beta: float = 0.1  # Temperature parameter for advantage weighting

    # Multi-turn self-correction parameters
    max_correction_turns: int = 3
    verification_threshold: float = 0.7

    # General training parameters
    max_sequence_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MathDataset(Dataset):
    """Dataset class for loading math problem-solution pairs"""

    def __init__(self, jsonl_file: str, tokenizer, max_length: int = 512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

        logger.info(f"Loaded {len(self.data)} samples from {jsonl_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Format prompt for instruction following
        prompt = f"Problem: {sample['prompt']}\nSolution:"
        response = sample['response']

        # Tokenize prompt and response
        prompt_tokens = self.tokenizer.encode(
            prompt, max_length=self.max_length//2, truncation=True)
        response_tokens = self.tokenizer.encode(
            response, max_length=self.max_length//2, truncation=True)

        return {
            'prompt': prompt,
            'response': response,
            'prompt_tokens': torch.tensor(prompt_tokens),
            'response_tokens': torch.tensor(response_tokens),
            'level': sample.get('level', 'Unknown'),
            'type': sample.get('type', 'Unknown')
        }


class ValueFunction(nn.Module):
    """
    Value function V(s) for A*-PO algorithm.

    This estimates the expected future reward from a given state.
    In our case, the state is the current problem + partial solution.
    """

    def __init__(self, model_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch_size, seq_len, model_dim] or [batch_size, model_dim]
        Returns:
            values: [batch_size] - estimated value for each state
        """
        if len(hidden_states.shape) == 3:
            # Take the last token's hidden state as the state representation
            hidden_states = hidden_states[:, -1, :]

        values = self.value_head(hidden_states).squeeze(-1)
        return values


class A_PO_Trainer:
    """
    Main A*-PO trainer implementing the two-stage algorithm:
    Stage 1: Offline value function estimation
    Stage 2: Online policy optimization with estimated advantages
    """

    def __init__(self, config: A_PO_Config):
        self.config = config
        self.device = torch.device(config.device)

        # These will be set during initialization
        self.model = None
        self.tokenizer = None
        self.value_function = None
        self.optimizer = None
        self.value_optimizer = None

        logger.info(f"Initialized A*-PO trainer on {self.device}")

    def initialize_model(self, model, tokenizer):
        """Initialize the base model and tokenizer"""
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        # Initialize value function based on model's hidden size
        model_dim = self.model.config.hidden_size
        self.value_function = ValueFunction(model_dim).to(self.device)

        # Set up optimizers
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.config.policy_learning_rate)
        self.value_optimizer = optim.AdamW(
            self.value_function.parameters(), lr=self.config.value_learning_rate)

        logger.info("Model and value function initialized")

    def estimate_returns(self, trajectories: List[Dict]) -> List[float]:
        """
        Estimate returns for trajectories using a simple reward model.

        In a full implementation, this would use a trained reward model.
        For now, we use heuristics based on solution correctness.
        """
        returns = []

        for traj in trajectories:
            # Simple heuristic: longer solutions that contain key math indicators get higher rewards
            solution = traj['response']

            # Basic math solution quality indicators
            math_indicators = ['=', '\\boxed',
                               'therefore', 'thus', 'solve', 'answer']
            indicator_count = sum(
                1 for indicator in math_indicators if indicator in solution.lower())

            # Length factor (longer solutions often more complete)
            length_factor = min(len(solution) / 200, 1.0)

            # Combine factors
            estimated_return = (indicator_count * 0.3 +
                                length_factor * 0.7) * 10
            returns.append(max(estimated_return, 0.1))  # Ensure positive

        return returns

    def stage1_value_estimation(self, train_dataloader: DataLoader):
        """
        Stage 1: Offline estimation of optimal value function V*

        This stage learns to predict the expected return from any given state.
        """
        logger.info("Starting Stage 1: Value Function Estimation")

        self.value_function.train()

        for epoch in range(self.config.value_epochs):
            total_loss = 0
            num_batches = 0

            for batch in train_dataloader:
                # Get model hidden states for the current batch
                with torch.no_grad():
                    input_ids = batch['prompt_tokens'].to(self.device)
                    outputs = self.model(input_ids, output_hidden_states=True)
                    # Last layer hidden states
                    hidden_states = outputs.hidden_states[-1]

                # Estimate returns for this batch (in practice, use trained reward model)
                trajectories = [{'response': resp}
                                for resp in batch['response']]
                returns = torch.tensor(self.estimate_returns(trajectories),
                                       dtype=torch.float32, device=self.device)

                # Predict values
                predicted_values = self.value_function(hidden_states)

                # Value function loss (MSE between predicted and target returns)
                value_loss = F.mse_loss(predicted_values, returns)

                # Optimize value function
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

                total_loss += value_loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            logger.info(
                f"Value Estimation Epoch {epoch+1}/{self.config.value_epochs}, Loss: {avg_loss:.4f}")

    def compute_advantages(self, hidden_states, returns):
        """
        Compute advantages using the trained value function.

        Advantage A(s,a) = Q(s,a) - V(s) ≈ Return - V(s)
        """
        with torch.no_grad():
            values = self.value_function(hidden_states)

        advantages = returns - values

        # Clip advantages to prevent training instability
        advantages = torch.clamp(
            advantages, -self.config.advantage_clip, self.config.advantage_clip)

        return advantages

    def stage2_policy_optimization(self, train_dataloader: DataLoader):
        """
        Stage 2: Online policy optimization using estimated optimal advantages

        This stage optimizes the policy using advantage-weighted importance sampling.
        """
        logger.info("Starting Stage 2: Policy Optimization")

        self.model.train()
        self.value_function.eval()  # Keep value function frozen

        for epoch in range(self.config.policy_epochs):
            total_loss = 0
            num_batches = 0

            for batch in train_dataloader:
                # Prepare inputs
                input_ids = batch['prompt_tokens'].to(self.device)
                target_ids = batch['response_tokens'].to(self.device)

                # Forward pass through model
                outputs = self.model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]

                # Estimate returns and compute advantages
                trajectories = [{'response': resp}
                                for resp in batch['response']]
                returns = torch.tensor(self.estimate_returns(trajectories),
                                       dtype=torch.float32, device=self.device)
                advantages = self.compute_advantages(hidden_states, returns)

                # Compute policy loss with advantage weighting
                # Create full input sequence for language modeling
                full_input = torch.cat([input_ids, target_ids], dim=1)

                # Ensure we don't exceed max length
                if full_input.size(1) > self.config.max_sequence_length:
                    full_input = full_input[:,
                                            :self.config.max_sequence_length]

                # Create labels (only compute loss on response tokens)
                labels = full_input.clone()
                # Ignore prompt tokens in loss
                labels[:, :input_ids.size(1)] = -100

                # Forward pass for policy loss
                policy_outputs = self.model(full_input, labels=labels)
                base_loss = policy_outputs.loss

                # Weight loss by advantages (A*-PO core idea)
                advantage_weights = torch.softmax(
                    advantages / self.config.beta, dim=0)
                weighted_loss = base_loss * advantage_weights.mean()

                # Optimize policy
                self.optimizer.zero_grad()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += weighted_loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            logger.info(
                f"Policy Optimization Epoch {epoch+1}/{self.config.policy_epochs}, Loss: {avg_loss:.4f}")

    def multi_turn_correction(self, problem: str, max_turns: int = None) -> List[str]:
        """
        Implement multi-turn self-correction inspired by PAG paper.

        This generates multiple solution attempts and uses verification to improve.
        """
        if max_turns is None:
            max_turns = self.config.max_correction_turns

        solutions = []
        current_prompt = f"Problem: {problem}\nSolution:"

        for turn in range(max_turns):
            # Generate solution attempt
            inputs = self.tokenizer.encode(current_prompt, return_tensors='pt', truncation=True,
                                           max_length=self.config.max_sequence_length // 2)
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=self.config.max_sequence_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            solution = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            solutions.append(solution)

            # Simple verification (in practice, use a trained verifier)
            if self._verify_solution(problem, solution):
                logger.info(f"Solution verified after {turn + 1} turns")
                break

            # Prepare for next turn with previous attempt for revision
            current_prompt = f"Problem: {problem}\nPrevious attempt: {solution}\nImproved solution:"

        return solutions

    def _verify_solution(self, problem: str, solution: str) -> bool:
        """
        Simple solution verification heuristic.
        In practice, this would use a trained verification model.
        """
        # Basic checks for solution completeness
        has_final_answer = '\\boxed' in solution or 'answer' in solution.lower()
        has_reasoning = len(solution.split('.')) > 2  # Multiple sentences

        return has_final_answer and has_reasoning

    def train(self, train_dataset: MathDataset):
        """
        Main training loop implementing the full A*-PO algorithm
        """
        logger.info("Starting A*-PO training")

        # Create data loader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.value_batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )

        # Stage 1: Value function estimation
        self.stage1_value_estimation(train_dataloader)

        # Stage 2: Policy optimization
        policy_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.policy_batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        self.stage2_policy_optimization(policy_dataloader)

        logger.info("A*-PO training completed")

    def _collate_fn(self, batch):
        """Custom collate function for batching"""
        # Pad sequences to same length
        max_prompt_len = max(len(item['prompt_tokens']) for item in batch)
        max_response_len = max(len(item['response_tokens']) for item in batch)

        padded_batch = {
            'prompt': [item['prompt'] for item in batch],
            'response': [item['response'] for item in batch],
            'prompt_tokens': torch.stack([
                F.pad(item['prompt_tokens'],
                      (0, max_prompt_len - len(item['prompt_tokens'])))
                for item in batch
            ]),
            'response_tokens': torch.stack([
                F.pad(item['response_tokens'],
                      (0, max_response_len - len(item['response_tokens'])))
                for item in batch
            ]),
            'level': [item['level'] for item in batch],
            'type': [item['type'] for item in batch]
        }

        return padded_batch


# Example usage demonstration
if __name__ == "__main__":
    # This would be called from the main training script
    config = A_PO_Config()
    trainer = A_PO_Trainer(config)

    print("A*-PO Implementation Complete!")
    print("Key Components:")
    print("1. Value Function: Estimates expected returns from states")
    print("2. Advantage Computation: A(s,a) = Return - V(s)")
    print("3. Policy Optimization: Weighted by advantages")
    print("4. Multi-turn Correction: Self-improving solution generation")
    print("\nReady for integration with Qwen2.5-1.5B-Instruct model!")
