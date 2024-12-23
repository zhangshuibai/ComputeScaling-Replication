import math
import statistics
from typing import Optional, List

import torch
from prm_interface import PRM, StepScore

from utils import condense_newlines
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore


class Llemma7bPRM(PRM):
    def __init__(
        self,
        aggregation: str = "min",
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the Llemma7bPRM class.

        Args:
            aggregation (str): Method to aggregate step probabilities ('min', 'max', 'mean', 'prod', 'last').
            quantization_config (Optional[BitsAndBytesConfig]): Configuration for model quantization.
            device (Optional[torch.device]): Device to run the model on ('cuda' or 'cpu').
        """
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "ScalableMath/llemma-7b-prm-prm800k-level-1to3-hf",
            quantization_config=quantization_config,
        )
        if not quantization_config:
            self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llemma_7b")
        if self.tokenizer.pad_token is None:
            # Option 2: Add a new pad_token '[PAD]' (Recommended)
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Encode special tokens without adding additional special tokens
        self.begin_solution_tokens = self.tokenizer.encode(
            "# Solution", add_special_tokens=False
        )[1:]
        self.scoring_tokens = self.tokenizer.encode("\n\n", add_special_tokens=False)[1:]
        self.eos_token = self.tokenizer.eos_token_id
        self.pad_token = self.tokenizer.pad_token_id

        self.aggregation = aggregation
        self.padding_side = self.tokenizer.padding_side  # 'right' or 'left'
        

    def __call_batch(self, beams: List[str]) -> List[StepScore]:
        """
        Process multiple beams in a batch and return corresponding StepScores.

        Args:
            beams (List[str]): Multiple reasoning beams.

        Returns:
            List[StepScore]: StepScores for each beam.
        """
        original_beams = beams  # Preserve original beams for final results
        beams = [condense_newlines(beam) for beam in beams]

        # Tokenize with padding
        encoded = self.tokenizer(
            beams,
            return_tensors='pt',
            padding=True,  # Automatically handle padding based on tokenizer's padding_side
            truncation=True,
            add_special_tokens=False
        )

        input_ids = encoded['input_ids'].to(self.device)  # Shape: (batch_size, seq_length)
        attention_mask = encoded['attention_mask'].to(self.device)  # Shape: (batch_size, seq_length)

        batch_size, seq_length = input_ids.size()

        # Find candidate positions for each beam
        candidate_positions_list = []
        # temp_pos_list = []

        for i in range(batch_size):
            ids = input_ids[i].tolist()
            candidate_positions = []
            # temp_pos = []
            begin_solution_flag = False

            for start_idx in range(len(ids)):
                # Avoid negative indexing and ensure the slice does not exceed sequence length
                end_idx_solution = start_idx + len(self.begin_solution_tokens)
                end_idx_scoring = start_idx + len(self.scoring_tokens)

                # Check for "# Solution" tokens
                if not begin_solution_flag and tuple(ids[start_idx:end_idx_solution]) == tuple(self.begin_solution_tokens):
                    begin_solution_flag = True

                # Check for scoring tokens after "# Solution"
                if begin_solution_flag and tuple(ids[start_idx:end_idx_scoring]) == tuple(self.scoring_tokens):
                    # Ensure the scoring tokens are not padding tokens
                    if not any(token == self.pad_token for token in ids[start_idx:end_idx_scoring]):
                        candidate_positions.append(start_idx)
                        # Extract surrounding tokens for debugging or additional processing
                        # surrounding_start = max(start_idx - 2, 0)
                        # surrounding_end = min(start_idx + len(self.scoring_tokens), len(ids))
                        # temp_pos.append(
                        #     self.tokenizer.decode(ids[surrounding_start:surrounding_end])
                        # )

                # Check for the end of sequence (EOS)
                if ids[start_idx] == self.eos_token:
                    # Ensure EOS is not a padding token
                    if ids[start_idx] != self.pad_token:
                        candidate_positions.append(start_idx)
                        # temp_pos.append(self.tokenizer.decode([ids[start_idx]]))
                    break  # Stop processing after EOS

            # Remove the first candidate position which corresponds to "# Solution"
            if candidate_positions:
                del candidate_positions[0]

            candidate_positions_list.append(candidate_positions)
            # temp_pos_list.append(temp_pos)

        # Determine the maximum number of candidates across all beams
        max_candidates = max(len(cands) for cands in candidate_positions_list)
        if max_candidates == 0:
            # If no candidates are found in any beam, return scores as 0.0
            return [StepScore(step=beam, score=0.0) for beam in original_beams]

        # Perform model inference
        with torch.no_grad():
            logits = self.model(input_ids).logits  # Shape: (batch_size, seq_length, vocab_size)
            scores = logits.mean(dim=-1)  # Shape: (batch_size, seq_length)

            # Initialize a list to store step probabilities for each beam
            step_probs_list = []

            for i, candidate_positions in enumerate(candidate_positions_list):
                if not candidate_positions:
                    # No candidates found for this beam
                    step_probs_list.append([])
                    continue

                # Extract scores for the candidate positions
                step_scores = scores[i, candidate_positions]  # Shape: (num_candidates,)
                # Apply sigmoid to convert logits to probabilities
                step_probs = torch.sigmoid(step_scores).cpu().tolist()
                step_probs_list.append(step_probs)

        # Aggregate scores based on the specified aggregation method
        results = []
        for i, beam in enumerate(original_beams):
            step_probs = step_probs_list[i]
            if not step_probs:
                score = 0.0
            elif self.aggregation == "min":
                score = min(step_probs)
            elif self.aggregation == "max":
                score = max(step_probs)
            elif self.aggregation == "mean":
                score = statistics.mean(step_probs)
            elif self.aggregation == "prod":
                score = math.prod(step_probs)
            elif self.aggregation == "last":
                score = step_probs[-1]
            else:
                raise NotImplementedError(f"Aggregation method {self.aggregation} not implemented.")
            results.append(StepScore(step=beam, score=score))

            # Uncomment the following lines for debugging purposes
            # print(f"Beam: {repr(beam)}")
            # print(f"Step Probs: {step_probs}")
            # print(len(step_probs))
            # print(f"Score: {score}")
            # print(temp_pos_list[i])
            # input()

        return results

    def __call__(self, steps: List[str]) -> List[StepScore]:
        """
        Process multiple reasoning beams and return their corresponding scores.

        Args:
            steps (List[str]): A list of reasoning beams.

        Returns:
            List[StepScore]: A list of StepScores for each beam.
        """
        return self.__call_batch(steps)