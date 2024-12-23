import math
import statistics
from typing import Optional
from peft import PeftModel
import torch,os
from prm_interface import PRM, StepScore
from transformers import BitsAndBytesConfig
from torch.types import Device
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import json

def read_json_file(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data



from tqdm import tqdm

def check_model_precision(model):
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, dtype: {param.dtype}")

def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  
    tokenizer.padding_side = 'left' 
    tokenizer.truncation_side = 'left'
    return tokenizer

class test_prm_dual(PRM):
    def __init__(
        self,
        aggregation: str = "full",
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[Device] = None,
    ) -> None:
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.quantization_config = quantization_config
        self.model_id = "mtzig/prm800k_llama_joint_checkpoint4500"

        self.tokenizer = get_tokenizer(self.model_id)

        self.candidate_tokens = [128254, 128255]

        if self.quantization_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, quantization_config=quantization_config
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

        if not quantization_config:
            self.model.to(self.device)

        self.aggregation = aggregation

    def __call_single(self, single_beam: str) -> list[float]:
        """
        Computes scores for each reasoning step in the single_beam.

        Args:
            single_beam (str): A single reasoning beam, consisting of Question + Solution.

        Returns:
            list[float]: The scores for each step in the Solution.
        """
        # Tokenize the entire input
        encoded = self.tokenizer.encode(single_beam, return_tensors='pt').to(self.device)
        input_ids = encoded[0]
        total_length = input_ids.size(0)

        # Get logits from the model
        with torch.no_grad():
            logits = self.model(encoded).logits  # Shape: [1, seq_length, vocab_size]

        # Identify the token IDs for separators
        separator = self.tokenizer.encode(" \n\n", add_special_tokens=False)
        sep_len = len(separator)

        sep_positions = []
        for i in range(total_length - sep_len + 1):
            if input_ids[i:i+sep_len].tolist() == separator:
                sep_positions.append(i)

        if not sep_positions:
            raise ValueError

        # Define step boundaries based on separators
        step_boundaries = [0] + [pos + sep_len for pos in sep_positions] + [total_length]

        step_scores = []

        for i in range(1, len(step_boundaries)-1):
            start = step_boundaries[i]
            end = sep_positions[i] if i < len(sep_positions) else total_length

            if start >= end:
                continue  

            step_tokens = input_ids[start:end]
            last_token_index = end - 1

            last_token_logits = logits[0, last_token_index, self.candidate_tokens]
            last_token_probs = torch.softmax(last_token_logits, dim=-1)

            # Select the probability of the "good" candidate token (assuming index 0)
            step_score = last_token_probs[1].item()
            step_scores.append(step_score)
        # print(step_scores)

        # step_probs  = step_scores.tolist()
        step_probs = step_scores

        if self.aggregation == "min":
            return min(step_probs)
        elif self.aggregation == "max":
            return max(step_probs)
        elif self.aggregation == "mean":
            return statistics.mean(step_probs)
        elif self.aggregation == "prod":
            return math.prod(step_probs)
        elif self.aggregation == "last":
            return step_probs[-1]
        elif self.aggregation == "full":
            return step_probs
        else:
            raise NotImplementedError


    def __call__(self, steps: list[str]) -> list[StepScore]:
        """
        Computes scores for a list of reasoning beams.

        Args:
            steps (list[str]): A list of reasoning beams.

        Returns:
            list[StepScore]: A list of StepScore objects, each containing step and score.
        """
        result = []

        for beam in steps:
            step_score = self.__call_single(beam)
            result.append(StepScore(step=beam, score=step_score))

        return result

if __name__ == "__main__":
    prm = test_prm_dual(
                aggregation="full", 
            )
    
    json_file_path = ""
    data = read_json_file(json_file_path)

    for each_data in tqdm(data):
        for cot in each_data["chain_of_thoughts"]:
            steps = [step.strip().replace("\n", "") for step in steps]
            question = each_data["question"].strip().replace("\n", "")
            updated_steps = []
            for index, step in enumerate(steps):
                indexed_step = f" \n\n{step}"
                updated_steps.append(indexed_step)
            steps = updated_steps
            steps_all = f"{question}" + "".join(steps)
            rewards = prm([steps_all])
            cot["prm_reward"] = rewards[0].score
    

