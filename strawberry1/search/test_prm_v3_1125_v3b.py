import math
import math
import statistics
from typing import Optional
from peft import PeftModel
import torch
from prm_interface import PRM, StepScore
from torch.types import Device
from transformers import (  # type: ignore  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
def check_model_precision(model):
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, dtype: {param.dtype}")

def get_tokenizer(model_id):
        
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token #llama doesn't define pad token, so we need to do this
    tokenizer.padding_side='left' # we want to pad from the left
    tokenizer.truncation_side = 'left'

    return tokenizer

class test_prm_v3(PRM):
    def __init__(
        self,
        aggregation: str = "min",
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[Device] = None,
    ) -> None:
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.quantization_config = quantization_config
        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = 'ки'
        self.model_id = "mtzig/v3b_mistral_lora"

        self.tokenizer = get_tokenizer(self.model_id)
        
        self.candidate_tokens = self.tokenizer.encode(f"{self.good_token} {self.bad_token}")[1:] # [648, 387]
        self.step_tag_id = self.tokenizer.encode(f"{self.step_tag}")[-1] # 12902
        ###
        if self.quantization_config:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id,
                                                       quantization_config=quantization_config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

        if not quantization_config:
            self.model.to(self.device)

        self.aggregation = aggregation

    def __call_single(self, single_beam: str) -> float | list[float]:
        input_for_prm = single_beam
        input_id = torch.tensor([self.tokenizer.encode(input_for_prm)]).to(self.device)

        with torch.no_grad():
            logits = self.model(input_id).logits[:,:,self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,0]#for the good token
            # print(scores)
            # print(scores.shape)
            step_scores = scores[input_id == self.step_tag_id]
            # print(single_beam)
            # print(input_id.shape)
            # print(step_scores)
       
        step_probs  = step_scores.tolist()

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
        Args:
            steps (list[str]): A list of reasoning beams.

        Returns:
            list[StepScore]: A list of dictionaries where each dictionary
        """

        result = []

        for beam in steps:
            step_score = self.__call_single(beam)
            result.append(StepScore(step=beam, score=step_score))

        return result

if __name__ == "__main__":
    pass



