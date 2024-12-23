import math,os
import statistics
from typing import Optional

import torch
from peft import PeftModel
from prm_interface import PRM, StepScore
from torch.types import Device
from transformers import (  # type: ignore  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from utils import condense_newlines
from transformers import BitsAndBytesConfig

class test_prm_v0(PRM):
    def __init__(
        self,
        aggregation: str = "min",
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[Device] = None,
    ) -> None:
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_id = 'mtzig/v0_mistral_lora_batch8'
        
        base_model_name = "peiyi9979/math-shepherd-mistral-7b-prm"
        model = AutoModelForCausalLM.from_pretrained(base_model_name,
                                                    quantization_config=quantization_config)

        lora_adapter_path = self.model_id
        self.model = PeftModel.from_pretrained(model, lora_adapter_path)
        ###

        if not quantization_config:
            self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.begin_solution_tokens = self.tokenizer.encode(
            "# Answer", add_special_tokens=False
        )[1:]
        self.scoring_tokens = self.tokenizer.encode("\n\n", add_special_tokens=False)[
            1:
        ]
        self.eos_token = self.tokenizer.eos_token_id

        self.aggregation = aggregation

    def __call_single(self, single_beam: str) -> float | list[float]:
        single_beam = condense_newlines(single_beam)
        input_ids = self.tokenizer.encode(single_beam)

        begin_solution_flag = False

        candidate_positions = []

        temp_pos = []


        for start_idx in range(len(input_ids)):
            if tuple(
                input_ids[start_idx : start_idx + len(self.begin_solution_tokens)]
            ) == tuple(self.begin_solution_tokens):
                begin_solution_flag = True

            if begin_solution_flag and tuple(
                input_ids[start_idx : start_idx + len(self.scoring_tokens)]
            ) == tuple(self.scoring_tokens):
                candidate_positions.append(start_idx)
                temp_pos.append(self.tokenizer.batch_decode([input_ids[start_idx-3 : start_idx + len(self.scoring_tokens)]]))

            if input_ids[start_idx] == self.eos_token:
                candidate_positions.append(start_idx)
                temp_pos.append(self.tokenizer.batch_decode([input_ids[start_idx]]))
                break

        if not candidate_positions:
            return 0
        # del candidate_positions[0]

        # if not candidate_positions:
        #     return 0

        input_tensor = torch.tensor([input_ids]).to(self.device)
        candidate_positions_tensor = torch.tensor(candidate_positions)

        ###
        # print(temp_pos)
        ###

        with torch.no_grad():
            logits = self.model(input_tensor).logits
            # print(logits.shape)
            # input()
            scores = logits[:,:,:2].softmax(dim=-1)[:,:,1]
            step_probs = scores[0][candidate_positions_tensor].tolist()

            # step_probs = torch.sigmoid(step_scores).tolist()
            # print(step_probs)
            # print(temp_pos)

        # print(repr(single_beam))
        # print(len(single_beam))
        # print(self.tokenizer.batch_decode(input_ids[102 : 102 + len(self.scoring_tokens)]))
        # print(len(step_probs))
        # print(step_probs)
        # print(candidate_positions)
        # print(self.tokenizer.batch_decode(input_ids[108 : 108 + len(self.scoring_tokens)]))
        # input()

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
    # pass
    qa_example = """# Question

Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$

# Answer To convert from rectangular to polar coordinates, I need to use the formulas $r = \sqrt{x^2 + y^2}$ and $\theta = \tan^{-1}(y/x).$

In this case, $x = 0$ and $y = 3,$ so I can plug them into the formulas.

For $r,$ I get $r = \sqrt{0^2 + 3^2} = \sqrt{9} = 3.$

For $\theta,$ I get $\theta = \tan^{-1}(3/0).$

This is undefined, since the tangent function is not defined at $0.$

However, I can use the fact that the point $(0,3)$ lies on the positive $y$-axis, which has an angle of $\pi/2$ radians or $90^\circ.$

Therefore, I can choose any angle in the range $(0,\pi/2)$ as the value of $\theta.$

I will choose $\theta = \pi/2,$ since it is the simplest and most natural choice.

Therefore, the polar coordinates of the point $(0,3)$ are $(3,\pi/2).$

"""
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    v0 = test_prm_v0(aggregation = "full", quantization_config=quantization_config)
    probs = v0([qa_example])
    print(probs)

    