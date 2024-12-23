import math,re
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
import sys
sys.path.append("api_model_single_call")
from gpt_agent import gpt_agent

class gpt4o_prm(PRM):
    def __init__(
        self,
        aggregation: str = "min",
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[Device] = None,
        debug = True,
    ) -> None:

        self.aggregation = aggregation
        self.debug = debug

        self.system_prompt = "You are an experienced grader with broad knowledge across subjects, focused on evaluating the correctness of reasoning steps in solutions. \n\n###Task###:\nYou will be given a question, a reference answer, and a student's step-by-step solution, which includes several intermediate steps and a final answer. Each step begins with ##Solution## and ends with a newline character.\n\nYour primary task is to identify the first reasoning step that is incorrect. Evaluate correctness based on the following criteria: \n1. Factual Accuracy - Each step should contain accurate information. \n2. Logical Validity - Each step should logically follow from prior steps. \n3. Consistency with Previous Steps - The step should build appropriately on preceding steps. \nA step that does not meet any of these criteria and leads the reasoning toward an incorrect solution is considered incorrect. \nFor each step, assess its correctness sequentially. Stop at the first incorrect step, if any, and output only that step's index. If all steps are correct, output error_idx: -1. \n\n###Output Format Example###: \n\nReasoning about correctness of step 1. \n\nReasoning about correctness of step 2 given step 1. \n\nReasoning about correctness of step 3 given steps 1 and 2. \n\n... \n\nerror_idx: INDEX_OF_FIRST_INCORRECT_STEP\n\nreference_answer: true/false\n\n"
        
        self.gpt4o_evaluator = gpt_agent(role_description=self.system_prompt, model_name= "gpt-4o", temperature=1.0)



        

    def __call_single(self, single_beam: str, len_step) -> float | list[float]:
        

        evaluation = self.gpt4o_evaluator(single_beam)

        try:
            # error_idx = int(error_idx)
            error_idx = int(re.search(r'-?\d+', evaluation.split('error_idx: ')[1]).group())

            if error_idx == -1:
                generated_rewards = [1.0] * len_step
            elif 1 <= error_idx <= len_step:
                generated_rewards = [1.0] * (error_idx - 1) + [-1.0] * (len_step - error_idx + 1)
            else:
                print('error_idx {} not matching the number of steps. Skip labeling this example.'.format(error_idx))
                generated_rewards = [-2.0] * len_step
        except:
            print('output not in the correct format. Skip labeling this example.')
            generated_rewards = [-3.0] * len_step

        step_probs = generated_rewards

        if self.aggregation == "full":
            if self.debug:
                return step_probs, evaluation
            else:
                return step_probs, None
        else:
            raise NotImplementedError

    def __call__(self, steps: list[str], len_steps: list[int]):
        """
        Args:
            steps (list[str]): A list of reasoning beams.

        Returns:
            list[StepScore]: A list of dictionaries where each dictionary
        """
        assert len(steps) == len(len_steps)
        result = []

        for beam, len_step in zip(steps,len_steps):
            step_score, evaluation_step = self.__call_single(beam, len_step)
            if self.debug:
                result.append((StepScore(step=beam, score=step_score), evaluation_step))
            else:
                result.append(StepScore(step=beam, score=step_score))

        return result

if __name__ == "__main__":
    pass



