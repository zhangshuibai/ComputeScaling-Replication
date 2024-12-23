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
sys.path.append("/home/ec2-user/strawberry/api_model_single_call")
from aws_agent import aws_agent

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# def get_tokenizer(model_id):
#     tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token = "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")
#     tokenizer.pad_token = tokenizer.eos_token  # llama doesn't define pad token, so we need to do this
#     tokenizer.padding_side = 'left'  # we want to pad from the left
#     tokenizer.truncation_side = 'left'
#     return tokenizer

# class llama_agent_local():

#     def __init__(self, role_description="", model_name="", temperature=0.7):
#         self.role_description = role_description
#         self.temperature = temperature

#         # Automatically set device to GPU if available, otherwise CPU
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Load model and move it to the selected device
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             use_auth_token = "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"
#         ).to(self.device)
#         self.tokenizer = get_tokenizer(model_name)

#         self.system_prompt = role_description

#     def __call__(self, prompt: str) -> str:
#         # Concatenate the system prompt and user prompt
#         full_prompt = self.system_prompt + "\n" + prompt

#         # Tokenize the input without padding
#         inputs = self.tokenizer(
#             full_prompt,
#             return_tensors="pt",
#             truncation=True,
#             max_length=2048  # Adjust max_length as needed
#         )

#         # Move input tensors to the same device as the model
#         input_ids = inputs["input_ids"].to(self.device)
#         attention_mask = inputs["attention_mask"].to(self.device)

#         # Generate output from the model
#         outputs = self.model.generate(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             max_length=2048,  # Adjust max_length as needed
#             temperature=self.temperature,
#             eos_token_id=self.tokenizer.eos_token_id,  # Optional: Stop generation at EOS token
#         )

#         # Decode the generated tokens to text
#         response = self.tokenizer.decode(
#             outputs[0],
#             skip_special_tokens=True
#         )

#         return response
    

class llama_zeorshot_prm(PRM):
    def __init__(
        self,
        aggregation: str = "full",
        device: Optional[Device] = None,
        debug = True,
    ) -> None:

        self.aggregation = aggregation
        self.debug = debug

        self.system_prompt = """
        You are an experienced evaluator specializing in assessing the quality of reasoning steps in problem-solving. Your task is to analyze and classify the last step in a student's solution based on its correctness, clarity, and contribution to the reasoning process.

        #### Task Description
        You will be provided with:
        1. A Question
        2. A Reference Answer
        3. A Student's Step-by-Step Solution, where each step begins with `##Solution##` and ends with a newline character.

        The provided solution may be partial or incomplete, so evaluate correctness only up to the given steps. Do not assume any unstated steps or context unless explicitly provided.

        Your job is to analyze the last step of the solution and classify it as one of the following:
        - GOOD
        - OK
        - BAD

        #### Criteria for Classification
        1. GOOD Step
        A step is classified as GOOD if it meets all of these criteria:
        - Correct: Everything stated is accurate and aligns with known principles or the given problem.
        - Verifiable: The step can be verified using common knowledge, simple calculations, or a quick reference (e.g., recalling a basic theorem). If verifying requires extensive effort (e.g., detailed calculations or obscure references), mark it BAD instead.
        - Appropriate: The step fits logically within the context of the preceding steps. If a prior mistake exists, a GOOD step can correct it.
        - Insightful: The step demonstrates reasonable problem-solving direction. Even if ultimately progress in the wrong direction, it is acceptable as long as it represents a logical approach.

        2. OK Step
        A step is classified as OK if it is:
        - Correct and Verifiable: Contains no errors and can be verified.
        - Unnecessary or Redundant: Adds little value, such as restating prior information or providing basic encouragement (e.g., “Good job!”).
        - Partially Progressing: Makes some progress toward the solution but lacks decisive or significant advancement.

        3. BAD Step
        A step is classified as BAD if it:
        - Is Incorrect: Contains factual errors, misapplies concepts, or derives an incorrect result.
        - Is Hard to Verify: Requires significant effort to confirm due to poor explanation.
        - Is Off-Topic: Includes irrelevant or nonsensical information.
        - Derails: Leads to dead ends, circular reasoning, or unreasonable approaches.

        #### Output Guidelines
        - Analyze the solution up to the last step, considering only the provided information.
        - Provide a step-by-step reasoning analysis of the last step's quality.
        - Conclude your analysis with the classification: STEP_CLASS: GOOD, STEP_CLASS: OK, or STEP_CLASS: BAD.

        #### Output Format Example
        [Step-by-step reasoning about how to classify the last step]
        STEP_CLASS: [GOOD | OK | BAD]
        """
        self.llama_evaluator = aws_agent(role_description=self.system_prompt, model_name= "meta.llama3-8b-instruct-v1:0", temperature=0.7)



    
    def __call_single(self, single_beam: str) -> float | list[float]:
        
        evaluation = self.llama_evaluator(single_beam)

        try:
            # error_idx = int(error_idx)
            pattern = r"STEP_CLASS:\s*(GOOD|OK|BAD)"
            match = re.search(pattern, evaluation)
            step_class =  -1 if match.group(1) == 'BAD' else 1
            
        except:
            print('output not in the correct format. Default labeling it -1')
            step_class = -3

        return float(step_class), evaluation
    

    def __call__(self, steps: list[str]):
        """
        Args:
            steps (list[str]): A list of reasoning beams.

        Returns:
            list[StepScore]: A list of dictionaries where each dictionary
        """
        result = []

        for beam in steps :
            step_score, evaluation_step = self.__call_single(beam)
            if self.debug:
                result.append((StepScore(step=beam, score=step_score), evaluation_step))
            else:
                result.append(StepScore(step=beam, score=step_score))

        return result


if __name__ == "__main__":
    pass



