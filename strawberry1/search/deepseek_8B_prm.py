import math
import statistics
from typing import Optional
from tqdm import tqdm
import torch
from prm_interface import PRM, StepScore
from torch.types import Device
from transformers import (  # type: ignore  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import json

def read_json_file(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


class Deepseek_RLHflow8bPRM(PRM):
    def __init__(
        self,
        aggregation: str = "full",#the way how prm step scores will be aggregated in a solution
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[Device] = None,
    ) -> None:
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )


        self.tokenizer = AutoTokenizer.from_pretrained('RLHFlow/Llama3.1-8B-PRM-Deepseek-Data')
        self.model = AutoModelForCausalLM.from_pretrained('RLHFlow/Llama3.1-8B-PRM-Deepseek-Data', torch_dtype=torch.bfloat16).eval().to(self.device)

        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        
        
        self.plus_tag_id = self.tokenizer.encode('+')[-1]
        self.minus_tag_id = self.tokenizer.encode('-')[-1]

        self.aggregation = aggregation

    def __call_single(self, single_beam: str) -> float | list[float]:


        single_beam_split = single_beam.split('\n\n')
        prompt = single_beam_split[0]
        ans_list = single_beam_split[1:]




        step_score = []
        conversation = []

        
        ans_list = [j.strip() for j in ans_list]
        for k in range(len(ans_list)):
            if k == 0:
                text = prompt + " " + ans_list[0]
            else:
                text = ans_list[k]
            conversation.append({"content":text,"role":"user"})
            conversation.append({"content":"+","role":"assistant"})

            input_ids = self.tokenizer.apply_chat_template(conversation,return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids).logits[:,-3,[self.plus_tag_id, self.minus_tag_id]] #simple version, the +/- is predicted by the '-3' position
                scores = logits.softmax(dim=-1)[:,0] # 0 means the prob of + (1 mean -)
                #print(scores)
                step_score.append(scores[0].detach().to('cpu', dtype=torch.float32).item())

                
        step_probs  = step_score

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
            steps (list[str]): A list of reasoning solutions.

        Returns:
            list[StepScore]: A list of dictionaries where each dictionary
        """

        result = []

        for beam in steps:#each beam is a cot solution, each step_score is a list of prm step scores for this solution(if aggregation methods is full)
            step_score = self.__call_single(beam)
            result.append(StepScore(step=beam, score=step_score))

        return result
    

if __name__ == "__main__":

    prm = Deepseek_RLHflow8bPRM(
                aggregation="full", 
            )
    
    json_file_path = "/home/ec2-user/strawberry/prmMixedDomain/prmMixedDomain.json"###mixedomain format json file
    data = read_json_file(json_file_path)

   #organizing input cot data format based on different prm data formats
    for each_data in tqdm(data):
        for cot in each_data["chain_of_thoughts"]:
            steps = cot["steps"]
            steps = [step.replace('\n', "") for step in steps]
            question = each_data["question"].replace('\n', "")
            steps_all = f"{question}\n\n" + "\n\n".join(steps)
            rewards = prm([steps_all])
            cot["prm_reward"] = rewards[0].score
            print(cot["prm_reward"])
            print(len(cot["prm_reward"]))
            print(len(steps))
            input()

    # #organizing input cot data format based on different prm data formats
    # for each_data in tqdm(data):
    #     for cot in each_data["chain_of_thoughts"]:
    #         steps = [step.replace(prm.step_tag, "") for step in steps]
    #         question = each_data["question"].replace(prm.step_tag, "")
    #         steps_all = f"{question}\n\n" + "\n\n".join(steps)
    #         rewards = prm([steps_all])
    #         cot["prm_reward"] = rewards[0].score
    

