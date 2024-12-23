from typing import Any, Optional

import torch
from model_interface import BaseModel
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

META_PROMPT = """
In additon to any other constraints, please solve the problem using a specific format. Each reasoning step should be separated by a blank line, and the final answer should be given in the format \\boxed{<answer>}.

\n\n# Solution

"""


class LlamaModel(BaseModel):
    def __init__(
        self,
        llama_model_name: str,
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            llama_model_name,
            quantization_config=quantization_config,
            token=hf_token,
        )

        if not quantization_config:
            self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.end_step_str = [" \n\n", "\n\n", ".\n\n"]
        self.end_step_id = [
            e[-1]
            for e in self.tokenizer(self.end_step_str)["input_ids"]  # type: ignore
        ]
        self.end_solution_str = self.tokenizer.eos_token
        self.end_solution_id = self.tokenizer.eos_token_id
        self.meta_prompt = META_PROMPT

    def generate(
        self,
        prompt: str,
        config: dict[str, Any] = {},
        add_meta_prompt: bool = False,
    ) -> list[str]:
        """Returns the decoded outputs of prompts."""
        if add_meta_prompt:
            prompt = self.add_meta_prompt(prompt)

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        outputs = self.model.generate(
            **inputs.to(self.device), **config, use_cache=True
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def add_meta_prompt(self, prompt: str) -> str:
        """Add the meta prompt to the beginning of the given prompt."""
        return prompt + self.meta_prompt

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the answer from the provided text using /boxed{<answer to extract>} pattern."""
        i = text.rfind("\\boxed{") + len("\\boxed{")
        if i == -1:
            return None

        j = i
        brace_count = 1
        while brace_count > 0 and j < len(text) - 1:
            j += 1
            if text[j] == "{":
                brace_count += 1
            elif text[j] == "}":
                brace_count -= 1

        if brace_count > 0:
            return None

        return text[i:j] if text[i:j] != "<answer>" else None

    def get_delimeter_ids(self) -> list[int]:
        """Returns a list of end-of-step and end-of-solution ids."""
        return self.end_step_id + [self.end_solution_id]

    def is_complete(self, text: str) -> bool:
        if self.end_solution_str in text:
            return True

        return self.extract_answer(text) is not None
