from beam_search import BeamSearch
from llama_model import LlamaModel
from math_eval import load_jsonl
from random_prm import RandomPRM
from transformers import BitsAndBytesConfig

llamma = LlamaModel(
    "meta-llama/Llama-3.2-1B-Instruct",
    hf_token="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",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
)

bs = BeamSearch(llamma, RandomPRM())

data = load_jsonl("prm800k/math_splits/test.jsonl")

bs(data[0]["problem"])
