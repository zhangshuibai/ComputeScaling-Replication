import json
import os
import random
import time
from typing import Any, Optional

import numpy as np
import pandas as pd  # type: ignore
import torch
import wandb
from beam_search import BeamSearch
from best_of_n import BestOfN
from llama_model import LlamaModel
from llemma_7b_prm import Llemma7bPRM
from no_search import NoSearch
from prm800k import grade_answer
from search_interface import Search
from transformers import BitsAndBytesConfig, set_seed

seed = 42
set_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

os.environ["HF_HOME"] = "/mnt/data/huggingface_cache"


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def math_eval(
    search: Search,
    data_path: str = "prm800k/math_splits/test.jsonl",
    max_n_test: Optional[int] = None,
    name: str = "math_eval",
) -> pd.DataFrame:
    """Evaluates search algorithm on the math dataset. Returns results as `DataFrame`."""
    run = wandb.init(
        dir="results",
        project="strawberry",
        name=name,
    )

    data = load_jsonl(data_path)
    if max_n_test is not None:
        data = data[:max_n_test]

    results = []
    n_correct = 0
    n_answered = 0
    n_total = 0

    print("Starting Eval")
    for i, x in enumerate(data):
        start = time.time()
        output = search(x["problem"])
        search_answer = output.answer
        end = time.time()
        print(f"search_time={end - start:.2f}, {search_answer=}", end=", ")

        if search_answer is None or search_answer == "<answer>":
            grade = False
            search_answer = None
        else:
            grade = grade_answer(search_answer, x["answer"])

        print(f"{grade=}")
        print(f"{output.output=}")

        n_answered += search_answer is not None
        n_correct += grade
        n_total += 1

        del x["problem"]
        del x["solution"]
        x["search_answer"] = search_answer if search_answer is not None else pd.NA
        x["output"] = output.output if output.output is not None else pd.NA
        x["grade"] = grade
        results.append(x)

        answer_rate = n_answered / n_total
        accuracy = n_correct / n_total
        print(f"[{i + 1}/{len(data)}] {accuracy=:.2f}, {answer_rate=:.2f}")
        wandb.log({"answer_rate": answer_rate, "accuracy": accuracy}, step=i)

    run.finish()
    return pd.DataFrame(results)


if __name__ == "__main__":
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    def get_no_search():
        return NoSearch(
            LlamaModel(
                "meta-llama/Llama-3.2-3B-Instruct",
                quantization_config=quantization_config,
                hf_token="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",
            ),
        )

    def get_best_of_n():
        return BestOfN(
            LlamaModel(
                "meta-llama/Llama-3.2-3B-Instruct",
                quantization_config=quantization_config,
                hf_token="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",
            ),
            Llemma7bPRM(quantization_config=quantization_config),
        )

    def get_beam_search():
        return BeamSearch(
            LlamaModel(
                "meta-llama/Llama-3.2-3B-Instruct",
                quantization_config=quantization_config,
                hf_token="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",
            ),
            Llemma7bPRM(quantization_config=quantization_config),
        )

    for name, get_search in [
        ("no_search", get_no_search),
        ("best_of_n", get_best_of_n),
        ("beam_search", get_beam_search),
    ]:
        search = get_search()
        results = math_eval(search, name=f"math_eval_{name}")
        results.to_csv(f"results/{name}.csv")
        print("Accuracy:", results["grade"].mean())
        print(f"{results['search_answer'].isna().sum()} questions unanswered")
        del search
