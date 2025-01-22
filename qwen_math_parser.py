import random
import regex
import re
import sympy
from latex2sympy2 import latex2sympy
from typing import TypeVar, Iterable, List, Union, Any, Dict
from word2number import w2n
from utils import *
from tqdm import tqdm
from parser import strip_string
from grader import math_equal

from datasets import load_dataset

def load_math_500_dataset():
    """
    Load the HuggingFaceH4/MATH-500 dataset and return it as a dictionary
    where each entry can be accessed by its index.

    Returns:
        dict: A dictionary where the keys are indices (integers) and the values
        are the corresponding dataset entries.
    """
    # Load the dataset from Hugging Face
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    # Convert the dataset to a dictionary with index keys
    data_dict = {i: entry for i, entry in enumerate(dataset)}
    
    return data_dict

import json

def read_json_file(file_path):
    """Reads a JSON file and returns its content as a Python dictionary.

    Args:
        file_path (str): The file path to the JSON file.

    Returns:
        dict: The content of the JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_json_file(data, file_path):
    """Saves a Python dictionary to a JSON file.

    Args:
        data (dict): The data to save.
        file_path (str): The file path to save the JSON file.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def extract_answer_map(pred_str, data_name, use_last_number=True):

    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        # minerva_math
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            a = ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        # Handle Chinese few-shot multiple choice problem answer extraction
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    else:  # use the last number
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""

    # # choice answer
    # if (
    #     data_name in ["sat_math", "aqua"]
    #     or "mmlu" in data_name
    # ):
    #     tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    #     if tmp:
    #         pred = tmp[-1]
    #     else:
    #         pred = pred.strip().strip(".")

    # multiple line
    pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred


if __name__ == "__main__":
    # original_file_path = "/home/ec2-user/strawberry/full_precision_results/transformed_llama1b_math500_reward_results/transformed_llama1b_math500_with_prm800k_llama_lora_reward/parsed_answer_meta-llama_Llama-3.2-1B-Instruct_HuggingFaceH4_MATH-500_temp0.8_samples256_max_new_tokens_2048_with_prm800k_llama_lora_rewards.json"
    # updated_qwen_math_parser_file_path = "/mnt/data2/straberry_data2/full_precision_results_updated_qwen_math_parser/transformed_llama1b_math500_reward_results/transformed_llama1b_math500_with_prm800k_llama_lora_reward/qwen_math_parser_updated_parsed_answer_meta-llama_Llama-3.2-1B-Instruct_HuggingFaceH4_MATH-500_temp0.8_samples256_max_new_tokens_2048_with_prm800k_llama_lora_rewards.json"

    original_file_path = "/home/ec2-user/strawberry/full_precision_results/transformed_llama1b_math500_reward_results/transformed_llama1b_math500_with_deepseek_8b_prm_reward/parsed_answer_meta-llama_Llama-3.2-1B-Instruct_HuggingFaceH4_MATH-500_temp0.8_samples256_max_new_tokens_2048_with_deepseek_8b_prm_rewards.json"
    updated_qwen_math_parser_file_path = "/mnt/data2/straberry_data2/full_precision_results_updated_qwen_math_parser/transformed_llama1b_math500_reward_results/transformed_llama1b_math500_with_deepseek_8b_prm_reward/qwen_math_parser_updated_parsed_answer_meta-llama_Llama-3.2-1B-Instruct_HuggingFaceH4_MATH-500_temp0.8_samples256_max_new_tokens_2048_with_deepseek_8b_prm_rewards.json"
    os.makedirs(os.path.dirname(updated_qwen_math_parser_file_path), exist_ok=True)

    data = read_json_file(original_file_path)

    math_500_data = load_math_500_dataset()

    updated_data = []

    for each_q in tqdm(data):
        index = each_q["split_index"].split("_")[1]
        ###
        each_q["qwen_math_parser_GT_answer"] = extract_answer_map(math_500_data[int(index)]["solution"], "math")
        ###
        for each_cot_dict in each_q["chain_of_thoughts"]:
            # each_cot_dict["qwen_math_parser_parsed_answer"] = extract_answer_map(each_cot_dict["original_response"], "math")
            # each_cot_dict["qwen_math_parser_parsed_answer_correctness"] = math_equal(prediction=each_cot_dict["qwen_math_parser_parsed_answer"], 
            #                                                                          reference=each_q["qwen_math_parser_GT_answer"])
            
            each_cot_dict["original_parsed_answer"] = each_cot_dict["parsed_answer"]
            each_cot_dict["original_parsed_answer_correctness"] = each_cot_dict["parsed_answer_correctness"]
            
            each_cot_dict["parsed_answer"] = extract_answer_map(each_cot_dict["original_response"], "math")
            each_cot_dict["parsed_answer_correctness"] = math_equal(prediction=each_cot_dict["parsed_answer"], 
                                                                    reference=each_q["qwen_math_parser_GT_answer"],
                                                                    timeout=True)
            
            each_cot_dict["qwen_math_parser_result_different"] = each_cot_dict["parsed_answer_correctness"] == each_cot_dict["original_parsed_answer_correctness"]
        
        updated_data.append(each_q)
        save_json_file(data=updated_data, file_path=updated_qwen_math_parser_file_path)
        # input()

