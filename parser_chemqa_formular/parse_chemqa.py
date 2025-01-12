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


import json


def read_jsonl(file_path):
    """
    Reads a JSONL (JSON Lines) file and returns the data as a list of dictionaries.

    Parameters:
        file_path (str): The path to the .jsonl file.

    Returns:
        list: A list of dictionaries containing the data from the JSON lines.
    """
    data = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    return data

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

def extract_answer(pred_str,use_last_number=True):

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

    # multiple line
    pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    return pred


if __name__ == "__main__":
    # file_path = "/home/ec2-user/chemqa_formular/chemqa-fewshot-ouput.out"
    # ref_json_file = ""
    # data = read_jsonl(file_path)
    # for each_dict in data:
    #     each_output = each_dict["modelOutput"]["generation"]
    #     answer = extract_answer(each_output)
    #     print(answer)
    #     input()
    file_path = "/home/ec2-user/chemqa_formular/chemqa_train.json"
    data = read_json_file(file_path=file_path)

    for each_dict in data:
        try:
            last_cot = each_dict["chain_of_thoughts"][0]["steps"][-1]
        except:
            continue
        extracted_answer = extract_answer(last_cot)
        parsed_answer_correctness = math_equal(prediction=extracted_answer.split(" ")[0], reference=each_dict["answer"].split(" ")[0], abs_tol=1, )
        print("extracted: "+ extracted_answer)
        print("GT: " + each_dict["answer"])
        print("correctness: " + str(parsed_answer_correctness))

        input("\n")

