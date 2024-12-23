
import sys,os,re
import json
import random
from collections import defaultdict  
from typing import List           
from latex2sympy2 import latex2sympy
from sympy import latex, simplify
from datasets import load_dataset
from prm800k_grader import grade_answer

import re

def split_and_clean_steps(markdown_text):
    """
    Splits a markdown text into individual steps, removing the step headers.

    Args:
        markdown_text (str): The markdown string containing steps.

    Returns:
        List[str]: A list of step descriptions without headers.
    """
    # Use regex to split the text at each step header.
    # The pattern looks for lines starting with ## Step followed by a number and a colon.
    if markdown_text is None:
        return []

    steps = re.split(r'## Step \d+:\s*', markdown_text)
    
    # The first split part might be empty if the string starts with a step, so remove it.
    if steps and not steps[0].strip():
        steps = steps[1:]
    
    # Optionally, you can strip leading/trailing whitespace from each step.
    steps = [step.strip() for step in steps]
    
    return steps

def read_json_file(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_json_to_file(data, filename):

    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

from sympy import simplify, sympify
def are_equivalent(expr1: str, expr2: str) -> bool:
    
    try:
        # Convert the expressions to SymPy objects
        sympy_expr1 = sympify(expr1)
        sympy_expr2 = sympify(expr2)
        
        # Check if their difference simplifies to zero
        return simplify(sympy_expr1 - sympy_expr2) == 0
    except Exception as e:
        # Handle any errors that occur during parsing or simplification
        print(f"Error comparing expressions: {e}")
        return None
    
def get_canonical_form(expression: str) -> str:
    # try:
    #     parsed_expr = latex2sympy(expression)
    # except Exception as e:
    #     # print(e)
    #     # print(expression)
    #     # input()
    #     # return None
    #     parsed_expr = expression
    parsed_expr = latex2sympy(expression)

    simplified_expr = simplify(parsed_expr)
    return latex(simplified_expr)

def find_majority_answer(answers: List[str]) -> str:
    canonical_groups = defaultdict(int) 
    canonical_to_original = {}

    for answer in answers:
        if not answer:
            continue

        canonical_form = get_canonical_form(answer)
        # if not canonical_form:
        #     continue

        # Increment count for the canonical form
        canonical_groups[canonical_form] += 1

        # Track the original answer for this canonical form
        if canonical_form not in canonical_to_original:
            canonical_to_original[canonical_form] = answer

    # Find the canonical form with the largest count
    max_count = max(canonical_groups.values())
    for canonical_form, count in canonical_groups.items():
        if count == max_count:
            # Return the first occurring group in case of a tie
            return canonical_to_original[canonical_form]

def parse_final_answer(s):
    """
    Parses and returns the content inside '\\boxed{}', including nested braces.

    Args:
        s (str): The input string containing '\\boxed{...}'.

    Returns:
        str: The content inside '\\boxed{}', or None if no match is found.
    """
    # Regular expression to match '\\boxed{...}' with potential nested braces
    match = re.search(r'\\boxed\{(.*)\}', s)
    if match:
        content = match.group(1)
        # Handle nested braces by only extracting the correct scope
        brace_count = 0
        extracted_content = []
        for char in content:
            if char == '{':
                brace_count += 1
            elif char == '}':
                if brace_count == 0:
                    break  # Stop when we reach the end of the top-level brace
                brace_count -= 1
            extracted_content.append(char)
        return ''.join(extracted_content)
    # return s
    else:

        # print(s)
        # input()
        return None
    


if __name__ == "__main__":
    parse_data_path = "/home/ec2-user/huggingface_reproduce/generated_outputs/meta-llama_Llama-3.2-1B-Instruct_HuggingFaceH4_MATH-500_temp0.8_samples256_max_new_tokens_2048.json"

    majority_results_path = f"/home/ec2-user/huggingface_reproduce/parsed_answer_{os.path.basename(parse_data_path)}"

    data_read = read_json_file(parse_data_path)
    updated_data = []
    GroundTruth_answers = load_dataset("HuggingFaceH4/MATH-500")["test"]

    for each_q in data_read:
        updated_each_q = each_q
        q_index = int(each_q["split_index"].split("_")[1])
        GT_answer = GroundTruth_answers["answer"][q_index]
        # print(GT_answer)
        # input("GT")

        parsed_answer_list = [parse_final_answer(response[2]["content"]) for response in each_q["sample_responses"]]
        cot_steps_list = [split_and_clean_steps(response[2]["content"]) for response in each_q["sample_responses"]]
        original_response_list = [response[2]["content"] for response in each_q["sample_responses"]]
        assert len(cot_steps_list) == len(parsed_answer_list)

        # parsed_answer_list = [each for each in parsed_answer_list if each]
        ###
        # parsed_answer_list = ["\\\\boxed{}".format("{"+each+"}") for each in parsed_answer_list if each]
        ###
        # majority_answer = find_majority_answer(parsed_answer_list)
        # print(majority_answer)

        parsed_answer_correctness_list = [grade_answer(given_answer=answer, ground_truth=GT_answer) for answer in parsed_answer_list]

        updated_each_q.pop("sample_responses")
        updated_each_q['question'] = updated_each_q.pop('prompt')
        updated_each_q["domain"] = os.path.basename(parse_data_path)
        updated_each_q["source"] = os.path.basename(parse_data_path)
        # updated_each_q["majority_answer"] = majority_answer
        # updated_each_q["majority_answer_correctness"] = majority_answer_correctness
        updated_each_q["GT_answer"] = GT_answer
        updated_each_q["chain_of_thoughts"] = [
            {
                "original_response":original_response,
                "steps": cot_steps, 
                "parsed_answer": parsed_answer, 
                "parsed_answer_correctness":parsed_answer_correctness
            } 
        for original_response, cot_steps, parsed_answer, parsed_answer_correctness in zip(original_response_list, cot_steps_list, parsed_answer_list, parsed_answer_correctness_list)
        ]

        updated_data.append(updated_each_q)
    
        save_json_to_file(data=updated_data, filename=majority_results_path)




