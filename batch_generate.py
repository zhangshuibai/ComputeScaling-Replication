import os
import json
import argparse
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime
import json

def save_json(data, file_path):


    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def process_dataset(dataset_name):
    """
    Extracts the 'problem' field from the specified dataset.

    Args:
        dataset_name (str): The Hugging Face dataset repository name.

    Returns:
        list: A list of math problems as prompts.
    """
    # Load the dataset's train split
    dataset = load_dataset(dataset_name, split='test')
    # Extract the 'problem' field
    problems = [row["problem"] for row in dataset]
    return problems


def generate_samples(pipe, system_prompt, prompt, temperature, num_samples, maximum_batch_size, max_new_tokens):
    """
    Generates samples for a single prompt using multiple pipeline calls if necessary.

    Args:
        pipe (transformers.pipeline): The text generation pipeline.
        full_prompt (str): The combined system prompt and user prompt.
        temperature (float): The sampling temperature.
        num_samples (int): Total number of samples to generate.
        maximum_batch_size (int): Maximum number of sequences to generate per pipeline call.

    Returns:
        list: A list of generated texts.
    """
    generated_texts = []
    remaining_samples = num_samples


    while remaining_samples > 0:
        current_batch_size = min(remaining_samples, maximum_batch_size)
        messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt},
]
        outputs = pipe(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=current_batch_size,
        )
        # Extract generated texts
        batch_texts = [output["generated_text"] for output in outputs]
        generated_texts.extend(batch_texts)
        remaining_samples -= current_batch_size

    return generated_texts

def batch_generate(model_name, dataset_name, system_prompt, temperature, num_samples, maximum_batch_size,max_new_tokens):
    """
    Generates text using the specified model and saves the output to a JSON file incrementally.

    Args:
        model_name (str): The Hugging Face model repository name.
        dataset_name (str): The Hugging Face dataset repository name.
        system_prompt (str): The system prompt for the model.
        temperature (float): The temperature for sampling.
        num_samples (int): Number of samples to generate per prompt.
        maximum_batch_size (int): Maximum number of samples per pipeline call to prevent memory issues.

    Returns:
        str: The path to the generated JSON file.
    """
    # Load the text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype="auto",
        device_map="auto",  # Automatically maps to GPU if available
    )

    # Process the dataset to extract prompts
    prompts = process_dataset(dataset_name)

    # Prepare output directory and filename
    sanitized_model_name = model_name.replace('/', '_')
    sanitized_dataset_name = dataset_name.replace('/', '_')

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    output_dir = os.path.join(script_dir, "generated_outputs")
    os.makedirs(output_dir, exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_filename = f"{sanitized_model_name}_{sanitized_dataset_name}_temp{temperature}_samples{num_samples}_{timestamp}.json"
    output_filename = f"{sanitized_model_name}_{sanitized_dataset_name}_temp{temperature}_samples{num_samples}_max_new_tokens_{max_new_tokens}.json"
    output_path = os.path.join(output_dir, output_filename)


    total_data = []
    try:
        # Iterate over prompts sequentially with progress bar
        for idx, prompt in enumerate(tqdm(prompts, desc="Generating prompts")):
            split_index = f"test_{idx}"
            print(f"Processing {split_index}...")

            # Combine system_prompt with the actual prompt
            # full_prompt = f"{system_prompt}\n\n{prompt}"

            # Generate multiple samples for the prompt using the helper function
            generated_texts = generate_samples(
                pipe=pipe,
                system_prompt=system_prompt,
                prompt=prompt,
                temperature=temperature,
                num_samples=num_samples,
                maximum_batch_size=maximum_batch_size,
                max_new_tokens=max_new_tokens
            )

            result = {
                "split_index": split_index,
                "prompt": prompt,
                "num_samples": num_samples,
                "temperature": temperature,
                "sample_responses": generated_texts
            }


            total_data.append(result)
            save_json(total_data, output_path)

    except Exception as e:
        print(f"An error occurred: {e}")


    return output_path

def main():
    # Set up argparse for command-line arguments
    parser = argparse.ArgumentParser(description="Batch generate text with a Hugging Face model.")
    parser.add_argument("--model_name", type=str, default= "meta-llama/Llama-3.2-1B-Instruct", help="Hugging Face model repository name.")
    parser.add_argument("--dataset_name", type=str, default= "HuggingFaceH4/MATH-500", help="Hugging Face dataset repository name.")
    parser.add_argument(
        "--system_prompt", 
        type=str, 
        default=
'''
Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.
''',
        help="System prompt for the model."
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--num_samples", type=int, default=256, help="Number of samples to generate per prompt.")
    parser.add_argument("--maximum_batch_size", type=int, default=32, help="Maximum number of samples per pipeline call to prevent memory issues.")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    # Parse arguments
    args = parser.parse_args()

    # Call the batch_generate function with parsed arguments
    batch_generate(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        num_samples=args.num_samples,
        maximum_batch_size=args.maximum_batch_size,
        max_new_tokens=args.max_new_tokens
    )

if __name__ == "__main__":
    main()