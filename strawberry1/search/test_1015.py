import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'meta-llama/Llama-3.2-1B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def generate_and_print_info(input_text, max_new_tokens=20, use_cache=True):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        return_dict_in_generate=True,
        output_scores=True
    )
    
    generated_sequence = output.sequences[0]
    past_key_values = output.past_key_values
    
    print(f"Input sequence shape: {input_ids.shape}")
    print(f"Generated sequence shape: {generated_sequence.shape}")
    print(f"Number of layers in past_key_values: {len(past_key_values)}")
    print(f"Shape of past_key_values for first layer: {past_key_values[0][0].shape}, {past_key_values[0][1].shape}")
    
    generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
    
    if use_cache:
        second_output = model.generate(
            generated_sequence.unsqueeze(0),
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            return_dict_in_generate=True,
            output_scores=True,
            past_key_values=past_key_values
        )
        
        second_generated_sequence = second_output.sequences[0]
        second_past_key_values = second_output.past_key_values
        
        print(f"\nSecond generation input shape: {generated_sequence.unsqueeze(0).shape}")
        print(f"Second generated sequence shape: {second_generated_sequence.shape}")
        print(f"Shape of second past_key_values for first layer: {second_past_key_values[0][0].shape}, {second_past_key_values[0][1].shape}")
        
        second_generated_text = tokenizer.decode(second_generated_sequence, skip_special_tokens=True)
        print(f"Second generated text: {second_generated_text}")

input_text = "Once upon a time, there was a"
generate_and_print_info(input_text, max_new_tokens=20, use_cache=True)

