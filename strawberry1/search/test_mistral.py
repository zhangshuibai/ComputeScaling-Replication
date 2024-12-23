# from transformers import AutoTokenizer
# from transformers import AutoModelForCausalLM
# import torch
from transformers import BitsAndBytesConfig, set_seed
# good_token = '+'
# bad_token = '-'
# step_tag = 'ки'

# tokenizer = AutoTokenizer.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm')
# candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:] # [648, 387]
# print("candidate_tokens")
# print(candidate_tokens)

# step_tag_id = tokenizer.encode(f"{step_tag}")[-1] # 12902
# model = AutoModelForCausalLM.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm').eval()

question = """Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
output1 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки""" # 18 is right
output2 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки""" # 17 is wrong
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
from mistral_7b_prm import Mistral7bPRM
prm = Mistral7bPRM(aggregation="full",quantization_config=quantization_config)

for output in [output1, output2]:
    input_for_prm = f"{question} {output}"
    output = prm([input_for_prm])
    print(output)
    # input_id = torch.tensor([tokenizer.encode(input_for_prm)])
    # print(input_id.shape)

    # with torch.no_grad():
    #     original_logits = model(input_id).logits
    #     print("original_logits:")
    #     print(original_logits.shape)
    #     logits = original_logits[:,:,candidate_tokens]
    #     print("logits:")
    #     print(logits.shape)
    #     scores = logits.softmax(dim=-1)[:,:,0] 
    #     print("scores")
    #     print(scores.shape)
    #     step_scores = scores[input_id == step_tag_id]
    #     print("step_scores")
    #     print(step_scores.shape)
    #     print(step_scores)

        
# tensor([0.9955, 0.9958, 0.9983, 0.9957])
# tensor([0.9955, 0.9958, 0.9983, 0.0240])
