# ComputeScaling-Replication
replication of part of the huggingface blog  https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute

Since the details of grading implementation in the blog is not enough to reproduce the results in the blog, i adapted the grading code in the https://github.com/openai/prm800k 

Using "last" as the aggregation method:
![replication results of math-psa](./comparison/parsed_answer_meta-llama_Llama-3.2-1B-Instruct_HuggingFaceH4_MATH-500_temp0.8_samples256_max_new_tokens_2048_with_math_psa_rewards/comparison_last.png)

Using "mean" as the aggregation method:
![replication results of math-psa](./comparison/parsed_answer_meta-llama_Llama-3.2-1B-Instruct_HuggingFaceH4_MATH-500_temp0.8_samples256_max_new_tokens_2048_with_math_psa_rewards/comparison_mean.png)

Using "min" as the aggregation method:
![replication results of math-psa](./comparison/parsed_answer_meta-llama_Llama-3.2-1B-Instruct_HuggingFaceH4_MATH-500_temp0.8_samples256_max_new_tokens_2048_with_math_psa_rewards/comparison_min.png)