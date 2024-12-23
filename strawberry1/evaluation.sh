#!/bin/bash

#working dir is in strawberry

# Stop the script if any command fails
# set -e

# Define an associative array mapping datasets to their corresponding output directories and metric file directories
declare -A CONFIGS=(
  # ["./prmMixedDomain"]="./full_precision_results/prmMixedDomain_reward_results ./full_precision_figures"
  # ["./v2_data/v2_test_data_transformed"]="./full_precision_results/transformed_v2_test ./full_precision_figures"
  # ["./v2_data/v2_eval_data_transformed"]="./full_precision_results/transformed_v2_eval ./full_precision_figures"
  # ["./synthetic_formalized_reasoning_data"]="./full_precision_results/synthetic_formalized_reward_results ./full_precision_figures"
  # ["./transformed_experiment_synthetic_data_held_out"]="./full_precision_results/transformed_experiment_synthetic_data_held_out_reward_results ./full_precision_figures"
  # ["./transformed_experiment_synthetic_data_validation"]="./full_precision_results/transformed_experiment_synthetic_data_validation_reward_results ./full_precision_figures"
  # ["./transformed_experiment_synthetic_v1_data_validation"]="./full_precision_results/transformed_experiment_synthetic_v1_data_validation_reward_results ./full_precision_figures"
  # ["./evaluate_prm800k/transformed_prm800k"]="./full_precision_results/transformed_prm800k_reward_results ./full_precision_figures"
  # ["./transformed_llamag70b_generated_sample100_prm800k"]="./full_precision_results/transformed_llamag70b_generated_sample100_prm800k_reward_results ./full_precision_figures"
  # ["./prm800k_best_of_n_sample100_openai"]="./full_precision_results/prm800k_best_of_n_sample100_openai_reward_results ./full_precision_figures"
  # ["./transformed_llamag70b_generated_sample1_prm800k"]="./full_precision_results/transformed_llamag70b_generated_sample1_prm800k_reward_results ./full_precision_figures"
  # ["./manual_synthetic_cot_12_11"]="./full_precision_results/manual_synthetic_cot_12_11_reward_results ./full_precision_figures"
  # ["./transformed_processbench"]="./full_precision_results/transformed_processbench_reward_results ./full_precision_figures"
  ["./transformed_llama1b_math500"]="./full_precision_results/transformed_llama1b_math500_reward_results ./full_precision_figures"
)
#strawberry/manual_synthetic_cot_12_11/manual_eval_synthetic_cots_12_11.json

# Define models to test
MODELS=("llemma" "mistral" "v1104" "v1105" "v1112" "v1_subset_1117" "v2_checkpoint_1118" "v2_1119" "v2_balanced_1120" "v2_1122_full_finetune" "v3_1123_lora" "v3_1124_checkpoint" "v3_1123_checkpoint1" "v3_1124_balanced" "1125_v3b_mistral_lora" "1125_v3c_mistral_lora" "v4_mistral_lora" "v3d_mistral_lora" "v4_llama_1203" "v3_llama_1205" "prm800k_mistral_full_1203_re" "llama_zeroshot_prm" "llama_zeroshot_prm_aws" "prm800k_llama_joint_checkpoint500" "prm800k_llama_joint_checkpoint1000" "prm800k_llama_joint_checkpoint2000" "reasoneval_7b" "math_psa" "qwen_7b_prm")
# MODELS=("v1_subset_1117")
# MODELS=("v2_checkpoint_1118")
# MODELS=("v2_1119")
# MODELS=("v2_1119" "v2_checkpoint_1118")
# MODELS=("v2_balanced_1120")
# MODELS=("v2_1122_full_finetune")
# MODELS=("v3_1123_lora")
# MODELS=("v3_1124_checkpoint")
# MODELS=("v3_1123_checkpoint1")
# MODELS=("v3_1124_balanced")
# MODELS=("1125_v3b_mistral_lora" "1125_v3c_mistral_lora" "v4_mistral_lora")
# MODELS=("v4_mistral_lora")
# MODELS=("1125_v3c_mistral_lora" "v4_mistral_lora")
# MODELS=("gpt4o")
# MODELS=("gpt4o_real")
# MODELS=("1125_v3c_mistral_lora" "llemma" "mistral" "v4_mistral_lora")
# MODELS=("v3d_mistral_lora")
# MODELS=("v4_llama_1203" "v3_llama_1205")
# MODELS=("v3_llama_1205")
# MODELS=("prm800k_mistral_full_1203_re")
# MODELS=("llama_zeroshot_prm")
# MODELS=("llama_zeroshot_prm_aws" "gpt4o_real")
# MODELS=("llama_zeroshot_prm_aws")
# MODELS=("prm800k_llama_joint_checkpoint500" "prm800k_llama_joint_checkpoint1000")
# MODELS=("prm800k_llama_joint_checkpoint1000")
MODELS=("math_psa" "rlhflow_8b_prm" "deepseek_8b_prm" "reasoneval_7b" "llemma" "mistral" "prm800k_llama_joint_checkpoint4500" "qwen_7b_prm" "prm800k_llama_joint_checkpoint2000" "prm800k_llama_joint_checkpoint1000" "v4_llama_1203" "v3_llama_1205")
# MODELS=("prm800k_llama_joint_checkpoint4500")
# MODELS=("reasoneval_7b")
# MODELS=("qwen_7b_prm")
# MODELS=("math_psa")
# MODELS=("math_psa" "qwen_7b_prm")
# MODELS=("rlhflow_8b_prm" "deepseek_8b_prm")
# MODELS=("qwen_7b_prm" "prm800k_llama_joint_checkpoint4500" "llemma" "mistral")

# Loop over each dataset configuration
for DATASET in "${!CONFIGS[@]}"; do
  # Split the configuration value into output directory and metric file directory
  IFS=' ' read -r OUTPUT_DIR METRIC_FILE_DIR <<< "${CONFIGS[$DATASET]}"
  
  # Loop over each model
  for MODEL in "${MODELS[@]}"; do
    echo "Processing dataset: $DATASET with model: $MODEL"
    
    # Run the Python script with current dataset, model, and output configuration
    python ./search/get_rewards_reasoning_step.py \
      --example_file_path_dir "$DATASET" \
      --test_prm "$MODEL" \
      --output_dir "$OUTPUT_DIR" \
      --metric_file_dir "$METRIC_FILE_DIR" \
      --do_not_calculate_metric \
      # --sample_10 \

    
    echo "Finished processing dataset: $DATASET with model: $MODEL"
  done
done

echo "All tasks completed successfully."
