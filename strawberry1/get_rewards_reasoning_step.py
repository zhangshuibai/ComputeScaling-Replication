from cgi import test
import json, os
from re import T
from shlex import join
from xml import dom
import random
import argparse
import sys
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np

def filter_first_error(labels, rewards):
    updated_labels = []
    updated_rewards = []
    for label,reward in zip(labels,rewards):
        updated_labels.append(label)
        updated_rewards.append(reward)
        if label == -1:
            break
    return labels, rewards

def update_ssl(ssl, steps, labels, rewards):
    assert len(steps) == len(labels) and len(labels) == len(rewards)

    cumulative_step = ""
    updated_labels = []
    updated_rewards = []

    for step, label, reward in zip(steps, labels, rewards):
        cumulative_step += step
        if cumulative_step in ssl:
            # print("==========")
            # print(cumulative_step)
            continue
        else:
            # print("++++++++")
            ssl.append(cumulative_step)
            updated_labels.append(label)
            updated_rewards.append(reward)
    
    return ssl, labels, rewards
        



def hash_nested_dict(d):
    if isinstance(d, dict):
        return hash(tuple((k, hash_nested_dict(v)) for k, v in sorted(d.items())))
    elif isinstance(d, list):
        return hash(tuple(hash_nested_dict(i) for i in d))
    elif isinstance(d, set):
        return hash(frozenset(hash_nested_dict(i) for i in d))
    else:
        return hash(d)


def contains_nan(lst):
    return any(np.isnan(x) for x in lst)
def save_dict_to_file(data, file_path):

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"successfully saved {file_path}")
    except Exception as e:
        print(f"Error: {e}")

def clean_lists(label_list, reward_list):
    assert len(label_list) == len(reward_list)

    cleaned_label_list = []
    cleaned_reward_list = []
    
    for label, reward in zip(label_list, reward_list):
        if label is not None and reward is not None:
            cleaned_label_list.append(label)
            cleaned_reward_list.append(reward)
    
    return cleaned_label_list, cleaned_reward_list

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, average_precision_score

def plot_accuracy_vs_threshold(labels: list[int], rewards: list[float], domain: str, figure_file_dir: str) -> None:
    """
    Based on different thresholds, binarize the rewards and calculate accuracy, precision, recall, F1 score, and specificity.
    Finally, plot accuracy, precision, recall, F1 score, and specificity vs. threshold and save the plot locally.

    Parameters:
    labels (List[int]): A list containing -1, 0, or 1, representing negative, positive, and neutral (treated as positive) labels.
    rewards (List[float]): A list of rewards, with values between 0 and 1.
    domain (str): The domain name or title to be used in the plot.

    Returns:
    None
    """
    # Convert input to numpy arrays
    labels_array = np.array(labels)
    rewards_array = np.array(rewards)

    # Define thresholds, typically from 0 to 1
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    specificities = []

    # Iterate over each threshold
    for threshold in thresholds:
        # rewards >= threshold are treated as positive (+1), otherwise negative (-1)
        predicted_labels = np.where(rewards_array >= threshold, 1, -1)
        
        # Treat 0 in labels as positive (+1)
        modified_labels = np.where(labels_array == 0, 1, labels_array)

        # Calculate accuracy
        accuracy = (predicted_labels == modified_labels).mean()
        accuracies.append(accuracy)

        # Calculate precision
        precision = precision_score(modified_labels, predicted_labels, pos_label=1, zero_division=0)
        precisions.append(precision)

        # Calculate recall
        recall = recall_score(modified_labels, predicted_labels, pos_label=1, zero_division=0)
        recalls.append(recall)

        # Calculate F1 score
        f1 = f1_score(modified_labels, predicted_labels, pos_label=1, zero_division=0)
        f1_scores.append(f1)

        # Calculate confusion matrix to extract TN and FP for Specificity
        tn, fp, _, _ = confusion_matrix(modified_labels, predicted_labels, labels=[-1, 1]).ravel()
        
        # Calculate specificity: TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)



    # Plot the graph
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, accuracies, label="Accuracy", color="b")
    plt.plot(thresholds, precisions, label="Precision", color="g")
    plt.plot(thresholds, recalls, label="Recall", color="orange")
    plt.plot(thresholds, f1_scores, label="F1 Score", color="r")
    plt.plot(thresholds, specificities, label="Specificity", color="purple")
    plt.title(f"Accuracy, Precision, Recall, F1 Score & Specificity vs. Threshold for {domain}")
    plt.xlabel("Threshold")
    plt.ylabel("Metrics")
    plt.grid(True)
    plt.legend()

    # Save the image locally
    save_path = os.path.join(figure_file_dir, domain + ".png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    # Additionally, print PR-AUC score
    pr_auc = average_precision_score(modified_labels, rewards_array)
    # print(f"PR-AUC for {domain}: {pr_auc:.4f}")


import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import os
import warnings

###1126
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import warnings

def compute_metrics_at_threshold(labels, rewards, threshold):
    """
    Compute classification metrics at a specific threshold.

    Parameters:
    labels (np.ndarray): Binary ground truth labels (0 or 1).
    rewards (np.ndarray): Predicted probabilities (between 0 and 1).
    threshold (float): Threshold for converting probabilities into binary predictions.

    Returns:
    dict: A dictionary containing accuracy, precision, recall, F1, and specificity.
    """
    predicted_labels = (rewards >= threshold).astype(int)

    accuracy = np.mean(predicted_labels == labels)
    precision = precision_score(labels, predicted_labels, average="binary", zero_division=0)
    recall = recall_score(labels, predicted_labels, average="binary", zero_division=0)
    f1 = f1_score(labels, predicted_labels, average="binary", zero_division=0)

    # Compute confusion matrix to get TN and FP for specificity calculation
    try:
        tn, fp, fn, tp = confusion_matrix(labels, predicted_labels).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except Exception as e:
        print(labels)
        print(predicted_labels)
        # print(confusion_matrix(labels, predicted_labels).ravel())
        specificity = 0
        # raise e

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
    }


def find_threshold_for_specificity(labels, rewards, target_specificity):
    """
    Find the threshold that achieves the closest specificity to the target value.

    Parameters:
    labels (np.ndarray): Binary ground truth labels (0 or 1).
    rewards (np.ndarray): Predicted probabilities (between 0 and 1).
    target_specificity (float): The desired specificity value.

    Returns:
    float: The threshold that achieves the closest specificity to the target value.
    """
    fpr, tpr, thresholds = roc_curve(labels, rewards, pos_label=1)
    specificity = 1 - fpr  # Specificity is 1 - FPR

    # Find the threshold where specificity is closest to the target
    # print(specificity)
    # input()
    idx = (np.abs(specificity - target_specificity)).argmin()
    return thresholds[idx]


def compute_auc_metrics(labels: list[int], rewards: list[float], ci: float = 0.95) -> dict:
    """
    Compute PR-AUC, ROC-AUC, their confidence intervals using bootstrapping, and accuracy at a threshold of 0.5
    and at the threshold where specificity equals 0.8.

    Parameters:
    labels (List[int]): A list containing -1, 0, or 1, representing negative (-1), neutral (0), and positive (1) labels.
    rewards (List[float]): A list of rewards, with values between 0 and 1.
    ci (float): Confidence interval level (default is 0.95, i.e., 95%).

    Returns:
    dict: A dictionary containing AUROC metrics, accuracy, precision, recall, F1 score, and specificity.
    """
    # Convert input to numpy arrays
    labels_array = np.array(labels)
    rewards_array = np.array(rewards)

    # Treat -1 as negative (0), and 0 and +1 as positive (1)
    modified_labels = np.where(labels_array == -1, 0, 1)

    # Compute Precision-Recall curve and PR-AUC
    precision, recall, _ = precision_recall_curve(modified_labels, rewards_array, pos_label=1)
    pr_auc = auc(recall, precision)

    # Compute ROC curve and ROC-AUC
    fpr, tpr, _ = roc_curve(modified_labels, rewards_array, pos_label=1)
    roc_auc_full = auc(fpr, tpr)

    # Bootstrapping to estimate confidence interval for ROC-AUC
    n_bootstraps = 1000
    rng_seed = 42  # For reproducibility
    bootstrapped_aucs = []
    rng = np.random.RandomState(rng_seed)
    
    for _ in range(n_bootstraps):
        # Resample with replacement
        indices = rng.choice(np.arange(len(rewards_array)), size=len(rewards_array), replace=True)
        if len(np.unique(modified_labels[indices])) < 2:
            # Skip this iteration if resampled data does not contain both classes
            continue
        fpr_boot, tpr_boot, _ = roc_curve(modified_labels[indices], rewards_array[indices], pos_label=1)
        roc_auc_boot = auc(fpr_boot, tpr_boot)
        bootstrapped_aucs.append(roc_auc_boot)

    # Check if bootstrapped_aucs has enough samples
    if len(bootstrapped_aucs) < n_bootstraps * 0.9:
        warnings.warn("Number of valid bootstrapped samples is low. Confidence intervals may be unreliable.")

    # Sort bootstrapped AUROCs and compute the mean
    sorted_aucs = np.sort(bootstrapped_aucs)
    auc_mean = np.mean(bootstrapped_aucs)

    # Compute the confidence interval bounds using percentiles
    lower_percentile = (1 - ci) / 2 * 100  # e.g., 2.5 for 95% CI
    upper_percentile = 100 - lower_percentile  # e.g., 97.5 for 95% CI

    if len(sorted_aucs) == 0:
        confidence_lower = 0.0
        confidence_upper = 0.0
    else:
        confidence_lower = np.percentile(sorted_aucs, lower_percentile)
        confidence_upper = np.percentile(sorted_aucs, upper_percentile)

    # Compute metrics at threshold = 0.5
    metrics_at_0_5 = compute_metrics_at_threshold(modified_labels, rewards_array, threshold=0.5)

    # Find the threshold for specificity = 0.8
    target_specificity = 0.6
    specificity_threshold = find_threshold_for_specificity(modified_labels, rewards_array, target_specificity)

    # Compute metrics at the specificity threshold
    metrics_at_specificity = compute_metrics_at_threshold(modified_labels, rewards_array, threshold=specificity_threshold)

    # Return metrics as a dictionary
    return {
        'auroc_full': roc_auc_full,
        'auroc_ci_lower': confidence_lower,
        'auroc_ci_upper': confidence_upper,
        'auroc_ci_mean': auc_mean,
        'pr_auc': pr_auc,

        'accuracy_at_threshold_0.5': metrics_at_0_5["accuracy"],
        'precision_at_threshold_0.5': metrics_at_0_5["precision"],
        'recall_at_threshold_0.5': metrics_at_0_5["recall"],
        'f1_at_threshold_0.5': metrics_at_0_5["f1"],
        'specificity_at_threshold_0.5': metrics_at_0_5["specificity"],

        'specificity_threshold': specificity_threshold,

        'accuracy_at_specificity_threshold': metrics_at_specificity["accuracy"],
        'precision_at_specificity_threshold': metrics_at_specificity["precision"],
        'recall_at_specificity_threshold': metrics_at_specificity["recall"],
        'f1_at_specificity_threshold': metrics_at_specificity["f1"],
        'specificity_at_specificity_threshold': metrics_at_specificity["specificity"],
    }
###

###1115
# predicted_labels = (rewards_array <= 0.5).astype(int)
# modified_labels = [1 if x == 0 else 0 for x in modified_labels]
###



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import os
import warnings


def filter_FALSE_AUGMENTED_data(labels, rewards):
    # Assert that labels and rewards have the same length
    assert len(labels) == len(rewards), "labels and rewards must have the same length"
    
    filtered_labels = []
    filtered_rewards = []

    # Iterate over the labels and rewards
    for i in range(len(labels)):
        if labels[i] == -1:
            # Keep the label and reward if the label is -1
            filtered_labels.append(labels[i])
            filtered_rewards.append(rewards[i])

    # Check if the length of filtered_labels and filtered_rewards is exactly 1
    if len(filtered_labels) != 1 or len(filtered_rewards) != 1:
        raise ValueError("The length of filtered_labels and filtered_rewards must be exactly 1")

    return filtered_labels, filtered_rewards
    


def main():
    parser = argparse.ArgumentParser(description="Process rewards for PRM models")

    # Add argument for the directory paths and test_prm
    parser.add_argument('--example_file_path_dir', type=str, required=True, help="Directory for example files")
    parser.add_argument('--test_prm', type=str, choices=['llemma', 'mistral', 'v1104', "v1105", "v1112", "gpt4o", "gpt4o_real", "v1_subset_1117", "v2_checkpoint_1118", "v2_1119", "v2_balanced_1120", "v2_1122_full_finetune", "v3_1123_lora", "v3_1124_checkpoint", "v3_1123_checkpoint1", "v3_1124_balanced", "1125_v3b_mistral_lora", "1125_v3c_mistral_lora", "v4_mistral_lora", "v3d_mistral_lora", "v4_llama_1203", "v3_llama_1205", "prm800k_mistral_full_1203_re", "llama_zeroshot_prm", "llama_zeroshot_prm_aws", "prm800k_llama_joint_checkpoint500", "prm800k_llama_joint_checkpoint1000", "prm800k_llama_joint_checkpoint2000", "prm800k_llama_joint_checkpoint4500", "reasoneval_7b", "math_psa", "qwen_7b_prm", "rlhflow_8b_prm", "deepseek_8b_prm", "prm800k_qwen_alt_lora"], required=True, help="PRM model to use")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the reward files")
    parser.add_argument("--metric_file_dir", type=str, required=True)
    parser.add_argument("--four_bit", action="store_true")
    parser.add_argument("--gpt4o_prm_debug", action="store_true")
    parser.add_argument("--sample_10", action="store_true")
    parser.add_argument("--do_not_calculate_metric", action="store_true")
    
    # Parse the arguments
    args = parser.parse_args()
    # print(args.metric_file_dir)
    # input()
    os.makedirs(args.metric_file_dir, exist_ok=True)
    # Set the directory paths from the parsed arguments
    example_file_path_dir = args.example_file_path_dir
    example_file_path_list = [os.path.join(example_file_path_dir, file) for file in os.listdir(example_file_path_dir)]
    
    test_prm = args.test_prm
    output_dir = args.output_dir
    
    # Determine reward file folder based on the `test_prm` argument
    test_dataset_name = example_file_path_dir.split("/")[-1]
    # if test_prm == "llemma":
    #     reward_file_folder_dir = os.path.join(output_dir, f"{test_dataset_name}_with_llemma_reward")
    # elif test_prm == "mistral":
    #     reward_file_folder_dir = os.path.join(output_dir, f"{test_dataset_name}_with_mistral_reward")
    # else:
    #     raise NotImplementedError
    reward_file_folder_dir = os.path.join(output_dir, f"{test_dataset_name}_with_{test_prm}_reward")
    
    os.makedirs(reward_file_folder_dir, exist_ok=True)
    
    # Configure quantization
    if args.four_bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    else:
        quantization_config = None

    all_file_exists = True
    for example_file_path in example_file_path_list:
        file_save_path = os.path.join(
            reward_file_folder_dir, os.path.basename(example_file_path).split(".js")[0]+f"_with_{test_prm}_rewards.json"
        )

        if not os.path.exists(file_save_path):
            all_file_exists = False

    if not all_file_exists:
        if test_prm == "llemma":
            from llemma_7b_prm import Llemma7bPRM
            prm = Llemma7bPRM(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "mistral":
            from mistral_7b_prm import Mistral7bPRM
            prm = Mistral7bPRM(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "v1104":
            from test_prm_v0_1103 import test_prm_v0
            prm = test_prm_v0(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "v1105":
            from test_prm_v1_1105 import test_prm_v0
            prm = test_prm_v0(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "v1112":
            from test_prm_v1_1112 import test_prm_v1
            prm = test_prm_v1(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "v1_subset_1117":
            from test_prm_v1_1117 import test_prm_v1
            prm = test_prm_v1(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "v2_checkpoint_1118":
            from test_prm_v2_1118 import test_prm_v2
            prm = test_prm_v2(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "v2_1119":
            from test_prm_v2_1119 import test_prm_v2
            prm = test_prm_v2(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "v2_balanced_1120":
            from test_prm_v2_1120 import test_prm_v2
            prm = test_prm_v2(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "v2_1122_full_finetune":
            from test_prm_v2_1122 import test_prm_v2
            prm = test_prm_v2(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "v3_1123_lora":
            from test_prm_v3_1123 import test_prm_v3
            prm = test_prm_v3(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "v3_1124_checkpoint":
            from test_prm_v3_1124_checkpoint import test_prm_v3
            prm = test_prm_v3(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "v3_1123_checkpoint1":
            from test_prm_v3_1124_checkpoint1 import test_prm_v3
            prm = test_prm_v3(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "v3_1124_balanced":
            from test_prm_v3_1124 import test_prm_v3
            prm = test_prm_v3(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "1125_v3b_mistral_lora":
            from test_prm_v3_1125_v3b import test_prm_v3
            prm = test_prm_v3(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "1125_v3c_mistral_lora":
            from test_prm_v3_1125_v3c import test_prm_v3
            prm = test_prm_v3(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "v4_mistral_lora":
            from test_prm_v4_1125 import test_prm_v4
            prm = test_prm_v4(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "v3d_mistral_lora":
            from test_prm_v3_1130 import test_prm_v3
            prm = test_prm_v3(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "gpt4o":
            from gpt4o_prm import gpt4o_prm
            prm = gpt4o_prm(
                aggregation="full", 
            )
        elif test_prm == "v4_llama_1203":
            from test_prm_v4_1203 import test_prm_v4
            prm = test_prm_v4(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "v3_llama_1205":
            from test_prm_v3_1205 import test_prm_v3
            prm = test_prm_v3(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "prm800k_mistral_full_1203_re":
            from test_prm_prm800k_mistral_full_1203_re import test_prm_v3
            prm = test_prm_v3(
                aggregation="full", 
                quantization_config=quantization_config
            )

        elif test_prm == "gpt4o_real":
            from gpt4o_prm_real import gpt4o_prm
            prm = gpt4o_prm(
                 aggregation="full", 
            )
        elif test_prm == "llama_zeroshot_prm":
            from llama_zeroshot_prm import llama_zeorshot_prm
            prm = llama_zeorshot_prm(
                aggregation="full"
            )
        elif test_prm == "llama_zeroshot_prm_aws":
            from llama_zeroshot_prm_aws import llama_zeorshot_prm
            prm = llama_zeorshot_prm(
                aggregation="full"
            )
        elif test_prm == "prm800k_llama_joint_checkpoint500":
            from test_prm800k_llama_joint_checkpoint500 import test_prm_dual
            prm = test_prm_dual(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "prm800k_llama_joint_checkpoint1000":
            from test_prm800k_llama_joint_checkpoint1000 import test_prm_dual
            prm = test_prm_dual(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "prm800k_llama_joint_checkpoint2000":
            from test_prm800k_llama_joint_checkpoint2000 import test_prm_dual
            prm = test_prm_dual(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "prm800k_llama_joint_checkpoint4500":
            from test_prm800k_llama_joint_checkpoint4500 import test_prm_dual
            prm = test_prm_dual(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "reasoneval_7b":
            from reasoneval_7b import ReasonEval7bPRM
            prm = ReasonEval7bPRM(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "math_psa":
            from math_psa import math_psa_prm
            prm = math_psa_prm(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "qwen_7b_prm":
            from qwen_7b_prm import Qwen7bPRM
            prm = Qwen7bPRM(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "rlhflow_8b_prm":
            from rlhflow_8B_prm import RLHflow8bPRM
            prm = RLHflow8bPRM(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "deepseek_8b_prm":
            from deepseek_8B_prm import Deepseek_RLHflow8bPRM
            prm = Deepseek_RLHflow8bPRM(
                aggregation="full", 
                quantization_config=quantization_config
            )
        elif test_prm == "prm800k_qwen_alt_lora":
            from prm800k_qwen_alt_lora import test_prm_dual
            prm = test_prm_dual(
                aggregation="full", 
            )
        else:
            raise NotImplementedError
        
    final_data_dict = {}

    # Process each example file

    for example_file_path in example_file_path_list:

        file_save_path = os.path.join(
            reward_file_folder_dir, os.path.basename(example_file_path).split(".js")[0]+f"_with_{test_prm}_rewards.json"
        )

        if os.path.exists(file_save_path):
            continue

        with open(example_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        ##
        if args.sample_10:
            random.seed(42)
            data = random.sample(data, 10)
        ##

        for each_data in tqdm(data, desc= f"processing {os.path.basename(example_file_path)}"):

            for cot in each_data["chain_of_thoughts"]:
                # print(each_data)
                # print(cot)
                # input("--")
                if "manually_inspected" in cot.keys():
                    if cot["manually_inspected"] == False:
                        continue

                steps = cot["steps"]
                # labels = cot["labels"]
                
                if test_prm == "llemma":
                    steps = [step.replace("\n\n", "") for step in steps]
                    steps_all = "# Question\n\n" + each_data["question"] + "\n\n" + "# Solution\n\n" + "\n\n".join(steps) + "\n\n"
                    rewards = prm([steps_all])
                    cot["prm_reward"] = rewards[0].score

                elif test_prm == "mistral":
                    steps = [step.replace(prm.step_tag, "") for step in steps]
                    ###
                    updated_steps = []
                    for index, step in enumerate(steps):
                        indexed_step = f"\nStep {str(index+1)}: {step} {prm.step_tag}"
                        updated_steps.append(indexed_step)
                    steps = updated_steps
                    ###
                    question = each_data["question"].replace(prm.step_tag, "")
                    steps_all = f"{question} " + "".join(steps)
                    rewards = prm([steps_all])
                    cot["prm_reward"] = rewards[0].score

                elif test_prm == "v1104":
                    steps = [step.replace("\n\n", "") for step in steps]
                    steps_all = each_data["question"] + "# Answer " + "\n\n".join(steps) + "\n\n"
                    rewards = prm([steps_all])
                    cot["prm_reward"] = rewards[0].score
                elif test_prm == "v1105" or test_prm == "v1112" or test_prm == "v1_subset_1117" or test_prm =="v2_checkpoint_1118" or test_prm == "v2_1119" or test_prm == "v2_balanced_1120" or test_prm == "v2_1122_full_finetune" or test_prm == "v3_1123_lora" or test_prm == "v3_1124_checkpoint" or test_prm == "v3_1123_checkpoint1" or test_prm == "v3_1124_balanced" or test_prm == "1125_v3b_mistral_lora" or test_prm ==  "1125_v3c_mistral_lora" or test_prm == "v4_mistral_lora" or test_prm == "v3d_mistral_lora" or test_prm == "v4_llama_1203" or test_prm == "v3_llama_1205" or test_prm == "prm800k_mistral_full_1203_re":
                    steps = [step.replace(prm.step_tag, "") for step in steps]
                    ###
                    updated_steps = []
                    for index, step in enumerate(steps):
                        indexed_step = f"\nStep {str(index+1)}: {step} {prm.step_tag}"
                        updated_steps.append(indexed_step)
                    steps = updated_steps
                    ###
                    question = each_data["question"].replace(prm.step_tag, "")
                    steps_all = f"{question} " + "".join(steps)
                    rewards = prm([steps_all])
                    cot["prm_reward"] = rewards[0].score
                elif test_prm in ["prm800k_llama_joint_checkpoint500", "prm800k_llama_joint_checkpoint1000", "prm800k_llama_joint_checkpoint2000", "prm800k_llama_joint_checkpoint4500"]:

                    steps = [step.strip().replace("\n", "") for step in steps]
                    question = each_data["question"].strip().replace("\n", "")
                    ###
                    updated_steps = []
                    for index, step in enumerate(steps):
                        indexed_step = f" \n\n{step}"
                        updated_steps.append(indexed_step)
                    steps = updated_steps
                    ###
                    # question = each_data["question"].replace(prm.step_tag, "")
                    steps_all = f"{question}" + "".join(steps)
                    rewards = prm([steps_all])
                    cot["prm_reward"] = rewards[0].score
                    # print(cot["prm_reward"])
                    # input()
                elif test_prm == "gpt4o":
                    cot_steps = ''
                    for index, step in enumerate(steps):
                        cot_steps += '##Solution##' + f"Step {str(index+1)}. " + step + '\n'
                    
                    user_prompt = ['###Question###: \n' + each_data['question'] + '\n', 
                                '###Reference Answer###: \n' + each_data['answer'] + '\n', 
                                '###Student Answer###: \n' + cot_steps + '\n']
                    steps_all = ''.join(user_prompt)

                    rewards = prm([steps_all])
                    cot["step_evaluation"] = rewards[0][1]
                    cot["prm_reward"] = rewards[0][0].score

                elif test_prm == "reasoneval_7b":
                    steps = [step.strip().replace("\n", "") for step in steps]
                    updated_steps = []
                    for index, step in enumerate(steps):
                        indexed_step = f"{index+1}. {step}"
                        updated_steps.append(indexed_step)
                    steps = updated_steps
                    question = each_data["question"].replace('\n', "")
                    steps_all = f"{question}\n\n" + "\n\n".join(steps)
                    rewards = prm([steps_all])
                    cot["prm_reward"] = rewards[0].score

                elif test_prm == "math_psa":
                    steps = [step.replace("\n", "") for step in steps]
                    question = each_data["question"].replace("\n", "")

                    updated_steps = []
                    for index, step in enumerate(steps):
                        indexed_step = f"Step {str(index+1)}: {step} \n\n\n\n\n "
                        updated_steps.append(indexed_step)
                    steps = updated_steps
                    steps_all = f"{question} " + "".join(steps)
                    rewards = prm([steps_all])
                    cot["prm_reward"] = rewards[0].score
                elif test_prm == "qwen_7b_prm":
                    steps = [step.replace(prm.step_tag, "") for step in steps]
                    updated_steps = []
                    for index, step in enumerate(steps):
                        indexed_step = f"{step} {prm.step_tag}"
                        updated_steps.append(indexed_step)
                    steps = updated_steps
                    question = each_data["question"].replace(prm.step_tag, "")

                    input_all = [question, "".join(steps)]
                    # steps_all = f"{question} " + "".join(steps)

                    rewards = prm([input_all])
                    # rewards = prm([steps_all])
                    cot["prm_reward"] = rewards[0].score[0]
                elif test_prm in ["rlhflow_8b_prm","deepseek_8b_prm"]:
                    steps = [step.replace('\n', "") for step in steps]
                    question = each_data["question"].replace('\n', "")
                    steps_all = f"{question}\n\n" + "\n\n".join(steps)
                    rewards = prm([steps_all])
                    cot["prm_reward"] = rewards[0].score
                elif test_prm in ["prm800k_qwen_alt_lora"]:
                    # steps = cot["steps"]
                    steps = [step.strip().replace(" \n\n\n\n", "") for step in steps]
                    question = each_data["question"].strip().replace(" \n\n\n\n", "")
                    updated_steps = []
                    for index, step in enumerate(steps):
                        indexed_step = f"{step} \n\n\n\n"
                        updated_steps.append(indexed_step)
                    steps = updated_steps
                    steps_all = f"{question} \n\n" + "".join(steps)
                    rewards = prm([steps_all])
                    cot["prm_reward"] = rewards[0].score


                elif test_prm == "gpt4o_real" or test_prm == "llama_zeroshot_prm" or test_prm == "llama_zeroshot_prm_aws":

                    rewards_list = []
                    step_evaluations_list = []

                    for step_index in range(len(steps)):
                        cot_steps = ''
                        for index, step in enumerate(steps[:step_index+1]):
                            cot_steps += '##Solution##' + f"Step {str(index+1)}. " + step + '\n'

                        # len_solution = len(steps[:step_index])
                        user_prompt = ['###Question###: \n' + each_data['question'] + '\n', 
                                    '###Answer###: \n' + cot_steps + '\n']
                        
                        steps_all = ''.join(user_prompt)

                        rewards = prm([steps_all])

                        rewards_list.append(rewards[0][0].score)
                        step_evaluations_list.append(rewards[0][1]) 


                    cot["step_evaluation"] = step_evaluations_list
                    cot["prm_reward"] = rewards_list
                

                else:
                    raise NotImplementedError
                # if gpt4o_prm_debug:
                    # cot["step_evaluation"] = rewards[0][1]
                    # cot["prm_reward"] = rewards[0][0].score
                # else:
                #     cot["prm_reward"] = rewards[0].score

                # if test_prm == "gpt4o" or test_prm == "gpt4o_each_step":
                #     cot["step_evaluation"] = rewards[0][1]
                #     cot["prm_reward"] = rewards[0][0].score
                # else:
                #     cot["prm_reward"] = rewards[0].score

                if contains_nan(cot["prm_reward"]):
                    print(steps_all)
                    print("debugggggggg")
                    rewards = prm([steps_all])
                    raise ValueError

        save_dict_to_file(data=data, file_path=file_save_path)

    
    if args.do_not_calculate_metric:
        print("evaluation done")
        sys.exit()

    
    # Aggregate rewards from all files and calculate AUC-ROC
    metric_file_dir = os.path.join(args.metric_file_dir, os.path.basename(args.example_file_path_dir), test_prm)
    os.makedirs(metric_file_dir, exist_ok=True)

    total_data_dict = {}
    total_data_hash_list = []

    ssl = []
    
    for file_path in os.listdir(reward_file_folder_dir):
        rewards_file_path = os.path.join(reward_file_folder_dir, file_path)
        
        
        with open(rewards_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        


        for each_data in data:
            
            data_hash = hash_nested_dict(each_data)
            if data_hash in total_data_hash_list:
                continue
            else:
                total_data_hash_list.append(data_hash)

            if each_data["domain"] not in total_data_dict:
                total_data_dict[each_data["domain"]] = []
            
            for cot in each_data["chain_of_thoughts"]:

                if "manually_inspected" in cot.keys():
                    if cot["manually_inspected"] == False:
                        continue

                steps = cot["steps"]
                # print(rewards_file_path)
                labels = cot["labels"]

                rewards = cot["prm_reward"]
                try:
                    # print("labels noooooo")
                    assert len(steps) == len(labels) and len(labels) == len(rewards)
                except AssertionError:
                    print(each_data["domain"])
                    print(len(steps))
                    print(len(labels))
                    print(len(rewards))
                    print(cot)
                    print(each_data)
                    # input()
                    continue

                ssl, labels, rewards = update_ssl(ssl, steps, labels, rewards)

                if "FALSE_AUGMENTED" in rewards_file_path:
                    # print("----")
                    labels, rewards = filter_FALSE_AUGMENTED_data(labels, rewards)
                    # print(labels)
                    # print(rewards)

                labels, rewards = filter_first_error(labels, rewards)
                total_data_dict[each_data["domain"]].append((labels, rewards))

    # Calculate AUC-ROC for each domain
    ###calculate total accuracy and auroc        

    total_label_list = []
    total_reward_list = []
    ###
    auc_roc = {}
    for domain in total_data_dict.keys():
        label_list = []
        reward_list = []
        for labels, rewards in total_data_dict[domain]:
            label_list.extend(labels)
            reward_list.extend(rewards)

        if not label_list or not reward_list:
            print("null domain: " + domain)
            input()

        # print(len(label_list))
        # print(len(reward_list))
        label_list,reward_list = clean_lists(label_list=label_list, reward_list=reward_list)
        ###
        total_label_list.extend(label_list)
        total_reward_list.extend(reward_list)
        ###
        
        plot_accuracy_vs_threshold(labels=label_list, rewards=reward_list, domain=domain, figure_file_dir=os.path.join(metric_file_dir, "acc_vs_threshold"))
        auc_roc[domain] = compute_auc_metrics(labels=label_list, rewards=reward_list)


    # print("---")
    total_dataset_metrics = compute_auc_metrics(labels=total_label_list, rewards=total_reward_list)
    # input("+++")
    auc_roc["total_dataset_metrics"] = total_dataset_metrics

    auc_roc_dir = os.path.join(metric_file_dir, "auc_roc")
    os.makedirs(auc_roc_dir, exist_ok=True)

    save_dict_to_file(data=auc_roc, file_path=os.path.join(auc_roc_dir, f"{test_prm}_{test_dataset_name}.json") )


    ###

    # save_dict_to_json(total_dataset_metrics, os.path.join(metric_file_dir, "total_metrics", f"{test_prm}_{test_dataset_name}.json"))
    ###



if __name__ == "__main__":
    main()