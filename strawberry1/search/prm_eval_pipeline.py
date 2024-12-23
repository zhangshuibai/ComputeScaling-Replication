
import argparse, os, json
from prm_interface import StepScore
from typing import List

def get_dataset_from_output_file(model_output_file_path: str) -> list[str]:
    with open(model_output_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError("The JSON file must contain a list of strings.")
    
    return data



def save_prm_results(data: List[StepScore], prm_results_save_path: str):
    data_list = []
    for each_step_score in data:
        data_list.append([each_step_score.step, each_step_score.score])

    with open(prm_results_save_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, ensure_ascii=False, indent=4)

    print(f"Data successfully saved to {prm_results_save_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Prm")

    parser.add_argument('--output_data_file', type=str, required=True)
    parser.add_argument('--test_dataset_name', type=str, required=True)
    parser.add_argument('--prm_results_dir', type=str, required=True)
    parser.add_argument('--prm', type=str, required=True, help="which prm to use")


    args = parser.parse_args()


    if args.prm == "Llemma7bPRM_single":
        from llemma_7b_prm import Llemma7bPRM as Llemma7bPRM_single
        prm_model = Llemma7bPRM_single()
    elif args.prm == "llemma_7b_prm_batched":
        from llemma_7b_prm_batched import Llemma7bPRM as Llemma_7b_prm_batched
        prm_model = Llemma_7b_prm_batched()
    else:
        raise NotImplementedError
    
    dataset = get_dataset_from_output_file(args.output_data_file)
    output_prm_resutls = prm_model(dataset)

    prm_results_save_path = os.path.join(args.prm_results_dir, args.prm, args.test_dataset_name+".json")
    os.makedirs(os.path.dirname(prm_results_save_path))

    save_prm_results(data=output_prm_resutls, prm_results_save_path=prm_results_save_path)







        
        







