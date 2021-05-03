import os

import numpy as np
from tqdm import tqdm

datasets = ["feifei_original", "georgia_tech_non_crop", "sof_original", "feifei_front_align",
            "youtube_faces_categories", "lfw", "our_dataset"]

# models = ['without_sigmoid_model-inter-695001.pt', 'model-inter-11251.pt']
# models = ['before_sim_conversion_model-inter-695001.pt', 'before_sim_conversion_model-inter-11251.pt', 'before_sim_conversion_model-inter-11001.pt', 'before_sim_conversion_model-inter-14501.pt']
models = ['model-inter-13751.pt',  'model-inter-11251.pt']


def get_similarity_score_from_file(file_name):
    print(file_name)
    with open(file_name, 'r', newline='') as file:
        evaluation_lines = file.readlines()
    evaluation_lines = np.array(evaluation_lines).astype(np.float32)
    return evaluation_lines


def write_results_to_output(output_file_name, comparison_score_list):
    print(output_file_name)
    with open(output_file_name, 'w', newline='') as file:
        for comparison_score in comparison_score_list:
            comparison_score_str = str(comparison_score) + "\n"
            file.write(comparison_score_str)


ensemble_folder_name = "x_"
for model in models:
    ensemble_folder_name += model[:-3]
    ensemble_folder_name += "_"
print(ensemble_folder_name)
ensemble_folder = f".\\outputs\\{ensemble_folder_name}\\"
if not os.path.isdir(ensemble_folder):
    os.mkdir(ensemble_folder)

for dataset in tqdm(datasets):
    scores = None
    output_file_name = f"{ensemble_folder}score_{dataset}.txt"
    for model_name in models:
        MODEL_NAME = model_name
        score_file_dir = f".\\outputs\\{MODEL_NAME[:-3]}"

        input_file = f"{score_file_dir}\\score_{dataset}.txt"
        similarity_scores = get_similarity_score_from_file(input_file)

        if scores is None:
            scores = similarity_scores
        else:
            scores += similarity_scores

    scores /= len(models)
    # scores = 1 / (1 + scores)
    write_results_to_output(output_file_name, scores)
