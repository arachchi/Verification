import csv
import os

import numpy as np
from tqdm import tqdm

from config import DATASET_BASE_FOLDER
from pipeline import Pipeline
from maskcomp_evaluation import evaluate

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# datasets = ["feifei_front_align", "feifei_original", "georgia_tech_crop", "georgia_tech_non_crop", "sof_original",
#             "sof_whole_images", "youtube_faces_categories", "lfw"]

datasets = ["feifei_original", "georgia_tech_non_crop", "sof_original", "feifei_front_align",
            "youtube_faces_categories", "lfw", "our_dataset"]

# datasets = ["our_dataset"]

evaluation_result_summary_csv = '../outputs/evaluation_dataset_results_new.csv'
trained_model = "./inputs/model_VGG16_CONTRASTIVE_VGGFACE2_5_2021_04_22_16_34_05.h5"


def __get_evaluation_line_details(line):
    reference, probe, label_reference, label_probe, true_label = line.split(" ")
    key = f"{reference.strip()}_{probe.strip()}_{label_reference.strip()}_{label_probe.strip()}"
    return key, true_label.strip()


def __get_input_evaluation_line_details(line):
    reference, probe, label_reference, label_probe = line.split(" ")
    key = f"{reference.strip()}_{probe.strip()}_{label_reference.strip()}_{label_probe.strip()}"
    return key


def get_true_labels(evaluation_file, input_evaluation_file):
    if os.path.isfile(input_evaluation_file):
        with open(input_evaluation_file, 'r', newline='') as file:
            input_evaluation_lines = file.readlines()

        if os.path.isfile(evaluation_file):
            with open(evaluation_file, 'r', newline='') as file:
                evaluation_lines = file.readlines()

            input_index = 0
            evaluation_index = 0
            input_true_labels = []

            while input_index < len(input_evaluation_lines):

                input_line = __get_input_evaluation_line_details(input_evaluation_lines[input_index])
                evaluation_line, true_label = __get_evaluation_line_details(evaluation_lines[evaluation_index])

                if input_line == evaluation_line:
                    input_true_labels.append(true_label)
                    evaluation_index += 1
                    input_index += 1
                else:
                    evaluation_index += 1

                if len(evaluation_lines) < evaluation_index:
                    print("something wrong with evaluation list.")
                    break
            return input_true_labels
        else:
            return []
    else:
        return []


base_folder = DATASET_BASE_FOLDER
with open(evaluation_result_summary_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    for dataset in tqdm(datasets):
        dataset_base_folder = f"{base_folder}\\{dataset}\\"

        evaluation_file = f".\\inputs\\evaluation_{dataset}.txt"
        input_evaluation_file = f".\\inputs\\landmark_evaluation_inut_{dataset}.txt"
        landmarks_file = f".\\inputs\\landmark_list_{dataset}.txt"
        output_file = f".\\outputs\\score_{dataset}.txt"

        true_labels = get_true_labels(evaluation_file, input_evaluation_file)
        true_labels = np.array(true_labels).astype(int)
        row = [trained_model, input_evaluation_file]

        pipeline = Pipeline(input_evaluation_file, landmarks_file, output_file, trained_model, dataset_base_folder)

        similarity_score = pipeline.process()
        reference_list, probe_list, label_reference_list, label_probe_list = pipeline.read_evaluation_list_file()
        values = [reference_list, probe_list, label_reference_list, label_probe_list]
        print(similarity_score)
        print(true_labels)

        [fnmr_fmr0, fnmr_fmr100, fnmr_fmr1000, eer] = evaluate(similarity_score, true_labels, dataset_base_folder,
                                                               values)
        row.extend([fnmr_fmr0, fnmr_fmr100, fnmr_fmr1000, eer])
        writer.writerow(row)
        print(input_evaluation_file, fnmr_fmr0, fnmr_fmr100, fnmr_fmr1000, eer)
        print(f"Accuracy {1 - eer}  Rank {fnmr_fmr100}")
