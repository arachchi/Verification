import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np


def err(FARs, FRRs):
    # find eer
    min_abs_diff = 5
    min_abs_diff_id = -1
    for i in range(0, FARs.__len__()):
        abs_diff = np.abs(FARs[i] - FRRs[i])
        if abs_diff < min_abs_diff:
            min_abs_diff = abs_diff
            min_abs_diff_id = i
    # print(min_abs_diff_id, min_abs_diff)
    err = (FARs[min_abs_diff_id] + FRRs[min_abs_diff_id]) / 2.0
    return (err)


def get_dist(references, probes):
    return np.linalg.norm(references - probes, axis=1)


def min_max_normalize(distance_matrix):
    temp = distance_matrix - np.min(distance_matrix)
    return temp / np.max(distance_matrix)


# call with two arrays one containing the reference features and the other containing the corresponding probe features
def get_similarity_scores(references, probes):
    if len(references) != len(probes):
        print("references and probes do not match in size")
        return

    distances = get_dist(references, probes)
    similarities = 1 / (1 + distances)
    return similarities


def save_reference_and_probe(dataset_base_folder, image_1_path, image_2_path, reference_label, probel_label,
                             match_status, sim, thresh, test_image_saving_folder):
    threshold_folder = f"{test_image_saving_folder}/{str(thresh)[:5]}"
    if not os.path.isdir(test_image_saving_folder):
        os.mkdir(test_image_saving_folder)
    if not os.path.isdir(threshold_folder):
        os.mkdir(threshold_folder)

    status_color = (0, 0, 255)

    image_1 = cv2.imread(f"{dataset_base_folder}/{image_1_path}")
    image_2 = cv2.imread(f"{dataset_base_folder}/{image_2_path}")

    name = image_1_path.split("/")[-1][:-4]
    name += "_"
    name += image_2_path.split("/")[-1]

    name = name.replace("\\","_")

    new_image_path = f"{threshold_folder}/{name}"

    if image_1.shape != image_2.shape:
        r = image_1.shape[0]
        c = image_1.shape[1]

        if r < image_2.shape[0]:
            r = image_2.shape[0]

        if c < image_2.shape[1]:
            c = image_2.shape[1]

        image_1 = cv2.resize(image_1, (r, c))
        image_2 = cv2.resize(image_2, (r, c))

    image = np.concatenate((image_1, image_2), axis=1)
    image = cv2.putText(image, match_status, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 1,
                        cv2.LINE_AA)
    image = cv2.putText(image,
                        f"reference {reference_label} probe : {probel_label} similarity : {sim} threshold : {thresh}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(new_image_path, image)


# similarity_score (for all test pairs): range [0,1], 0 - dissimilar, 1 - similar
# actual label: 0 negative pair, 1 positive pair
def evaluate(similarity_score, actual_label, dataset_base_folder=None, values=None):
    if len(similarity_score) != len(actual_label):
        print("similarity score and actual labels do not match in size")
        return

    reference_list, probe_list, label_reference_list, label_probe_list = values

    thresholds = np.linspace(1, 0, num=50)

    FNMRs = []
    FMRs = []
    thresholds_list = []
    thresholds_false_images_dict = dict()
    for thresh in thresholds:
        false_non_match = 0.0
        true_non_match = 0.0
        false_match = 0.0
        true_match = 0.0
        match_status = ""
        status_color = (255, 255, 255)
        save_image = False
        data_row = []
        for idx, sim in enumerate(similarity_score):
            if sim > thresh:  # decision - same user
                if actual_label[idx] == 1:  # actual - same user
                    true_match += 1
                    match_status = "true match"
                    save_image = False
                else:  # actual - different users
                    false_match += 1
                    match_status = "false match"
                    save_image = True

            if sim <= thresh:  # decision - different users
                if actual_label[idx] == 1:  # actual - same user
                    false_non_match += 1
                    match_status = "false non match"
                    save_image = True
                else:  # actual - different users
                    true_non_match += 1
                    match_status = "true non match"
                    save_image = False

            if save_image:
                image_1_path, image_2_path = reference_list[idx], probe_list[idx]
                reference_label, probe_label = label_reference_list[idx], label_probe_list[idx]
                data_row.append([dataset_base_folder, image_1_path, image_2_path, reference_label, probe_label,
                                 match_status, sim, thresh])

                # save_reference_and_probe(dataset_base_folder, image_1_path, image_2_path, reference_label, probe_label,
                #                          match_status, sim, thresh)

        thresholds_false_images_dict[thresh] = data_row
        FNMR = false_non_match / (false_non_match + true_match)  # divide by all correct samples
        FMR = false_match / (true_non_match + false_match)  # divide by all wrong samples
        # when thresh == 0, FNMR = 0
        # when thresh == 1, FMR = 0

        FNMRs.append(FNMR)
        FMRs.append(FMR)
        thresholds_list.append(thresh)

    fnmr_fmr100 = 1.0
    fnmr_fmr1000 = 1.0
    fnmr_fmr0 = 1.0
    threshold_1 = thresholds_list[0]
    threshold_2 = thresholds_list[0]
    threshold_3 = thresholds_list[0]

    for idx, fmr in enumerate(FMRs):
        if fmr < 0.01:  # FMR100
            if fnmr_fmr100 > FNMRs[idx]:
                fnmr_fmr100 = FNMRs[idx]
                threshold_3 = threshold_2
                threshold_2 = threshold_1
                threshold_1 = thresholds_list[idx]

        if fmr < 0.001:  # FMR1000
            if fnmr_fmr1000 > FNMRs[idx]:
                fnmr_fmr1000 = FNMRs[idx]
        if fmr == 0:  # FMR0
            if fnmr_fmr0 > FNMRs[idx]:
                fnmr_fmr0 = FNMRs[idx]

    # Calculate EER

    plt.plot(thresholds, FMRs, '-b', label='FMR')
    plt.plot(thresholds, FNMRs, '--r', label='FNMR')
    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.title('FMR vs FNMR')
    plt.xlabel('Thresholds')
    plt.ylabel('score')
    date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dataset_name = dataset_base_folder.split("\\")[-1]
    if dataset_name == '':
        dataset_name = dataset_base_folder.split("\\")[-2]
    saving_folder_path = f"{dataset_base_folder}\\..\\evaluation_results\\result_{dataset_name}"

    if not os.path.isdir(f"{dataset_base_folder}\\..\\evaluation_results\\"):
        os.mkdir(f"{dataset_base_folder}\\..\\evaluation_results")
    if not os.path.isdir(saving_folder_path):
        os.mkdir(saving_folder_path)
    plt.savefig(f'{saving_folder_path}\\FMR_FNMR{dataset_name}_{date_time}', bbox_inches='tight')

    image_saving_thresholds = [threshold_1, threshold_2, threshold_3]
    for threshold in image_saving_thresholds:
        image_data = thresholds_false_images_dict.get(threshold)
        for row in image_data:
            dataset_base_folder, image_1_path, image_2_path, reference_label, probe_label, match_status, sim, thresh = row
            save_reference_and_probe(dataset_base_folder, image_1_path, image_2_path, reference_label, probe_label,
                                     match_status, sim, thresh, saving_folder_path)

    eer = err(FMRs, FNMRs)

    return [fnmr_fmr0, fnmr_fmr100, fnmr_fmr1000, eer]
