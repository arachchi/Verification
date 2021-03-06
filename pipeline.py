import os

import cv2
import numpy as np
import tensorflow.keras.backend as K
from keras_preprocessing.image import img_to_array
from keras_vggface.utils import preprocess_input
from tqdm import tqdm

from build_networks import build_siamese_network
from config import *


class Pipeline:

    def __init__(self, evaluation_list_file, landmarks_file, output_path, model_path=None, dataset_base_folder=None):
        self.evaluation_list_file = evaluation_list_file

        if PRODUCTION:
            self.model_path = SIAMESE_MODEL_PATH
            self.dataset_base_folder = DATASET_BASE_FOLDER
        else:
            self.model_path = model_path
            self.dataset_base_folder = dataset_base_folder

        self.bbox_dict = dict()
        self.facial_key_points_dict = dict()
        if os.path.isfile(landmarks_file):
            with open(landmarks_file, 'r', newline='') as file:
                lines = file.readlines()
                for line in lines:
                    file_name, bbox, facial_key_points = self.__get_landmark_line_details(line)
                    self.bbox_dict[file_name] = bbox
                    self.facial_key_points_dict[file_name] = facial_key_points

        else:
            print(f"Landmark file is not accessible. Please check the path {landmarks_file}.")
            exit(0)

        self.output_path = output_path
        self.siamese_model = self.build_models()

    def build_models(self):
        model, sister_network = build_siamese_network_vgg16(fine_tune_percentage=0.01)

        model.load_weights(self.model_path)
        sister_network.summary()
        return sister_network

    def evaluate_images(self, reference_image, reference_bbox, probe_image, probe_bbox, true_label):
        pass

    def __image_margin(self, image):
        shape = IMG_SHAPE
        background = np.zeros(shape, dtype=np.uint8)
        image_shape = image.shape

        r = int((shape[0] - image_shape[0]) / 2)
        c = int((shape[1] - image_shape[1]) / 2)

        background[r:r + image_shape[0], c:c + image_shape[1], :] = image

        return background

    def scale_and_resize_image(self, image):
        shape = IMG_SHAPE
        image_shape = image.shape

        if image_shape[0] > image_shape[1]:
            # print((image_shape), (shape[0], int(image_shape[1] * (shape[1] / image_shape[0]))))
            resized_image = cv2.resize(image, (int(image_shape[1] * (shape[1] / image_shape[0])), shape[0]))
        else:
            # print((image_shape), (int(image_shape[0] * (shape[1] / image_shape[1])), shape[1]))
            resized_image = cv2.resize(image, (shape[0], int(image_shape[0] * (shape[1] / image_shape[1]))))
        # print(image_shape, resized_image.shape)
        return self.__image_margin(resized_image)

    def __crop_image(self, image, x1, y1, x2, y2, ratio=1.0):

        y1_ = int(y1 - (y2 - y1) * ratio)
        x1_ = int(x1 - (x2 - x1) * ratio)
        y2_ = int(y2 + (y2 - y1) * ratio)
        x2_ = int(x2 + (x2 - x1) * ratio)

        if x1_ <= 0:
            x1_ = 0
        if y1_ <= 0:
            y1_ = 0
        if x2_ > image.shape[1]:
            x2_ = image.shape[1]
        if y2_ > image.shape[0]:
            y2_ = image.shape[0]

        cropped_image = image[y1_:y2_, x1_:x2_]

        if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
            print(image.shape, x1, x2, y1, y2, x1_, x2_, y1_, y2_)

        return self.scale_and_resize_image(cropped_image)

    def open_image(self, image_path):
        if not PRODUCTION:
            image_path = f"{self.dataset_base_folder}\\{image_path}"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __get_processed_inference_images(self, image_path):
        images = []
        original_image = self.open_image(image_path)
        scaled_original_image = self.scale_and_resize_image(original_image)
        images.append(scaled_original_image)

        face_bbox = self.bbox_dict.get(image_path)
        if face_bbox is not None:
            for ratio in [0.5, 0.6, 0.7]:
                cropped_image = self.__crop_image(original_image, face_bbox[0], face_bbox[1], face_bbox[2],
                                                  face_bbox[3],
                                                  ratio=ratio)
                #cropped_image = img_to_array(cropped_image)
                #cropped_image = preprocess_input(cropped_image, version=1)
                images.append(cropped_image)

        return np.array(images)

    @staticmethod
    def euclidean_distance(vectors):
        # unpack the vectors into separate lists
        (featsA, featsB) = vectors
        # compute the sum of squared distances between the vectors
        sumSquared = K.square(featsA - featsB)
        # return the euclidean distance between the vectors
        return K.sqrt(sumSquared)
    @staticmethod
    def exposure_correcion(image1, image2):
        # find darker image
        mean1 = np.average(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY))
        mean2 = np.average(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY))
        # print(mean1, mean2)
        if mean1 < mean2:
            matched = match_histograms(image1, image2, multichannel=True)
            return matched, image2
        else:
            matched = match_histograms(image2, image1, multichannel=True)
            return image1, matched

    @staticmethod
    def last_preprocess_steps(image):
        image = img_to_array(image)
        image = preprocess_input(image, version=1)
        return image

    def process(self):
        comparison_scores = []
        reference_list, probe_list, label_reference_list, label_probe_list = self.read_evaluation_list_file()
        for idx, reference in enumerate(tqdm(reference_list)):
            probe = probe_list[idx]
            reference_images = self.__get_processed_inference_images(reference)
            probe_images = self.__get_processed_inference_images(probe)

            for k, refs in enumerate(reference_images):
                # show1 = np.hstack((reference_images[k], probe_images[k]))
                temp1, temp2 = self.exposure_correcion(reference_images[k], probe_images[k])
                reference_images[k] = self.last_preprocess_steps(temp1)
                probe_images[k] = self.last_preprocess_steps(temp2)
                # show2= np.hstack((reference_images[k], probe_images[k]))
                # show  = np.vstack((show1, show2))
                # cv2.imshow("this", show)
                # cv2.waitKey(0)

            reference_vectors = self.siamese_model.predict([reference_images])
            probe_vectors = self.siamese_model.predict([probe_images])
            distances = []
            for j, reference_vector in enumerate(reference_vectors):
                distance = self.euclidean_distance([reference_vector, probe_vectors[j]])
                distance = 1 / (1 + distance)
                distances.append(distance)
            average_score = np.average(distances)
            comparison_scores.append(average_score)

        self.write_results_to_output(comparison_scores)

        return comparison_scores

    def __get_landmark_line_details(self, line):
        values = line.split(" ")
        file_name = values[0]
        bbox = np.array(values[1:5]).astype(int)
        facial_key_points = np.array(values[5:]).astype(int)

        bbox = bbox.tolist()
        facial_key_points = facial_key_points.tolist()

        return file_name, bbox, facial_key_points

    def __get_evaluation_line_details(self, line):
        reference, probe, label_reference, label_probe = line.split(" ")

        return reference.strip(), probe.strip(), label_reference.strip(), label_probe.strip()

    def read_evaluation_list_file(self):
        reference_list, probe_list, label_reference_list, label_probe_list = [], [], [], []
        if os.path.isfile(self.evaluation_list_file):
            with open(self.evaluation_list_file, 'r', newline='') as file:
                lines = file.readlines()
                i = 0
                for line in lines:
                    i += 1

                    reference, probe, label_reference, label_probe = self.__get_evaluation_line_details(line)
                    reference_list.append(reference)
                    probe_list.append(probe)
                    label_reference_list.append(label_reference)
                    label_probe_list.append(label_probe)

        else:
            print(f"Evaluation list is not accessible. Please check the path {self.evaluation_list_file}.")

        return reference_list, probe_list, label_reference_list, label_probe_list

    def write_results_to_output(self, comparison_score_list):
        output_file_name = self.output_path
        if os.path.isdir(self.output_path):
            output_file_name = f"{output_file_name}\\scores.txt"

        with open(output_file_name, 'w', newline='') as file:
            for comparison_score in comparison_score_list:
                comparison_score_str = str(comparison_score) + "\n"
                file.write(comparison_score_str)
