import cv2
# import time
import mediapipe as mp
import math
import numpy as np
import os
import csv
import random


class Get_csv_from_images():
    def __init__(self, images_folder,
                 csvs_out_folder,
                 detector,
                 Pose_embedder):
        self.images_folder = images_folder
        self.csvs_out_folder = csvs_out_folder
        self.detector = detector
        self.Pose_embedder = Pose_embedder
        self.pose_class_names = [name for name in os.listdir(self.images_folder)]

    def __call__(self):
        if not os.path.exists(self.csvs_out_folder):
            os.makedirs(self.csvs_out_folder)

        for image_name in self.pose_class_names:
            csv_out_path = os.path.join(self.csvs_out_folder, image_name + '.csv')
            with open(csv_out_path, 'w', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)

                image_path = os.path.join(self.images_folder, image_name)
                image_names = os.listdir(image_path)
                for name in image_names:
                    img = cv2.imread(os.path.join(image_path, name))
                    img = self.detector.findPose(img, draw=False)
                    lmlist = np.array(self.detector.findPoints(img), dtype=np.float32)
                    if len(lmlist) != 0:
                        pose_landmarks = lmlist[:, 1:]
                        pose_landmarks = self.Pose_embedder(pose_landmarks)
                        csv_writer.writerow([name] + pose_landmarks.flatten().astype(np.str).tolist())


class PoseEmbedder():

    def __init__(self, torso_size_multiplier=2.5):
        self._torso_size_multiplier = torso_size_multiplier
        self._landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]

    def __call__(self, landmarks):
        assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(
            landmarks.shape[0])
        # 获取 landmarks.
        landmarks = np.copy(landmarks)
        # Normalize landmarks.
        landmarks = self._normalize_pose_landmarks(landmarks)
        # Get embedding.
        embedding = self._get_pose_distance_embedding(landmarks)
        return embedding

    def _normalize_pose_landmarks(self, landmarks):
        landmarks = np.copy(landmarks)
        # Normalize translation.
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center
        # Normalize scale.
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # Multiplication by 100 is not required, but makes it eaasier to debug.
        landmarks *= 100

        return landmarks

    def _get_pose_center(self, landmarks):
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self, landmarks, torso_size_multiplier):
        landmarks = landmarks[:, :2]

        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        hips = (left_hip + right_hip) * 0.5

        left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
        right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        torso_size = np.linalg.norm(shoulders - hips)

        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _get_pose_distance_embedding(self, landmarks):
        embedding = np.array([

            self._get_distance(
                self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
                self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow'),

            self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_knee', 'right_ankle'),

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
            self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
            self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle'),
        ])

        return embedding

    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from


class sample_from_csv():
    def __init__(self, name, class_name, landmarks,embedder):
        self.name = name
        self.class_name = class_name
        self.landmarks = landmarks
        self.embedder = embedder


class PoseClassify_with_knn():
    def __init__(self, csv_file, Pose_embedder,smooth_result, k, weights_axes=(1., 1., 0.2)):
        self.csv_file = csv_file
        self.Pose_embedder = Pose_embedder
        self.smooth_result = smooth_result
        self.k = k
        # self.k_max = k_max
        self.weights = weights_axes
        # self.pose_sample = sample_from_csv(name)
        self.pose_sample = self.get_sample_csv(csv_file, Pose_embedder)
        random.shuffle(self.pose_sample)
        n = len(self.pose_sample) // 10
        self.test_sample = self.pose_sample[:n]
        self.train_sample = self.pose_sample[n:]

    def get_sample_csv(self, csv_file, Pose_embedder):
        files_name = [name for name in os.listdir(csv_file)]
        class_name = []
        for name in files_name[:2]:
            class_name.append(name[:-(len('csv') + 1)])

        pose_sample = []
        for i, name in enumerate(class_name):
            path = os.path.join(csv_file, name + '.csv')

            with open(path) as f:
                csv_read = csv.reader(f, delimiter=',')
                datas = [row for row in csv_read]

            for _, row in enumerate(datas):
                assert len(row) == 33 * 3 + 1, 'Wrong number of values: {}'.format(len(row))
                landmarks = np.array(row[1:], np.float32).reshape(33, 3)
                embedder = Pose_embedder(landmarks)
                # pose_sample.append(sample_from_csv(name=row[0], class_name=name,
                #                                    landmarks=landmarks, embedder=embedder))
                pose_sample.append(sample_from_csv(name=row[0], class_name=name,
                                                   landmarks=landmarks,embedder= embedder))

        return pose_sample

    def __call__(self, pose_landmarks):
        pose_embedder = self.Pose_embedder(pose_landmarks)
        flipped_pose_embedder = self.Pose_embedder(pose_landmarks*np.array([-1, 1, 1]))
        # res = self.distance(pose_embedder)
        res = self.distance_Euclidean(pose_embedder,flipped_pose_embedder)
        class_names = []
        for _, sample_res in enumerate(res):
            class_names.append(sample_res[1])
        result = {class_name: class_names.count(class_name) for class_name in set(class_names)}
        result = self.smooth_result(result)
        return result

    def distance_Euclidean(self, pose_embedder,flipped_pose_embedder):
        # pose_embedder = self.Pose_embedder(Pose_landmarks)
        res_max = []
        for id, sample in enumerate(self.train_sample):
            dis = min(np.abs(np.max(pose_embedder - sample.embedder)),np.abs(np.max(flipped_pose_embedder - sample.embedder)))
            res_max.append([dis, sample])
        res_max = sorted(res_max, key=lambda x: x[0])
        res_max = np.array(res_max[:self.k*3])
        samples = [sample for sample in res_max[:,-1]]

        res = []
        for _, sample in enumerate(samples):
            dis_Euclidean = min(np.mean(np.sqrt(np.sum((np.abs(sample.embedder - pose_embedder) ** 2) * self.weights, axis=1))),
                                 np.mean(np.sqrt(np.sum((np.abs(sample.embedder - flipped_pose_embedder) ** 2) * self.weights, axis=1))))
            res.append([dis_Euclidean, sample.class_name])
        res = sorted(res, key=lambda x: x[0])
        res = res[:self.k]
        #
        # print(res)

        return res

    def test(self):
        correct = 0
        num = 0
        for test in self.test_sample:
            result = test.class_name
            # print(result)
            res = self.distance_Euclidean(test.embedder,test.embedder*np.array([-1,1,1]))
            class_names = []
            for _, sample_res in enumerate(res):
                class_names.append(sample_res[1])
            result2_dist = {class_name: class_names.count(class_name) for class_name in set(class_names)}
            # print(result2_dist)
            if 'push_down' in result2_dist and result2_dist['push_down'] >= 10:
                result2 = 'push_down'
            elif 'push_up' in result2_dist and result2_dist['push_up'] >= 10:
                result2 = 'push_up'
            else:
                result2 = 'None'
            # print(result2)
            if result == result2:
                correct += 1
            num += 1
        print('准确率：', correct * 100 / num, '%')

        #     print(count_list)
        # print(0)


class EMADictSmoothing():

    def __init__(self, window_size=10, alpha=0.2):
        self._window_size = window_size
        self._alpha = alpha

        self._data_in_window = []

    def __call__(self, data):
        # Add new data to the beginning of the window for simpler code.
        self._data_in_window.insert(0, data)
        self._data_in_window = self._data_in_window[:self._window_size]

        # Get all keys.
        keys = set([key for data in self._data_in_window for key, _ in data.items()])

        # Get smoothed values.
        smoothed_data = dict()
        for key in keys:
            factor = 1.0
            top_sum = 0.0
            bottom_sum = 0.0
            for data in self._data_in_window:
                value = data[key] if key in data else 0.0

                top_sum += factor * value
                bottom_sum += factor

                # Update factor.
                factor *= (1.0 - self._alpha)

            smoothed_data[key] = top_sum / bottom_sum

        return smoothed_data

# class count():
#     def __init__(self):
#         pass
