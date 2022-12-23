import csv
import os
import cv2
# import time
import mediapipe
import numpy as np
import pyautogui
import PoseModule
import PosedetectWithKnn
import codecs


def test_angle_with_z():
    detector = PoseModule.PoseDetector()

    img = cv2.imread('photo/3.png')
    img = detector.findPose(img, draw=False)
    lmlist = detector.findPoints(img)

    angle_d1 = detector.findangle_z(12, 14, 16)
    angle_d2 = detector.findangle_z(11, 13, 15)

    cv2.putText(img, str(int(angle_d1)), (200, 350),
                cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
    cv2.putText(img, str(int(angle_d2)), (400, 350),
                cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)

    cv2.imshow("photo", img)
    cv2.waitKey(0)


def test_posemodule_with_game():
    pyautogui.press('space')


# def test_dtype():
#     a = np.array([1., 2., 0.3])
#     s = np.dtype('a')
#     print()
#     print(s)


class sample_from_csv():
    def __init__(self, name, class_name, landmarks, embedder):
        self.name = name
        self.class_name = class_name
        self.landmarks = landmarks
        self.embedder = embedder


def test_csv_find():
    Pose_embedder = PosedetectWithKnn.PoseEmbedder()
    files_name = [name for name in os.listdir('fitness_poses_csvs_out')]
    print()
    # print(files_name)
    path = []
    class_name = []
    for name in files_name[:2]:
        class_name.append(name[:-(len('csv') + 1)])
    print(class_name)

    pose_sample = []

    for i, name in enumerate(class_name):
        path = os.path.join('fitness_poses_csvs_out', name + '.csv')

        with open(path) as f:
            csv_read = csv.reader(f, delimiter=',')
            datas = [row for row in csv_read]

            for _, row in enumerate(datas):
                assert len(row) == 33 * 3 + 1, 'Wrong number of values: {}'.format(len(row))
                landmarks = np.array(row[1:], np.float32).reshape(33, 3)
                pose_sample.append(sample_from_csv(name=row[0], class_name=name,
                                                   landmarks=landmarks, embedder=Pose_embedder(landmarks)))

    # print(len(pose_sample))
    # for i, sample in enumerate(pose_sample):
    #     print(i,':')
    #     print(sample.embedder)

    with codecs.open('csv_embedder.txt', mode='a', encoding='utf-8') as file_txt:
        for i, sample in enumerate(pose_sample):
            file_txt.write(str(sample.class_name) + '\n' + str(sample.embedder) + '\n' + '\n')
    # datas = np.array(datas)
    # print(datas)
    # print(datas.shape)


def test_mutiply_of_np2np():
    a = np.array((1, 1, 0.2))
    b = np.array([(1, 1, 2), (2, 3, 4), (10, 2, 5)])
    b2 = np.array([(2, 3, 2), (1, 5, 2), (1, 5, 7)])

    c = np.abs(b - b2)
    d = c * a
    e = np.max(d)
    print()
    print(c)
    print()
    print(d)
    print()
    print(e)


def test_sort():
    max_dist_heap = np.array([(1, 5, 2), (2, 2, 5), (0, 5, 2)])
    max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
    print(max_dist_heap)


def test_set():
    a = np.array([1, 2, 2, 2, 5, 1, 1, 5, 6])
    print(set(a))
    b = {1, 1, 1, 2}
    # print(set(b))
    c = [n for n in set(b)]
    print(c)


# def test_2d():
#     a = ([1, 3], [2, 5], [0, 6], [4, 6])
#     a = sorted(a, key=lambda x: x[0])
#     a = a[:3]
#
#     result = {class_name: a[:][1].count(class_name) for class_name in set(a[:][1])}
#     print()
#     print(a)
#     print(a[:][1])
#     print(result)


def test_knn():
    img = cv2.imread('photo/6.png')
    detector = PoseModule.PoseDetector()
    Pose_embedder = PosedetectWithKnn.PoseEmbedder()
    pose_classfication = PosedetectWithKnn.PoseClassify_with_knn(csv_file='fitness_poses_csvs_out',
                                                                 Pose_embedder=Pose_embedder,
                                                                 k=13, weights_axes=(1, 1, 0.2))
    img = detector.findPose(img, draw=False)
    lmlist = np.array(detector.findPoints(img), dtype=np.float32)
    pose_landmarks = lmlist[:, 1:]
    result = pose_classfication(pose_landmarks)
    print(result)


def test_get_image_data():
    image_folder = 'photo'
    data = [image for image in os.listdir(image_folder)]
    print(data)


#
# def test_get_coloum():
#     a = ([1, 2, 5], [2, 3, 5])
#     for i, j in a:
#         pass

# def test_res_mean():
#     a = np.array(([1, 2, 4], [2, 5, 6]))
#     b = np.array(([1, -5, 6], [6, 3, 6]))
#     c = np.abs(a - b)
#     print()
#     print(c)
#     d = c * (1, 1, 0.2)
#     print(d)
#     e = np.mean(d)
#     print(e)


def test_dis_oushi():
    a = np.array(([1, 2, 4], [2, 5, 6]))
    b = np.array(([1, 2, 6], [3, 6, 1]))
    c = (np.abs(a - b) ** 2) * (1, 1, 0.2)
    d = np.sum(c, axis=1)
    e = np.sqrt(d)
    f = np.mean(e)
    print()
    print(c)
    print(d)
    print(e)
    print(f)


def test_get_image():
    image_names = [name for name in os.listdir('image_all')]
    image_up_path = os.path.join('image_all', image_names[1])
    image_name = os.listdir(image_up_path)
    print(image_name[1])
    print(os.path.join(image_up_path, image_name[1]))
    img = cv2.imread(os.path.join(image_up_path, image_name[1]))
    cv2.imshow('photo', img)
    cv2.waitKey(0)


def test_get_csv():
    detector = PoseModule.PoseDetector()
    Pose_embedder = PosedetectWithKnn.PoseEmbedder()
    Getcsv = PosedetectWithKnn.Get_csv_from_images(images_folder='image_all', csvs_out_folder='csv_push',
                                                   detector=detector, Pose_embedder=Pose_embedder)
    Getcsv.__call__()


def test_dist():
    a = {'a': 14, 'b': 12}
    b = a['a'] / a['b']
    print(b)


def test_ravel():
    idx = np.array([0, 0, 1, 1, 2, 2, 1, 0, 1, 2])
    indics = np.where(idx == 1)
    a = np.array(([3, 2], [4, 4], [5, 4], [7, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]))
    print(indics)
    l = len(indics[0])
    print(l)
    b = np.sum(a[indics, :], axis=1)
    print(b)
    print(np.where(idx == 1)[0])
    print(a[np.where(idx == 0)[0],:])
