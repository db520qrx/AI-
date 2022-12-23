import cv2
import PoseModule
import PosedetectWithKnn
import numpy as np
import csv
import os

# with open('')

cap = cv2.VideoCapture('video/19.mp4')
detector = PoseModule.PoseDetector()
Pose_embedder = PosedetectWithKnn.PoseEmbedder()
smooth_result = PosedetectWithKnn.EMADictSmoothing()
pose_classfication = PosedetectWithKnn.PoseClassify_with_knn(csv_file='fitness_poses_csvs_out',
                                                             Pose_embedder= Pose_embedder,
                                                             smooth_result= smooth_result,
                                                             k=15)

# 0 = up , 1 = down
dir = 1
count = 0

# 测试阶段
pose_classfication.test()

while True:
    success, img = cap.read()
    img = detector.findPose(img, draw=False)
    h,w,c = img.shape
    img = cv2.resize(img,(w//2,h//2))
    lmlist = np.array(detector.findPoints(img), dtype=np.float32)
    pose_landmarks = lmlist[:, 1:]
    result = pose_classfication(pose_landmarks)
    # pose_classfication.test()
    # print(result)
    color = (0, 0, 255)
    if 'push_down' in result and result['push_down'] >= 10:
        color = (0, 255, 0)
        if dir == 1:
            count += 0.5
            dir = 0

    if 'push_up' in result and result['push_up'] >= 10:
        color = (0, 255, 0)
        if dir == 0:
            count += 0.5
            dir = 1

    detector.drawangle(img, 11, 13, 15, color)
    detector.drawangle(img, 23, 25, 27, color)
    detector.drawangle(img, 11, 23, 25, color)
    cv2.rectangle(img, (0, 0), (100, 100), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, str(int(count)), (25, 70),
               cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
    # result_test = pose_classfication.test()
    # print(result)
    # embeding = cvtembeder.__call__(pose_landmarks)
    # print(embeding)
    # print(embeding.shape)
    cv2.imshow('video', img)
    cv2.waitKey(1)
