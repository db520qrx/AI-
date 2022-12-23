
import cv2 as cv
# import time
import mediapipe
import numpy as np

import PoseModule

cap = cv.VideoCapture('video/12.mp4')
# cap = cv.VideoCapture(0)
detector = PoseModule.PoseDetector()
dir = 1
count = 0
judge = False

while True:
    # fps = cap.get(cv.CAP_PROP_FPS)
    success, img = cap.read()
    img = cv.resize(img, (950, 550))
    img = detector.findPose(img)
    lmlist = detector.findPoints(img)
    # print(lmlist)

    # angle_1 = detector.findangle(12, 14, 16)
    # angle_2 = detector.findangle(24, 26, 28)
    # angle_3 = detector.findangle(12, 24, 26)
    # print(angle_1, angle_2, angle_3)
    #
    #
    # per_1 = np.interp(angle_1, (70, 170), (0, 100))
    # # per_2 = np.interp(angle_2, (170, 180), (0, 100))
    # # per_3 = np.interp(angle_3, (170, 180), (0, 100))
    #
    # color_1 = (0, 0, 255)
    # color_2 = (0, 0, 255)
    # color_3 = (0, 0, 255)
    #
    # if angle_2 > 170 and angle_3 > 170:
    #     judge = True
    # else:
    #     judge = False
    #
    # print(judge)
    #
    # if angle_2 > 170:
    #     color_2 = (0, 255, 0)
    #
    # if angle_3 > 170:
    #     color_3 = (0, 255, 0)
    #
    # if per_1 == 100:
    #     color_1 = (0, 255, 0)
    #     if dir == 0 and judge:
    #         count += 0.5
    #         dir = 1
    # if per_1 == 0:
    #     color_1 = (0, 255, 0)
    #     if dir == 1 and judge:
    #         count += 0.5
    #         dir = 0


    # cv.rectangle(img, (0, 0), (100, 100), (255, 255, 255), cv.FILLED)
    # cv.putText(img, str(int(count)), (25, 70),
    #            cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
    # cv.putText(img, str(int(angle_2)), (400, 550),
    #            cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
    #
    # detector.drawangle(img, 12, 14, 16, color_1)
    # detector.drawangle(img, 24, 26, 28, color_2)
    # detector.drawangle(img, 12, 24, 26, color_3)


    cv.imshow("video", img)
    cv.waitKey(1)
