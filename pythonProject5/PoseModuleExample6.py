import cv2 as cv
# import time
import mediapipe
import numpy as np
import PoseModule
import pyautogui

# cap = cv.VideoCapture('video/17.mp4')
cap = cv.VideoCapture(0)
detector = PoseModule.PoseDetector()
# 1 = in , 0 = out
dir = 1
count = 0
# judge = True
# list_dir = []
# list_per = []


while True:
    # fps = cap.get(cv.CAP_PROP_FPS)
    success, img = cap.read()
    # img = cv.resize(img, (750, 600))
    img = detector.findPose(img, draw=False)
    lmlist = detector.findPoints(img)

    angle_1 = detector.findangle(11, 13, 15)

    per = np.interp(angle_1, (90, 170), (0, 100))
    # print(int(per))

    color_1 = (0, 0, 255)

    # judge std of  arm
    if per == 100:
        color_1 = (0, 255, 0)
        if dir == 0:
            count += 0.5
            dir = 1

    if per == 0:
        color_1 = (0, 255, 0)
        if dir == 1:
            count += 0.5
            dir = 0

    cv.rectangle(img, (0, 0), (100, 100), (255, 255, 255), cv.FILLED)
    cv.putText(img, str(int(count)), (15, 70),
               cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
    # cv.putText(img, str(int(angle_1)), (400, 550),
    #            cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    detector.drawangle(img, 11, 13, 15, color_1)

    cv.imshow("video", img)
    cv.waitKey(1)
