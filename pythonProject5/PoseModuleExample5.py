import math

import cv2 as cv
# import time
import mediapipe
import numpy as np

import PoseModule

cap = cv.VideoCapture('video/17.mp4')
# cap = cv.VideoCapture(0)
detector = PoseModule.PoseDetector()
#1 = in , 0 = out
dir = 1
count = 0
judge = True
list_dir = []
list_per = []
action = 0


while True:
    # fps = cap.get(cv.CAP_PROP_FPS)
    success, img = cap.read()
    # img = cv.resize(img, (750, 600))
    img = detector.findPose(img, draw=False)
    lmlist = detector.findPoints(img)
    # print(lmlist)
    # if len(lmlist) != 0:
    #     print(abs(lmlist[16][1] - lmlist[28][1]))

    angle_1 = detector.findangle(12, 14, 16)
    angle_2 = detector.findangle(24, 26, 28)
    angle_3 = detector.findangle(12, 24, 26)

    per = np.interp(angle_3, (90, 170), (0, 100))
    print(int(per))
    list_per.append(int(per))

    length = len(list_dir)

    if len(list_per) > 2:
        if list_per[len(list_per)-1] - list_per[len(list_per)-2] > 0 \
                and list_per[len(list_per) - 2] - list_per[len(list_per) - 3] > 0:
            dir_list = 0
            list_dir.append(dir_list)
        if list_per[len(list_per)-1] - list_per[len(list_per)-2] < 0 \
                and list_per[len(list_per)-2] - list_per[len(list_per)-3] < 0 :
            dir_list = 1
            list_dir.append(dir_list)

    # print(list_dir)


    if len(list_dir) > 1 and  length < len(list_dir) :
        if list_dir[len(list_dir)-1] != list_dir[len(list_dir)-2]:
            action += 0.5

    color_1 = (0, 0, 255)
    color_2 = (0, 0, 255)
    color_3 = (0, 0, 255)

    if angle_2 >= 170:
        color_2 = (0, 255, 0)
    if angle_1 >= 155:
        color_1 = (0, 255, 0)

    if angle_2 < 150 or angle_1 < 150:
        judge = False

    # print(judge)

    if abs(lmlist[16][1] - lmlist[28][1]) < 25:
        judge_2 = True
    else:
        judge_2 = False

    # judge std of  arm
    if per == 100:
        color_3 = (0, 255, 0)
        if dir == 0 and judge:
            count += 0.5
            dir = 1

    if per == 0:
        color_3 = (0, 255, 0)
        if dir == 1 and judge and judge_2:
            count += 0.5
            dir = 0


    if per == 100 and action > 0:
        action = 0
        list_dir = []
        list_per = []
        judge = True

    if action == 1:
        action = 0
        list_dir = []
        list_per = []
        judge = True

    cv.rectangle(img, (0, 0), (100, 100), (255, 255, 255), cv.FILLED)
    cv.putText(img, str(int(count)), (25, 70),
               cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
    # cv.putText(img, str(action), (25, 570),
    #            cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
    # cv.putText(img, str(judge), (25, 370),
    #            cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
    #
    # cv.putText(img, str(int(angle_1)), (400, 550),
    #            cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    detector.drawangle(img, 12, 14, 16, color_1)
    detector.drawangle(img, 24, 26, 28, color_2)
    detector.drawangle(img, 12, 24, 26, color_3)


    cv.imshow("video", img)
    cv.waitKey(1)
