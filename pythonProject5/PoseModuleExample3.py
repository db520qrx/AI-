import math

import cv2 as cv
# import time
import mediapipe
import numpy as np

import PoseModule

cap = cv.VideoCapture('video/4.mp4')
# cap = cv.VideoCapture(0)
detector = PoseModule.PoseDetector()
#1 = down , 0 = up
dir = 1
count = 0
judge = True
list_dir = []
list_per = []
action = 0


while True:
    # fps = cap.get(cv.CAP_PROP_FPS)
    success, img = cap.read()
    h,w,c = img.shape
    img = cv.resize(img, (w//2,h//2))
    img = detector.findPose(img, draw=False)
    lmlist = detector.findPoints(img)
    # print(lmlist)

    angle_1 = detector.findangle(12, 14, 16)
    angle_2 = detector.findangle(24, 26, 28)
    angle_3 = detector.findangle(12, 24, 26)

    per = np.interp(angle_1, (70, 170), (0, 100))
    # print(int(per))
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
    if angle_3 >= 154:
        color_3 = (0, 255, 0)

    if angle_2 < 170 or angle_3 < 154:
        judge = False

    # print(judge)

    # judge std of  arm
    if per == 100:
        color_1 = (0, 255, 0)
        if dir == 0 and judge:
            count += 0.5
            dir = 1

    if per == 0:
        color_1 = (0, 255, 0)
        if dir == 1 and judge:
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

    # if len(lmlist) !=0:
    #     cv.circle(img, (x2, y2),
    #               10, (255, 255, 255), cv.FILLED)
    #     cv.circle(img, (x1, y1),
    #               10, (255, 255, 255), cv.FILLED)
    #     cv.circle(img, (x3, y3),
    #               10, (255, 255, 255), cv.FILLED)
    #     cv.line(img,(x2, y2),(x3, y3),color,4)
    #     cv.line(img,(x2, y2),(x1, y1),color,4)

    # cv.line(img,(lmlist[12][1], lmlist[12][2]),(lmlist[16][1], lmlist[16][2]),(255,255,255),3)

    # results = pose.process(img)
    # img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    # cv.putText(img, str(int(fps)), (20, 20),
    #            cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255))