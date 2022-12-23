import math

import cv2 as cv
# import time
import mediapipe
import numpy as np

import PoseModule

cap = cv.VideoCapture('video/4.mp4')
# cap = cv.VideoCapture(0)
detector = PoseModule.PoseDetector()
dir = 1
count = 0
while True:
    # fps = cap.get(cv.CAP_PROP_FPS)
    success, img = cap.read()
    h,w,c = img.shape
    img = cv.resize(img, (w//2, h//2))
    img = detector.findPose(img, draw=False)
    lmlist = detector.findPoints(img)
    # print(lmlist)

    angle = detector.findangle(12,14,16)
        # angle -= 180
    per = np.interp(angle, (70, 170), (0, 100))
    bar = np.interp(angle, (70, 170), (400, 100))
    # print(angle,per)
    color = (0, 0, 255)
    if per == 100:
        color = (0, 255, 0)
        if dir == 0:
            count += 0.5
            dir = 1
    if per == 0:
        color = (0, 255, 0)
        if dir == 1:
            count += 0.5
            dir = 0
    # print(count)
    cv.rectangle(img, (750, 400), (800, 100), color, 2)
    cv.rectangle(img, (750, 400), (800, int(bar)), color, cv.FILLED)
    cv.putText(img, str(int(per)), (650, 450),
               cv.FONT_HERSHEY_PLAIN, 3, color, 2)

    cv.rectangle(img, (0, 0), (100, 100), (255, 255, 255), cv.FILLED)
    cv.putText(img, str(int(count)), (25, 70),
               cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    detector.drawangle(img,12,14,16,color)

        # cv.line(img,(lmlist[12][1], lmlist[12][2]),(lmlist[16][1], lmlist[16][2]),(255,255,255),3)

    # results = pose.process(img)
    # img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    # cv.putText(img, str(int(fps)), (20, 20),
    #            cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255))

    cv.imshow("video", img)
    cv.waitKey(1)
