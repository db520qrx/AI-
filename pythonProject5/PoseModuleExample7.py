
import cv2 as cv
# import time
import mediapipe
import numpy as np

import PoseModule


cap = cv.VideoCapture('video/14.mp4')
# cap = cv.VideoCapture(0)
detector = PoseModule.PoseDetector()
dir = 1
count = 0
judge = False

while True:
    # fps = cap.get(cv.CAP_PROP_FPS)
    success, img = cap.read()
    img = cv.resize(img, (750, 600))
    img = detector.findPose(img, draw=False)
    lmlist = detector.findPoints(img)
    # print(lmlist)

    angle_1 = detector.findangle(12, 14, 16)
    angle_2 = detector.findangle(11, 13, 15)

    angle_d1 = detector.findangle_z(12,14,16)
    angle_d2 = detector.findangle_z(11,13,15)

    per_1 = np.interp(angle_1, (70, 170), (0, 100))
    # per_2 = np.interp(angle_2, (170, 180), (0, 100))
    # per_3 = np.interp(angle_3, (170, 180), (0, 100))
    color_1 = (0, 0, 255)
    # color_2 = (0, 0, 255)
    # color_3 = (0, 0, 255)


    # print(count)
    # cv.rectangle(img,(650,400),(700,100),color,2)
    # cv.rectangle(img,(650,400),(700,int(bar)),color,cv.FILLED)
    # cv.putText(img, str(int(per)), (650,450),
    #            cv.FONT_HERSHEY_PLAIN, 3, color,2)

    cv.rectangle(img, (0, 0), (100, 100), (255, 255, 255), cv.FILLED)
    cv.putText(img, str(int(count)), (25, 70),
               cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
    cv.putText(img, str(int(angle_d1)), (200, 350),
               cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
    cv.putText(img, str(int(angle_d2)), (400, 350),
               cv.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)


    detector.drawangle(img, 12, 14, 16, color_1)
    detector.drawangle(img, 11, 13, 15, color_1)
    # detector.drawangle(img, 24, 26, 28, color_2)
    # detector.drawangle(img, 12, 24, 26, color_3)


    cv.imshow("video", img)
    cv.waitKey(1)