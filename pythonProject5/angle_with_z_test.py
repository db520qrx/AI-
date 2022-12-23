import cv2
# import time
import mediapipe
import numpy as np
import pyautogui
import PoseModule

cap = cv2.VideoCapture("video/14.mp4")
detector = PoseModule.PoseDetector()
# 1 = in , 0 = out

while True:
    success, img = cap.read()
    # img = cv2.resize(img, (750, 600))
    img = detector.findPose(img, draw=False)
    lmlist_3D = detector.findPoints_3D(img)
    lmlist = detector.findPoints(img)

    #
    angle_d1 = detector.findangle_z(12, 14, 16)
    angle_d2 = detector.findangle_z(11, 13, 15)
    #
    cv2.putText(img, str(int(angle_d1)), (200, 350),
                cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
    cv2.putText(img, str(int(angle_d2)), (650, 350),
                cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)

    detector.drawangle(img, 11, 13, 15, (255, 0, 0))
    detector.drawangle(img, 12, 14, 16, (0, 255, 0))

    cv2.imshow("video", img)
    cv2.waitKey(1)