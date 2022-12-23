import math
import cv2 as cv
# import time
# import mediapipe
import numpy as np
import PoseModule

cap = cv.VideoCapture('video/11.mp4')
# cap = cv.VideoCapture(0)
detector = PoseModule.PoseDetector()
dir = 1
count = 0
while True:
    fps = cap.get(cv.CAP_PROP_FPS)
    success, img = cap.read()
    img = cv.resize(img, (750,600))
    img = detector.findPose(img,draw = False)
    lmlist = detector.findPoints(img)
    # print(lmlist)
    x1 = lmlist[24][1]
    x2 = lmlist[26][1]
    x3 = lmlist[28][1]

    y1 = lmlist[24][2]
    y2 = lmlist[26][2]
    y3 = lmlist[28][2]

    angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))
    print(angle)

    per = np.interp(angle,(60,170),(0,100))
    bar = np.interp(angle,(60,170),(400,100))
    # print(angle,per)
    color = (0,0,255)
    if per == 100:
        color = (0,255,0)
        if dir == 0:
            count += 0.5
            dir = 1
    if per == 0:
        color = (0,255,0)
        if dir == 1:
            count += 0.5
            dir = 0

    # print(count)
    cv.rectangle(img,(650,400),(700,100),color,2)
    cv.rectangle(img,(650,400),(700,int(bar)),color,cv.FILLED)
    cv.putText(img, str(int(per)), (650,450),
               cv.FONT_HERSHEY_PLAIN, 3, color,2)


    cv.rectangle(img,(0,0),(100,100),(255,255,255),cv.FILLED)
    cv.putText(img, str(int(count)), (25, 70),
               cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0),5)

    cv.putText(img, str(int(angle)), (x2+50, y2+50),
               cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0),5)


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

    cv.imshow("video", img)
    cv.waitKey(1)